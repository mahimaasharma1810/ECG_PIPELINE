"""agent_bridge.py — the missing live populator for MedGemma-Agent's `ecg_risk`,
plus a standalone transparent clinical report generator (raw ECG -> RiskReport
JSON -> clinician narrative).

Confirmed via direct inspection of MedGemma-Agent (2026-07-20): the
`ecg_risk` schema field, `escalate_with_ecg`, and the "ECG FINDINGS" prompt
section are all fully wired, but nothing in the live request path (
`api/routes/vitals.py`) ever populates `ecg_risk` — it's a pure pass-through
of whatever the HTTP caller sends. The `push` subcommand below is that
caller.

The `report` subcommand (added 2026-07-24) is a separate, self-contained
flow: raw ECG file -> ECGPipeline (frozen production classifier + risk
cascade, both untouched) -> a fully transparent RiskReport JSON (every
number, every threshold, an ordered rule_trace showing exactly which rule
decided the level) -> an optional MedGemma narrative layer that can only
quote those numbers and escalate, never invent or downgrade. It does not
depend on the MedGemma-Agent FastAPI app at all — only on a MedGemma model
being reachable via Ollama (`call_medgemma` in ecg_pipeline_core.py). If
that's unreachable, it falls back to a deterministic template built
directly from rule_trace, so the report is always complete on its own.

Vitals (for `push`) are simulated: real-vitals pairing is still an open
question (see top-level README "Where this goes next" / project memory), so
that path borrows vitals VALUES from MedGemma-Agent's own
vitals/test_cases.yaml, but the patient_id and timing come from the real ECG
segment -- that's the actual pairing key that subcommand exists to
establish. `report` has no vitals dependency at all.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import requests

from ecg_pipeline.ecg_pipeline_core import (
    AAMI_CLASSES, DATA_RAW, MEDGEMMA_MODEL, MODELS_DIR, OLLAMA_URL, RISK,
    TARGET_FS,
    AuditLog, ECGPipeline, FiveClassBeatClassifier, PipelineResult, Recording,
    RiskReport,
    call_medgemma, discover_vitalpatch_files, merge_decision,
    parse_vitalpatch_ecg, parse_wfdb_record,
    to_agent_ecg_risk_summary,
)

AGENT_DIR = Path(__file__).resolve().parent.parent / "MedGemma-Agent"
DEFAULT_AGENT_URL = "http://localhost:8000/api/v1/vitals/snapshot"
# Matches MedGemma-Agent/.env's API_KEYS_CLINICIAN local-dev default.
DEFAULT_API_KEY = "change-me-clinician-key"

# Report-assembly-time guard only (2026-07-24): below this many analyzed beats,
# the frozen cascade's "no thresholds exceeded" fallback is indistinguishable
# from a genuine clean LOW reading, because it never actually saw enough
# signal to evaluate anything. This floor does NOT touch score_recording(),
# the classifier, or RiskThresholds -- it only decides, after the fact, whether
# the resulting risk_level is trustworthy enough to report as "LOW" at all.
MIN_BEATS_FOR_ASSESSMENT = 5


def load_mock_vitals_case() -> dict:
    if str(AGENT_DIR) not in sys.path:
        sys.path.insert(0, str(AGENT_DIR))
    from vitals.mock_producer import list_test_cases
    return random.choice(list_test_cases())


def load_classifier(classifier_path: Path) -> FiveClassBeatClassifier:
    classifier = FiveClassBeatClassifier()
    classifier.load(classifier_path)
    if not classifier.is_trained:
        raise RuntimeError(f"Failed to load a trained classifier from {classifier_path}")
    return classifier


def run_ecg_segment(recording, classifier: FiveClassBeatClassifier):
    pipeline = ECGPipeline(classifier=classifier)
    result = pipeline.run(recording)
    return to_agent_ecg_risk_summary(result.risk_report), result


def submit_to_agent(patient_id: str, vitals_values: dict, ecg_risk: dict,
                     agent_url: str, api_key: str, timeout: float = 150.0) -> dict:
    payload = {"patient_id": patient_id, **vitals_values, "ecg_risk": ecg_risk}
    resp = requests.post(agent_url, json=payload, headers={"X-API-Key": api_key}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ============================================================================
# Transparent RiskReport builder
#
# Everything below reads already-computed PipelineResult / RiskReport /
# AuditLog fields. It does NOT alter the classifier, the AAMI scheme, the
# risk cascade, or its thresholds (all imported read-only from RISK/
# ecg_pipeline_core.py) -- it only re-renders what score_recording() already
# decided into a fully transparent, ordered trace, so every number and
# threshold in the eventual report is traceable back to that one frozen
# source of truth.
# ============================================================================

# Reporting-layer calibration notes only, from Docs/BEAT_CLASSIFICATION_SUMMARY.md
# DS2 (held-out, 22-patient) numbers for the current production model
# (five_class_xgb.json). Not used by the classifier or cascade -- purely to
# render an honest confidence caveat alongside beat counts.
CLASS_CONFIDENCE_NOTES = {
    "N": {"confidence": "HIGH",
          "note": "Majority class; reliable in practice though not broken out separately in the DS2 macro-F1 table."},
    "S": {"confidence": "LOW",
          "note": "DS2 held-out F1 = 0.139 (production five_class_xgb.json). Frequent S<->N confusion -- "
                  "treat S counts as a screening signal, not a diagnosis."},
    "V": {"confidence": "MODERATE-HIGH",
          "note": "DS2 held-out F1 = 0.826 (production five_class_xgb.json). The most reliable class this "
                  "model produces."},
    "F": {"confidence": "LOW",
          "note": "DS2 held-out F1 = 0.011 (production five_class_xgb.json). Essentially unsolved -- any F "
                  "count should be treated as noise, not a finding."},
    "Q": {"confidence": "N/A",
          "note": "Assigned to beats rejected by the deterministic quality gate, not predicted by the classifier."},
}


def _get_audit_payload(audit: AuditLog, event_type: str) -> dict | None:
    for entry in reversed(audit.to_list()):
        if entry["event_type"] == event_type:
            return entry["payload"]
    return None


def _beat_summary(result: PipelineResult) -> dict:
    labels = result.beat_labels
    n = len(labels)
    summary = {}
    for c in AAMI_CLASSES:
        count = labels.count(c)
        summary[c] = {
            "count": count,
            "pct_of_analyzed_beats": round(100.0 * count / n, 2) if n else 0.0,
            **CLASS_CONFIDENCE_NOTES[c],
        }
    return summary


def _rhythm_findings_json(result: PipelineResult) -> list[dict]:
    beats = result.beats
    out = []
    for f in result.rhythm_findings:
        start_beat = beats[f.start_beat_idx] if f.start_beat_idx < len(beats) else None
        end_beat = beats[f.end_beat_idx] if f.end_beat_idx < len(beats) else None
        start_ms = start_beat.r_peak_ms if start_beat is not None else None
        end_ms = end_beat.r_peak_ms if end_beat is not None else None
        duration_s = round((end_ms - start_ms) / 1000.0, 3) if (start_ms is not None and end_ms is not None) else None

        if f.kind == "VT_RUN":
            evidence_text = (f"{f.detail['beat_count']} consecutive V beats "
                              f"(rhythm-engine threshold: >={RISK.vt_run_beats} consecutive V beats = 1 run)")
        elif f.kind == "BIGEMINY":
            evidence_text = f"{f.detail['repeats']}x repeats of the N-V pattern (threshold: >=4 repeats)"
        elif f.kind == "TRIGEMINY":
            evidence_text = f"{f.detail['repeats']}x repeats of the N-N-V pattern (threshold: >=3 repeats)"
        elif f.kind == "AFIB_SUSPECTED":
            evidence_text = (f"RR coefficient-of-variation {f.detail['rr_cv']:.3f} > 0.15 "
                              f"over a 20-beat rolling window (beats {f.start_beat_idx}-{f.end_beat_idx})")
        else:
            evidence_text = str(f.detail)

        out.append({
            "kind": f.kind,
            "start_beat_idx": f.start_beat_idx,
            "end_beat_idx": f.end_beat_idx,
            "start_time_s": round(start_ms / 1000.0, 3) if start_ms is not None else None,
            "duration_s": duration_s,
            "evidence": dict(f.detail),
            "evidence_text": evidence_text,
        })
    return out


def _build_rule_trace(risk_report: RiskReport, audit: AuditLog,
                       thresholds=RISK) -> list[dict]:
    """Re-renders score_recording()'s ordered if/elif cascade as a full,
    non-short-circuited trace: every rule the cascade *would* consult, in
    the same order, each independently evaluated against the same
    already-computed RiskReport numbers, with `fired` = whether that
    condition alone is true. Because the cascade's own order is preserved
    exactly, the first fired entry here is always the same rule that
    actually decided risk_report.alert_level -- this function only makes
    that visible, it does not re-decide it.
    """
    hrv_payload = _get_audit_payload(audit, "STAGE6_FEATURES") or {}
    sdnn_ms = hrv_payload.get("hrv", {}).get("sdnn_ms", 0.0)

    trace = [
        {
            "condition": "PVC burden > CRITICAL threshold",
            "measured_value": round(risk_report.pvc_burden_pct, 3),
            "threshold": thresholds.pvc_burden_critical_pct,
            "fired": risk_report.pvc_burden_pct > thresholds.pvc_burden_critical_pct,
            "would_set_level": "CRITICAL",
            "depends_on_low_confidence_class": False,
        },
        {
            "condition": f"VT run count > 0 (a run = >={thresholds.vt_run_beats} consecutive V beats)",
            "measured_value": risk_report.vt_run_count,
            "threshold": 0,
            "fired": risk_report.vt_run_count > 0,
            "would_set_level": "CRITICAL",
            "depends_on_low_confidence_class": False,
        },
        {
            "condition": "PVC burden > HIGH threshold",
            "measured_value": round(risk_report.pvc_burden_pct, 3),
            "threshold": thresholds.pvc_burden_high_pct,
            "fired": risk_report.pvc_burden_pct > thresholds.pvc_burden_high_pct,
            "would_set_level": "HIGH",
            "depends_on_low_confidence_class": False,
        },
        {
            "condition": "PAC burden > HIGH threshold OR AFib burden > HIGH threshold",
            "measured_value": {"pac_burden_pct": round(risk_report.pac_burden_pct, 3),
                                "afib_burden_pct": round(risk_report.afib_burden_pct, 3)},
            "threshold": {"pac_burden_pct": thresholds.pac_burden_high_pct,
                          "afib_burden_pct": thresholds.afib_burden_high_pct},
            "fired": (risk_report.pac_burden_pct > thresholds.pac_burden_high_pct
                      or risk_report.afib_burden_pct > thresholds.afib_burden_high_pct),
            "would_set_level": "HIGH",
            "depends_on_low_confidence_class": risk_report.pac_burden_pct > thresholds.pac_burden_high_pct,
        },
        {
            "condition": "Sustained HRV suppression: SDNN < threshold",
            "measured_value": round(sdnn_ms, 3),
            "threshold": thresholds.hrv_sdnn_suppressed_ms,
            "fired": risk_report.hrv_suppressed,
            "would_set_level": "MEDIUM",
            "depends_on_low_confidence_class": False,
        },
    ]

    deciding = next((rule for rule in trace if rule["fired"]), None)
    baseline_level = deciding["would_set_level"] if deciding else "LOW"

    news2_rule = {
        "condition": "NEWS2 safety override (>= critical threshold)",
        "measured_value": risk_report.news2_score,
        "threshold": thresholds.news2_critical_threshold,
        "fired": (risk_report.news2_score is not None
                  and risk_report.news2_score >= thresholds.news2_critical_threshold
                  and baseline_level != "CRITICAL"),
        "would_set_level": "CRITICAL",
        "depends_on_low_confidence_class": False,
        "evaluated": risk_report.news2_score is not None,
    }
    trace.append(news2_rule)
    if news2_rule["fired"]:
        baseline_level = "CRITICAL"
        deciding = deciding or news2_rule

    qsofa_rule = {
        "condition": "qSOFA safety override (>= high threshold)",
        "measured_value": risk_report.qsofa_score,
        "threshold": thresholds.qsofa_high_threshold,
        "fired": (risk_report.qsofa_score is not None
                  and risk_report.qsofa_score >= thresholds.qsofa_high_threshold
                  and baseline_level not in ("HIGH", "CRITICAL")),
        "would_set_level": "HIGH",
        "depends_on_low_confidence_class": False,
        "evaluated": risk_report.qsofa_score is not None,
    }
    trace.append(qsofa_rule)
    if qsofa_rule["fired"]:
        baseline_level = "HIGH"
        deciding = deciding or qsofa_rule

    if deciding is None:
        trace.append({
            "condition": "No thresholds exceeded",
            "measured_value": None, "threshold": None, "fired": True,
            "would_set_level": "LOW", "depends_on_low_confidence_class": False,
        })
        deciding = trace[-1]
        baseline_level = "LOW"

    for rule in trace:
        rule["is_deciding_rule"] = rule is deciding

    if baseline_level != risk_report.alert_level:
        trace.append({
            "condition": "INTERNAL CONSISTENCY WARNING",
            "measured_value": baseline_level, "threshold": risk_report.alert_level,
            "fired": True, "would_set_level": None, "is_deciding_rule": False,
            "note": ("Reconstructed trace level does not match RiskReport.alert_level -- "
                     "report this immediately, do not trust risk_level below."),
        })

    return trace


def _confidence_statement(deciding_rule: dict) -> dict:
    depends_on_s_or_f = bool(deciding_rule.get("depends_on_low_confidence_class"))
    if depends_on_s_or_f:
        tier = "LOW"
        statement = ("The deciding rule depends on S (supraventricular) beat counts. The production "
                      "classifier's held-out S-class F1 is 0.139 -- this level should be treated as a "
                      "screening flag, not a reliable diagnosis, until confirmed by clinician review.")
    elif "VT run" in deciding_rule["condition"] or "PVC" in deciding_rule["condition"]:
        tier = "MODERATE-HIGH"
        statement = ("The deciding rule depends on V (ventricular) beat counts. The production classifier's "
                      "held-out V-class F1 is 0.826 -- the most reliable class this model produces, though "
                      "still not a clinical-grade guarantee.")
    elif "HRV" in deciding_rule["condition"] or "SDNN" in deciding_rule["condition"]:
        tier = "MODERATE"
        statement = ("The deciding rule is an RR-interval timing metric (SDNN), independent of the beat "
                      "classifier's per-class accuracy, but the suppression threshold itself is a fixed "
                      "conservative cutoff, not a calibrated probability.")
    elif "AFib" in deciding_rule["condition"]:
        tier = "MODERATE"
        statement = ("AFib-suspected findings are computed from RR-interval irregularity directly, not from "
                      "the beat classifier's S/F labels, so this is not subject to the S/F low-confidence "
                      "caveat -- but it is a heuristic (RR coefficient-of-variation), not a rhythm-strip "
                      "diagnosis.")
    else:
        tier = "BASELINE"
        statement = ("No threshold was exceeded (LOW). Because the S and F classes are the model's weakest "
                      "(held-out F1 0.139 and 0.011), a LOW read here does not rule out under-counted S/F "
                      "events -- it reliably rules out V-burden, VT runs, and HRV suppression, which this "
                      "model detects well.")
    return {
        "tier": tier,
        "statement": statement,
        "caveat": ("This is a proxy/heuristic confidence assessment produced by this reporting layer, not a "
                   "statistically calibrated probability (no conformal calibration set has been loaded for "
                   "this run) and not a clinical guarantee."),
    }


def _safety_overrides_json(risk_report: RiskReport, rule_trace: list[dict]) -> list[dict]:
    overrides = []
    for name, key in [("NEWS2", "NEWS2 safety override (>= critical threshold)"),
                       ("qSOFA", "qSOFA safety override (>= high threshold)")]:
        rule = next(r for r in rule_trace if r["condition"] == key)
        if not rule.get("evaluated", True):
            overrides.append({"override": name, "applied": False,
                               "note": f"No {name} score was supplied to this run -- override not evaluated. "
                                       f"This bridge is ECG-only; vitals/NEWS2/qSOFA pairing is a separate, "
                                       f"open integration question (see project memory)."})
        else:
            overrides.append({"override": name, "score": rule["measured_value"], "threshold": rule["threshold"],
                               "applied": rule["fired"]})
    return overrides


def build_risk_report_json(result: PipelineResult) -> dict:
    r = result.recording
    duration_s = (float(r.timestamps_ms[-1] - r.timestamps_ms[0]) / 1000.0) if len(r.timestamps_ms) > 1 else 0.0
    n_quality_rejected_beats = sum(1 for b in result.beats if b.quality_rejected)

    rule_trace = _build_rule_trace(result.risk_report, result.audit)
    deciding_rule = next(r for r in rule_trace if r.get("is_deciding_rule"))

    n_beats_analyzed = result.n_beats_accepted
    assessable = n_beats_analyzed >= MIN_BEATS_FOR_ASSESSMENT
    risk_level = result.risk_report.alert_level if assessable else "NOT_ASSESSABLE"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recording": {
            "source": r.source,
            "patient_id": r.patient_id,
            "segment_id": r.segment_id,
            "duration_s": round(duration_s, 2),
            "n_raw_samples": result.n_raw_samples,
            "fs_nominal_hz": r.fs_nominal,
            "fs_processed_hz": TARGET_FS,
            "quality_score": round(1.0 - result.sqi_rejection_rate, 4),
            "sqi_window_rejection_rate": round(result.sqi_rejection_rate, 4),
            "n_beats_detected": len(result.beats),
            "n_beats_analyzed": n_beats_analyzed,
            "n_beats_flagged_low_quality": n_quality_rejected_beats,
        },
        "beat_summary": _beat_summary(result),
        "rhythm_findings": _rhythm_findings_json(result),
        "assessable": assessable,
        "risk_level": risk_level,
        "rule_trace": rule_trace,
        "deciding_rule": deciding_rule,
        "confidence": _confidence_statement(deciding_rule),
        "safety_overrides": _safety_overrides_json(result.risk_report, rule_trace),
        "known_limitations": [
            "Beat classifier is the frozen production XGBoost model (five_class_xgb.json). DS2 held-out "
            "per-class F1: V=0.826 (reliable), S=0.139 (LOW-CONFIDENCE), F=0.011 (LOW-CONFIDENCE, essentially "
            "unsolved). Any finding driven primarily by S or F counts is a screening flag, not a diagnosis.",
            "NEWS2/qSOFA vitals pairing is not wired into this ECG-only bridge -- safety_overrides above are "
            "reported as not-evaluated rather than fabricated.",
        ],
    }

    if not assessable:
        report["not_assessable_reason"] = (
            f"Only {n_beats_analyzed} beat(s) survived quality gating / resampling "
            f"(floor: {MIN_BEATS_FOR_ASSESSMENT}). The underlying risk cascade's alert_level "
            f"({result.risk_report.alert_level!r}) reflects its 'no thresholds exceeded' fallback on "
            f"empty/near-empty input, NOT a genuine clean reading -- there was not enough signal to "
            f"assess. sqi_window_rejection_rate={round(result.sqi_rejection_rate, 4)}, "
            f"quality_score={round(1.0 - result.sqi_rejection_rate, 4)}."
        )
        report["known_limitations"].append(
            "This report is NOT_ASSESSABLE: too few beats survived the quality gate to draw any "
            "conclusion. Do not treat this as a clean/LOW result -- treat it as missing data and "
            "investigate signal quality at the source."
        )

    return report


# ============================================================================
# MedGemma narrative layer -- escalate-only, CRITICAL-bypass, deterministic
# fallback. Reuses call_medgemma() and merge_decision() from
# ecg_pipeline_core.py verbatim (same Ollama endpoint, same escalate-only /
# disagreement-rejection / CRITICAL-bypass safety logic already used by the
# pipeline's own Stage 9) -- this module only supplies a different prompt,
# one built strictly from the transparency JSON above instead of the
# trend/similar-cases prompt ECGPipeline.run() uses internally.
# ============================================================================

TRANSPARENCY_PROMPT_TEMPLATE = """You are rendering an ALREADY-DECIDED, deterministic ECG risk assessment \
into a clinician-facing narrative. A rule cascade computed the risk level below BEFORE you were called. \
You do not decide it. You may only ESCALATE it, never lower it.

STRICT RULES:
1. Use ONLY numbers, counts, and thresholds that appear in the JSON below. Never invent, estimate, or \
   recompute a number that isn't already there.
2. Quote the "deciding_rule" verbatim: its condition, measured_value, and threshold.
3. List supporting findings using ONLY numbers from rule_trace / rhythm_findings / beat_summary.
4. State the "confidence" field's statement and, if beat_summary marks S or F as LOW confidence and the \
   deciding rule depends on them, say so explicitly.
5. Frame any next step as decision-support ("clinician review suggested"), never as a diagnosis or a \
   treatment instruction.
6. If you believe the level should be raised given the evidence, say so explicitly via "escalate": true \
   and a reason. You may never set "escalate" to lower the level.

## Structured RiskReport JSON (the only source of numbers you may use)
{report_json}

## Required output format
Give brief reasoning, then a final JSON object of the form:
{{"risk_level": "LOW|MEDIUM|HIGH|CRITICAL", "narrative": "...", "disclaimer": "...", \
"escalate": false, "escalate_reason": null}}
"""


def build_transparency_prompt(report_json: dict) -> str:
    return TRANSPARENCY_PROMPT_TEMPLATE.format(report_json=json.dumps(report_json, indent=2, default=str))


def _deterministic_narrative(report_json: dict, header: str = "") -> str:
    deciding = report_json["deciding_rule"]
    lines = [header] if header else []
    lines.append(f"Risk level: {report_json['risk_level']}.")
    lines.append(f"Deciding rule: {deciding['condition']} -- measured {deciding['measured_value']} "
                 f"vs threshold {deciding['threshold']} (fired={deciding['fired']}).")
    other_fired = [r for r in report_json["rule_trace"]
                   if r.get("fired") and not r.get("is_deciding_rule") and r["condition"] != deciding["condition"]]
    if other_fired:
        lines.append("Other rules that also fired: " + "; ".join(
            f"{r['condition']} ({r['measured_value']} vs {r['threshold']})" for r in other_fired))
    lines.append(f"Confidence ({report_json['confidence']['tier']}): {report_json['confidence']['statement']}")
    for lim in report_json["known_limitations"]:
        lines.append(f"Limitation: {lim}")
    lines.append("Clinician review suggested -- this is decision support, not a diagnosis.")
    return "\n".join(lines)


def render_narrative(report_json: dict, risk_report: RiskReport, audit: AuditLog) -> dict:
    """Returns {"narrative", "medgemma_status", "final_risk_level"}.

    medgemma_status is one of: SKIPPED_NOT_ASSESSABLE | SKIPPED_CRITICAL |
    UNAVAILABLE_FALLBACK | REJECTED_FALLBACK | ACCEPTED -- always logged so
    it's clear whether the LLM layer ran or fell back (per the task's
    ROBUSTNESS requirement).
    """
    if report_json["risk_level"] == "NOT_ASSESSABLE":
        audit.append("TRANSPARENCY_MEDGEMMA_SKIPPED_NOT_ASSESSABLE",
                      {"reason": report_json.get("not_assessable_reason", "insufficient beats analyzed")})
        narrative = _deterministic_narrative(
            report_json, header="*** NOT ASSESSABLE -- insufficient signal survived quality gating; "
                                 "this is missing data, not a clean reading ***")
        return {"narrative": narrative, "medgemma_status": "SKIPPED_NOT_ASSESSABLE",
                "final_risk_level": "NOT_ASSESSABLE", "endpoint": OLLAMA_URL, "model": MEDGEMMA_MODEL}

    if report_json["risk_level"] == "CRITICAL":
        audit.append("TRANSPARENCY_MEDGEMMA_SKIPPED_CRITICAL",
                      {"reason": "CRITICAL level bypasses MedGemma entirely, per safety constraint"})
        narrative = _deterministic_narrative(
            report_json, header="*** CRITICAL -- IMMEDIATE CLINICIAN REVIEW REQUIRED ***")
        return {"narrative": narrative, "medgemma_status": "SKIPPED_CRITICAL",
                "final_risk_level": "CRITICAL", "endpoint": OLLAMA_URL, "model": MEDGEMMA_MODEL}

    prompt = build_transparency_prompt(report_json)
    llm_output = call_medgemma(prompt, model=MEDGEMMA_MODEL)
    merged = merge_decision(risk_report, llm_output, audit)

    if llm_output is None:
        return {"narrative": _deterministic_narrative(report_json),
                "medgemma_status": "UNAVAILABLE_FALLBACK",
                "final_risk_level": merged.final_decision["risk_level"],
                "endpoint": OLLAMA_URL, "model": MEDGEMMA_MODEL}

    if merged.llm_rejected_reason:
        narrative = _deterministic_narrative(report_json) + (
            f"\n(MedGemma output rejected: {merged.llm_rejected_reason} -- deterministic fallback used instead.)")
        return {"narrative": narrative, "medgemma_status": f"REJECTED_FALLBACK:{merged.llm_rejected_reason}",
                "final_risk_level": merged.final_decision["risk_level"],
                "endpoint": OLLAMA_URL, "model": MEDGEMMA_MODEL}

    narrative = merged.final_decision.get("narrative") or _deterministic_narrative(report_json)
    return {"narrative": narrative, "medgemma_status": "ACCEPTED",
            "final_risk_level": merged.final_decision["risk_level"],
            "endpoint": OLLAMA_URL, "model": MEDGEMMA_MODEL}


def run_full_report(recording: Recording, classifier: FiveClassBeatClassifier) -> tuple[dict, PipelineResult]:
    """The one callable: a Recording -> a complete, transparent RiskReport
    dict (JSON-serializable) with the narrative + MedGemma status attached.
    """
    pipeline = ECGPipeline(classifier=classifier)
    result = pipeline.run(recording)
    report_json = build_risk_report_json(result)
    render = render_narrative(report_json, result.risk_report, result.audit)
    report_json["narrative"] = render["narrative"]
    report_json["medgemma"] = {"status": render["medgemma_status"], "endpoint": render["endpoint"],
                                "model": render["model"]}
    report_json["final_risk_level"] = render["final_risk_level"]
    return report_json, result


# ============================================================================
# CLI
# ============================================================================

def _load_recordings(source: str, path: Path) -> list[Recording]:
    if source == "vitalpatch":
        return parse_vitalpatch_ecg(path)
    if source == "wfdb":
        # ann_extension left at its default ("atr") purely so meta["beat_symbols"]
        # is available for the --verify-against-annotations cross-check below;
        # it is NEVER used to classify -- classification always goes through the
        # frozen production classifier via ECGPipeline, same as any other source.
        return [parse_wfdb_record(path)]
    raise ValueError(f"Unknown source: {source}")


def _verify_against_annotations(recording: Recording, result: PipelineResult) -> dict | None:
    """Sanity check only (not part of the report): for WFDB records with
    ground-truth annotations available, compare the raw V-beat count our
    pipeline produced against len(labels)-independent ground truth, so a
    reviewer can see rule_trace's PVC numbers aren't fabricated.
    """
    symbols = recording.meta.get("beat_symbols")
    if not symbols:
        return None
    from ecg_pipeline.ecg_pipeline_tools import AAMI_SYMBOL_MAP
    gt_v = sum(1 for s in symbols if AAMI_SYMBOL_MAP.get(s) == "V")
    pred_v = result.beat_labels.count("V")
    return {"ground_truth_annotation_V_count": gt_v, "pipeline_predicted_V_count": pred_v,
            "pipeline_n_beats_analyzed": len(result.beat_labels)}


def report_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="raw ECG file -> transparent RiskReport JSON + narrative")
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--source", choices=["vitalpatch", "wfdb"], required=True)
    parser.add_argument("--classifier", type=Path, default=MODELS_DIR / "five_class_xgb.json")
    parser.add_argument("--limit", type=int, default=1, help="max segments/recordings to process")
    parser.add_argument("--out-dir", type=Path, default=None, help="save each report JSON here")
    args = parser.parse_args(argv)

    classifier = load_classifier(args.classifier)
    recordings = _load_recordings(args.source, args.file)

    for recording in recordings[:args.limit]:
        report_json, result = run_full_report(recording, classifier)
        verify = _verify_against_annotations(recording, result)
        if verify:
            report_json["_annotation_cross_check"] = verify

        print("=" * 80)
        print(f"{recording.source} / {recording.patient_id} / {recording.segment_id}")
        print("=" * 80)
        print(json.dumps(report_json, indent=2, default=str))
        print("\n--- NARRATIVE ---")
        print(report_json["narrative"])

        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.out_dir / f"{recording.source}_{recording.patient_id}_{recording.segment_id}.json"
            out_path.write_text(json.dumps(report_json, indent=2, default=str))
            print(f"\n[saved -> {out_path}]")


def push_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vitalpatch-root", type=Path, default=DATA_RAW / "vitalpatch")
    parser.add_argument("--classifier", type=Path, default=MODELS_DIR / "five_class_xgb.json")
    parser.add_argument("--agent-url", default=DEFAULT_AGENT_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--limit", type=int, default=1, help="number of ECG segments to process")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    random.seed(args.seed)

    classifier = load_classifier(args.classifier)
    print(f"Loaded trained classifier from {args.classifier} "
          f"(classes: {list(classifier._label_encoder.classes_)})")

    files = discover_vitalpatch_files(args.vitalpatch_root)
    if not files:
        raise SystemExit(f"No VitalPatch ECG files found under {args.vitalpatch_root}")

    n_done = 0
    for f in files:
        if n_done >= args.limit:
            break
        for recording in parse_vitalpatch_ecg(f):
            if n_done >= args.limit:
                break
            ecg_risk, result = run_ecg_segment(recording, classifier)
            mock_case = load_mock_vitals_case()
            response = submit_to_agent(recording.patient_id, mock_case["vitals"], ecg_risk,
                                        args.agent_url, args.api_key)
            print(json.dumps({
                "segment_id": recording.segment_id,
                "patient_id": recording.patient_id,
                "n_beats": len(result.beats),
                "ecg_risk": ecg_risk,
                "vitals_source_case": mock_case["id"],
                "agent_response": response,
            }, indent=2, default=str))
            n_done += 1


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("push", "report"):
        mode, rest = sys.argv[1], sys.argv[2:]
    else:
        mode, rest = "push", sys.argv[1:]  # unchanged default behavior for existing callers

    if mode == "report":
        report_main(rest)
    else:
        push_main(rest)


if __name__ == "__main__":
    main()
