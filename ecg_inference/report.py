"""ecg_inference.report -- Stage 9: the transparent structured RiskReport
JSON builder, plus the optional deterministic/MedGemma narrative layer.

Extracted verbatim from two sources:
  - ecg_pipeline_core.py's similar_cases.py and report.py sections
    (SimilarCaseIndex, call_medgemma, merge_decision, generate_report --
    the machinery ECGPipeline.run() itself uses for its internal Stage 9
    call).
  - ecg_pipeline/agent_bridge.py's report-builder functions
    (build_risk_report_json and everything it depends on, plus the
    deterministic/MedGemma narrative renderer) -- this is the actual
    "transparent RiskReport JSON" the rest of the codebase (and this
    package's run_inference.py) uses; it reads only already-computed
    PipelineResult/RiskReport/AuditLog fields and never touches the
    classifier, the AAMI scheme, or any cascade threshold (all imported
    read-only from .preprocess / .classifier).

Excluded from the original agent_bridge.py (not needed for inference,
and the excluded parts are what would have pulled in a training-adjacent
import): `load_mock_vitals_case`, `submit_to_agent`, `push_main` (a
separate live-agent-vitals integration, unrelated to producing a RiskReport
for one ECG file) and `_verify_against_annotations`/`report_main` (the
original CLI's ground-truth-annotation cross-check, which is the ONE
place agent_bridge.py imported from ecg_pipeline_tools -- a training-side
module -- confirming this package needs none of that to run inference).
`run_full_report` (recording -> full report+narrative in one call) is not
reproduced here either: it would need to import ECGPipeline from
.pipeline, which itself imports generate_report from this module --
run_inference.py performs that same handful of calls directly instead of
introducing a circular import for a two-line convenience wrapper.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

from .classifier import FiveClassBeatClassifier, RiskReport
from .preprocess import AAMI_CLASSES, AuditLog, RISK, RISK_LEVELS, TARGET_FS



@dataclass
class CaseRecord:
    case_id: str
    patient_id: str
    embedding: np.ndarray
    outcome_label: str          # e.g. final alert_level for that recording/beat
    metadata: dict = field(default_factory=dict)


class SimilarCaseIndex:
    def __init__(self):
        self._records: list[CaseRecord] = []
        self._nn = None

    def add(self, record: CaseRecord) -> None:
        self._records.append(record)
        self._nn = None  # invalidate, rebuild lazily

    def add_many(self, records: list[CaseRecord]) -> None:
        self._records.extend(records)
        self._nn = None

    def _ensure_index(self):
        from sklearn.neighbors import NearestNeighbors
        if self._nn is None and self._records:
            X = np.vstack([r.embedding for r in self._records])
            self._nn = NearestNeighbors(n_neighbors=min(5, len(self._records)), metric="cosine").fit(X)

    def query(self, embedding: np.ndarray, k: int = 5) -> list[tuple[CaseRecord, float]]:
        self._ensure_index()
        if self._nn is None:
            return []
        k = min(k, len(self._records))
        dist, idx = self._nn.kneighbors(embedding.reshape(1, -1), n_neighbors=k)
        return [(self._records[i], float(d)) for i, d in zip(idx[0], dist[0])]

    def __len__(self) -> int:
        return len(self._records)

    def save(self, path: Path) -> None:
        import pickle
        Path(path).write_bytes(pickle.dumps(self._records))

    def load(self, path: Path) -> None:
        import pickle
        self._records = pickle.loads(Path(path).read_bytes())
        self._nn = None


OLLAMA_URL = "http://localhost:11434/api/generate"
MEDGEMMA_MODEL = "medgemma"
CRITICAL_LATENCY_TARGET_MS = 500


PROMPT_TEMPLATE = """You are assisting clinical staff monitoring a post-operative patient's ECG.

## Deterministic ECG risk summary
{ecg_summary}

## Rolling risk trend (last {trend_window} minutes)
{trend_summary}

## Similar past cases (nearest neighbours by waveform embedding)
{similar_cases_summary}

## EHR context
{ehr_context}

## Instructions
Think through this step by step BEFORE giving your final answer:
1. Summarize what the deterministic ECG metrics indicate.
2. Note any disagreement between the metrics and the similar past cases.
3. State your reasoning for a risk level.
4. Only then give the final answer.

You may only RAISE the deterministic risk level below, never lower it:
  deterministic_alert_level = {deterministic_level}

Respond with your step-by-step reasoning, then a final JSON object of the form:
{{"risk_level": "LOW|MEDIUM|HIGH|CRITICAL", "narrative": "...", "disclaimer": "..."}}
"""


@dataclass
class MergedDecision:
    deterministic_decision: dict
    llm_decision: dict | None
    llm_rejected_reason: str | None
    final_decision: dict
    bypassed_llm: bool


def build_prompt(risk_report: RiskReport, trend: dict, similar_cases_summary: str,
                  ehr_context: str = "Not available", trend_window_minutes: float = 15.0) -> str:
    ecg_summary = (
        f"PVC burden: {risk_report.pvc_burden_pct:.1f}% | PAC burden: {risk_report.pac_burden_pct:.1f}% | "
        f"VT runs: {risk_report.vt_run_count} | AFib burden: {risk_report.afib_burden_pct:.1f}% | "
        f"HRV suppressed: {risk_report.hrv_suppressed} | QRS width trend: {risk_report.qrs_width_trend:.2f} | "
        f"NEWS2: {risk_report.news2_score} | qSOFA: {risk_report.qsofa_score}"
    )
    trend_summary = (
        f"Slope: {trend.get('slope_per_minute', 0.0):.3f} risk-levels/min | "
        f"Worsening: {trend.get('worsening', False)} | Snapshots: {trend.get('n_snapshots', 0)}"
    )
    return PROMPT_TEMPLATE.format(
        ecg_summary=ecg_summary, trend_window=trend_window_minutes, trend_summary=trend_summary,
        similar_cases_summary=similar_cases_summary, ehr_context=ehr_context,
        deterministic_level=risk_report.alert_level,
    )


def call_medgemma(prompt: str, model: str = MEDGEMMA_MODEL, timeout_s: float = 10.0) -> dict | None:
    """Calls a locally-deployed MedGemma via Ollama's HTTP API. Returns
    None (not raises) if Ollama isn't reachable, so the pipeline degrades
    to rule-based-only mode instead of crashing — this is itself logged
    to the audit trail by the caller."""
    try:
        resp = requests.post(OLLAMA_URL, json={"model": model, "prompt": prompt, "stream": False},
                              timeout=timeout_s)
        resp.raise_for_status()
        text = resp.json().get("response", "")
    except (requests.RequestException, ValueError):
        return None

    try:
        json_start = text.rindex("{")
        # raw_decode stops at the JSON value's matching closing brace instead of
        # requiring the rest of the string to parse too -- MedGemma wraps its
        # JSON answer in a ```json ... ``` markdown fence, and a naive
        # json.loads(text[json_start:]) chokes on the trailing ``` as "Extra data".
        parsed, _end = json.JSONDecoder().raw_decode(text, json_start)
        parsed["_raw_reasoning"] = text[:json_start].strip()
        return parsed
    except (ValueError, json.JSONDecodeError):
        return None


def merge_decision(risk_report: RiskReport, llm_output: dict | None, audit: AuditLog) -> MergedDecision:
    deterministic = {"risk_level": risk_report.alert_level, "reasons": risk_report.alert_reasons}

    if risk_report.alert_level == "CRITICAL":
        audit.append("MEDGEMMA_BYPASSED", {"reason": "CRITICAL alert bypasses LLM",
                                            "latency_target_ms": CRITICAL_LATENCY_TARGET_MS})
        return MergedDecision(deterministic, None, None, deterministic, bypassed_llm=True)

    if llm_output is None:
        audit.append("MEDGEMMA_UNAVAILABLE", {"fallback": "rule_based_only"})
        return MergedDecision(deterministic, None, "MedGemma unavailable", deterministic, bypassed_llm=False)

    llm_level = llm_output.get("risk_level")
    if llm_level not in RISK_LEVELS:
        audit.append("MEDGEMMA_REJECTED", {"reason": "invalid risk_level in response", "raw": llm_output})
        return MergedDecision(deterministic, llm_output, "invalid risk_level", deterministic, bypassed_llm=False)

    det_idx, llm_idx = RISK_LEVELS.index(risk_report.alert_level), RISK_LEVELS.index(llm_level)

    if abs(llm_idx - det_idx) > 1:
        audit.append("MEDGEMMA_REJECTED", {"reason": "disagreement > 1 severity level",
                                            "deterministic": risk_report.alert_level, "llm": llm_level})
        return MergedDecision(deterministic, llm_output, "disagreement > 1 severity level", deterministic,
                               bypassed_llm=False)

    final_level = RISK_LEVELS[max(det_idx, llm_idx)]  # LLM may only raise, never lower
    final = {"risk_level": final_level, "narrative": llm_output.get("narrative", ""),
              "disclaimer": llm_output.get("disclaimer", "This is not a substitute for clinical judgement."),
              "reasoning": llm_output.get("_raw_reasoning", "")}
    audit.append("MEDGEMMA_ACCEPTED", {"deterministic": risk_report.alert_level, "llm": llm_level,
                                        "final": final_level})
    return MergedDecision(deterministic, llm_output, None, final, bypassed_llm=False)


def generate_report(risk_report: RiskReport, trend: dict, similar_cases_summary: str,
                     audit: AuditLog, ehr_context: str = "Not available") -> MergedDecision:
    if risk_report.alert_level == "CRITICAL":
        return merge_decision(risk_report, None, audit)

    prompt = build_prompt(risk_report, trend, similar_cases_summary, ehr_context)
    audit.append("MEDGEMMA_PROMPT_BUILT", {"prompt_chars": len(prompt)})
    llm_output = call_medgemma(prompt)
    return merge_decision(risk_report, llm_output, audit)




# ============================================================================
# Transparent RiskReport builder (from agent_bridge.py)
# ============================================================================

# Report-assembly-time guard only (2026-07-24): below this many analyzed beats,
# the frozen cascade's "no thresholds exceeded" fallback is indistinguishable
# from a genuine clean LOW reading, because it never actually saw enough
# signal to evaluate anything. This floor does NOT touch score_recording(),
# the classifier, or RiskThresholds -- it only decides, after the fact, whether
# the resulting risk_level is trustworthy enough to report as "LOW" at all.
MIN_BEATS_FOR_ASSESSMENT = 5


def load_classifier(classifier_path: Path) -> FiveClassBeatClassifier:
    classifier = FiveClassBeatClassifier()
    classifier.load(classifier_path)
    if not classifier.is_trained:
        raise RuntimeError(f"Failed to load a trained classifier from {classifier_path}")
    return classifier


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
            t0 = round(start_ms / 1000.0, 1) if start_ms is not None else None
            t1 = round(end_ms / 1000.0, 1) if end_ms is not None else None
            burden_pct = result.risk_report.afib_burden_pct
            burden_thresh = RISK.afib_burden_high_pct
            # AFIB_SUSPECTED is NOT informational-only in the real cascade -- it feeds
            # afib_burden_pct (fraction of examined 20-beat windows flagged), which DOES
            # raise risk to HIGH once it exceeds afib_burden_high_pct (see score_recording's
            # "PAC burden > HIGH OR AFib burden > HIGH" rule). A single flagged window like
            # this one genuinely does not cross that bar on its own -- stated explicitly here
            # so a reader doesn't read "AFIB_SUSPECTED" + "risk LOW" as contradictory.
            raised_risk = burden_pct > burden_thresh
            effect = ("this crosses the burden threshold and raises risk to HIGH." if raised_risk else
                      "flagged for clinician review; on its own, one flagged window does not raise the risk level.")
            evidence_text = (
                f"RR coefficient-of-variation {f.detail['rr_cv']:.3f} > 0.15 over a 20-beat rolling window "
                f"(beats {f.start_beat_idx}-{f.end_beat_idx}, t={t0}-{t1}s). "
                f"AFib burden this recording: {burden_pct:.1f}% of examined rolling windows "
                f"(risk-raising threshold: >{burden_thresh:.1f}%) -- {effect}"
            )
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
1. Use ONLY numbers, counts, and thresholds that appear below. Never invent, estimate, or recompute a \
   number that isn't already there.
2. Every rule below already has its comparison verdict computed for you ("EXCEEDS" or "does NOT exceed"). \
   Copy that verdict verbatim. NEVER decide for yourself whether a measured value crosses a threshold, and \
   NEVER say a value "exceeds"/"crosses" its threshold unless the line below already says EXCEEDS. If a \
   line says "does NOT exceed", you must not describe it as exceeding, triggering, or crossing anything.
3. Quote the deciding rule's condition and its already-computed verdict, not your own comparison.
4. If beat_summary marks a class (S or F) as LOW confidence and the deciding rule depends on it, say so \
   explicitly.
5. Frame any next step as decision-support ("clinician review suggested"), never as a diagnosis or a \
   treatment instruction.
6. If you believe the level should be raised given the evidence, say so explicitly via "escalate": true \
   and a reason. You may never set "escalate" to lower the level.

## Beat classification summary (counts already computed -- use these numbers)
{beat_summary_block}

## Rhythm findings (already detected -- use these as-is)
{rhythm_findings_block}

## Rule cascade status (verdict already computed for every rule -- copy verbatim, do not recompute)
{rule_status_block}

## Deciding rule (this is the one that set the risk level)
{deciding_rule_block}

## Final deterministic risk level
{risk_level}

## Required narrative structure -- cover ALL five points below, in this order
(a) Beat summary: state each class's count/percentage from the block above, including the S/F \
    low-confidence caveat if either appears.
(b) Rhythm findings: describe each finding above (or state plainly that none were found).
(c) Deciding rule: state its real value(s) vs threshold(s) and its verdict, copied verbatim from above.
(d) Final risk level: state it plainly.
(e) End with exactly: "Clinician review suggested -- this is decision support, not a diagnosis."

## Full structured RiskReport JSON (reference only, for any additional detail -- same numbers as above)
{report_json}

## Required output format
Give brief reasoning, then a final JSON object of the form:
{{"risk_level": "LOW|MEDIUM|HIGH|CRITICAL", "narrative": "...", "disclaimer": "...", \
"escalate": false, "escalate_reason": null}}
"""


def _fmt_num(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return str(round(v, 3))
    return str(v)


def _deciding_rule_sentence(rule: dict) -> str:
    """Plain, reader-facing summary of the deciding rule -- unlike
    _rule_verdict_line (dev-facing, feeds the MedGemma prompt), this never
    prints a raw "None"/"null" for the catch-all "no thresholds exceeded"
    rule (which genuinely has no measured_value/threshold -- it's what's
    left when nothing else fired) and never labels that catch-all "FIRED"/
    "fired=True", which would misleadingly imply a real threshold was
    crossed."""
    condition = rule["condition"]
    measured = rule.get("measured_value")
    threshold = rule.get("threshold")
    if measured is None and threshold is None:
        return f"No dangerous thresholds exceeded -> risk {rule.get('would_set_level') or 'LOW'}."
    fired = rule.get("fired")
    if isinstance(measured, dict) and isinstance(threshold, dict):
        parts = [f"{k}={_fmt_num(m)} (threshold {_fmt_num(threshold.get(k))})"
                 for k, m in measured.items()
                 if isinstance(m, (int, float)) and isinstance(threshold.get(k), (int, float))]
        verdict = "EXCEEDED" if fired else "not exceeded"
        return f"{condition}: {', '.join(parts)} -> {verdict}."
    if isinstance(measured, (int, float)) and isinstance(threshold, (int, float)):
        verdict = "EXCEEDED" if fired else "not exceeded"
        return f"{condition}: measured {_fmt_num(measured)} vs threshold {_fmt_num(threshold)} -> {verdict}."
    return f"{condition}."


def _rule_verdict_line(rule: dict) -> str:
    """Pre-computes the exceeds/does-not-exceed verdict in Python (matching
    score_recording's own strict `>` comparisons exactly) so MedGemma only
    has to copy a verdict, never infer one -- this is what closes the
    "PAC burden 8.642 exceeds 15.0" defect: the model was doing its own
    (wrong) arithmetic because the prompt only gave it raw numbers."""
    condition = rule["condition"]
    fired = rule.get("fired")
    measured = rule.get("measured_value")
    threshold = rule.get("threshold")
    if rule.get("evaluated") is False:
        return f"- {condition}: NOT EVALUATED (no input score supplied) -> fired=False"
    if isinstance(measured, dict) and isinstance(threshold, dict):
        parts = []
        for k, m in measured.items():
            t = threshold.get(k)
            if not isinstance(m, (int, float)) or not isinstance(t, (int, float)):
                continue
            verdict = "EXCEEDS" if m > t else "does NOT exceed"
            parts.append(f"{k}={_fmt_num(m)} vs its threshold {_fmt_num(t)} -> {verdict}")
        detail = "; ".join(parts)
        overall = "EXCEEDS (fired=True)" if fired else "does NOT exceed (fired=False)"
        return f"- {condition}:\n    {detail}\n    overall -> {overall}"
    if isinstance(measured, (int, float)) and isinstance(threshold, (int, float)):
        verdict = "EXCEEDS" if fired else "does NOT exceed"
        return f"- {condition}: measured {_fmt_num(measured)} vs threshold {_fmt_num(threshold)} -> {verdict} (fired={fired})"
    return f"- {condition}: fired={fired}"


def _render_rule_status_block(rule_trace: list[dict]) -> str:
    return "\n".join(_rule_verdict_line(r) for r in rule_trace) or "(no rules evaluated)"


def _render_beat_summary_block(beat_summary: dict) -> str:
    lines = []
    for cls, info in beat_summary.items():
        caveat = f" [{info['confidence']} CONFIDENCE -- {info['note']}]" if info.get("confidence") == "LOW" else ""
        lines.append(f"- {cls}: {info['count']} beats ({_fmt_num(info['pct_of_analyzed_beats'])}% "
                      f"of analyzed beats){caveat}")
    return "\n".join(lines) if lines else "(no beats survived quality gating)"


def _render_rhythm_findings_block(rhythm_findings: list[dict]) -> str:
    if not rhythm_findings:
        return "(none detected)"
    lines = []
    for f in rhythm_findings:
        lines.append(f"- {f['kind']}: beats {f['start_beat_idx']}-{f['end_beat_idx']} "
                      f"(starts at {_fmt_num(f['start_time_s'])}s, duration {_fmt_num(f['duration_s'])}s) "
                      f"-- {f['evidence_text']}")
    return "\n".join(lines)


def build_transparency_prompt(report_json: dict) -> str:
    return TRANSPARENCY_PROMPT_TEMPLATE.format(
        beat_summary_block=_render_beat_summary_block(report_json.get("beat_summary", {})),
        rhythm_findings_block=_render_rhythm_findings_block(report_json.get("rhythm_findings", [])),
        rule_status_block=_render_rule_status_block(report_json.get("rule_trace", [])),
        deciding_rule_block=_rule_verdict_line(report_json["deciding_rule"]),
        risk_level=report_json["risk_level"],
        report_json=json.dumps(report_json, indent=2, default=str),
    )


def _deterministic_narrative(report_json: dict, header: str = "") -> str:
    deciding = report_json["deciding_rule"]
    lines = [header] if header else []
    lines.append(f"Risk level: {report_json['risk_level']}.")
    lines.append(f"Deciding rule: {_deciding_rule_sentence(deciding)}")
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

    llm_narrative = merged.final_decision.get("narrative") or ""
    if llm_narrative and _narrative_asserts_false_exceed(llm_narrative, report_json.get("rule_trace", [])):
        audit.append("MEDGEMMA_NARRATIVE_FALSE_EXCEED_CLAIM_FILTERED",
                      {"reason": "free text asserted an 'exceeds' comparison for a rule with fired=False",
                       "raw_narrative": llm_narrative})
        llm_narrative = ""
    # Always route through the deterministic (a)-(e) composition -- even
    # when llm_narrative was filtered to "" above, this still guarantees
    # beat_summary/rhythm_findings/deciding_rule/risk_level/disclaimer are
    # present; only the free-text "Clinical interpretation" section is
    # dropped, per _compose_structured_narrative's own empty-string check.
    narrative = _compose_structured_narrative(report_json, llm_narrative)
    return {"narrative": narrative, "medgemma_status": "ACCEPTED",
            "final_risk_level": merged.final_decision["risk_level"],
            "endpoint": OLLAMA_URL, "model": MEDGEMMA_MODEL}


_EXCEED_WORDS = ("exceed", "cross", "trigger", "surpass")
_NEGATION_WINDOW = ("not ", "n't ", "no ", "below", "under", "less than", "does not", "did not")


def _narrative_asserts_false_exceed(text: str, rule_trace: list[dict]) -> bool:
    """Even when the prompt hands the model the correct verdict, its own
    free-text sometimes still independently asserts "X exceeds Y" for a rule
    that did NOT fire (observed directly: "PAC burden is at 8.642%, which
    exceeds the HIGH threshold of 15.0%" -- false, 8.642 < 15.0). The
    deterministic wrapper in _compose_structured_narrative can't fix wording
    inside the model's own sentences, so this scans for that specific
    failure mode and lets the caller drop the free text rather than save a
    false clinical claim."""
    text_lower = text.lower()
    for rule in rule_trace:
        if rule.get("fired") is not False or rule.get("evaluated") is False:
            continue
        measured = rule.get("measured_value")
        values = list(measured.values()) if isinstance(measured, dict) else [measured]
        for v in values:
            if not isinstance(v, (int, float)) or isinstance(v, bool) or float(v) in (0.0, 1.0):
                continue
            for rep in {str(v), str(round(float(v), 1)), str(round(float(v), 2)), str(int(v)) if float(v).is_integer() else None}:
                if not rep:
                    continue
                for m in re.finditer(r'(?<![\d.])' + re.escape(rep) + r'(?![\d.])', text_lower):
                    window = text_lower[max(0, m.start() - 50):m.end() + 50]
                    if any(w in window for w in _EXCEED_WORDS) and not any(n in window for n in _NEGATION_WINDOW):
                        return True
    return False


def _compose_structured_narrative(report_json: dict, llm_narrative: str) -> str:
    """Wraps MedGemma's own free-text narrative with the (a)-(e) structure
    deterministically, instead of trusting a 4B local model to reproduce it
    every time across a batch of hundreds of segments. The prompt already
    asks for this structure and pre-computed verdicts; this guarantees it
    lands even when the model's compliance is inconsistent (observed: beat
    counts and the closing disclaimer were both sometimes dropped across
    repeated calls on the same input). MedGemma's own sentence(s) are kept
    verbatim as the "clinical interpretation" -- nothing it wrote is
    discarded, only wrapped."""
    deciding = report_json["deciding_rule"]
    parts = [
        f"Beat summary:\n{_render_beat_summary_block(report_json.get('beat_summary', {}))}",
        f"Rhythm findings:\n{_render_rhythm_findings_block(report_json.get('rhythm_findings', []))}",
        f"Deciding rule: {_deciding_rule_sentence(deciding)}",
        f"Final risk level: {report_json['risk_level']}",
    ]
    # The model sometimes echoes the closing disclaimer itself (rule 5 of the
    # prompt asks for it); strip a trailing copy so the deterministic footer
    # below isn't printed twice.
    disclaimer = "clinician review suggested -- this is decision support, not a diagnosis."
    clean_llm_narrative = llm_narrative.strip()
    if clean_llm_narrative.lower().rstrip(".").endswith(disclaimer.rstrip(".")):
        clean_llm_narrative = clean_llm_narrative[:clean_llm_narrative.lower().rindex(
            disclaimer.split(" -- ")[0].lower())].strip().rstrip(".").strip()
    if clean_llm_narrative:
        parts.append(f"Clinical interpretation: {clean_llm_narrative}")
    parts.append("Clinician review suggested -- this is decision support, not a diagnosis.")
    return "\n\n".join(parts)


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


def render_report_markdown(report_json: dict) -> str:
    """Human-readable companion to the saved JSON report -- just the
    clinical report prose plus enough of a header to identify which
    recording it's for, for showing a report without reading JSON."""
    r = report_json["recording"]
    lines = [
        f"# ECG Clinical Report -- {r['source']} / {r['patient_id']} / {r['segment_id']}",
        "",
        f"- Duration: {r['duration_s']}s",
        f"- Beats detected / analyzed: {r['n_beats_detected']} / {r['n_beats_analyzed']}",
        f"- Assessable: {report_json['assessable']}",
        f"- Final risk level: {report_json['final_risk_level']}",
        f"- MedGemma status: {report_json['medgemma']['status']}",
        "",
        "---",
        "",
        report_json["narrative"],
    ]
    return "\n".join(lines) + "\n"


def save_report(report_json: dict, out_dir: Path, segment_id: str) -> tuple[Path, Path]:
    """Writes both the structured JSON and the human-readable .md companion
    for one segment, returning (json_path, md_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{segment_id}.json"
    md_path = out_dir / f"{segment_id}.md"
    json_path.write_text(json.dumps(report_json, indent=2, default=str))
    md_path.write_text(render_report_markdown(report_json))
    return json_path, md_path

