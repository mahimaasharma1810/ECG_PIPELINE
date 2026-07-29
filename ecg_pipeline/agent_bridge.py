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
import csv
import json
import random
import re
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


# LEGACY: used simulated vitals from MedGemma-Agent/vitals/test_cases.yaml
# Replaced by load_real_vitals() which reads actual VitalPatch vitals CSVs.
# Kept for reference only -- not called by any production path.
# See agent_bridge.py push_main() for the replacement.
def _legacy_mock_vitals_case() -> dict:
    if str(AGENT_DIR) not in sys.path:
        sys.path.insert(0, str(AGENT_DIR))
    from vitals.mock_producer import list_test_cases
    return random.choice(list_test_cases())


# ORIGINAL (before MedGemma-Agent's partial-vitals schema change): heart_rate,
# spo2, systolic_bp, diastolic_bp were ALL required (Field(...), no default) in
# MedGemma-Agent/vitals/schemas.py:66-70 -- none could be omitted or null.
# REQUIRED_AGENT_VITALS_FIELDS = ["heart_rate", "spo2", "systolic_bp", "diastolic_bp"]
#
# MedGemma-Agent/vitals/schemas.py now makes spo2/systolic_bp/diastolic_bp
# Optional[float] = None; heart_rate is still the only required field (it's
# VitalPatch's one guaranteed vital, and NEWS2/qSOFA can't score anything
# without at least one real value). This list must be kept in sync with that
# schema by hand -- it is not introspected at runtime.
REQUIRED_AGENT_VITALS_FIELDS = ["heart_rate"]
_AGENT_FIELD_TO_LOCAL_KEY = {
    "heart_rate": "hr", "spo2": "spo2", "systolic_bp": "sbp", "diastolic_bp": "dbp",
}

# Default pairing tolerance for load_real_vitals()'s nearest-timestamp match -- see
# that function's docstring for how this number was derived (not a guess).
DEFAULT_VITALS_MAX_OFFSET_MS = 30_000


def _extract_numeric_column(vitals_path: Path, col_idx: int, lo: float, hi: float) -> list[float]:
    values: list[float] = []
    with open(vitals_path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) <= col_idx or not row[col_idx]:
                continue
            try:
                v = float(row[col_idx])
            except ValueError:
                continue
            if lo <= v <= hi:
                values.append(v)
    return values


def _numeric_stat_dict(values: list[float], source: str, note: str | None = None) -> dict:
    if not values:
        return {"value": None, "source": source, "real": False,
                "note": "no valid readings in vitals file"}
    d = {
        "value": round(sum(values) / len(values), 1),
        "min": min(values),
        "max": max(values),
        "n_readings": len(values),
        "source": source,
        "real": True,
    }
    if note:
        d["note"] = note
    return d


def load_real_vitals(ecg_path: str, vitals_root: str,
                      max_offset_ms: int = DEFAULT_VITALS_MAX_OFFSET_MS) -> dict:
    """Loads real VitalPatch vitals for the ECG segment at ecg_path, from the
    companion vitals CSV under vitals_root/<same Patch_<ID> folder>/.

    CORRECTIONS made here vs. task assumptions, each verified directly against
    the real files on disk before writing/extending this function (not
    assumed):

    1. FILE PAIRING IS NOT AN EXACT FILENAME MATCH. Checked all 424 ECG files
       for Patch_184B2F against all 425 vitals files in the same folder -- an
       exact "_ecg.csv"->"_vitals.csv" substring swap matched 0/424. The two
       collector streams log independent timestamps for the same capture
       session. Nearest-timestamp offsets: p50=4.5s, p90=11.4s, 95.3% within
       15s, 98.3% within 30s, then a hard plateau (the remaining ~1.7% are
       genuine gaps, not a tolerance problem). 30s is the default cutoff. The
       actual offset used is always returned in "vitals_time_offset_ms" so a
       caller can judge staleness of any specific pairing directly.

    2. COLUMN 7 IS NOT SPO2. Col 1 (HR, bpm) is a genuine, varying sensor
       signal. Col 7 is NOT SpO2: within any single file it is one constant
       value repeated ~300 times (zero variance), and that constant varies
       wildly and non-physiologically ACROSS files for the same patient
       (checked 8 files: 2, 3, 7, 16, 17, 20, 32, 47, 75). This is a per-file
       device/session metadata value, not a measurement -- confirmed by
       parse_vitalpatch_vitals()'s own docstring in ecg_pipeline_core.py:
       "VitalPatch has no SpO2/BP sensor on this hardware, so those columns
       are sentinel values if present." SpO2 is always reported unavailable,
       never fabricated from col 7.

    3. COLUMN 3 (temperature) NEEDS A MULTI-FILE CHECK, NOT A SINGLE-FILE SPOT
       CHECK. The specific file used to verify correction #2 (offset=78ms
       match) happens to have ZERO col-3 readings for that time window --
       checking only that one file would have wrongly concluded "temperature
       is never present." Checked 20 files instead: 18/20 have real col-3
       data, physiologically plausible (35.83-37.36C) and drifting smoothly
       across consecutive time windows (a real physiological signal, not
       noise) -- confirms col 3 = temperature. This is why file-format claims
       here are verified across multiple files, not one.

    Additional columns confirmed and now extracted: col 2 = respiratory rate
    (breaths/min -- NOT the cardiac RR-interval; that is a separate field,
    col 6, not extracted by this function), col 5 = posture (categorical
    string, e.g. "Standing"/"Walking"/"LeaningBack"/"Unknown").

    Only hr, respiratory_rate, temperature, and posture can ever be marked
    "real": True by this function -- spo2/sbp/dbp never can, on this hardware.
    """
    ecg_path = Path(ecg_path)
    vitals_root = Path(vitals_root)
    patient_dir = vitals_root / ecg_path.parent.name

    try:
        ecg_ts = int(ecg_path.name.split("_")[0])
    except (ValueError, IndexError):
        ecg_ts = None

    best_path, best_offset = None, None
    if ecg_ts is not None and patient_dir.is_dir():
        for candidate in patient_dir.glob("*_vitals.csv"):
            try:
                cand_ts = int(candidate.name.split("_")[0])
            except ValueError:
                continue
            offset = abs(cand_ts - ecg_ts)
            if best_offset is None or offset < best_offset:
                best_offset, best_path = offset, candidate

    vitals_path = best_path if (best_path is not None and best_offset <= max_offset_ms) else None

    not_available_bp_note = "VitalPatch has no BP sensor"
    not_available_spo2 = {"value": None, "source": "not_available", "real": False,
                           "note": "VitalPatch has no SpO2 sensor"}
    not_available_sbp = {"value": None, "source": "not_available", "real": False,
                          "note": not_available_bp_note}
    not_available_dbp = {"value": None, "source": "not_available", "real": False,
                          "note": not_available_bp_note}

    if vitals_path is None:
        attempted = (str(best_path) if best_path is not None
                     else str(patient_dir / ecg_path.name.replace("_ecg.csv", "_vitals.csv")))
        not_found = {"value": None, "source": "not_found", "real": False}
        return {
            "hr": dict(not_found),
            "respiratory_rate": dict(not_found),
            "temperature": dict(not_found),
            "posture": {"values": [], "most_common": None, "source": "not_found", "real": False},
            "spo2": not_available_spo2,
            "sbp": not_available_sbp,
            "dbp": not_available_dbp,
            "vitals_file": attempted,
            "vitals_file_found": False,
            "vitals_time_offset_ms": best_offset,
        }

    hr = _numeric_stat_dict(_extract_numeric_column(vitals_path, 1, 20, 300),
                             "vitalpatch_col1")
    respiratory_rate = _numeric_stat_dict(
        _extract_numeric_column(vitals_path, 2, 4, 60), "vitalpatch_col2",
        note="VitalPatch-derived RR -- not spirometry")
    temperature = _numeric_stat_dict(
        _extract_numeric_column(vitals_path, 3, 35.0, 42.0), "vitalpatch_col3",
        note="Skin temperature -- may underestimate core temp by ~0.5C")

    posture_values: list[str] = []
    with open(vitals_path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) <= 5 or not row[5]:
                continue
            posture_values.append(row[5])
    if posture_values:
        counts: dict[str, int] = {}
        for p in posture_values:
            counts[p] = counts.get(p, 0) + 1
        posture = {
            "values": sorted(set(posture_values)),
            "most_common": max(counts, key=counts.get),
            "source": "vitalpatch_col5",
            "real": True,
        }
    else:
        posture = {"values": [], "most_common": None, "source": "vitalpatch_col5",
                    "real": False, "note": "no posture readings in vitals file"}

    return {
        "hr": hr,
        "respiratory_rate": respiratory_rate,
        "temperature": temperature,
        "posture": posture,
        "spo2": not_available_spo2,
        "sbp": not_available_sbp,
        "dbp": not_available_dbp,
        "vitals_file": str(vitals_path),
        "vitals_file_found": True,
        "vitals_time_offset_ms": best_offset,
    }


# ── Partial NEWS2 (Royal College of Physicians, National Early Warning Score
# (NEWS) 2, 2017) -- only the 3 of 6 standard components VitalPatch can supply
# (HR, respiratory rate, temperature) are scored from real data. SpO2,
# systolic BP, and consciousness (AVPU) are not measurable on this hardware
# and are always scored 0 -- the conservative direction, since it can only
# ever UNDER-count risk, never inflate it -- and always listed explicitly in
# missing_components, never silently dropped. This is a LOWER BOUND, not a
# clinically complete NEWS2: a missing component could itself be abnormal and
# this score has no way to detect that.
# ────────────────────────────────────────────────────────────────────────────

def _news2_hr_score(hr: float) -> int:
    if hr <= 40: return 3
    if hr <= 50: return 1
    if hr <= 90: return 0
    if hr <= 110: return 1
    if hr <= 130: return 2
    return 3


def _news2_rr_score(rr: float) -> int:
    if rr <= 8: return 3
    if rr <= 11: return 1
    if rr <= 20: return 0
    if rr <= 24: return 2
    return 3


def _news2_temp_score(temp: float) -> int:
    if temp <= 35.0: return 3
    if temp <= 36.0: return 1
    if temp <= 38.0: return 0
    if temp <= 39.0: return 1
    return 2


def compute_partial_news2(vitals: dict) -> dict:
    hr_v = vitals["hr"]["value"]
    rr_v = vitals["respiratory_rate"]["value"]
    temp_v = vitals["temperature"]["value"]

    hr_score = _news2_hr_score(hr_v) if hr_v is not None else 0
    rr_score = _news2_rr_score(rr_v) if rr_v is not None else 0
    temp_score = _news2_temp_score(temp_v) if temp_v is not None else 0
    total = hr_score + rr_score + temp_score  # spo2/sbp/consciousness always contribute 0

    missing = ["spo2", "sbp", "consciousness"]
    for key, val in (("hr", hr_v), ("respiratory_rate", rr_v), ("temperature", temp_v)):
        if val is None:
            missing.append(key)
    scored_count = sum(v is not None for v in (hr_v, rr_v, temp_v))

    if total >= 5:
        interpretation = "HIGH"
    elif total >= 3:
        interpretation = "MEDIUM"
    else:
        interpretation = "LOW"

    return {
        "partial_news2_score": total,
        "components": {
            "hr_score": hr_score, "hr_value": hr_v,
            "rr_score": rr_score, "rr_value": rr_v,
            "temp_score": temp_score, "temp_value": temp_v,
            "spo2_score": 0, "spo2_value": None, "spo2_note": "not available",
            "sbp_score": 0, "sbp_value": None, "sbp_note": "not available",
            "consciousness_score": 0, "consciousness_note": "assumed alert",
        },
        "missing_components": missing,
        "coverage": f"{scored_count}/6 NEWS2 components scored from real data",
        "interpretation": interpretation,
        "caveat": ("Partial NEWS2 -- missing SpO2, BP, consciousness. Score is a lower "
                   "bound. Full NEWS2 requires bedside SpO2 and BP measurement."),
    }


def compute_qsofa_proxy(vitals: dict) -> dict:
    """Degenerate proxy for qSOFA (Sepsis-3, Singer et al. JAMA 2016), NOT a real
    qSOFA score. Standard qSOFA has 3 independent criteria (SBP<=100, HR>90,
    a respiratory-dysfunction proxy) and is "high risk" at score>=2 -- see
    MedGemma-Agent/guardrails/clinical_rules.py:calculate_qsofa for the real
    3-criteria version this project already has for comparison. VitalPatch can
    only ever evaluate 1 of the 3 (HR).

    SBP is deliberately NOT scored 0 when unavailable -- scoring it 0 would
    silently assert "not hypotensive," which cannot be verified from this
    hardware. It is left out of the total entirely and reported as
    unevaluated, not assumed safe.
    """
    hr = vitals["hr"]["value"]
    hr_flag = bool(hr is not None and hr > 90)
    score = int(hr_flag)
    return {
        "qsofa_proxy_score": score,
        "hr_flag": hr_flag,
        "hr_value": hr,
        "sbp_flag": None,
        "sbp_note": ("SBP criterion cannot be evaluated -- VitalPatch has no BP sensor. "
                     "Not scored as 0 (would assert 'not hypotensive', unverifiable) -- "
                     "left out of the total and reported as unevaluated."),
        "note": ("1-of-3-criteria proxy, not a real qSOFA score. Standard qSOFA's "
                 "high-risk threshold is 2; this proxy's maximum possible value is 1, "
                 "so it can never independently trigger the ECG cascade's qSOFA "
                 "override (RISK.qsofa_high_threshold=2) from VitalPatch data alone."),
    }


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


# ORIGINAL (pre-connection-failure-hardening): raised whatever requests threw
# (ConnectionError/Timeout/HTTPError/etc.) straight up to the caller. This was
# never actually exercised until MedGemma-Agent's schema stopped requiring
# SpO2/BP -- before that, push_main's own completeness check always skipped
# before reaching this call. Now that the call is reachable, an unhandled
# exception here would crash the whole batch on the first unreachable/erroring
# segment, which is worse than reporting it and moving on.
# def submit_to_agent(patient_id: str, vitals_values: dict, ecg_risk: dict,
#                      agent_url: str, api_key: str, timeout: float = 150.0) -> dict:
#     payload = {"patient_id": patient_id, **vitals_values, "ecg_risk": ecg_risk}
#     resp = requests.post(agent_url, json=payload, headers={"X-API-Key": api_key}, timeout=timeout)
#     resp.raise_for_status()
#     return resp.json()
def submit_to_agent(patient_id: str, vitals_values: dict, ecg_risk: dict,
                     agent_url: str, api_key: str, timeout: float = 150.0) -> dict:
    """Returns {"push_status", "error", "agent_response"} -- never raises.
    push_status is one of SUCCESS / AGENT_UNREACHABLE / AGENT_TIMEOUT /
    AGENT_HTTP_ERROR / AGENT_ERROR (mirrors the status-string pattern this
    codebase already uses elsewhere, e.g. report.py's medgemma_status:
    SKIPPED_CRITICAL/UNAVAILABLE_FALLBACK/REJECTED_FALLBACK/ACCEPTED).

    timeout stays at the existing 150s default, not a shorter one -- the
    Agent's /vitals/snapshot handler runs a synchronous LLM analysis step
    (monitoring_agent.invoke -> llm_analyzer -> Ollama) inside the request,
    which can legitimately take much longer than a plain API call.
    """
    payload = {"patient_id": patient_id, **vitals_values, "ecg_risk": ecg_risk}
    try:
        resp = requests.post(agent_url, json=payload, headers={"X-API-Key": api_key}, timeout=timeout)
        resp.raise_for_status()
        return {"push_status": "SUCCESS", "error": None, "agent_response": resp.json()}
    except requests.exceptions.ConnectionError:
        return {
            "push_status": "AGENT_UNREACHABLE",
            "error": f"Could not connect to MedGemma Agent at {agent_url}. Is the server "
                     f"running? Start with: cd MedGemma-Agent && ./start.sh",
            "agent_response": None,
        }
    except requests.exceptions.Timeout:
        return {
            "push_status": "AGENT_TIMEOUT",
            "error": f"MedGemma Agent at {agent_url} did not respond within {timeout}s.",
            "agent_response": None,
        }
    except requests.exceptions.HTTPError as e:
        return {
            "push_status": "AGENT_HTTP_ERROR",
            "error": f"Agent returned HTTP {e.response.status_code}: {e.response.text[:200]}",
            "agent_response": None,
        }
    except requests.exceptions.RequestException as e:
        return {
            "push_status": "AGENT_ERROR",
            "error": f"Unexpected error contacting Agent: {e}",
            "agent_response": None,
        }


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
    from training.ecg_pipeline_tools import AAMI_SYMBOL_MAP  # training-side module, relocated under training/
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
    parser.add_argument(
        "--vitalpatch-root", type=Path, default=DATA_RAW / "vitalpatch",
        help="Root directory containing Patch_<ID>/*_ecg.csv files -- must be the "
             "PARENT of one or more Patch_* dirs, not a Patch_* dir itself (this CLI "
             "has no per-file --input flag; files are discovered via glob and taken "
             "in sorted order up to --limit). For a quick test with real HR+RR+Temp "
             "all present, use Patch_184B2F/1778329087549_VC2B008BF_184B2F_ecg.csv "
             "(confirmed this session: HR=93.7, RR=17.5, Temp=36.0) -- point this flag "
             "at a directory containing (e.g. a symlink to) only that one Patch_184B2F "
             "folder and pass --limit 3 to land on it deterministically.")
    parser.add_argument("--vitals-root", type=Path,
                         default=Path("/home2/mahimakopalley/projects/data/vitals_downloads"),
                         help="Root directory of VitalPatch vitals CSV files")
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
            vitals = load_real_vitals(ecg_path=str(f), vitals_root=str(args.vitals_root))

            if not vitals["vitals_file_found"]:
                print(f"WARNING: No vitals file found within "
                      f"{DEFAULT_VITALS_MAX_OFFSET_MS}ms of {f} "
                      f"(nearest candidate: {vitals['vitals_file']}, "
                      f"offset={vitals['vitals_time_offset_ms']}ms)")
                print("WARNING: Skipping push to MedGemma Agent for this segment -- "
                      "will not send None vitals to clinical endpoint.")
                n_done += 1
                continue

            # Print provenance clearly before sending -- see load_real_vitals()'s
            # docstring for why spo2/sbp/dbp can never be "real" for this device.
            print("\n=== VITALS PROVENANCE ===")
            print(f"  vitals file: {vitals['vitals_file']} "
                  f"(time offset from ECG segment: {vitals['vitals_time_offset_ms']}ms)")
            for key, unit in [("hr", "bpm"), ("respiratory_rate", "/min"), ("temperature", "C")]:
                v = vitals[key]
                real_flag = "REAL" if v["real"] else "NOT REAL"
                if v["real"]:
                    print(f"  {key.upper():17}: {v['value']}{unit} [{real_flag} -- {v['source']} -- "
                          f"n={v['n_readings']} readings, range {v['min']}-{v['max']}] {v.get('note', '')}")
                else:
                    print(f"  {key.upper():17}: {v['value']} [{real_flag} -- {v['source']}] {v.get('note', '')}")
            p = vitals["posture"]
            if p["real"]:
                print(f"  {'POSTURE':17}: {p['most_common']} (most common) "
                      f"[REAL -- {p['source']}] seen: {p['values']}")
            else:
                print(f"  {'POSTURE':17}: None [NOT REAL -- {p['source']}] {p.get('note', '')}")
            for key in ["spo2", "sbp", "dbp"]:
                v = vitals[key]
                print(f"  {key.upper():17}: {v['value']} [NOT AVAILABLE -- {v['source']}] {v.get('note', '')}")
            print("=========================\n")

            # Compute partial NEWS2 locally regardless of whether the push below
            # succeeds -- see compute_partial_news2()'s module comment.
            partial_news2 = compute_partial_news2(vitals)
            c = partial_news2["components"]
            print(f"=== PARTIAL NEWS2 ({partial_news2['coverage'].split(' ')[0]} components) ===")
            print(f"  HR score:   {c['hr_score']}  (HR={c['hr_value']})")
            print(f"  RR score:   {c['rr_score']}  (RR={c['rr_value']})")
            print(f"  Temp score: {c['temp_score']}  (Temp={c['temp_value']})")
            print(f"  SpO2:       -- (not available)")
            print(f"  SBP:        -- (not available)")
            print(f"  AVPU:       -- (assumed alert=0)")
            print(f"  Partial NEWS2: {partial_news2['partial_news2_score']} "
                  f"({partial_news2['coverage']}) = {partial_news2['interpretation']} "
                  f"(conservative lower bound)")
            print("=========================\n")

            # Feed the LOCALLY computed partial NEWS2 / qSOFA proxy into the ECG
            # pipeline's own risk cascade (score_recording()'s news2_score/
            # qsofa_score params, ecg_pipeline_core.py:2041-2042, wired through
            # ECGPipeline.run() at line 2481) -- these fields have existed since
            # the original audit but were permanently null because nothing ever
            # called them with a real value. This is that missing caller. Not a
            # round trip through MedGemma-Agent (that push still always skips,
            # see below) -- purely local, computed straight from VitalPatch data.
            qsofa_proxy = compute_qsofa_proxy(vitals)
            ecg_only_risk = result.risk_report.alert_level
            combined_result = ECGPipeline(classifier=classifier).run(
                recording,
                news2_score=partial_news2["partial_news2_score"],
                qsofa_score=qsofa_proxy["qsofa_proxy_score"],
            )
            combined_risk = combined_result.risk_report.alert_level
            override_activated = combined_risk != ecg_only_risk

            print("=== COMBINED RISK ===")
            print(f"  ECG-only:      {ecg_only_risk}")
            print(f"  Partial NEWS2: {partial_news2['partial_news2_score']} "
                  f"({partial_news2['coverage']} -- conservative lower bound; "
                  f"cascade's NEWS2-critical threshold is {RISK.news2_critical_threshold})")
            print(f"  qSOFA proxy:   {qsofa_proxy['qsofa_proxy_score']} "
                  f"(HR criterion only -- {qsofa_proxy['sbp_note']})")
            print(f"  Combined:      {combined_risk}")
            print(f"  Override:      {override_activated}")
            print("=========================\n")

            # MedGemma-Agent's /vitals/snapshot schema requires all four fields as
            # non-null floats (REQUIRED_AGENT_VITALS_FIELDS, sourced from
            # MedGemma-Agent/vitals/schemas.py:67-70) -- check completeness rather
            # than assuming; do not invent values to satisfy the schema.
            #
            # TODO (not done here -- MedGemma-Agent/ is read-only from this repo):
            # to accept real VitalPatch pushes, MedGemma-Agent/vitals/schemas.py's
            # VitalsSnapshotInput needs (a) systolic_bp/diastolic_bp/spo2 (lines
            # 67-70) changed from `Field(...)` (required) to
            # `Optional[float] = None`, PLUS matching `if v is None: return v`
            # guards added to validate_sbp/validate_dbp/validate_spo2 (lines
            # 80-106) and to validate_sbp_gt_dbp (lines 108-114) -- and
            # MedGemma-Agent/guardrails/clinical_rules.py's calculate_news2/
            # calculate_qsofa and their _score_* helpers (lines 28-124) need
            # equivalent None-handling, since they currently crash (not
            # degrade) on a None input. Confirmed by direct read this session --
            # not a one-line schema change. (b) respiratory_rate and
            # temperature have NO field at all in VitalsSnapshotInput today --
            # adding them would be a new field, not a relaxation of an existing
            # one. Neither (a) nor (b) is done here.
            missing_required = [
                agent_field for agent_field in REQUIRED_AGENT_VITALS_FIELDS
                if vitals[_AGENT_FIELD_TO_LOCAL_KEY[agent_field]]["value"] is None
            ]
            if missing_required:
                print(f"SKIP: MedGemma-Agent's /vitals/snapshot requires "
                      f"{REQUIRED_AGENT_VITALS_FIELDS} as non-null values "
                      f"(MedGemma-Agent/vitals/schemas.py:67-70, VitalsSnapshotInput) -- "
                      f"this segment is missing real values for {missing_required}. "
                      f"VitalPatch hardware cannot supply these (no SpO2/BP sensor -- see "
                      f"parse_vitalpatch_vitals()'s docstring in ecg_pipeline_core.py). "
                      f"To enable push: make these fields Optional in "
                      f"MedGemma-Agent/vitals/schemas.py (see TODO comment above for the "
                      f"full scope of that change). Skipping push rather than inventing "
                      f"values to satisfy the schema.")
                print(json.dumps({
                    "segment_id": recording.segment_id,
                    "patient_id": recording.patient_id,
                    "n_beats": len(result.beats),
                    "ecg_risk": ecg_risk,
                    "vitals_provenance": vitals,
                    "partial_news2": partial_news2,
                    "news2_source": "vitalpatch_partial_local",
                    "news2_score_used": partial_news2["partial_news2_score"],
                    "news2_partial": True,
                    "news2_missing_components": partial_news2["missing_components"],
                    "qsofa_proxy": qsofa_proxy["qsofa_proxy_score"],
                    "qsofa_proxy_note": qsofa_proxy["note"] + " " + qsofa_proxy["sbp_note"],
                    "ecg_only_risk": ecg_only_risk,
                    "combined_risk": combined_risk,
                    "override_activated": override_activated,
                    "agent_response": None,
                    "agent_push_status": "SKIPPED_MISSING_REQUIRED_FIELDS",
                    "agent_push_error": f"missing required Agent fields: {missing_required}",
                    "push_skipped_reason": f"missing required Agent fields: {missing_required}",
                }, indent=2, default=str))
                n_done += 1
                continue

            # ORIGINAL (before RR/temp were wired into the Agent payload): only
            # sent heart_rate/spo2/systolic_bp/diastolic_bp -- respiratory_rate
            # and temperature were computed by load_real_vitals() but never
            # actually included in the HTTP payload, so even a fully successful
            # push only ever gave the Agent 1 of VitalPatch's 3 real components.
            # vitals_values = {
            #     "heart_rate": vitals["hr"]["value"],
            #     "spo2": vitals["spo2"]["value"],
            #     "systolic_bp": vitals["sbp"]["value"],
            #     "diastolic_bp": vitals["dbp"]["value"],
            # }
            vitals_values = {
                "heart_rate":       vitals["hr"]["value"],
                "spo2":             vitals["spo2"]["value"],         # None -- no sensor
                "systolic_bp":      vitals["sbp"]["value"],          # None -- no sensor
                "diastolic_bp":     vitals["dbp"]["value"],          # None -- no sensor
                "respiratory_rate": vitals["respiratory_rate"]["value"],  # real if available
                "temperature":      vitals["temperature"]["value"],       # real if available
            }
            # submit_to_agent() now never raises -- it returns a structured
            # {"push_status", "error", "agent_response"} dict instead (see its
            # own docstring). This is the first point in the pipeline where the
            # live HTTP call is actually reachable (every prior run hit the SKIP
            # branch above first, so an unreachable/erroring Agent server was
            # never exercised before). The Agent push is additive: its failure
            # must never suppress the local multimodal result already computed
            # above (ecg_only_risk/combined_risk/override_activated), so it's
            # always written to the output JSON regardless of push_status.
            agent_result = submit_to_agent(recording.patient_id, vitals_values, ecg_risk,
                                            args.agent_url, args.api_key)
            push_status = agent_result["push_status"]

            # Extract the Agent's own live NEWS2/qSOFA scores. Verified by direct
            # read this session (MedGemma-Agent/vitals/schemas.py): AlertResponse.news2
            # is a NEWS2Breakdown with .total_score/.coverage; AlertResponse.qsofa is a
            # QSOFABreakdown with .score. agent_response here is the JSON-decoded dict
            # from resp.json() (submit_to_agent), so these are dict lookups, not
            # attribute access on the Pydantic model itself.
            agent_response = agent_result.get("agent_response") or {}
            news2_block = agent_response.get("news2") or {}
            qsofa_block = agent_response.get("qsofa") or {}
            agent_news2_score = news2_block.get("total_score")
            agent_news2_coverage = news2_block.get("coverage")
            agent_qsofa_score = qsofa_block.get("score")

            if push_status != "SUCCESS":
                print(f"\n=== AGENT PUSH STATUS: {push_status} ===")
                print(f"  {agent_result.get('error', 'unknown error')}")
                print(f"  Combined risk still computed locally from partial NEWS2.")
                print(f"  NEWS2 loopback from Agent: not available this run.")
            else:
                print(f"\n=== AGENT PUSH: SUCCESS ===")

            # If the Agent returned a real NEWS2 score, it supersedes the local
            # partial_news2 approximation already computed above -- re-run the ECG
            # cascade with the Agent's live values instead of trusting the local
            # one. combined_risk/override_activated are reassigned here only in
            # that case; local_partial_news2 below always preserves the original
            # local-only number for comparison, regardless of which one won.
            news2_source = "local_partial"
            if agent_news2_score is not None:
                print(f"\n=== AGENT NEWS2 RECEIVED ===")
                print(f"  Score:    {agent_news2_score}")
                print(f"  Coverage: {agent_news2_coverage}")
                print(f"  qSOFA:    {agent_qsofa_score}")

                agent_scored_result = ECGPipeline(classifier=classifier).run(
                    recording, news2_score=agent_news2_score, qsofa_score=agent_qsofa_score,
                )
                combined_risk = agent_scored_result.risk_report.alert_level
                override_activated = combined_risk != ecg_only_risk
                news2_source = "agent_live"

                print(f"\n=== COMBINED RISK (Agent-driven) ===")
                print(f"  ECG only:   {ecg_only_risk}")
                print(f"  Combined:   {combined_risk}")
                print(f"  Override:   {override_activated}")
            else:
                reason = "push did not succeed" if push_status != "SUCCESS" else "Agent response had no NEWS2 block"
                print(f"\n=== COMBINED RISK (local partial fallback) ===")
                print(f"  ECG only:  {ecg_only_risk}")
                print(f"  Combined:  {combined_risk}")
                print(f"  Note: {reason} -- using locally-computed partial NEWS2 instead")

            print(json.dumps({
                "segment_id": recording.segment_id,
                "patient_id": recording.patient_id,
                "n_beats": len(result.beats),
                "ecg_risk": ecg_risk,
                "vitals_provenance": vitals,
                "partial_news2": partial_news2,
                "agent_push_status": push_status,
                "agent_news2_score": agent_news2_score,
                "agent_news2_coverage": agent_news2_coverage,
                "agent_qsofa_score": agent_qsofa_score,
                "local_partial_news2": partial_news2["partial_news2_score"],
                "news2_source": news2_source,
                "news2_missing_components": partial_news2["missing_components"],
                "qsofa_proxy": qsofa_proxy["qsofa_proxy_score"],
                "qsofa_proxy_note": qsofa_proxy["note"] + " " + qsofa_proxy["sbp_note"],
                "ecg_only_risk": ecg_only_risk,
                "combined_risk": combined_risk,
                "override_activated": override_activated,
                "agent_response": agent_result["agent_response"],
                "agent_push_error": agent_result.get("error"),
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
