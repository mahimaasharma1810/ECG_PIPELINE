"""generate_clinician_report.py -- one visual, clinician-facing PDF per segment
in data/reports/clinician_review_sample.csv, so a clinician can adjudicate a
CRITICAL call by looking at the actual ECG, not just a risk_level string.

WHAT THIS SCRIPT DOES NOT DO: it does not decide, recompute, or alter any
classification, risk_level, or rule_trace value. Every number shown anywhere
in this report -- the plain-language Key Findings, the Interpretation
paragraph, Suggested Actions, AI Confidence tier, and the full appendix --
is derived from the already-saved JSON report in
data/reports/vitalpatch/<patient_id>/<segment_id>.json, the same file
batch_vitalpatch_report.py wrote weeks ago. Plain-language sections are a
presentation-layer translation of that saved data (fired rule -> sentence),
never a new judgment.

WHY IT STILL RE-RUNS THE PIPELINE: the saved JSON only stores AGGREGATE beat
counts (e.g. "V: 112 beats, 42.42%"), not a per-beat N/S/V/F/Q label sequence,
and does not store the filtered waveform or R-peak sample positions at all.
None of that exists anywhere on disk in reloadable form. To draw the R-peak
overlay and the beat-by-beat classification strip, this script re-runs
stages 1-7 (parse -> SQI gate -> resample -> filter chain -> R-peak detection
-> beat segmentation -> classification) using the exact same frozen
classifier weights (ecg_pipeline/models/five_class_xgb.json) that produced
the saved report. This is deterministic, not a new decision: same frozen
model, same rule-based cascade, same input file.

SAFETY NET: before rendering anything, the freshly recomputed report JSON
(same code path save_report's callers use, agent_bridge.run_full_report) is
compared against the saved JSON on disk. Two checks:
  - decision_matches(): does final_risk_level + deciding_rule match exactly?
    If not, the segment is MISMATCH and NOT rendered at all.
  - reports_match(): does EVERYTHING else also match (rhythm_findings,
    every rule_trace row, beat_summary), ignoring only generated_at/
    narrative/medgemma? A saved report generated before a later pipeline
    code change (e.g. the 2026-07-28 AFib-threshold fix) can differ here
    even though the decision itself never changed. When this happens, the
    segment IS rendered (using the saved data throughout, as always), but a
    plain-language banner is shown prominently near the top of page 1 --
    not buried in a technical diff -- naming exactly what current code
    finds differently, per the same "never hide a material difference"
    principle. The full technical diff is additionally kept in the
    appendix for anyone who wants to check the raw values.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ecg_pipeline.ecg_pipeline_core import (  # noqa: E402
    DATA_RAW, MODELS_DIR, TARGET_FS,
    apply_filter_chain, parse_vitalpatch_ecg, run_sqi_gate, to_target_rate,
)
from ecg_pipeline.agent_bridge import load_classifier, run_full_report  # noqa: E402

SAMPLE_CSV = REPO_ROOT / "data" / "reports" / "clinician_review_sample.csv"
REPORT_ROOT = REPO_ROOT / "data" / "reports" / "vitalpatch"
MULTIMODAL_MANIFEST = REPO_ROOT / "data" / "reports" / "multimodal_batch" / "multimodal_manifest.json"
OUT_DIR = REPO_ROOT / "data" / "reports" / "clinician_review_visual"

BEAT_COLORS = {"N": "#2ca02c", "S": "#e6b800", "V": "#d62728", "F": "#ff7f0e", "Q": "#7f7f7f"}
BEAT_LABEL_NAMES = {"N": "Normal", "S": "Supraventricular", "V": "Ventricular",
                     "F": "Fusion", "Q": "Unknown/rejected"}

# Docs/PIPELINE_METHODS_AND_RESULTS.md Sec 3 / README.md "What is verified
# working" Sec 1 -- production classifier, DS2 held-out, 45,804 beats. Real
# measured numbers, not re-derived here. Shown in full only in the appendix;
# the main page shows only the plain-language tier derived from these.
CLASSIFIER_METRICS = {
    "N": {"sensitivity": 0.972, "precision": 0.973, "f1": 0.972, "support": 40634},
    "S": {"sensitivity": 0.130, "precision": 0.150, "f1": 0.139, "support": 1795},
    "V": {"sensitivity": 0.912, "precision": 0.754, "f1": 0.826, "support": 3005},
    "F": {"sensitivity": 0.005, "precision": 0.125, "f1": 0.011, "support": 363},
    "Q": {"sensitivity": 0.000, "precision": 0.000, "f1": 0.000, "support": 7},
}
DISCLAIMER = "Clinician review suggested -- this is decision support, not a diagnosis."

SUGGESTED_ACTIONS = {
    "CRITICAL": "Review ECG manually. Assess symptoms (chest pain, syncope, palpitations). "
                "Consider urgent cardiology consultation. If clinically unstable, follow emergency protocol.",
    "HIGH": "Review ECG manually. Assess symptoms. Consider cardiology consultation. Continue monitoring.",
    "MEDIUM": "Continue monitoring. Reassess if symptoms develop.",
    "LOW": "No immediate action indicated. Continue routine monitoring.",
    "NOT_ASSESSABLE": "Signal quality insufficient for automated assessment. Manual review of the "
                       "raw recording is recommended if clinically indicated.",
}

# Fields legitimately allowed to differ between the saved report and a fresh
# re-run of the same deterministic pipeline: a timestamp, and narrative/LLM
# status text.
_IGNORE_KEYS = {"generated_at", "narrative", "medgemma"}


# ---------------------------------------------------------------------------
# Loading / cross-referencing saved data (no computation of new decisions)
# ---------------------------------------------------------------------------

def find_raw_ecg_path(patient_id: str, segment_id: str) -> Path:
    stem = re.sub(r"_seg\d+$", "", segment_id)
    return DATA_RAW / "vitalpatch" / f"Patch_{patient_id}" / f"{stem}.csv"


def load_saved_report(patient_id: str, segment_id: str) -> dict:
    path = REPORT_ROOT / patient_id / f"{segment_id}.json"
    return json.loads(path.read_text())


def _strip_ignored(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in _IGNORE_KEYS}


def diff_reports(saved: dict, fresh: dict, path: str = "") -> list[str]:
    """Short list of human-readable differences (first divergence per key),
    recursing into nested dicts. Technical detail -- shown in the appendix,
    never the only place a material difference is surfaced (see
    build_drift_banner for the plain-language version)."""
    diffs = []
    a, b = _strip_ignored(saved) if not path else saved, _strip_ignored(fresh) if not path else fresh
    keys = set(a.keys()) | set(b.keys())
    for k in sorted(keys):
        p = f"{path}.{k}" if path else k
        av, bv = a.get(k, "<missing>"), b.get(k, "<missing>")
        if isinstance(av, dict) and isinstance(bv, dict):
            diffs.extend(diff_reports(av, bv, p))
        elif av != bv:
            diffs.append(f"{p}: saved={av!r} fresh={bv!r}")
    return diffs


def reports_match(saved: dict, fresh: dict) -> bool:
    return _strip_ignored(saved) == _strip_ignored(fresh)


def decision_matches(saved: dict, fresh: dict) -> bool:
    """The actual safety-critical claim: does the final risk level, and the
    specific rule that produced it, match exactly between the saved report
    and a fresh re-run? Non-deciding rule_trace rows and rhythm_findings can
    legitimately drift when a saved report predates a later code change
    without that drift ever having been able to change what was actually
    decided -- lower-priority rules cannot become the deciding rule
    retroactively."""
    return (saved["final_risk_level"] == fresh["final_risk_level"]
            and saved["deciding_rule"] == fresh["deciding_rule"])


def classify_match(saved: dict, fresh: dict) -> tuple[str, list[str]]:
    """Returns (status, raw_diffs). status is one of MATCH / DECISION_OK_DRIFT
    / MISMATCH. raw_diffs is the technical diff list (empty for MATCH),
    used for the appendix and for console reporting on MISMATCH."""
    if reports_match(saved, fresh):
        return "MATCH", []
    if not decision_matches(saved, fresh):
        return "MISMATCH", diff_reports(saved, fresh)
    return "DECISION_OK_DRIFT", diff_reports(saved, fresh)


def build_drift_banner(saved: dict, fresh: dict) -> list[str]:
    """Plain-language, clinically-focused summary of what current pipeline
    code finds differently from this saved report -- only called when
    classify_match() returned DECISION_OK_DRIFT (final risk level and
    deciding rule are already confirmed identical). Describes ADDED/REMOVED
    rhythm_findings by name and, for an added AFIB_SUSPECTED finding, the
    real burden percentage from the fresh rule_trace -- never a raw dict
    dump. This is the banner shown prominently on page 1, not buried in a
    diff a clinician would have to parse by eye."""
    def _key(f):
        return (f["kind"], f["start_time_s"])

    saved_keys = {_key(f) for f in saved["rhythm_findings"]}
    fresh_keys = {_key(f) for f in fresh["rhythm_findings"]}
    added = [f for f in fresh["rhythm_findings"] if _key(f) not in saved_keys]
    removed = [f for f in saved["rhythm_findings"] if _key(f) not in fresh_keys]

    fresh_rt = {r["condition"]: r for r in fresh["rule_trace"]}
    pac_afib_fresh = fresh_rt.get("PAC burden > HIGH threshold OR AFib burden > HIGH threshold")

    def _describe_group(group: list[dict]) -> list[str]:
        """One description per KIND, not per individual finding -- an
        AFIB_SUSPECTED burden % is a recording-level aggregate shared by
        every AFIB_SUSPECTED episode, so repeating the same percentage once
        per episode would just be noise. Non-AFib kinds still get their own
        time-stamped description since they don't share an aggregate."""
        by_kind: dict[str, list[dict]] = {}
        for f in group:
            by_kind.setdefault(f["kind"], []).append(f)
        descs = []
        for kind, items in by_kind.items():
            if kind == "AFIB_SUSPECTED" and pac_afib_fresh:
                burden = round(pac_afib_fresh["measured_value"].get("afib_burden_pct", 0), 1)
                threshold = pac_afib_fresh["threshold"].get("afib_burden_pct")
                n = len(items)
                descs.append(f"AFib-suspected pattern ({n} episode{'s' if n != 1 else ''}), "
                              f"{burden}% burden (threshold {threshold}%)")
            else:
                for f in items:
                    descs.append(f"{kind} ({f['start_time_s']:.1f}s-{f['start_time_s'] + f['duration_s']:.1f}s)")
        return descs

    lines = ["NOTE: This report was generated from a saved result."]
    if added:
        lines.append("Re-running with current pipeline code finds ADDITIONAL finding(s) not shown "
                      "below: " + "; ".join(_describe_group(added)) + ".")
    if removed:
        lines.append("Re-running with current pipeline code no longer finds: "
                      + "; ".join(_describe_group(removed))
                      + " (the version shown below is the saved one).")
    if not added and not removed:
        lines.append("Non-deciding numeric details differ from what current code computes "
                      "(see appendix for exact values).")
    lines.append("Final risk level and deciding rule are CONFIRMED UNCHANGED by current pipeline "
                  "code -- only the detail(s) above differ. Consider requesting a fresh report for "
                  "this segment if the added/removed finding(s) above matter to your review.")
    return lines


def recompute_filtered_signal(recording) -> tuple[np.ndarray, np.ndarray]:
    """Re-derives a DISPLAY-ONLY version of the filtered/resampled signal
    (stages 2-4, skipping the final Kalman EMG-suppression step) --
    deterministic signal processing, no classifier involved, needed only
    because PipelineResult does not expose this intermediate array. Returns
    (filtered, t_resampled_ms). Empty arrays if nothing survives the quality
    gate.

    WHY THE KALMAN STEP IS SKIPPED HERE (display only -- never for
    classification/decisions): emg_suppress_kalman's fixed absolute
    variances (process_var/meas_var, see ecg_pipeline_core.py) assume a
    small-mV-scale signal. VitalPatch's raw signal is uncalibrated ADC
    counts, 2-3 orders of magnitude larger in amplitude -- at that scale the
    filter's gain collapses and it smears every QRS into a decaying blob
    instead of tracking it (verified: peak-to-peak amplitude on a real
    segment dropped ~12x, from ~1420 to ~120, after this one step). This
    array is used ONLY to draw the waveform panels and to read off each
    beat's y-coordinate for the R-peak/classification markers -- it never
    feeds beat windowing, feature extraction, or the classifier (those come
    from `result`/`saved`, computed separately by the frozen pipeline, and
    are byte-identical whether or not this function skips Kalman). This
    mirrors a choice the pipeline already makes elsewhere: R-peak DETECTION
    also runs on a Kalman-skipped signal (see detect_and_segment), while
    only feature extraction uses the full Kalman'd output -- see
    emg_suppress_kalman's docstring for why that one can't be changed."""
    keep_mask, _ = run_sqi_gate(recording.signal_mv, recording.timestamps_ms,
                                 recording.fs_nominal, clip_value=None)
    signal_clean = recording.signal_mv.copy()
    signal_clean[~keep_mask] = np.nan

    resampled, t_resampled = to_target_rate(signal_clean, recording.timestamps_ms,
                                             recording.fs_nominal, TARGET_FS)
    valid = ~np.isnan(resampled)
    if len(resampled) == 0 or not valid.any():
        return np.array([]), np.array([])
    resampled_filled = np.interp(t_resampled, t_resampled[valid], resampled[valid])
    filtered = apply_filter_chain(resampled_filled, TARGET_FS,
                                   already_bandpass_filtered=recording.already_bandpass_filtered,
                                   skip_emg_suppress=True)
    return filtered, t_resampled


def load_vitals_contribution(segment_id: str) -> dict | None:
    """Looks up this segment in the (already-saved) multimodal batch
    manifest, if present. Returns None if not found -- never fabricated."""
    if not MULTIMODAL_MANIFEST.exists():
        return None
    if not hasattr(load_vitals_contribution, "_cache"):
        data = json.loads(MULTIMODAL_MANIFEST.read_text())
        load_vitals_contribution._cache = {r["segment"]: r for r in data["summary"]}
    return load_vitals_contribution._cache.get(segment_id)


# ---------------------------------------------------------------------------
# Plain-language derivations (all from already-computed saved values)
# ---------------------------------------------------------------------------

def _vitals_sentence(vitals_row: dict | None) -> str:
    if vitals_row is None:
        return "No matched vitals data is available for this segment; this reflects ECG findings only."
    if vitals_row["combined_risk"] != vitals_row["ecg_only_risk"]:
        return (f"Vital signs independently raised the risk level from {vitals_row['ecg_only_risk']} "
                f"to {vitals_row['combined_risk']}.")
    return "Vital signs do not increase the ECG-derived risk level."


def build_key_findings(saved: dict, vitals_row: dict | None) -> list[str]:
    """Translates fired rule_trace rows and rhythm_findings into plain
    bullets. Every bullet maps to a real fired rule, a real rhythm_findings
    entry, or a real vitals value already present in the saved/cross-
    referenced data -- nothing here is a new clinical claim."""
    bullets = []
    rt = {r["condition"]: r for r in saved["rule_trace"]}
    kind_counts = Counter(f["kind"] for f in saved["rhythm_findings"])

    crit = rt.get("PVC burden > CRITICAL threshold")
    high = rt.get("PVC burden > HIGH threshold")
    if crit and crit["fired"]:
        bullets.append(f"PVC burden {crit['measured_value']}% (frequent ventricular ectopy) -- "
                        f"exceeds the critical threshold ({crit['threshold']}%)")
    elif high and high["fired"]:
        bullets.append(f"PVC burden {high['measured_value']}% -- exceeds the high-concern threshold "
                        f"({high['threshold']}%)")
    elif high:
        bullets.append(f"PVC burden {high['measured_value']}% -- within normal range")

    if kind_counts.get("BIGEMINY"):
        n = kind_counts["BIGEMINY"]
        bullets.append(f"Ventricular bigeminy observed ({n} episode{'s' if n != 1 else ''})")
    if kind_counts.get("TRIGEMINY"):
        n = kind_counts["TRIGEMINY"]
        bullets.append(f"Ventricular trigeminy observed ({n} episode{'s' if n != 1 else ''})")

    vt = rt.get("VT run count > 0 (a run = >=3 consecutive V beats)")
    if vt:
        if vt["fired"]:
            bullets.append(f"Ventricular tachycardia run(s) detected: {vt['measured_value']} "
                            f"run(s) of >=3 consecutive V beats")
        else:
            bullets.append("No ventricular tachycardia detected")

    if kind_counts.get("AFIB_SUSPECTED"):
        n = kind_counts["AFIB_SUSPECTED"]
        bullets.append(f"AFib-suspected pattern observed ({n} episode{'s' if n != 1 else ''})")

    pac_afib = rt.get("PAC burden > HIGH threshold OR AFib burden > HIGH threshold")
    if pac_afib and pac_afib["fired"]:
        mv, th = pac_afib["measured_value"], pac_afib["threshold"]
        if mv.get("pac_burden_pct", 0) > th.get("pac_burden_pct", float("inf")):
            bullets.append(f"Frequent atrial ectopy: PAC burden {mv['pac_burden_pct']}% "
                            f"(threshold {th['pac_burden_pct']}%)")
        if mv.get("afib_burden_pct", 0) > th.get("afib_burden_pct", float("inf")):
            bullets.append(f"Elevated AFib burden: {mv['afib_burden_pct']}% of examined windows "
                            f"(threshold {th['afib_burden_pct']}%)")

    hrv = rt.get("Sustained HRV suppression: SDNN < threshold")
    if hrv and hrv["fired"]:
        bullets.append(f"Sustained HRV suppression (SDNN {hrv['measured_value']}ms < "
                        f"{hrv['threshold']}ms)")

    if vitals_row is not None and vitals_row.get("agent_news2_score") is not None:
        bullets.append(f"NEWS2 = {vitals_row['agent_news2_score']} "
                        f"({'vitals raise concern' if vitals_row['combined_risk'] != vitals_row['ecg_only_risk'] else 'vitals do not raise concern'})")
    else:
        bullets.append("NEWS2/qSOFA: not evaluated for this segment (no matched vitals data)")

    return bullets


def build_interpretation(saved: dict, vitals_row: dict | None) -> str:
    """One paragraph, selected by the deciding_rule's condition and filled
    in with the real measured values from that same rule_trace row -- no
    clinical language beyond translating those numbers into a sentence."""
    deciding = saved["deciding_rule"]
    condition = deciding["condition"]
    mv, th = deciding["measured_value"], deciding["threshold"]
    vitals_sentence = _vitals_sentence(vitals_row)
    kind_counts = Counter(f["kind"] for f in saved["rhythm_findings"])

    if condition == "PVC burden > CRITICAL threshold":
        n = kind_counts.get("BIGEMINY", 0)
        bigeminy = f" Ventricular bigeminy was also observed ({n} episode{'s' if n != 1 else ''})." if n else ""
        return (f"Frequent ventricular ectopic activity was detected. The PVC burden ({mv}%) "
                f"exceeds the predefined threshold for a critical ECG alert ({th}%).{bigeminy} "
                f"{vitals_sentence}")
    if condition == "PVC burden > HIGH threshold":
        return (f"Ventricular ectopic activity was detected at an elevated rate. The PVC burden "
                f"({mv}%) exceeds the threshold for a high-concern ECG alert ({th}%). {vitals_sentence}")
    if condition == "VT run count > 0 (a run = >=3 consecutive V beats)":
        return (f"A run of {mv} or more consecutive ventricular beats was detected, meeting the "
                f"criteria for a ventricular tachycardia run. {vitals_sentence}")
    if condition == "PAC burden > HIGH threshold OR AFib burden > HIGH threshold":
        parts = []
        if mv.get("pac_burden_pct", 0) > th.get("pac_burden_pct", float("inf")):
            parts.append(f"frequent atrial ectopy (PAC burden {mv['pac_burden_pct']}%, "
                          f"threshold {th['pac_burden_pct']}%)")
        if mv.get("afib_burden_pct", 0) > th.get("afib_burden_pct", float("inf")):
            parts.append(f"an atrial-fibrillation-suspected pattern (burden {mv['afib_burden_pct']}%, "
                          f"threshold {th['afib_burden_pct']}%)")
        return f"This segment shows {' and '.join(parts)}. {vitals_sentence}"
    if condition == "Sustained HRV suppression: SDNN < threshold":
        return (f"Sustained suppression of heart-rate variability was detected (SDNN {mv}ms, "
                f"below the {th}ms threshold). {vitals_sentence}")
    if condition == "NEWS2 safety override (>= critical threshold)":
        return (f"Vital signs alone met the criteria for a critical safety override (NEWS2 = {mv}, "
                f"threshold {th}), independent of the ECG findings. {vitals_sentence}")
    if condition == "qSOFA safety override (>= high threshold)":
        return (f"Vital signs alone met the criteria for a high-risk safety override (qSOFA = {mv}, "
                f"threshold {th}), independent of the ECG findings. {vitals_sentence}")
    if condition == "No thresholds exceeded":
        return (f"No rhythm pattern or ectopy burden crossed a risk-raising threshold in this "
                f"segment. {vitals_sentence}")
    return f"Deciding rule: {condition} (measured {mv} vs threshold {th}). {vitals_sentence}"


def determine_ai_confidence(saved: dict) -> tuple[str, str]:
    """Returns (tier, basis). Derived only from which deciding_rule condition
    fired and, for the combined PAC/AFib rule, which sub-condition actually
    exceeded threshold -- using the classifier's own documented per-class
    metrics (CLASSIFIER_METRICS) and the deterministic rule that already
    fired. No new clinical judgment."""
    condition = saved["deciding_rule"]["condition"]
    mv, th = saved["deciding_rule"]["measured_value"], saved["deciding_rule"]["threshold"]

    if condition in ("PVC burden > CRITICAL threshold", "PVC burden > HIGH threshold"):
        return "Moderate-High", "ventricular (V) beat"
    if condition == "PAC burden > HIGH threshold OR AFib burden > HIGH threshold":
        pac_exceeds = mv.get("pac_burden_pct", 0) > th.get("pac_burden_pct", float("inf"))
        afib_exceeds = mv.get("afib_burden_pct", 0) > th.get("afib_burden_pct", float("inf"))
        if pac_exceeds:
            return "Low", "supraventricular (S) beat"
        if afib_exceeds:
            return "High", "rhythm/timing (RR-interval) pattern"
        return "Moderate", "atrial ectopy/rhythm"
    if condition == "VT run count > 0 (a run = >=3 consecutive V beats)":
        return "High", "rhythm/timing pattern (consecutive-beat run)"
    if condition == "Sustained HRV suppression: SDNN < threshold":
        return "High", "rhythm/timing (heart-rate variability) pattern"
    if condition in ("NEWS2 safety override (>= critical threshold)",
                     "qSOFA safety override (>= high threshold)"):
        return "High", "vital-sign (NEWS2/qSOFA) score"
    if condition == "No thresholds exceeded":
        return "High", "absence of any rhythm or ectopy finding"
    return "Moderate", "the deciding rule shown in the appendix"


# ---------------------------------------------------------------------------
# Panels -- Page 1: header, drift banner, vitals
# ---------------------------------------------------------------------------

def _panel_header(ax, saved: dict):
    ax.axis("off")
    rec = saved["recording"]
    deciding = saved["deciding_rule"]
    lines = [
        "ECG Clinical Report -- Clinician Visual Review",
        "",
        f"Patient ID (VitalPatch): {rec['patient_id']}",
        f"Segment ID: {rec['segment_id']}",
        f"Source device: {rec['source']}",
        f"Duration: {rec['duration_s']}s   |   Nominal sample rate: {rec['fs_nominal_hz']} Hz"
        f"   |   Processed at: {rec['fs_processed_hz']} Hz",
        f"Beats detected / analyzed: {rec['n_beats_detected']} / {rec['n_beats_analyzed']}"
        f"   |   Quality score: {rec['quality_score']}",
        "",
        f"FINAL RISK LEVEL: {saved['final_risk_level']}",
        f"Deciding rule: {deciding['condition']}  "
        f"(measured {deciding['measured_value']} vs threshold {deciding['threshold']})",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=10,
            family="monospace", transform=ax.transAxes)


def _panel_drift_banner(ax, banner_lines: list[str]):
    ax.axis("off")
    if not banner_lines:
        return
    wrapped = "\n".join(textwrap.fill(line, width=100) for line in banner_lines)
    ax.text(0.02, 0.9, "⚠ " + wrapped, va="top", ha="left", fontsize=9,
            transform=ax.transAxes, color="#7a4a00",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#fff3cd", edgecolor="#e0a800"))


def _panel_vitals(ax, vitals_row: dict | None):
    ax.axis("off")
    ax.set_title("Vital Signs", fontsize=12, loc="left", fontweight="bold")

    if vitals_row is None:
        ax.text(0.0, 0.8, "No matched vitals data available for this segment.\n"
                           "This report reflects ECG-only findings.",
                va="top", ha="left", fontsize=10, transform=ax.transAxes)
        return

    def _fmt(v, unit):
        return f"{v}{unit}" if v is not None else "not available"

    rows = [
        ["HR", _fmt(vitals_row["hr"], " bpm")],
        ["RR", _fmt(vitals_row["rr"], " /min")],
        ["Temp", _fmt(vitals_row["temp"], "°C")],
        ["NEWS2", str(vitals_row["agent_news2_score"]) if vitals_row["agent_news2_score"] is not None else "not available"],
        ["qSOFA", str(vitals_row["agent_qsofa_score"]) if vitals_row["agent_qsofa_score"] is not None else "not available"],
    ]
    table = ax.table(cellText=rows, colLabels=["Vital", "Value"], loc="upper left",
                      cellLoc="left", colWidths=[0.18, 0.22], bbox=[0.0, 0.05, 0.4, 0.75])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    ax.text(0.48, 0.6, _vitals_sentence(vitals_row), va="top", ha="left", fontsize=10,
            transform=ax.transAxes, wrap=True)


# ---------------------------------------------------------------------------
# Panels -- Page 2: clinical summary, interpretation, actions, confidence
# ---------------------------------------------------------------------------

def _panel_key_findings(ax, saved: dict, vitals_row: dict | None):
    ax.axis("off")
    ax.set_title("Key Findings", fontsize=12, loc="left", fontweight="bold")
    bullets = build_key_findings(saved, vitals_row)
    text = "\n".join(f"• {b}" for b in bullets)
    ax.text(0.0, 0.95, text, va="top", ha="left", fontsize=10, transform=ax.transAxes)


def _panel_interpretation(ax, saved: dict, vitals_row: dict | None):
    ax.axis("off")
    ax.set_title("Interpretation", fontsize=12, loc="left", fontweight="bold")
    paragraph = textwrap.fill(build_interpretation(saved, vitals_row), width=100)
    ax.text(0.0, 0.85, paragraph, va="top", ha="left", fontsize=10, transform=ax.transAxes)


def _panel_suggested_actions(ax, saved: dict):
    ax.axis("off")
    ax.set_title("Suggested Clinical Actions", fontsize=12, loc="left", fontweight="bold")
    action = SUGGESTED_ACTIONS.get(saved["final_risk_level"], "No mapped action for this risk level.")
    paragraph = textwrap.fill(action, width=100)
    ax.text(0.0, 0.85, paragraph, va="top", ha="left", fontsize=10, transform=ax.transAxes)
    ax.text(0.0, 0.15, "This is decision support, not a diagnosis or treatment instruction.",
            va="top", ha="left", fontsize=8.5, style="italic", transform=ax.transAxes)


def _panel_ai_confidence(ax, saved: dict):
    ax.axis("off")
    ax.set_title("AI Confidence", fontsize=12, loc="left", fontweight="bold")
    tier, basis = determine_ai_confidence(saved)
    text = (f"AI Confidence: {tier}\n"
            f"Decision is primarily based on {basis} detection. Manual confirmation is "
            f"recommended before clinical action.\n\n"
            f"Full classifier reliability metrics per beat class are in the appendix.")
    ax.text(0.0, 0.85, text, va="top", ha="left", fontsize=10, transform=ax.transAxes)


# ---------------------------------------------------------------------------
# Panels -- ECG waveform panels (Phase 4 redesign: standard ECG-paper grid,
# stacked 10s strips at a fixed y-scale, a zoomed panel on the segment's
# longest rhythm finding with per-beat labels, and a one-time plain-language
# legend. Same underlying recording/filtered/beats/saved data as before --
# presentation only, no new computation.)
# ---------------------------------------------------------------------------

STRIP_SECONDS = 10.0
STRIPS_PER_PAGE = 4


def _draw_ecg_grid(ax, t0: float, t1: float, y0: float, y1: float):
    """Standard ECG-paper convention: minor gridline every 0.04s (1mm at
    25mm/s paper speed), major every 0.2s (5 minor divisions per major,
    matching a real strip's small/large box pattern). Horizontal gridlines
    are evenly spaced for visual rhythm/amplitude reference only -- this
    signal is uncalibrated, so they carry no mV meaning.

    Grid density and tick-label density are deliberately independent: a
    labeled tick every 0.2s would overlap into an unreadable smear, so grid
    lines are drawn directly (not via the tick locator) and labels are
    placed at a much sparser, span-dependent interval."""
    ax.set_xlim(t0, t1)
    ax.set_ylim(y0, y1)
    ax.set_facecolor("#fff6f6")

    minor_x = np.arange(np.ceil(t0 / 0.04) * 0.04, t1 + 1e-9, 0.04)
    major_x = np.arange(np.ceil(t0 / 0.2) * 0.2, t1 + 1e-9, 0.2)
    ax.vlines(minor_x, y0, y1, color="#f5cccc", linewidth=0.25, alpha=0.5, zorder=0)
    ax.vlines(major_x, y0, y1, color="#e08a8a", linewidth=0.5, alpha=0.7, zorder=0)
    y_major = np.linspace(y0, y1, 7)
    ax.hlines(y_major, t0, t1, color="#e08a8a", linewidth=0.5, alpha=0.7, zorder=0)
    ax.set_yticks([])

    span = t1 - t0
    label_step = 1.0 if span <= 15 else (2.0 if span <= 40 else 5.0)
    ticks = np.arange(np.ceil(t0 / label_step) * label_step, t1 + 1e-9, label_step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:.0f}" for v in ticks], fontsize=8)
    ax.tick_params(axis="x", length=3)


def _y_range_with_margin(y: np.ndarray, margin_frac: float = 0.15) -> tuple[float, float]:
    finite = y[np.isfinite(y)]
    if len(finite) == 0:
        return -1.0, 1.0
    lo, hi = float(np.min(finite)), float(np.max(finite))
    span = (hi - lo) or 1.0
    return lo - margin_frac * span, hi + margin_frac * span


def _beat_r_xy(beats: list, filtered: np.ndarray):
    """Splits beats into (accepted, rejected) (t_s, y) coordinate arrays for
    scatter-plotting R-peaks over the filtered signal."""
    accepted_t, accepted_y, rejected_t, rejected_y = [], [], [], []
    for b in beats:
        r_s = b.r_peak_ms / 1000.0
        y = filtered[b.r_peak_idx] if 0 <= b.r_peak_idx < len(filtered) else np.nan
        if b.quality_rejected:
            rejected_t.append(r_s); rejected_y.append(y)
        else:
            accepted_t.append(r_s); accepted_y.append(y)
    return ((np.array(accepted_t), np.array(accepted_y)),
            (np.array(rejected_t), np.array(rejected_y)))


def _panel_legend(ax):
    ax.axis("off")
    ax.set_title("How to read the ECG plots below", fontsize=12, loc="left", fontweight="bold")
    beat_key = ", ".join(f"{k}={BEAT_LABEL_NAMES[k]}" for k in BEAT_COLORS)
    lines = [
        "Each strip below shows 10 seconds of signal, left to right, like a printed ECG strip.",
        "The pink grid follows standard ECG paper convention: small boxes = 0.04s, "
        "large boxes (5 small boxes) = 0.2s.",
        "Amplitude is the device's raw, uncalibrated units -- NOT millivolts. Compare shape and "
        "timing, not absolute height, across recordings or patients.",
        f"Colored markers show each detected heartbeat's classification: {beat_key}.",
        "A zoomed panel later in this report shows, beat by beat, the specific rhythm episode "
        "that drove this segment's risk level.",
    ]
    ax.text(0.0, 0.82, "\n".join(f"• {textwrap.fill(l, width=100, subsequent_indent='    ')}"
                                  for l in lines),
            va="top", ha="left", fontsize=9.5, transform=ax.transAxes)


def _panel_overview_raw(ax, recording, y_range: tuple[float, float]):
    t_s = (recording.timestamps_ms - recording.timestamps_ms[0]) / 1000.0
    ax.plot(t_s, recording.signal_mv, color="#1f77b4", linewidth=0.4)
    ax.set_title("Raw ECG, full-segment overview (before filtering/SQI gating) -- "
                  "see 10s strips below for detail. Amplitude is raw device units, NOT calibrated mV.",
                  fontsize=9.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude\n(raw device units)")
    ax.set_xlim(t_s[0] if len(t_s) else 0, t_s[-1] if len(t_s) else 1)
    ax.set_ylim(*y_range)


def _panel_overview_filtered(ax, filtered: np.ndarray, accepted, rejected, rej_rate: float,
                              y_range: tuple[float, float]):
    t_s = np.arange(len(filtered)) / TARGET_FS
    ax.plot(t_s, filtered, color="#1f77b4", linewidth=0.4, zorder=1)
    at, ay = accepted
    rt, ry = rejected
    ax.scatter(at, ay, color="red", s=14, zorder=3, label="Accepted R-peak")
    ax.scatter(rt, ry, color="grey", s=22, marker="x", zorder=3, label="Rejected (quality gate)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(f"Filtered signal, full-segment overview (SQI window rejection rate: {rej_rate:.2%}) "
                  "-- see 10s strips below for detail", fontsize=9.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude\n(raw device units)")
    ax.set_xlim(t_s[0] if len(t_s) else 0, t_s[-1] if len(t_s) else 1)
    ax.set_ylim(*y_range)


def _plot_strip(ax, t_full: np.ndarray, y_full: np.ndarray, t0: float, t1: float,
                 y_range: tuple[float, float], accepted=None, rejected=None):
    mask = (t_full >= t0) & (t_full < t1)
    ax.plot(t_full[mask], y_full[mask], color="#222222", linewidth=0.7, zorder=2)
    _draw_ecg_grid(ax, t0, t1, *y_range)
    ax.set_ylabel(f"{t0:.0f}s", rotation=0, ha="right", va="center", fontsize=8, labelpad=14)
    if accepted is not None:
        at, ay = accepted
        m = (at >= t0) & (at < t1)
        ax.scatter(at[m], ay[m], color="red", s=12, zorder=3)
    if rejected is not None:
        rt, ry = rejected
        m = (rt >= t0) & (rt < t1)
        ax.scatter(rt[m], ry[m], color="grey", s=18, marker="x", zorder=3)


def _stacked_strip_pages(pdf, t_full: np.ndarray, y_full: np.ndarray, seg_duration_s: float,
                          y_range: tuple[float, float], page_title: str, accepted=None, rejected=None,
                          strips_per_page: int = STRIPS_PER_PAGE):
    """One or more pages of stacked STRIP_SECONDS-wide panels, all sharing
    `y_range` so amplitude is visually comparable strip-to-strip -- the
    primary diagnostic view (the single full-segment overview panel is for
    orientation only, not for reading individual QRS complexes)."""
    n_strips = max(1, int(np.ceil(seg_duration_s / STRIP_SECONDS)))
    strip_starts = [i * STRIP_SECONDS for i in range(n_strips)]
    for page_start in range(0, n_strips, strips_per_page):
        page_strips = strip_starts[page_start: page_start + strips_per_page]
        fig = plt.figure(figsize=(11, 2.3 * len(page_strips) + 1))
        gs = fig.add_gridspec(len(page_strips), 1, hspace=0.6)
        for i, t0 in enumerate(page_strips):
            t1 = min(t0 + STRIP_SECONDS, seg_duration_s)
            ax = fig.add_subplot(gs[i])
            _plot_strip(ax, t_full, y_full, t0, t1, y_range, accepted=accepted, rejected=rejected)
            if i == 0:
                ax.set_title(f"{page_title} ({page_strips[0]:.0f}-{min(page_strips[-1] + STRIP_SECONDS, seg_duration_s):.0f}s "
                              f"of {seg_duration_s:.0f}s)", fontsize=10, loc="left")
            if i == len(page_strips) - 1:
                ax.set_xlabel("Time (s)")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _longest_finding(saved: dict) -> dict | None:
    findings = saved["rhythm_findings"]
    if not findings:
        return None
    return max(findings, key=lambda f: f["duration_s"])


def _panel_zoomed_finding(pdf, filtered: np.ndarray, beats: list, beat_labels: list[str], saved: dict):
    """Zooms into the segment's single longest-duration rhythm finding
    (padded by 1s each side), overlaying each beat's N/S/V/F/Q label so
    individual QRS complexes and their classification are readable -- the
    detail the compressed overview strip can't show."""
    finding = _longest_finding(saved)
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 0.6, 0.6], hspace=0.8)
    ax_wave = fig.add_subplot(gs[0])
    ax_track = fig.add_subplot(gs[1])
    ax_caption = fig.add_subplot(gs[2])
    ax_caption.axis("off")

    if finding is None:
        ax_wave.axis("off")
        ax_track.axis("off")
        ax_wave.text(0.5, 0.5, "No rhythm findings in this segment -- no episode to zoom into.",
                     ha="center", va="center", fontsize=10, style="italic", transform=ax_wave.transAxes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    pad_s = 1.0
    t0 = max(0.0, finding["start_time_s"] - pad_s)
    t1 = finding["start_time_s"] + finding["duration_s"] + pad_s
    i0 = int(t0 * TARGET_FS)
    i1 = min(len(filtered), int(round(t1 * TARGET_FS)))
    t = np.arange(i0, i1) / TARGET_FS
    y = filtered[i0:i1]
    y_range = _y_range_with_margin(y, margin_frac=0.35)

    ax_wave.plot(t, y, color="#222222", linewidth=0.9, zorder=2)
    _draw_ecg_grid(ax_wave, t0, t1, *y_range)
    ax_wave.set_ylabel("Amplitude\n(raw device units)")

    in_window = [i for i, b in enumerate(beats) if t0 <= b.r_peak_ms / 1000.0 <= t1]
    for i in in_window:
        b = beats[i]
        rt = b.r_peak_ms / 1000.0
        ry = filtered[b.r_peak_idx] if 0 <= b.r_peak_idx < len(filtered) else np.nan
        label = beat_labels[i]
        color = BEAT_COLORS.get(label, "#000000")
        ax_wave.scatter([rt], [ry], color=color, s=36, zorder=4, edgecolor="white", linewidth=0.5)
        ax_wave.annotate(label, (rt, ry), textcoords="offset points", xytext=(0, 11),
                         ha="center", fontsize=9, fontweight="bold", color=color)
    ax_wave.set_title(f"Zoomed view: {finding['kind']} "
                       f"({finding['start_time_s']:.1f}s-{finding['start_time_s'] + finding['duration_s']:.1f}s), "
                       "the longest rhythm finding in this segment", fontsize=10, loc="left")

    window_idx_labels = [(beats[i].r_peak_ms / 1000.0, beat_labels[i]) for i in in_window]
    ax_track.set_xlim(t0, t1)
    ax_track.set_yticks([])
    ax_track.set_title("Beat-by-beat classification, this window only (wider strip -- individually countable)",
                        fontsize=9, loc="left")
    for wt, wl in window_idx_labels:
        ax_track.axvline(wt, color=BEAT_COLORS.get(wl, "#000000"), linewidth=3.0)
    ax_track.set_xlabel("Time (s)")

    kind_counts = Counter(f["kind"] for f in saved["rhythm_findings"])
    n_similar = kind_counts.get(finding["kind"], 1) - 1
    similar_note = (f" {n_similar} more episode{'s' if n_similar != 1 else ''} of {finding['kind']} "
                     f"occur elsewhere in this segment." if n_similar > 0 else
                     f" This is the only {finding['kind']} episode in this segment.")
    label_counts = Counter(wl for _, wl in window_idx_labels)
    count_str = ", ".join(f"{v} {BEAT_LABEL_NAMES[k]}" for k, v in label_counts.items()) or "no beats detected"
    caption_lines = [
        f"Evidence: {finding.get('evidence_text', '')}",
        f"This {t1 - t0:.1f}s window contains {len(in_window)} beat(s): {count_str}.{similar_note}",
    ]
    wrapped = "\n".join(textwrap.fill(line, width=110, subsequent_indent="    ") for line in caption_lines)
    ax_caption.text(0.0, 1.0, wrapped, va="top", ha="left", fontsize=8.5, transform=ax_caption.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _panel_beat_track(ax, beats: list, beat_labels: list[str]):
    """A continuous color strip: each beat owns the time span from the
    midpoint to its previous neighbor through the midpoint to its next
    neighbor, colored by its class."""
    ax.set_yticks([])
    ax.set_title("Beat-by-beat classification (colored strip, one segment per beat)")
    ax.set_xlabel("Time (s)")
    if not beats:
        return

    times = [b.r_peak_ms / 1000.0 for b in beats]
    n = len(times)
    bounds = [times[0] - (times[1] - times[0]) / 2 if n > 1 else times[0] - 0.5]
    for i in range(n - 1):
        bounds.append((times[i] + times[i + 1]) / 2)
    bounds.append(times[-1] + (times[-1] - times[-2]) / 2 if n > 1 else times[0] + 0.5)

    spans = [(bounds[i], bounds[i + 1] - bounds[i]) for i in range(n)]
    colors = [BEAT_COLORS.get(label, "#000000") for label in beat_labels]
    ax.broken_barh(spans, (0, 1), facecolors=colors, edgecolor="none")

    handles = [plt.Line2D([0], [0], color=c, lw=6, label=f"{k} ({BEAT_LABEL_NAMES[k]})")
               for k, c in BEAT_COLORS.items()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.35),
              ncol=5, fontsize=7, frameon=False)
    ax.set_ylim(0, 1)
    ax.set_xlim(bounds[0], bounds[-1])


def _panel_rhythm_timeline(ax, saved: dict):
    """Bars show WHERE each finding occurred, with a short on-chart label
    that never overflows the axis; full evidence text is numbered on the
    chart and listed below it."""
    findings = saved["rhythm_findings"]
    duration = saved["recording"]["duration_s"]
    ax.set_xlim(0, duration)
    ax.set_ylim(0, max(len(findings), 1) + 1)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Rhythm findings timeline ({len(findings)} finding(s))")

    if not findings:
        ax.text(0.5, 0.5, "No rhythm findings in this segment.", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, style="italic")
        return

    kind_colors = {"VT_RUN": "#d62728", "BIGEMINY": "#ff7f0e", "TRIGEMINY": "#e6b800",
                   "AFIB_SUSPECTED": "#9467bd"}
    evidence_lines = []
    for i, f in enumerate(findings):
        start = f["start_time_s"]
        dur = max(f["duration_s"], 0.3)
        color = kind_colors.get(f["kind"], "#1f77b4")
        row = i + 1
        ax.barh(row, dur, left=start, height=0.6, color=color, alpha=0.85)
        short_label = f"[{i + 1}] {f['kind']} {start:.1f}-{start + f['duration_s']:.1f}s"
        ax.text(min(start, duration * 0.75), row + 0.4, short_label, fontsize=7, va="bottom", ha="left")
        wrapped = textwrap.fill(f"[{i + 1}] {f['evidence_text']}", width=110, subsequent_indent="    ")
        evidence_lines.append(wrapped)

    ax.text(0.0, -0.25, "\n".join(evidence_lines), transform=ax.transAxes,
            fontsize=6.5, va="top", ha="left")


# ---------------------------------------------------------------------------
# Panels -- Clinician verdict
# ---------------------------------------------------------------------------

def _panel_verdict(ax):
    ax.axis("off")
    lines = [
        DISCLAIMER,
        "",
        "Clinician verdict:",
        "  [ ] Agree -- CRITICAL is clinically justified",
        "  [ ] Disagree -- this appears to be a false alarm",
        "  [ ] Uncertain -- needs more context",
        "",
        "Notes: " + "_" * 70,
        "",
        "_" * 90,
        "_" * 90,
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=10, transform=ax.transAxes)


# ---------------------------------------------------------------------------
# Panels -- Appendix (technical detail, for reference only)
# ---------------------------------------------------------------------------

def _fmt_rule_value(v) -> str:
    if isinstance(v, dict):
        return "\n".join(f"{k}={v2}" for k, v2 in v.items())
    return str(v)


def _panel_appendix_rule_cascade(ax, saved: dict):
    ax.axis("off")
    ax.set_title("Appendix -- Technical Detail (For Reference Only)\nFull rule_trace "
                  "(highlighted row = deciding rule)", fontsize=10, loc="left")
    rule_trace = saved["rule_trace"]
    col_labels = ["Rule", "Threshold", "Measured Value", "Fired?", "Would Set Level"]
    cell_text = []
    row_colors = []
    for r in rule_trace:
        condition = textwrap.fill(r["condition"], width=38)
        cell_text.append([
            condition,
            _fmt_rule_value(r["threshold"]),
            _fmt_rule_value(r["measured_value"]),
            "YES" if r.get("fired") else "no",
            r["would_set_level"],
        ])
        row_colors.append("#ffe08a" if r.get("is_deciding_rule") else "#ffffff")

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="left",
                      colWidths=[0.38, 0.14, 0.22, 0.10, 0.16])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 2.4)
    for ri, color in enumerate(row_colors, start=1):
        for ci in range(len(col_labels)):
            table[ri, ci].set_facecolor(color)
            if color != "#ffffff":
                table[ri, ci].set_text_props(weight="bold")


def _panel_appendix_metrics_and_diff(ax, saved: dict, raw_diffs: list[str]):
    ax.axis("off")
    lines = ["Production classifier per-class metrics (DS2 held-out, 45,804 beats):"]
    for cls, m in CLASSIFIER_METRICS.items():
        lines.append(f"  {cls}: sensitivity={m['sensitivity']}, precision={m['precision']}, "
                      f"F1={m['f1']}, support={m['support']}")
    lines.append("")
    lines.append(f"Reporting layer's own confidence statement: \"{saved['confidence']['statement']}\"")
    lines.append(f"Caveat: {saved['confidence']['caveat']}")
    lines.append("")
    lines.append("Confidence tier shown in this report is a heuristic, not a calibrated "
                  "probability (Docs/README.md, 'Known and unfixed').")
    lines.append("")
    lines.append("Waveform panels in this report show a DISPLAY-ONLY reconstruction of the filtered "
                  "signal that skips the pipeline's final EMG-smoothing (Kalman) step -- that step's "
                  "fixed noise-variance constants assume a much smaller signal scale than this device "
                  "produces and otherwise smear every QRS complex into an unreadable blob. This "
                  "affects plotting only: beat classification, rhythm findings, and the risk decision "
                  "above are all computed by the pipeline's own frozen output and are unaffected.")

    if raw_diffs:
        lines.append("")
        lines.append("Raw saved-vs-fresh technical diff (see page 1 banner for the plain-language "
                      "version of this):")
        for d in raw_diffs:
            lines.append(f"  - {d}")

    wrapped = []
    for line in lines:
        wrapped.append(textwrap.fill(line, width=110, subsequent_indent="    ") if line else line)
    ax.text(0.0, 1.0, "\n".join(wrapped), va="top", ha="left", fontsize=7.5, family="monospace",
            transform=ax.transAxes)


# ---------------------------------------------------------------------------
# Per-segment orchestration
# ---------------------------------------------------------------------------

def render_segment(patient_id: str, segment_id: str, classifier, out_dir: Path) -> dict:
    try:
        saved = load_saved_report(patient_id, segment_id)
    except FileNotFoundError as e:
        return {"segment_id": segment_id, "status": "ERROR", "detail": f"saved report missing: {e}"}

    raw_path = find_raw_ecg_path(patient_id, segment_id)
    if not raw_path.exists():
        return {"segment_id": segment_id, "status": "ERROR", "detail": f"raw ECG file not found: {raw_path}"}

    recordings = parse_vitalpatch_ecg(raw_path)
    recording = next((r for r in recordings if r.segment_id == segment_id), None)
    if recording is None:
        return {"segment_id": segment_id, "status": "ERROR",
                "detail": f"segment_id not found among {len(recordings)} recording(s) re-parsed from {raw_path}"}

    fresh_report, result = run_full_report(recording, classifier)

    match_status, raw_diffs = classify_match(saved, fresh_report)
    if match_status == "MISMATCH":
        return {"segment_id": segment_id, "status": "MISMATCH", "detail": raw_diffs}
    banner_lines = build_drift_banner(saved, fresh_report) if match_status == "DECISION_OK_DRIFT" else []

    filtered, _ = recompute_filtered_signal(recording)
    if len(filtered) == 0:
        return {"segment_id": segment_id, "status": "ERROR",
                "detail": "no signal survived the quality gate on re-run (unexpected given saved report is assessable)"}

    vitals_row = load_vitals_contribution(segment_id)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{segment_id}.pdf"
    n_findings = len(saved["rhythm_findings"])

    with PdfPages(pdf_path) as pdf:
        # Page 1: header, drift banner (if any), vitals
        fig = plt.figure(figsize=(11, 9 if banner_lines else 7.5))
        ratios = [1.1, 0.9, 1.6] if banner_lines else [1.1, 1.6]
        gs = fig.add_gridspec(len(ratios), 1, height_ratios=ratios, hspace=0.5)
        _panel_header(fig.add_subplot(gs[0]), saved)
        if banner_lines:
            _panel_drift_banner(fig.add_subplot(gs[1]), banner_lines)
            _panel_vitals(fig.add_subplot(gs[2]), vitals_row)
        else:
            _panel_vitals(fig.add_subplot(gs[1]), vitals_row)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: clinical summary, interpretation, suggested actions, AI confidence
        fig = plt.figure(figsize=(11, 10))
        gs = fig.add_gridspec(4, 1, height_ratios=[1.3, 1.1, 0.9, 0.9], hspace=0.6)
        _panel_key_findings(fig.add_subplot(gs[0]), saved, vitals_row)
        _panel_interpretation(fig.add_subplot(gs[1]), saved, vitals_row)
        _panel_suggested_actions(fig.add_subplot(gs[2]), saved)
        _panel_ai_confidence(fig.add_subplot(gs[3]), saved)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: legend + raw-waveform full-segment overview
        seg_duration_s = saved["recording"]["duration_s"]
        raw_t_s = (recording.timestamps_ms - recording.timestamps_ms[0]) / 1000.0
        raw_y_range = _y_range_with_margin(recording.signal_mv)
        fig = plt.figure(figsize=(11, 8.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.3], hspace=0.5)
        _panel_legend(fig.add_subplot(gs[0]))
        _panel_overview_raw(fig.add_subplot(gs[1]), recording, raw_y_range)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page(s): raw waveform, stacked 10s strips at a consistent y-scale
        _stacked_strip_pages(pdf, raw_t_s, recording.signal_mv, seg_duration_s, raw_y_range,
                              page_title="Raw ECG, 10-second strips")

        # Page: filtered signal, full-segment overview with R-peaks
        filtered_t_s = np.arange(len(filtered)) / TARGET_FS
        filtered_y_range = _y_range_with_margin(filtered)
        accepted, rejected = _beat_r_xy(result.beats, filtered)
        rej_rate = saved["recording"]["sqi_window_rejection_rate"]
        fig = plt.figure(figsize=(11, 4.5))
        _panel_overview_filtered(fig.add_subplot(111), filtered, accepted, rejected, rej_rate, filtered_y_range)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page(s): filtered signal, stacked 10s strips with R-peaks marked
        _stacked_strip_pages(pdf, filtered_t_s, filtered, seg_duration_s, filtered_y_range,
                              page_title="Filtered signal, 10-second strips (R-peaks marked)",
                              accepted=accepted, rejected=rejected)

        # Page: zoomed view of the segment's longest rhythm finding, beat-by-beat
        _panel_zoomed_finding(pdf, filtered, result.beats, result.beat_labels, saved)

        # Page: beat track (compressed overview strip) + rhythm timeline (unchanged)
        timeline_ratio = 0.9 + 0.18 * n_findings
        fig = plt.figure(figsize=(11, min(7 + 0.3 * n_findings, 14)))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, timeline_ratio], hspace=0.5)
        _panel_beat_track(fig.add_subplot(gs[0]), result.beats, result.beat_labels)
        _panel_rhythm_timeline(fig.add_subplot(gs[1]), saved)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page: clinician verdict
        fig = plt.figure(figsize=(11, 5))
        ax = fig.add_subplot(111)
        _panel_verdict(ax)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page: appendix -- technical detail, for reference only
        fig = plt.figure(figsize=(11, 11))
        gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.6], hspace=0.5)
        _panel_appendix_rule_cascade(fig.add_subplot(gs[0]), saved)
        _panel_appendix_metrics_and_diff(fig.add_subplot(gs[1]), saved, raw_diffs)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    status = "OK_WITH_DRIFT_BANNER" if banner_lines else "OK"
    return {"segment_id": segment_id, "status": status, "path": str(pdf_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="process only the first N sample rows (0 = all)")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(SAMPLE_CSV)))
    if args.limit:
        rows = rows[: args.limit]

    classifier = load_classifier(MODELS_DIR / "five_class_xgb.json")

    results = []
    for row in rows:
        res = render_segment(row["patient_id"], row["segment_id"], classifier, OUT_DIR)
        results.append(res)
        rendered = res["status"] in ("OK", "OK_WITH_DRIFT_BANNER")
        print(f"[{res['status']}] {res['segment_id']}"
              + (f" -> {res.get('path')}" if rendered else f" -- {res.get('detail')}"))

    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_drift = sum(1 for r in results if r["status"] == "OK_WITH_DRIFT_BANNER")
    n_mismatch = sum(1 for r in results if r["status"] == "MISMATCH")
    n_error = sum(1 for r in results if r["status"] == "ERROR")
    print(f"\nDONE: {n_ok} OK, {n_drift} OK_WITH_DRIFT_BANNER, "
          f"{n_mismatch} MISMATCH, {n_error} ERROR (of {len(results)})")
    if n_mismatch or n_error:
        print("MISMATCH/ERROR segments were NOT rendered. See detail above for each.")


if __name__ == "__main__":
    main()
