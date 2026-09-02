"""MedGemma JSON: an ALLOW-LIST, not a dump.

The narrative validator requires every number in the prose to appear in the
evidence JSON. The JSON therefore DEFINES THE VOCABULARY of what the model is
permitted to say - a number absent from it cannot be spoken.

Design consequences:
  * only numbers a clinician needs are included;
  * raw ECG samples are excluded (large, and unspeakable anyway);
  * derived-but-unvalidated quantities are excluded - if you do not want a
    number spoken, keep it out;
  * provisional status and the scope statement travel in EVERY file.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from . import SCOPE_STATEMENT
from .verdict import THRESHOLD_CV, REFUSAL_HALF_WIDTH, THRESHOLD_PROVENANCE, UNABLE
from .smoothing import DEFAULT_N_CONSECUTIVE
from .sqi import BSQI_TOL_MS

SCHEMA_VERSION = "1.0"

REQUIRED_TOP = ("schema_version", "window_id", "patient_id", "recorded_at_utc",
                "verdict", "verdict_reason", "evidence", "decision", "quality",
                "provenance", "scope_statement", "artifacts")
REQUIRED_EVIDENCE = ("n_beats", "n_intervals", "mean_rr_ms", "heart_rate_bpm",
                     "rr_cv", "rmssd_ms", "n_intervals_flagged",
                     "pct_intervals_flagged")

# --- HARD DISPLAY CONSTRAINT, enforced structurally --------------------------
# A rhythm verdict must NEVER be serialisable without the heart rate beside it.
# Measured reason: 350 windows (7.7%) across MITDB+SVDB read REGULAR at HR>100
# bpm, up to 161 bpm - sustained fast rhythms are evenly spaced, so a
# regularity-only view calls them normal. A further 339 windows (7.5%) read
# REGULAR below 60 bpm. The rate layer catches 100% of both, but only if the
# rate travels with the verdict.
#
# This is enforced here rather than in documentation because if the pair CAN be
# separated in the schema, something will eventually render the verdict alone.
VERDICT_REQUIRES = ("heart_rate_bpm", "rate_state")
# Exact key names that would carry raw signal into the document. Matched
# exactly, not as substrings: an earlier substring rule rejected the legitimate
# key "L1_signal_quality" because it contains "signal".
FORBIDDEN = frozenset(("samples", "amplitude", "amplitudes", "raw_ecg", "ecg",
                       "ecg_clean", "waveform", "signal", "x", "sig"))


def _r(v, dp):
    return None if v is None or v != v else round(float(v), dp)


class DisplayConstraintError(ValueError):
    """Raised when a rhythm verdict would be emitted without its rate."""


def build(window_id: str, patient_id: str, recorded_at_utc: str,
          verdict, features, sqi, sqi_passed: bool,
          consecutive_agreeing: int, rate=None, events=None,
          morphology=None, artifacts: dict | None = None) -> dict:
    """Assemble the MedGemma JSON for one window.

    `rate` is REQUIRED whenever a rhythm verdict is present. Omitting it raises
    rather than producing a document that could render a verdict alone.
    """
    if rate is None:
        raise DisplayConstraintError(
            "a rhythm verdict may not be serialised without its rate layer: "
            "350 windows in public data read REGULAR at HR>100 bpm (max 161), "
            "and a REGULAR verdict at that rate is actively misleading alone")
    return {
        "schema_version": SCHEMA_VERSION,
        "window_id": window_id,
        "patient_id": str(patient_id),
        "recorded_at_utc": recorded_at_utc,

        "verdict": verdict.verdict,
        "verdict_reason": None if verdict.verdict != UNABLE else verdict.reason,

        "evidence": {
            "n_beats": int(features.n_intervals + 1),
            "n_intervals": int(features.n_intervals),
            "mean_rr_ms": _r(features.mean_rr_ms, 0),
            "heart_rate_bpm": _r(features.hr_bpm, 0),
            "rr_cv": _r(features.rr_cv, 3),
            "rmssd_ms": _r(features.rmssd_ms, 0),
            "n_intervals_flagged": int(features.n_flagged),
            "pct_intervals_flagged": _r(features.flagged_fraction * 100, 1),
        },

        "decision": {
            "threshold_rr_cv": THRESHOLD_CV,
            "refusal_band": [round(THRESHOLD_CV - REFUSAL_HALF_WIDTH, 4),
                             round(THRESHOLD_CV + REFUSAL_HALF_WIDTH, 4)],
            "hysteresis_n": DEFAULT_N_CONSECUTIVE,
            "consecutive_agreeing_windows": int(consecutive_agreeing),
        },

        "quality": {
            "bsqi": _r(sqi.bsqi, 2),
            "sqi_passed": bool(sqi_passed),
            "gate": "bsqi>=0.95 AND (sdnn<=15ms OR lag1>=-0.5)",
        },

        # --- layers, each carrying its own status -------------------------
        "layers": {
            "L1_signal_quality": {"passed": bool(sqi_passed),
                                  "status": "PROVISIONAL"},
            "L2_rate": {"state": rate.state, "hr_bpm": _r(rate.hr_bpm, 0),
                        "extreme": bool(rate.extreme),
                        "status": "PROVISIONAL"},
            "L3_regularity": {"verdict": verdict.verdict,
                              "actionable": bool(verdict.actionable),
                              "status": "PROVISIONAL - PRIMARY, DETERMINISTIC"},
            "L4_ectopy_burden": (
                {"burden": events.burden, "couplets": events.ectopic_couplets,
                 "pauses": events.pause_count, "reports_type": False,
                 "status": "PROVISIONAL"} if events is not None else
                {"status": "NOT_COMPUTED"}),
            "L5_ectopy_type": (
                morphology if morphology is not None else
                {"status": "UNTESTED-ON-DEVICE - not computed"}),
        },

        "provenance": {
            "threshold_status": THRESHOLD_PROVENANCE["status"],
            "threshold_source": (
                f"LTAFDB n={THRESHOLD_PROVENANCE['derived_on_records']} records, "
                f"Youden, resampled to {THRESHOLD_PROVENANCE['resampled_rate_hz']} Hz"),
            "validated_on_device": THRESHOLD_PROVENANCE["validated_on_device"],
            "held_out_database": (
                f"MITDB: Se {THRESHOLD_PROVENANCE['held_out_se']} "
                f"Sp {THRESHOLD_PROVENANCE['held_out_sp']} "
                f"PPV {THRESHOLD_PROVENANCE['held_out_ppv']} "
                f"NPV {THRESHOLD_PROVENANCE['held_out_npv']}"),
            "shippable_blocked_on": THRESHOLD_PROVENANCE["shippable_blocked_on"],
        },

        "scope_statement": SCOPE_STATEMENT,
        "artifacts": artifacts or {},
    }


def validate_schema(doc: dict) -> tuple[bool, list]:
    """Structural check. Returns (ok, failures)."""
    f = []
    for k in REQUIRED_TOP:
        if k not in doc:
            f.append(f"missing top-level key {k!r}")
    for k in REQUIRED_EVIDENCE:
        if k not in doc.get("evidence", {}):
            f.append(f"missing evidence key {k!r}")
    # the display constraint, re-checked on the finished document
    ev = doc.get("evidence", {})
    if doc.get("verdict") and not all(
            ev.get(k) is not None or doc.get("layers", {}).get("L2_rate", {}).get(k.replace("heart_rate_bpm", "hr_bpm"))
            is not None for k in ("heart_rate_bpm",)):
        f.append("verdict present without heart rate - display constraint violated")
    if doc.get("verdict") and "L2_rate" not in doc.get("layers", {}):
        f.append("verdict present without the L2 rate layer - display constraint violated")
    if doc.get("scope_statement") != SCOPE_STATEMENT:
        f.append("scope_statement absent or altered")
    if "PROVISIONAL" not in str(doc.get("provenance", {}).get("threshold_status", "")):
        f.append("provisional status does not travel in this file")

    def scan(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                if str(k).lower() in FORBIDDEN:
                    f.append(f"forbidden raw-signal key at {path}{k}")
                scan(v, f"{path}{k}.")
        elif isinstance(o, list) and len(o) > 64:
            f.append(f"suspiciously long array at {path} (raw samples?)")
    scan(doc)
    return (not f), f
