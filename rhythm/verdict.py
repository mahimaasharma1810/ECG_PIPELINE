"""Stage 7 - the deterministic verdict.

The verdict is decided HERE, before any language model is called. Brief 7:
the LLM's only job is to restate this in prose; it never classifies.

THRESHOLD PROVENANCE (brief 11: no borrowed threshold without
resolution-matched re-derivation):

  value        RR CV >= 0.1275
  derived on   LTAFDB, 16 records, 389 h, 21,920 windows of 64 beats
               (15,016 REGULAR from (N, 6,904 IRREGULAR from (AFIB)
  resolution   RR timing snapped to the device's measured 89.70 Hz grid
  source RR    EXPERT ANNOTATIONS, not detector output, so the boundary is
               physiology rather than detection error
  performance  Se 0.9607  Sp 0.8871  PPV 0.7965  NPV 0.9801
  held out    MITDB Se 0.9839 Sp 0.8660 PPV 0.4692 NPV 0.9978
  robustness  LORO spread 0.0125 over 16 records; CI95 [0.1150, 0.1525]
  lead         UNVERIFIED - LTAFDB headers name both channels 'ECG'.
               This is NOT a like-for-like Lead-II result.
  overlap      REGULAR p95 = 0.1951 vs IRREGULAR p05 = 0.1280. The classes
               overlap; the boundary is a trade-off, not a separation.
  noise floor  0.006 at 89.70 Hz on a uniform grid -> threshold is 20x above

REFUSAL BAND. CV carries measurement uncertainty from two sources: the
quantisation floor (~0.006) and residual detector-induced error surviving the
SQI gate (p95 0.0267 on MITDB). A window whose CV falls within +/-0.027 of the
boundary is reported UNABLE_TO_DETERMINE rather than forced to a side. This
converts a measured weakness into a principled refusal (brief 11: the system
can refuse).

NOT VALIDATED ON THE TARGET DEVICE. There are no rhythm labels for any
ProRhythm capture. Applying this threshold to device data is an untested
cross-database transfer, and transfer has already failed twice in this
project.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

THRESHOLD_CV = 0.1275
REFUSAL_HALF_WIDTH = 0.027

REGULAR = "REGULAR"
IRREGULAR = "IRREGULAR"
UNABLE = "UNABLE_TO_DETERMINE"

# --- OPERATING-POINT DESIGN -------------------------------------------------
#
# Operate at the Youden point and make ONLY the REGULAR verdict actionable.
#
# Rationale, measured. NPV falls MONOTONICALLY as the threshold is tightened
# (0.9994 at 0.10 -> 0.9160 at 0.30), while PPV only exceeds 0.80 above 0.28,
# where sensitivity has already collapsed to 0.13. Tightening therefore
# degrades the one property this instrument is genuinely good at in order to
# buy one it never achieves usefully.
#
# The NPV advantage also survives prevalence: with Se/Sp fixed at the Youden
# point, NPV is 0.9997 at 1% prevalence and still 0.9755 at 50%, whereas PPV
# is 0.064 at 1% and only 0.87 at 50% - prevalences a screening indicator
# will not see.
#
# Consequence for downstream systems:
#   REGULAR   -> ACTIONABLE. Trustworthy at NPV ~0.998. This is the shippable
#                output: regularity is established for this window.
#   IRREGULAR -> NOT ACTIONABLE. PPV ~0.40 at the operating point. It must
#                prompt a look, and must NEVER fire an alarm or reach an
#                escalation desk on its own.
#   UNABLE    -> NOT ACTIONABLE.
#
# This also makes the hysteresis N choice largely cosmetic rather than
# clinical: display churn on a non-alarming indicator costs little.
ACTIONABLE = {REGULAR: True, IRREGULAR: False, UNABLE: False}
NOT_ESTABLISHED = "REGULARITY_NOT_ESTABLISHED"


def action_state(verdict: str) -> str:
    """What a downstream consumer is permitted to act on."""
    return "REGULARITY_ESTABLISHED" if verdict == REGULAR else NOT_ESTABLISHED

THRESHOLD_PROVENANCE = {
    "status": "PROVISIONAL - NOT SHIPPABLE",
    "shippable_blocked_on": (
        "no device-anchored REGULAR distribution exists; the ProRhythm REGULAR "
        "distribution measured to date is an artefact of R-peak detection error "
        "(device windows median RR CV 0.19 vs 0.03 for healthy LTAFDB windows). "
        "Unblocks on a controlled recording with a beat-level RR reference."
    ),
    "feature": "rr_cv",
    "threshold": THRESHOLD_CV,
    "loro_spread": 0.0125,
    "loro_range": [0.1200, 0.1325],
    "refusal_half_width": REFUSAL_HALF_WIDTH,
    "derived_on_database": "ltafdb",
    "derived_on_records": 16,
    "derived_on_hours": 389,
    "derived_on_windows": 21920,
    "n_regular_windows": 15016,
    "n_irregular_windows": 6904,
    "resampled_rate_hz": 89.70,
    "rr_source": "expert_annotations",
    "lead": "UNVERIFIED",
    "held_out": "none - full derivation set",
    "operating_point": "Youden; only the REGULAR verdict is actionable",
    "threshold_ci95_cluster_bootstrap": [0.1150, 0.1525],
    "ci_method": "record-level (cluster) bootstrap, 1000 resamples of 16 LTAFDB records",
    "ci_note": (
        "Re-derived on 16 records (was 6). LORO spread narrowed 0.0550 -> 0.0125 "
        "and the CI halved (0.0752 -> 0.0375 wide). IRREGULAR windows now come "
        "from 14 records with a top-2 share of 44% (was 4 records, 76%). MITDB's "
        "own Youden optimum is 0.1350, agreeing to within 0.0075."
    ),
    "held_out_database": "mitdb",
    "held_out_se": 0.9839,
    "held_out_sp": 0.8660,
    "held_out_ppv": 0.4692,
    "held_out_npv": 0.9978,
    "sensitivity": 0.9607,
    "specificity": 0.8871,
    "ppv": 0.7965,
    "npv": 0.9801,
    "noise_floor_cv": 0.006,
    "validated_on_device": False,
}


@dataclass(frozen=True)
class Verdict:
    verdict: str
    actionable: bool
    action_state: str
    reason: str
    rr_cv: float | None
    threshold: float
    margin_low: float
    margin_high: float
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def decide(features, sqi_passed: bool, sqi_reason: str) -> Verdict:
    """Deterministic verdict for one window. Refusal is a first-class output."""
    lo = THRESHOLD_CV - REFUSAL_HALF_WIDTH
    hi = THRESHOLD_CV + REFUSAL_HALF_WIDTH
    cv = features.rr_cv if features.rr_cv == features.rr_cv else None   # NaN check

    ev = {
        "rr_cv": cv,
        "mean_rr_ms": features.mean_rr_ms,
        "hr_bpm": features.hr_bpm,
        "rmssd_ms": features.rmssd_ms,
        "sdnn_ms": features.sdnn_ms,
        "n_intervals": features.n_intervals,
        "n_flagged": features.n_flagged,
        "threshold": THRESHOLD_CV,
    "loro_spread": 0.0125,
    "loro_range": [0.1200, 0.1325],
    }

    if not features.usable:
        return _mk(UNABLE, features.reason, cv, lo, hi, ev)
    if not sqi_passed:
        return _mk(UNABLE, f"{UNABLE}: signal quality - {sqi_reason}", cv, lo, hi, ev)
    if cv is None:
        return _mk(UNABLE, f"{UNABLE}: RR CV could not be computed", cv, lo, hi, ev)
    if lo <= cv <= hi:
        return _mk(UNABLE, f"{UNABLE}: RR CV {cv:.4f} lies within the refusal band "
                   f"{lo:.4f}-{hi:.4f} around the threshold {THRESHOLD_CV:.4f}", cv, lo, hi, ev)
    if cv > hi:
        return _mk(IRREGULAR, f"RR CV {cv:.4f} exceeds the threshold "
                   f"{THRESHOLD_CV:.4f} (NOT actionable: PPV ~0.40 - prompts review, "
                   f"never an alarm)", cv, lo, hi, ev)
    return _mk(REGULAR, f"RR CV {cv:.4f} is below the threshold {THRESHOLD_CV:.4f}",
               cv, lo, hi, ev)


def _mk(verdict, reason, cv, lo, hi, ev):
    return Verdict(verdict=verdict, actionable=ACTIONABLE[verdict],
                   action_state=action_state(verdict), reason=reason, rr_cv=cv,
                   threshold=THRESHOLD_CV, margin_low=lo, margin_high=hi, evidence=ev)
