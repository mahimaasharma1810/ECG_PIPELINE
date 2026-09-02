"""RATE axis - bradycardic / normal / tachycardic / undetermined.

THRESHOLD PROVENANCE, stated rather than assumed:

  Normal band 60-100 bpm is ADOPTED clinical convention. It is SUPPORTED, not
  replaced, by measurement: across 15,016 LTAFDB windows labelled (N (sinus
  rhythm), heart rate runs p05 63.7 - p95 91.1 bpm, median 75.0. The derived
  band sits inside the convention.

  CAVEAT on that support: LTAFDB is a cardiac-patient cohort at rest, not a
  general healthy population, so it evidences the convention rather than
  establishing a population reference.

  Extreme bands 40 / 150 bpm are ADOPTED convention with NO derivation and NO
  validation in this project. They are marked as such in output.

RATE HAS ITS OWN TRUST CRITERION, NOT THE REGULARITY GATE.

The bSQI gate exists to protect beat-POSITION precision, which the regularity
axis needs because it measures interval-to-interval variation. Rate is a MEAN
over 64 beats and is far more tolerant of small position errors - the device
cross-check shows mean rate accurate to ~4% even while regularity is unusable.

Gating rate on bSQI therefore refuses a trustworthy measurement for a reason
that does not apply to it, and defeats the purpose of independent axes.
Measured: doing so marks rate UNDETERMINED in 97% of device windows, while the
rate values themselves are physiologically sound (median 85.1 bpm, p05 70.9,
p95 99.8, zero extremes across 1,210 windows of healthy adults).

Rate instead requires SIGNAL-level quality only - that a real ECG is present:
kurtosis, QRS-band power, flatline and saturation. Not detector agreement.

RATE IS AN INDEPENDENT AXIS AND MUST NEVER FEED THE REGULARITY DECISION.
Heart rate looks like a strong predictor of irregularity (AUC 0.9221) but that
is a COHORT ARTEFACT: irregular episodes in these databases happen to run fast
(median 112 vs 75 bpm). A model given rate would learn "fast means irregular".
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

BRADY_CRITICAL = 40.0
NORMAL_LOW = 60.0
NORMAL_HIGH = 100.0
TACHY_CRITICAL = 150.0

BRADYCARDIC = "BRADYCARDIC"
NORMAL = "NORMAL"
TACHYCARDIC = "TACHYCARDIC"
UNDETERMINED = "UNDETERMINED"

PROVENANCE = {
    "normal_low_bpm": NORMAL_LOW,
    "normal_high_bpm": NORMAL_HIGH,
    "normal_band_basis": "ADOPTED convention, SUPPORTED by measurement",
    "normal_band_evidence": ("LTAFDB sinus-rhythm windows (n=15,016): "
                             "HR p05 63.7, median 75.0, p95 91.1 bpm"),
    "extreme_low_bpm": BRADY_CRITICAL,
    "extreme_high_bpm": TACHY_CRITICAL,
    "extreme_band_basis": "ADOPTED convention - NOT derived, NOT validated here",
    "status": "PROVISIONAL - NOT SHIPPABLE",
}


@dataclass(frozen=True)
class RateAxis:
    state: str
    hr_bpm: float | None
    extreme: bool          # outside 40-150 bpm
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def rate_trustworthy(sqi, features) -> tuple[bool, str]:
    """Rate's OWN trust criterion: is a real ECG present?

    Deliberately EXCLUDES bSQI and RR lag-1, which guard beat-position
    precision for the regularity axis and are far stricter than a mean rate
    requires.
    """
    if not features.usable:
        return False, features.reason
    if sqi is None:
        return True, "ok"
    if sqi.ksqi < 2.0:
        return False, f"kSQI {sqi.ksqi:.2f} - no QRS-like peaks"
    if sqi.psqi < 0.10:
        return False, f"pSQI {sqi.psqi:.3f} - little QRS-band power"
    if sqi.flatline_frac > 0.30:
        return False, f"flatline {sqi.flatline_frac:.0%}"
    if sqi.sat_frac > 0.20:
        return False, f"saturated {sqi.sat_frac:.0%}"
    return True, "ok"


def assess_rate(features, trustworthy: bool, reason: str = "") -> RateAxis:
    """Classify heart rate. Refuses independently of the other axes."""
    hr = features.hr_bpm
    if not trustworthy:
        return RateAxis(UNDETERMINED, None, False,
                        reason or "signal quality insufficient for a rate")
    if hr is None or hr != hr:
        return RateAxis(UNDETERMINED, None, False, "heart rate not computable")

    hr = float(hr)
    if hr < BRADY_CRITICAL:
        return RateAxis(BRADYCARDIC, hr, True,
                        f"{hr:.0f} bpm is below {BRADY_CRITICAL:.0f} "
                        f"(adopted convention, not derived here)")
    if hr > TACHY_CRITICAL:
        return RateAxis(TACHYCARDIC, hr, True,
                        f"{hr:.0f} bpm is above {TACHY_CRITICAL:.0f} "
                        f"(adopted convention, not derived here)")
    if hr < NORMAL_LOW:
        return RateAxis(BRADYCARDIC, hr, False,
                        f"{hr:.0f} bpm is below the normal band "
                        f"{NORMAL_LOW:.0f}-{NORMAL_HIGH:.0f}")
    if hr > NORMAL_HIGH:
        return RateAxis(TACHYCARDIC, hr, False,
                        f"{hr:.0f} bpm is above the normal band "
                        f"{NORMAL_LOW:.0f}-{NORMAL_HIGH:.0f}")
    return RateAxis(NORMAL, hr, False,
                    f"{hr:.0f} bpm is within {NORMAL_LOW:.0f}-{NORMAL_HIGH:.0f}")
