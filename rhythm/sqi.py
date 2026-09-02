"""Stage 3 - signal quality index, computed per window and used as a GATE.

The system must be able to say "the signal is too poor to judge". A
classifier that always answers is lying some of the time.

Design is driven by the measured failure mode (reports_rhythm/s04_*): the
detector fails mainly by OVER-DETECTION, and a false R-peak splits one RR
interval into two, manufacturing irregularity that no threshold can survive
(record 113: expert CV 0.099 -> detector CV 0.349).

The primary component is therefore bSQI - agreement between two independent
R-peak detectors - which is precisely what collapses when peaks are spurious.
Supporting components catch the signal-level causes.

None of these use annotations, so every one is computable on device data.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import signal as sps
from scipy.stats import kurtosis

BSQI_TOL_MS = 150.0


@dataclass(frozen=True)
class SQI:
    bsqi: float          # 0-1, two-detector agreement (primary)
    rr_lag1: float       # lag-1 autocorr of RR; strongly negative = alternation
    ksqi: float          # kurtosis; QRS-dominated signal is peaky
    psqi: float          # fraction of 1-40 Hz power sitting in the QRS band
    flatline_frac: float # fraction of samples in a dead/flat run
    sat_frac: float      # fraction of samples pinned at the amplitude extremes
    passed: bool
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def bsqi(peaks_a_ms: np.ndarray, peaks_b_ms: np.ndarray,
         tol_ms: float = BSQI_TOL_MS) -> float:
    """Clifford bSQI: |matched| / |union of both detectors' peaks|.

    1.0 = the two detectors agree exactly. Spurious peaks are found by one
    detector and not the other, so they drive this down.
    """
    a = np.sort(np.asarray(peaks_a_ms, dtype=float))
    b = np.sort(np.asarray(peaks_b_ms, dtype=float))
    if a.size == 0 and b.size == 0:
        return 0.0
    if a.size == 0 or b.size == 0:
        return 0.0
    used = np.zeros(b.size, bool)
    matched = 0
    j = 0
    for t in a:
        while j < b.size and b[j] < t - tol_ms:
            j += 1
        k, best, bd = j, -1, tol_ms
        while k < b.size and b[k] <= t + tol_ms:
            if not used[k] and abs(b[k] - t) <= bd:
                bd, best = abs(b[k] - t), k
            k += 1
        if best >= 0:
            used[best] = True
            matched += 1
    union = a.size + b.size - matched
    return float(matched / union) if union else 0.0


def rr_lag1(rr_ms: np.ndarray) -> float:
    """Lag-1 autocorrelation of the RR series.

    Catches what bSQI cannot: CORRELATED over-detection, where both detectors
    make the same T-wave error so they agree with each other and bSQI stays
    high. Inserting a spurious peak produces short/long alternation, which
    shows up as strongly negative lag-1 autocorrelation.

    Measured on MITDB (reports_rhythm/s05b_sqi_rr.csv): median +0.011 on
    windows the detector gets right, -0.065 on windows it gets wrong; among
    windows passing bSQI>=0.95 but still wrong, -0.082 vs +0.020.

    Note: a genuinely alternating rhythm (bigeminy) also lowers this. The
    consequence is a refusal, not a misclassification, which is the safe
    direction for a regularity indicator.
    """
    rr = np.asarray(rr_ms, float)
    rr = rr[np.isfinite(rr) & (rr > 0)]
    if rr.size < 4:
        return 0.0
    d = rr - rr.mean()
    denom = float(np.sum(d * d))
    return float(np.sum(d[:-1] * d[1:]) / denom) if denom > 0 else 0.0


def signal_quality(x: np.ndarray, fs: float) -> dict:
    """Signal-level SQI components for one window."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < int(fs):
        return dict(ksqi=0.0, psqi=0.0, flatline_frac=1.0, sat_frac=1.0)

    xc = x - np.mean(x)
    scale = np.std(xc)

    ksqi = float(kurtosis(xc, fisher=False)) if scale > 0 else 0.0

    # power in the QRS band relative to the diagnostic band
    nper = min(int(fs * 4), xc.size)
    f, p = sps.welch(xc, fs=fs, nperseg=max(nper, 16))
    band = (f >= 1) & (f <= min(40.0, fs / 2 - 1))
    qrs = (f >= 5) & (f <= min(15.0, fs / 2 - 1))
    tot = float(p[band].sum())
    psqi = float(p[qrs].sum() / tot) if tot > 0 else 0.0

    # flat runs: consecutive samples with near-zero derivative
    d = np.abs(np.diff(xc))
    thr = max(1e-12, 0.001 * scale)
    flat = d < thr
    flatline_frac = float(flat.mean()) if flat.size else 1.0

    # saturation: samples pinned at the observed extremes
    lo, hi = xc.min(), xc.max()
    rng = hi - lo
    sat = ((xc <= lo + 1e-6 * rng) | (xc >= hi - 1e-6 * rng)) if rng > 0 else np.ones_like(xc, bool)
    sat_frac = float(sat.mean())

    return dict(ksqi=ksqi, psqi=psqi, flatline_frac=flatline_frac, sat_frac=sat_frac)


def assess(x: np.ndarray, fs: float,
           peaks_a_ms: np.ndarray, peaks_b_ms: np.ndarray,
           rr_ms: np.ndarray | None = None,
           *, bsqi_min: float = 0.95, lag1_min: float = -0.5, ksqi_min: float = 2.0,
           psqi_min: float = 0.10, flatline_max: float = 0.30,
           sat_max: float = 0.20) -> SQI:
    """Combine components into a pass/fail gate with an explicit reason.

    Thresholds are parameters, not constants: bsqi_min in particular is
    DERIVED from data (see scripts_rhythm/s05_sqi_derive.py) and must be
    reported alongside the resolution it was derived at.
    """
    comp = signal_quality(x, fs)
    b = bsqi(peaks_a_ms, peaks_b_ms)
    if rr_ms is None:
        rr_ms = np.diff(np.sort(np.asarray(peaks_a_ms, float)))
    lag1 = rr_lag1(rr_ms)

    reasons = []
    if b < bsqi_min:
        reasons.append(f"bSQI {b:.3f} < {bsqi_min:.3f} (detectors disagree)")
    if lag1 < lag1_min:
        reasons.append(f"RR lag-1 {lag1:+.3f} < {lag1_min:+.3f} (short/long alternation)")
    if comp["ksqi"] < ksqi_min:
        reasons.append(f"kSQI {comp['ksqi']:.2f} < {ksqi_min:.2f} (no QRS-like peaks)")
    if comp["psqi"] < psqi_min:
        reasons.append(f"pSQI {comp['psqi']:.3f} < {psqi_min:.3f} (little QRS-band power)")
    if comp["flatline_frac"] > flatline_max:
        reasons.append(f"flatline {comp['flatline_frac']:.0%} > {flatline_max:.0%}")
    if comp["sat_frac"] > sat_max:
        reasons.append(f"saturated {comp['sat_frac']:.0%} > {sat_max:.0%}")

    return SQI(bsqi=b, rr_lag1=lag1, **comp, passed=not reasons,
               reason="ok" if not reasons else "; ".join(reasons))
