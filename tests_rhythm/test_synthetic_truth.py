"""TASK B - synthetic ground truth: signals where the answer is known exactly.

The quantisation-alternation bug and the 3-spurious-peaks finding both came
from ad hoc synthetic tests. Formalising them means they stay caught.

Run:  python3 -m pytest tests_rhythm/test_synthetic_truth.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm.features import (window_features, screen_rr, segment_aware_rmssd,
                             BEATS_PER_WINDOW, INTERVALS_PER_WINDOW)
from rhythm.verdict import decide, THRESHOLD_CV, REFUSAL_HALF_WIDTH, UNABLE, REGULAR, IRREGULAR
from rhythm.sqi import assess, rr_lag1, signal_quality
from rhythm.degrade import DEVICE_FS

GRID_MS = 1000.0 / DEVICE_FS
SEED = 20260829


def quantise(peaks_ms):
    """Snap true peak times onto the device's 89.70 Hz sample grid."""
    return np.round(np.asarray(peaks_ms, float) / GRID_MS) * GRID_MS


def regular_peaks(rr_ms, n=BEATS_PER_WINDOW):
    return quantise(rr_ms * np.arange(n, dtype=float))


def synth_ecg(peaks_ms, fs=DEVICE_FS, width_s=0.012):
    """A QRS-like spike train at the given peak times."""
    dur = peaks_ms[-1] / 1000.0 + 1.0
    t = np.arange(0, dur, 1.0 / fs)
    x = np.zeros_like(t)
    for p in peaks_ms / 1000.0:
        x += np.exp(-((t - p) ** 2) / (2 * width_s ** 2))
    return t, x


# ---------------------------------------------------------------- regular
@pytest.mark.parametrize("hr,rr", [(50, 1200.0), (75, 800.0), (100, 600.0)])
def test_perfectly_regular_is_REGULAR_at_the_noise_floor(hr, rr):
    f = window_features(np.diff(regular_peaks(rr)))
    v = decide(f, sqi_passed=True, sqi_reason="ok")
    assert v.verdict == REGULAR, f"HR {hr}: got {v.verdict}"
    # quantisation floor measured at 0.0044-0.0071 across 50-100 bpm
    assert f.rr_cv < 0.010, f"HR {hr}: CV {f.rr_cv:.4f} above the ~0.006 floor"


# ---------------------------------------------------------------- AF-like
def test_af_like_rr_is_IRREGULAR():
    rng = np.random.default_rng(SEED)
    # AF-like: broad, near-exponential RR scatter around ~700 ms
    rr = np.clip(rng.gamma(shape=6.0, scale=115.0, size=INTERVALS_PER_WINDOW), 350, 1600)
    # build peak TIMES, quantise to the device grid, then take RR from them -
    # so the test exercises the same resolution the real pipeline sees
    peaks = quantise(np.r_[0.0, np.cumsum(rr)])
    f = window_features(np.diff(peaks))
    v = decide(f, sqi_passed=True, sqi_reason="ok")
    assert f.rr_cv > THRESHOLD_CV + REFUSAL_HALF_WIDTH, f"CV {f.rr_cv:.4f} too low"
    assert v.verdict == IRREGULAR


# ------------------------------------------------- 3 spurious peaks/window
def test_three_spurious_peaks_reproduce_measured_cv():
    """Regression on the finding that ~3 false peaks per window give CV ~0.156."""
    rr = np.full(57, 700.0)
    split = np.repeat([350.0], 6)          # 3 splits -> 6 short intervals
    f = window_features(np.r_[rr, split])
    assert 0.14 < f.rr_cv < 0.17, f"CV {f.rr_cv:.4f} outside the measured 0.156 band"
    v = decide(f, sqi_passed=True, sqi_reason="ok")
    assert v.verdict == IRREGULAR


# ---------------------------------------------------------------- bigeminy
def test_bigeminy_is_refused_by_rr_lag1_not_called_irregular():
    """Alternating short/long RR is ambiguous - the system must refuse."""
    rr = np.tile([500.0, 1000.0], INTERVALS_PER_WINDOW // 2 + 1)[:INTERVALS_PER_WINDOW]
    f = window_features(rr)
    assert rr_lag1(rr) < -0.5, f"lag-1 {rr_lag1(rr):+.3f} should be strongly negative"
    peaks = np.r_[0, np.cumsum(rr)]
    _, x = synth_ecg(peaks)
    q = assess(x, DEVICE_FS, peaks, peaks, rr_ms=rr)
    assert not q.passed and "lag-1" in q.reason
    v = decide(f, sqi_passed=q.passed, sqi_reason=q.reason)
    assert v.verdict == UNABLE


def test_regular_rhythm_survives_the_lag1_guard():
    """The variance guard must not let lag-1 reject a steady rhythm.

    On the 11.148 ms grid a perfectly regular rhythm alternates between two
    quantisation levels, giving lag-1 down to -0.56 at RR 1200. Unguarded,
    this rejected the MOST regular signals - worst at low heart rates.
    """
    rr = np.diff(regular_peaks(1200.0))
    assert rr_lag1(rr) < -0.5, "precondition: quantisation alternation present"
    f = window_features(rr)
    assert f.sdnn_ms <= 15.0, "precondition: SDNN at the quantisation floor"
    # pipeline applies the guard: lag-1 is ignored below the SDNN threshold
    v = decide(f, sqi_passed=True, sqi_reason="ok")
    assert v.verdict == REGULAR


# ------------------------------------------------------------ RR artefacts
def test_dropped_sample_run_flags_the_interval():
    rr = np.full(INTERVALS_PER_WINDOW, 800.0)
    rr[20] = 2500.0                       # a gap left by dropped samples
    f = window_features(rr)
    assert f.n_flagged == 1
    flagged = screen_rr(rr)
    rmssd, pairs, segs = segment_aware_rmssd(rr, flagged)
    assert segs == 2 and pairs == INTERVALS_PER_WINDOW - 3
    assert rmssd == 0.0, "RMSSD must not span the excluded interval"


def test_over_20pct_flagged_is_UNABLE():
    rr = np.full(INTERVALS_PER_WINDOW, 800.0)
    rr[:20] = 2500.0                      # 20/63 = 31.7%
    f = window_features(rr)
    v = decide(f, sqi_passed=True, sqi_reason="ok")
    assert f.flagged_fraction > 0.20
    assert not f.usable and v.verdict == UNABLE
    assert "outside" in v.reason and "300" in v.reason


def test_fewer_than_64_beats_is_UNABLE():
    f = window_features(np.full(40, 800.0))
    v = decide(f, sqi_passed=True, sqi_reason="ok")
    assert v.verdict == UNABLE and "40 intervals" in v.reason


# ------------------------------------------------------------ signal faults
def test_flatline_is_refused():
    x = np.zeros(int(DEVICE_FS * 45))
    c = signal_quality(x, DEVICE_FS)
    assert c["flatline_frac"] > 0.30


def test_saturated_signal_is_refused():
    x = np.tile([1000.0, -1000.0], int(DEVICE_FS * 45) // 2)
    c = signal_quality(x, DEVICE_FS)
    assert c["sat_frac"] > 0.20


# ------------------------------------------------------------ refusal band
def test_cv_inside_refusal_band_is_UNABLE():
    lo, hi = THRESHOLD_CV - REFUSAL_HALF_WIDTH, THRESHOLD_CV + REFUSAL_HALF_WIDTH
    rng = np.random.default_rng(SEED)
    target = (lo + hi) / 2
    rr = 800.0 * (1 + rng.normal(0, target, INTERVALS_PER_WINDOW))
    rr = 800.0 + (rr - rr.mean()) * (target * 800.0 / rr.std(ddof=1))
    f = window_features(rr)
    assert lo <= f.rr_cv <= hi, f"setup: CV {f.rr_cv:.4f} not in band"
    v = decide(f, sqi_passed=True, sqi_reason="ok")
    assert v.verdict == UNABLE and "refusal band" in v.reason


def test_64_beats_is_63_intervals():
    assert INTERVALS_PER_WINDOW == 63 and BEATS_PER_WINDOW == 64
    f = window_features(np.diff(regular_peaks(800.0, BEATS_PER_WINDOW)))
    assert f.n_intervals == 63
