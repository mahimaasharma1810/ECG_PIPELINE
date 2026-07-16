"""Stage 6 — Feature extraction.

Two parallel outputs:
  1. The 56-dimensional per-beat feature vector (Zhu et al. 2021): 5
     morphological + 51 wavelet coefficients. Kept as the interpretable,
     no-training-data-required baseline feature set — SHAP-ranked and fed
     to the XGBoost classifier in `classify.py`.
  2. Recording-level HRV/frequency features (SDNN, RMSSD, pNN50, LF/HF,
     QRS-width trend) for the stage 8 risk scorer.

The learned encoder in `encoder.py` (recommendation #1/#2) is a SEPARATE,
richer representation computed from the same beat windows — it does not
replace this module, it supplements it, so the pipeline still works with
zero training data (this module) while the encoder is being pretrained.

NOTE (2026-07-14): a 7-feature local-rhythm-context extension (rolling
RR-ratio at K=8/16/32, rr_pre/rr_post ratio, RR CV, normalized
prematurity score, compensatory-pause flag) was built and evaluated here
as an experiment to fix S-class underperformance. It improved the DS1_VAL
validation split (Macro-F1 0.3526->0.3775, S F1 0.200->0.270) but caused
a real regression on the DS2 held-out test set (V F1 0.856->0.762, driven
by a jump in true-S-beats-predicted-as-V from 22%->62%) that the small
4-record validation split failed to catch. Reverted per explicit decision
after reviewing both results — see ABLATION_REPORT.md for the full
before/after tables and confusion matrices. Kept as documented history so
this isn't silently re-attempted without re-reading why it was dropped.
"""
from __future__ import annotations

import warnings

import numpy as np
import pywt

from .beats import Beat
from .filters import robust_zscore

# The spec's 75-sample primary window (200ms pre + R + 400ms post @ 125Hz) is
# shorter than what a level-4 db4 decomposition ideally wants, so every
# beat trips pywt's boundary-effects warning. The coefficients are still
# valid (just more boundary-influenced) and are resampled to a fixed
# length below regardless, so the warning is expected noise, not a bug.
warnings.filterwarnings("ignore", message="Level value of.*is too high", module="pywt")

N_MORPHOLOGICAL = 5
N_WAVELET = 51
N_FEATURES = N_MORPHOLOGICAL + N_WAVELET


def _wavelet_features(window: np.ndarray, wavelet: str = "db4") -> np.ndarray:
    """Discrete wavelet decomposition -> 14 approximation (a4) + 23 detail
    (d3) + 14 detail (d4) coefficients, resampled to fixed lengths so the
    feature vector has constant dimensionality regardless of input length.
    """
    coeffs = pywt.wavedec(window, wavelet, level=4)
    # coeffs = [a4, d4, d3, d2, d1]
    a4, d4, d3 = coeffs[0], coeffs[1], coeffs[2]

    def _fixed_len(arr: np.ndarray, target_len: int) -> np.ndarray:
        if len(arr) == target_len:
            return arr
        x_old = np.linspace(0, 1, len(arr))
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_old, arr)

    return np.concatenate([
        _fixed_len(a4, 14),
        _fixed_len(d3, 23),
        _fixed_len(d4, 14),
    ])


def _morphological_features(window: np.ndarray, primary_pre_samples: int,
                             rr_pre_ms: float | None, rr_post_ms: float | None) -> np.ndarray:
    r_idx = primary_pre_samples

    rr_pre = rr_pre_ms if rr_pre_ms is not None else 0.0
    local_hrv = (rr_post_ms - rr_pre_ms) if (rr_pre_ms and rr_post_ms) else 0.0

    left = window[:r_idx]
    right = window[r_idx:]
    left_area, right_area = float(np.sum(np.abs(left))), float(np.sum(np.abs(right)))
    area_ratio = left_area / right_area if right_area > 1e-9 else 0.0

    r_amp = window[r_idx]
    above = window[window > 0]
    below = window[window < 0]
    above_below_ratio = (np.sum(above) / abs(np.sum(below))) if len(below) and np.sum(below) != 0 else 0.0

    amplitude_range = float(np.ptp(window))

    return np.array([rr_pre, local_hrv, area_ratio, above_below_ratio, amplitude_range])


def beat_feature_vector(beat: Beat, primary_pre_samples: int) -> np.ndarray | None:
    """Step 6 of the filter chain (robust Z-score, per beat window) is
    applied here, right before feature extraction — quality rejection
    upstream (`beats.py`) intentionally runs on the raw window instead, so
    normalization can't mask a genuinely flat/low-amplitude beat."""
    if beat.primary_window is None or beat.quality_rejected:
        return None
    normalized = robust_zscore(beat.primary_window)
    morph = _morphological_features(normalized, primary_pre_samples, beat.rr_pre_ms, beat.rr_post_ms)
    wavelet = _wavelet_features(normalized)
    return np.concatenate([morph, wavelet])


def batch_feature_matrix(beats: list[Beat], primary_pre_samples: int) -> tuple[np.ndarray, list[int]]:
    """Returns (feature_matrix, indices_into_beats_used) — skipping
    rejected/out-of-bounds beats but preserving which original beat each
    row corresponds to."""
    rows, idxs = [], []
    for i, beat in enumerate(beats):
        vec = beat_feature_vector(beat, primary_pre_samples)
        if vec is not None:
            rows.append(vec)
            idxs.append(i)
    if not rows:
        return np.zeros((0, N_FEATURES)), []
    return np.vstack(rows), idxs


def recording_level_hrv(beats: list[Beat]) -> dict:
    """SDNN, RMSSD, pNN50, LF/HF power ratio, QRS-width trend — used by
    the stage 8 risk scorer, not the beat classifier."""
    rr = np.array([b.rr_post_ms for b in beats if b.rr_post_ms is not None and not b.rr_flagged])
    if len(rr) < 3:
        return {"sdnn_ms": 0.0, "rmssd_ms": 0.0, "pnn50_pct": 0.0, "lf_hf_ratio": 0.0, "qrs_width_trend": 0.0}

    sdnn = float(np.std(rr, ddof=1))
    diffs = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) else 0.0
    pnn50 = float(np.mean(np.abs(diffs) > 50.0) * 100) if len(diffs) else 0.0
    lf_hf = _lf_hf_ratio(rr)

    widths = [float(np.ptp(np.where(np.abs(b.primary_window) > np.std(b.primary_window))[0]))
              for b in beats if b.primary_window is not None and not b.quality_rejected]
    qrs_width_trend = float(np.polyfit(np.arange(len(widths)), widths, 1)[0]) if len(widths) > 2 else 0.0

    return {
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "pnn50_pct": pnn50,
        "lf_hf_ratio": lf_hf,
        "qrs_width_trend": qrs_width_trend,
    }


def _lf_hf_ratio(rr_ms: np.ndarray) -> float:
    """Rough LF (0.04-0.15Hz)/HF (0.15-0.4Hz) power ratio from the RR
    tachogram via Lomb-Scargle (unevenly sampled RR series)."""
    from scipy.signal import lombscargle
    t = np.cumsum(rr_ms) / 1000.0
    rr_detrended = rr_ms - np.mean(rr_ms)
    freqs = np.linspace(0.01, 0.5, 200) * 2 * np.pi
    try:
        power = lombscargle(t, rr_detrended, freqs, normalize=True)
    except (ZeroDivisionError, ValueError):
        return 0.0
    f_hz = freqs / (2 * np.pi)
    lf_power = float(np.sum(power[(f_hz >= 0.04) & (f_hz < 0.15)]))
    hf_power = float(np.sum(power[(f_hz >= 0.15) & (f_hz < 0.4)]))
    return lf_power / hf_power if hf_power > 1e-9 else 0.0
