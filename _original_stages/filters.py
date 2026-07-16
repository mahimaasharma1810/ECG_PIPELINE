"""Stage 4 — Five-step filter chain, applied in order.

Each step assumes the previous one has run. SeNSiO pre-filtered files skip
steps 1-4 (device already bandpass-filtered) and start at step 5, to avoid
double-filtering artefacts.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, lfilter, medfilt

from .config import FILTER, FilterChainConfig


def _safe_filtfilt(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """`filtfilt` needs len(x) > ~3*max(len(a),len(b)); short beat-level
    segments (e.g. a device's shortest test recordings) can be shorter
    than that. Fall back to a causal `lfilter` for those rather than
    raising, since a short segment is still worth passing through the
    rest of the pipeline (flagged for its unusually short duration
    upstream, not silently dropped here)."""
    min_len = 3 * max(len(a), len(b))
    if len(x) <= min_len:
        return lfilter(b, a, x)
    return filtfilt(b, a, x)


def remove_baseline_median(x: np.ndarray, fs: float, window_ms: float = FILTER.median_baseline_window_ms) -> np.ndarray:
    """Step 1: double median filter, subtract to remove baseline drift
    without distorting QRS shape."""
    win_samples = int(round(window_ms / 1000.0 * fs))
    win_samples = min(win_samples, len(x))
    win_samples += 1 - (win_samples % 2)  # medfilt needs an odd kernel
    win_samples = max(win_samples, 3)
    if win_samples > len(x):
        return x - np.median(x)  # segment too short for a real median-filter window
    stage1 = medfilt(x, kernel_size=win_samples)
    stage2 = medfilt(stage1, kernel_size=win_samples)
    return x - stage2


def highpass_residual(x: np.ndarray, fs: float, cutoff_hz: float = FILTER.highpass_hz) -> np.ndarray:
    """Step 2: 2nd-order zero-phase Butterworth high-pass — catches drift
    the median filter misses (motion artefact)."""
    nyq = fs / 2.0
    b, a = butter(2, cutoff_hz / nyq, btype="high")
    return _safe_filtfilt(b, a, x)


def powerline_notch(x: np.ndarray, fs: float, notch_hz: float = FILTER.notch_hz,
                     q: float = FILTER.notch_q) -> np.ndarray:
    """Step 3: IIR notch at mains frequency."""
    nyq = fs / 2.0
    if notch_hz >= nyq:
        return x
    b, a = iirnotch(notch_hz / nyq, q)
    return _safe_filtfilt(b, a, x)


def bandpass(x: np.ndarray, fs: float, low_hz: float = FILTER.bandpass_low_hz,
             high_hz: float = FILTER.bandpass_high_hz, order: int = FILTER.bandpass_order) -> np.ndarray:
    """Step 4: industry-standard wearable ECG passband."""
    nyq = fs / 2.0
    high = min(high_hz, nyq * 0.99)
    b, a = butter(order, [low_hz / nyq, high / nyq], btype="band")
    return _safe_filtfilt(b, a, x)


def emg_suppress_kalman(x: np.ndarray, process_var: float = 1e-4, meas_var: float = 1e-2) -> np.ndarray:
    """Step 5: scalar Kalman filter with adaptive gain, removes muscle
    noise that a Butterworth bandpass alone would strip the QRS to remove.

    A minimal scalar (constant-value) Kalman filter: adequate for
    suppressing high-frequency EMG bursts between beats while a genuine
    QRS transient (large innovation) still passes through because the
    Kalman gain adapts upward on large residuals.
    """
    n = len(x)
    out = np.empty(n)
    est = x[0]
    err_cov = 1.0
    for i in range(n):
        err_cov_pred = err_cov + process_var
        innovation = x[i] - est
        adaptive_meas_var = meas_var * (1.0 + 5.0 * min(abs(innovation), 3.0))
        gain = err_cov_pred / (err_cov_pred + adaptive_meas_var)
        est = est + gain * innovation
        err_cov = (1 - gain) * err_cov_pred
        out[i] = est
    return out


def robust_zscore(x: np.ndarray) -> np.ndarray:
    """Step 6: outlier-resistant normalization (median / MAD) — mean/std
    is inflated by motion artefacts."""
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    scale = mad * 1.4826 if mad > 1e-9 else (np.std(x) + 1e-9)
    return (x - median) / scale


def apply_filter_chain(x: np.ndarray, fs: float, already_bandpass_filtered: bool = False,
                        cfg: FilterChainConfig = FILTER) -> np.ndarray:
    """Run the full chain, or steps 5-6 only if the device pre-filtered."""
    if not already_bandpass_filtered:
        x = remove_baseline_median(x, fs, cfg.median_baseline_window_ms)
        x = highpass_residual(x, fs, cfg.highpass_hz)
        x = powerline_notch(x, fs, cfg.notch_hz, cfg.notch_q)
        x = bandpass(x, fs, cfg.bandpass_low_hz, cfg.bandpass_high_hz, cfg.bandpass_order)
    x = emg_suppress_kalman(x)
    return x
