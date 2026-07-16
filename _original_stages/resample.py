"""Stage 3 — Uniform resample to the common target rate (125 Hz)."""
from __future__ import annotations

import numpy as np
from scipy.signal import decimate

from .config import TARGET_FS


def resample_linear(signal: np.ndarray, timestamps_ms: np.ndarray,
                     target_fs: float = TARGET_FS) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation onto a strict uniform time grid.

    Linear rather than cubic, to avoid ringing near sharp QRS peaks
    (cubic splines overshoot around the steep R-wave edge).
    """
    valid = ~np.isnan(signal)
    t = timestamps_ms[valid]
    x = signal[valid]
    if len(t) < 2:
        return np.array([]), np.array([])

    step_ms = 1000.0 / target_fs
    t_uniform = np.arange(t[0], t[-1], step_ms)
    x_uniform = np.interp(t_uniform, t, x)
    return x_uniform, t_uniform


def resample_decimate(signal: np.ndarray, fs_in: float,
                       target_fs: float = TARGET_FS) -> np.ndarray:
    """FIR anti-aliased decimation for higher native rates (e.g. Icentia11k
    250 Hz -> 125 Hz). Naive every-other-sample downsampling would alias
    high-frequency content into the QRS complex; `scipy.signal.decimate`
    applies a proper anti-aliasing filter first.
    """
    ratio = fs_in / target_fs
    if abs(ratio - round(ratio)) > 1e-6:
        raise ValueError(f"decimate requires an integer ratio, got {ratio} "
                          f"({fs_in} Hz -> {target_fs} Hz)")
    return decimate(signal, int(round(ratio)), ftype="fir", zero_phase=True)


def to_target_rate(signal: np.ndarray, timestamps_ms: np.ndarray, fs_nominal: float,
                    target_fs: float = TARGET_FS) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch: decimate when downsampling from a clean integer-ratio
    source (WFDB-style fixed fs), otherwise linear-interpolate irregular
    wearable timestamps (VitalPatch/SeNSiO).
    """
    if abs(fs_nominal - target_fs) < 1e-6:
        return signal, timestamps_ms

    ratio = fs_nominal / target_fs
    if fs_nominal > target_fs and abs(ratio - round(ratio)) < 1e-6:
        out = resample_decimate(signal, fs_nominal, target_fs)
        step_ms = 1000.0 / target_fs
        t_out = timestamps_ms[0] + np.arange(len(out)) * step_ms
        return out, t_out

    return resample_linear(signal, timestamps_ms, target_fs)
