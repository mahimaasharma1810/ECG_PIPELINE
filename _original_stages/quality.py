"""Stage 2 — Pre-filter Signal Quality Index (SQI) gate.

Runs on non-overlapping windows *before* any filtering. A window that
fails any threshold is rejected and logged with a quality code, metric
value, and timestamp range — never silently dropped — forming a
per-patient signal-quality audit trail (see `audit.py`).

Recommendation #6 fix ("Fix the signal-quality check so it doesn't reject
real AFib patients"): the baseline gate is documented as using a single
"regularity"-flavoured kurtosis threshold that, in practice, also caught
genuinely irregular-but-clean rhythms (AFib, frequent ectopy) because
irregular RR spacing lowers window-level kurtosis in the same direction as
motion artefact does. Here:
  1. Kurtosis is computed only as a *morphology* impulse-character check
     (peakedness of the amplitude distribution), which is legitimately
     noise-sensitive and rhythm-agnostic.
  2. RR-interval irregularity is never part of this gate at all — it is
     computed later (stage 5/7) and fed to the rhythm classifier as a
     clinical signal (AFib burden), not treated as a quality defect.
  3. RR-interval outliers detected downstream are flagged, never dropped
     (see `beats.py`), so a fast, irregular but real AFib strip survives
     end to end instead of being silently discarded as "noisy".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .config import SQI, SQIThresholds


@dataclass
class WindowVerdict:
    start_idx: int
    end_idx: int
    start_ms: float
    end_ms: float
    passed: bool
    reject_code: str | None
    metrics: dict


def _longest_equal_runs_frac(x: np.ndarray, min_run_samples: int, at_values: np.ndarray | None = None) -> float:
    """Fraction of samples that belong to a run of >= min_run_samples
    consecutive identical values (optionally restricted to runs sitting at
    specific values, e.g. the window's own min/max for railing/clipping).
    Distinguishes real "stuck sensor" flatline / ADC railing from ordinary
    quantization noise, which produces only short scattered equal-adjacent
    pairs and would otherwise false-trigger on a real, clean recording.
    """
    if len(x) < 2:
        return 0.0
    is_repeat = np.diff(x) == 0
    if at_values is not None:
        at_mask = np.isin(x[1:], at_values) & np.isin(x[:-1], at_values)
        is_repeat = is_repeat & at_mask

    flagged = np.zeros(len(x), dtype=bool)
    run_start = None
    for i, rep in enumerate(np.append(is_repeat, False)):
        if rep:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_len = i - run_start + 1  # +1: diff index i means samples[i] == samples[i+1]
                if run_len >= min_run_samples:
                    flagged[run_start:i + 1] = True
            run_start = None
    return float(np.mean(flagged))


def _flatline_frac(x: np.ndarray, fs: float, run_ms: float) -> float:
    min_run = max(2, int(round(run_ms / 1000.0 * fs)))
    return _longest_equal_runs_frac(x, min_run)


def _clipping_frac(x: np.ndarray, fs: float, run_ms: float, clip_value: float | None) -> float:
    """Railing detector: a sustained run sitting at the window's own
    observed extremes (or at an explicit device clip_value, if known)."""
    x_valid = x[~np.isnan(x)]
    if len(x_valid) < 2:
        return 0.0
    if clip_value is not None:
        rail_values = np.array([clip_value, -clip_value])
    else:
        rail_values = np.array([np.max(x_valid), np.min(x_valid)])
    min_run = max(2, int(round(run_ms / 1000.0 * fs)))
    return _longest_equal_runs_frac(x, min_run, at_values=rail_values)


def _missing_frac(x: np.ndarray) -> float:
    return float(np.mean(np.isnan(x)))


def _morphology_kurtosis(x: np.ndarray) -> float:
    """Amplitude-distribution kurtosis — QRS impulse character.

    Deliberately NOT a function of beat timing/RR spacing (see module
    docstring, recommendation #6): a perfectly regular flat noise signal
    and a perfectly irregular but clean AFib signal can both have "normal"
    RR patterns or not — this metric only asks whether the waveform shape
    has the sharp/peaky character of real QRS complexes.
    """
    x = x[~np.isnan(x)]
    if len(x) < 8:
        return 0.0
    return float(stats.kurtosis(x, fisher=True))


def _baseline_wander_ratio(x: np.ndarray, fs: float) -> float:
    """Peak-to-peak low-frequency drift, expressed relative to the
    window's own robust dynamic range (IQR) rather than an absolute mV
    number — see `SQIThresholds.baseline_wander_ratio_max` docstring for
    why (unknown per-device ADC scale)."""
    from scipy.signal import butter, filtfilt
    x = x[~np.isnan(x)]
    if len(x) < 20:
        return 0.0
    nyq = fs / 2.0
    b, a = butter(2, 0.5 / nyq, btype="low")
    try:
        low = filtfilt(b, a, x)
    except ValueError:
        return 0.0
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    scale = abs(iqr) if abs(iqr) > 1e-9 else (np.std(x) + 1e-9)
    return float(np.ptp(low)) / scale


def _snr_db(x: np.ndarray, fs: float) -> float:
    from scipy.signal import butter, filtfilt
    x = x[~np.isnan(x)]
    if len(x) < 20:
        return 0.0
    nyq = fs / 2.0
    b, a = butter(4, [0.5 / nyq, 40 / nyq], btype="band")
    try:
        signal_band = filtfilt(b, a, x)
    except ValueError:
        return 0.0
    noise = x - signal_band
    sig_power = np.mean(signal_band ** 2)
    noise_power = np.mean(noise ** 2) + 1e-12
    return float(10 * np.log10(sig_power / noise_power)) if sig_power > 0 else -999.0


def evaluate_window(x: np.ndarray, t_ms: np.ndarray, fs: float,
                     clip_value: float | None, thresholds: SQIThresholds = SQI) -> WindowVerdict:
    metrics = {
        "flatline_frac": _flatline_frac(x, fs, thresholds.flatline_run_ms),
        "clipping_frac": _clipping_frac(x, fs, thresholds.clipping_run_ms, clip_value),
        "missing_frac": _missing_frac(x),
        "morphology_kurtosis": _morphology_kurtosis(x),
        "baseline_wander_ratio": _baseline_wander_ratio(x, fs),
        "snr_db": _snr_db(x, fs),
    }

    reject_code = None
    if metrics["flatline_frac"] > thresholds.flatline_frac_max:
        reject_code = "FLATLINE"
    elif metrics["clipping_frac"] > thresholds.clipping_frac_max:
        reject_code = "CLIPPING"
    elif metrics["missing_frac"] > thresholds.missing_frac_max:
        reject_code = "MISSING_SAMPLES"
    elif metrics["morphology_kurtosis"] < thresholds.kurtosis_min:
        reject_code = "NO_QRS_IMPULSE_CHARACTER"
    elif metrics["baseline_wander_ratio"] > thresholds.baseline_wander_ratio_max:
        reject_code = "BASELINE_WANDER"
    elif metrics["snr_db"] < thresholds.snr_db_min:
        reject_code = "LOW_SNR"

    return WindowVerdict(
        start_idx=0, end_idx=len(x),
        start_ms=float(t_ms[0]) if len(t_ms) else 0.0,
        end_ms=float(t_ms[-1]) if len(t_ms) else 0.0,
        passed=reject_code is None,
        reject_code=reject_code,
        metrics=metrics,
    )


def run_sqi_gate(signal: np.ndarray, timestamps_ms: np.ndarray, fs: float,
                  clip_value: float | None = None,
                  thresholds: SQIThresholds = SQI) -> tuple[np.ndarray, list[WindowVerdict]]:
    """Split into non-overlapping windows, evaluate each, return a boolean
    keep-mask (same length as signal) plus the full per-window audit trail.
    """
    win_len = max(1, int(round(thresholds.window_seconds * fs)))
    n = len(signal)
    keep_mask = np.zeros(n, dtype=bool)
    verdicts: list[WindowVerdict] = []

    for start in range(0, n, win_len):
        end = min(start + win_len, n)
        window = signal[start:end]
        t_window = timestamps_ms[start:end]
        verdict = evaluate_window(window, t_window, fs, clip_value, thresholds)
        verdict.start_idx, verdict.end_idx = start, end
        verdicts.append(verdict)
        if verdict.passed:
            keep_mask[start:end] = True

    return keep_mask, verdicts


def rejection_rate(verdicts: list[WindowVerdict]) -> float:
    if not verdicts:
        return 0.0
    return sum(1 for v in verdicts if not v.passed) / len(verdicts)
