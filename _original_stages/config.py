"""Shared constants for the Cliniaura ECG pipeline (new methodology).

Every threshold here is either carried over from the existing baseline
(documented in PROJECT_OVERVIEW.md / the architecture PDF) or introduced by
one of the "Top Recommendations" in the internal review PDF. Each new
constant says which recommendation it implements.
"""
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

TARGET_FS = 125.0  # Hz, common resample target (VitalPatch native rate)

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


@dataclass
class SQIThresholds:
    """Stage 2 pre-filter Signal Quality Index gate.

    Recommendation #6 ("Fix the signal-quality check so it doesn't reject
    real AFib patients"): the baseline gate used a single "regularity"
    style kurtosis check that conflated *rhythm* irregularity (real AFib,
    frequent ectopy) with *noise* irregularity (motion artefact). Here the
    kurtosis/impulse check is scoped to *within-beat* QRS morphology only,
    never to RR-interval regularity, so an irregular-but-clean AFib strip
    no longer fails the gate. Window-level irregular RR is instead handed
    downstream to the rhythm classifier, not rejected here.
    """

    window_seconds: float = 5.0
    flatline_run_ms: float = 400.0  # a "stuck sensor" run of >=400ms identical samples, not scattered quantization repeats
    flatline_frac_max: float = 0.05
    clipping_run_ms: float = 200.0  # a railed run of >=200ms sitting at the window's own min/max
    clipping_frac_max: float = 0.02
    missing_frac_max: float = 0.03
    kurtosis_min: float = 1.5  # QRS impulse character, computed per-beat-window, NOT RR-interval based
    # Real device recordings arrive in arbitrary firmware-scaled raw ADC counts with no published
    # mV-per-count constant (VitalPatch and SeNSiO both do this). Absolute-mV thresholds would be
    # meaningless without that calibration constant, so baseline wander is expressed as a ratio of
    # the window's own robust dynamic range (IQR) instead of a hard mV number.
    baseline_wander_ratio_max: float = 1.2
    snr_db_min: float = 5.0
    adc_clip_value: float | None = None  # only set for a device whose true ADC full-scale is known


@dataclass
class FilterChainConfig:
    median_baseline_window_ms: float = 200.0
    highpass_hz: float = 0.5
    notch_hz: float = 50.0
    notch_q: float = 30.0
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 40.0
    bandpass_order: int = 4


@dataclass
class BeatWindowConfig:
    primary_pre_ms: float = 200.0
    primary_post_ms: float = 400.0
    wide_pre_ms: float = 500.0
    wide_post_ms: float = 500.0
    rr_min_ms: float = 300.0
    rr_max_ms: float = 2000.0
    # Amplitude floor expressed relative to the window's own sample-to-sample noise floor
    # (median absolute deviation of the first difference), not an absolute mV number — see
    # SQIThresholds.baseline_wander_ratio_max for why (unknown per-device ADC scale).
    beat_amplitude_min_noise_ratio: float = 3.0


@dataclass
class RiskThresholds:
    pvc_burden_high_pct: float = 10.0
    pvc_burden_critical_pct: float = 20.0
    pac_burden_high_pct: float = 15.0
    vt_run_beats: int = 3  # >=3 consecutive V beats = VT run


@dataclass
class ConformalConfig:
    """Recommendation #4: statistically guaranteed confidence ranges.

    Split-conformal prediction over the 4 risk levels: given a calibration
    set of (softmax scores, true label), compute the (1 - alpha) quantile
    of nonconformity scores, then at inference time return the *set* of
    risk levels whose score falls within that quantile, instead of a bare
    single label. Coverage is guaranteed marginally, not per-class.
    """

    alpha: float = 0.1  # -> 90% coverage guarantee
    min_calibration_size: int = 30


@dataclass
class TemporalTrackingConfig:
    """Recommendation #5: track risk over time, not just one snapshot."""

    window_minutes: float = 15.0
    trend_slope_alert_threshold: float = 0.15  # risk-score units per minute
    history_max_windows: int = 500


SQI = SQIThresholds()
FILTER = FilterChainConfig()
BEATS = BeatWindowConfig()
RISK = RiskThresholds()
CONFORMAL = ConformalConfig()
TEMPORAL = TemporalTrackingConfig()
