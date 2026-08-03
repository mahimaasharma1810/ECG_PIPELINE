"""ecg_inference.features -- Stage 6: the 56-dim handcrafted per-beat
feature vector (5 morphological + 51 wavelet) and recording-level HRV
features. Extracted verbatim from ecg_pipeline_core.py's features.py
section. Excludes the encoder.py section (self-supervised learned
embedding) -- confirmed unused by every real deployment call site (no
production ECGPipeline(...) construction ever passes an encoder), so it
is dead weight for inference and pulls in a torch dependency this
package doesn't need. See ecg_inference/pipeline.py's ECGPipeline.__init__
docstring for the same note.
"""
from __future__ import annotations

import warnings

import numpy as np
import pywt

from .detector import Beat
from .preprocess import TARGET_FS, robust_zscore


# The spec's 75-sample primary window (200ms pre + R + 400ms post @ 125Hz) is
# shorter than what a level-4 db4 decomposition ideally wants, so every
# beat trips pywt's boundary-effects warning. The coefficients are still
# valid (just more boundary-influenced) and are resampled to a fixed
# length below regardless, so the warning is expected noise, not a bug.
warnings.filterwarnings("ignore", message="Level value of.*is too high", module="pywt")

N_MORPHOLOGICAL = 5
N_WAVELET = 51
N_TIMING = 7
N_QRS_SHAPE = 4
N_FEATURES = N_MORPHOLOGICAL + N_WAVELET  # unchanged: production's dimensionality
N_FEATURES_WITH_TIMING = N_FEATURES + N_TIMING


def _feature_width(include_timing: bool, drop_compensatory_pause: bool = False,
                    timing_only: bool = False, include_r_amp: bool = False,
                    include_qrs_shape: bool = False) -> int:
    timing_width = (N_TIMING - (1 if drop_compensatory_pause else 0)) if (include_timing or timing_only) else 0
    if timing_only:
        return timing_width
    base_width = N_FEATURES + (1 if include_r_amp else 0)
    return base_width + timing_width + (N_QRS_SHAPE if include_qrs_shape else 0)


TIMING_FEATURE_NAMES = ["rr_ratio_k8", "rr_ratio_k16", "rr_ratio_k32", "rr_pre_post_ratio",
                         "rr_cv_local", "prematurity_score", "compensatory_pause_flag"]


def _timing_features(beat: "Beat", drop_compensatory_pause: bool = False) -> np.ndarray:
    """`drop_compensatory_pause` is a single-purpose ablation knob (not a
    general feature-selection mechanism) for the timing-features
    micro-experiment in ABLATION_REPORT.md: compensatory_pause_flag was
    found to be the single most important feature in the 63-dim model
    (rank 1/63, ~32% of total gain) and fires for both premature S and
    premature V beats, the suspected driver of the S->V confusion increase
    that regressed V's DS2 F1. Drops the column entirely (6-dim timing
    block) rather than zeroing it, so the ablation removes the information,
    not just its typical value."""
    vals = [beat.rr_ratio_k8, beat.rr_ratio_k16, beat.rr_ratio_k32,
            beat.rr_pre_post_ratio, beat.rr_cv_local,
            beat.prematurity_score, beat.compensatory_pause_flag]
    if drop_compensatory_pause:
        vals = vals[:-1]
    return np.array(vals)


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


# Column index of local_hrv within the morphological block (and so within the full
# feature vector, morphology being first). Mirrors ecg_pipeline_core.py.
LOCAL_HRV_FEATURE_IDX = 1


def _morphological_features(window: np.ndarray, primary_pre_samples: int,
                             rr_pre_ms: float | None, rr_post_ms: float | None,
                             rr_flagged: bool = False) -> np.ndarray:
    """[rr_pre, local_hrv, area_ratio, above_below_ratio, amplitude_range].

    `rr_flagged` means an adjacent RR interval is out of physiological range -- in
    practice a missed-beat gap across an SQI-rejected stretch. When set, local_hrv
    is NaN rather than the raw difference, so a detection gap cannot enter the
    classifier as physiology. Measured motivation: VitalPatch 184B27/seg1 beat 27
    carried local_hrv = 5104 ms (a 5.1 s gap), and TreeSHAP made it the largest
    single attribution in the segment, pushing that beat toward class V.

    NaN rather than 0.0 because 0.0 is a real and common local_hrv (regular
    rhythm); median-filled by fill_missing_local_hrv() on the batch path, and
    treated natively as `missing` by XGBoost otherwise.

    Mirrors ecg_pipeline/ecg_pipeline_core.py:_morphological_features -- keep the
    two in sync (the SDNN 0.0 sentinel fix was applied to only one copy and stayed
    live in this package as a result).
    """
    r_idx = primary_pre_samples

    rr_pre = rr_pre_ms if rr_pre_ms is not None else 0.0
    if rr_flagged:
        local_hrv = np.nan
    else:
        local_hrv = (rr_post_ms - rr_pre_ms) if (rr_pre_ms and rr_post_ms) else 0.0

    left = window[:r_idx]
    right = window[r_idx:]
    left_area, right_area = float(np.sum(np.abs(left))), float(np.sum(np.abs(right)))
    area_ratio = left_area / right_area if right_area > 1e-9 else 0.0

    above = window[window > 0]
    below = window[window < 0]
    above_below_ratio = (np.sum(above) / abs(np.sum(below))) if len(below) and np.sum(below) != 0 else 0.0

    amplitude_range = float(np.ptp(window))

    return np.array([rr_pre, local_hrv, area_ratio, above_below_ratio, amplitude_range])


def _r_amp_feature(window: np.ndarray, primary_pre_samples: int) -> np.ndarray:
    """R-peak amplitude (post-`robust_zscore` normalized window). This was
    originally computed inline in `_morphological_features` (as `r_amp`)
    but never returned -- dead code, silently discarded, found during the
    two-stage-classifier prerequisite work (ABLATION_REPORT.md,
    "Prerequisite 2"). Split into its own function and appended at the END
    of the full feature vector in `beat_feature_vector` (not inserted
    between the 5 morphological features and the 51 wavelet features),
    because inserting it there was tried first and shifted every wavelet
    column's index by one -- caught by the train/inference parity check,
    which is exactly what that check is for. Appending at the end keeps
    every existing index (0-55) meaning exactly what it always meant;
    this is index 56, opt-in only via include_r_amp."""
    return np.array([float(window[primary_pre_samples])])


def _qrs_width_amplitude_crossing(window: np.ndarray, r_idx: int, fs: float,
                                   fraction: float = 0.15,
                                   width_radius_samples: int = 15) -> float:
    """FIX (2026-07-23): replaces the derivative-threshold QRS-width walk,
    which collapsed to a 1-sample floor on ~9% of true-V beats -- a
    fragmented/notched wide V complex can have a local derivative dip below
    a 10%-of-peak-derivative threshold in the MIDDLE of the complex, so that
    walk could terminate one sample from R even though the true complex is
    still wide. Amplitude relative to baseline doesn't have this failure
    mode: a wide complex stays far from baseline throughout its width even
    if notched, so a level-crossing on amplitude is far more robust than a
    threshold on the derivative.

    Baseline = median of the whole (already robust_zscore-normalized)
    window -- the beat spends most of its ~600ms span near isoelectric
    baseline even within this short window, so the median is a simple,
    adequate isoelectric estimate without needing a separate PQ-segment
    detector. R height = |R amplitude - baseline|. Onset/offset are found
    by walking outward from R until |signal - baseline| crosses below
    `fraction` (default 15%) of R height, searched within
    `width_radius_samples` (default 15 = 120ms @ 125Hz) of R on each side --
    a fixed physiological window, per the guard-rail requirement. If no
    crossing is found in that window (e.g. baseline is itself noisy, or R
    height is ~0 for a degenerate beat), falls back to the WINDOW EDGE,
    never to 1 sample -- the exact failure mode being fixed here."""
    lo = max(0, r_idx - width_radius_samples)
    hi = min(len(window) - 1, r_idx + width_radius_samples)
    baseline = float(np.median(window))
    r_height = abs(float(window[r_idx]) - baseline)

    onset, offset = lo, hi  # guard-rail fallback: window edge, not 1 sample
    if r_height > 1e-9:
        level = fraction * r_height
        for i in range(r_idx, lo - 1, -1):
            if abs(window[i] - baseline) < level:
                onset = i
                break
        for i in range(r_idx, hi + 1):
            if abs(window[i] - baseline) < level:
                offset = i
                break
    onset, offset = min(onset, r_idx), max(offset, r_idx)
    return float((offset - onset) / fs * 1000.0)


def _qrs_shape_features(window: np.ndarray, primary_pre_samples: int,
                         fs: float = TARGET_FS, search_radius_samples: int = 20,
                         width_fraction: float = 0.15,
                         width_radius_ms: float = 120.0) -> np.ndarray:
    """QRS-complex shape discriminators: qrs_width_ms, qrs_abs_area,
    slope_pre_r, slope_post_r. Added to help Stage 2 separate narrow-QRS
    beats (N, S -- supraventricular origin) from wide-QRS beats (V --
    ventricular origin) -- the existing 56-dim vector has no direct measure
    of QRS width itself (rr_pre/local_hrv in `_morphological_features` are
    RR-timing, not QRS shape; area_ratio/above_below_ratio/amplitude_range
    describe the whole ~600ms window, not specifically the QRS complex).
    See ABLATION_REPORT.md "QRS morphology discriminators" for the DS2
    confusion-matrix evidence (N->V and S->V leakage) this addresses.

    `qrs_abs_area` and the two slopes are computed from a derivative-
    threshold onset/offset walk outward from R (searched within
    `search_radius_samples` of R on each side, stopping where the
    derivative magnitude drops below 10% of the local peak derivative) --
    UNCHANGED since the feature was first added. `qrs_width_ms` (2026-07-23
    fix) uses a SEPARATE, more robust amplitude-crossing measure instead
    (`_qrs_width_amplitude_crossing`) -- see that function's docstring for
    why the derivative-threshold walk was unreliable specifically for
    width. Operates on the SAME normalized window passed to
    `_morphological_features`/`_wavelet_features`, at TARGET_FS (both
    training, via `_load_record_beats`, and runtime inference, via
    `ECGPipeline.run`, resample to TARGET_FS before beat segmentation --
    see `to_target_rate` call sites -- so this fixed `fs` default is valid
    for both code paths, not just one)."""
    r_idx = primary_pre_samples
    lo = max(0, r_idx - search_radius_samples)
    hi = min(len(window), r_idx + search_radius_samples)
    d = np.diff(window)  # d[i] = window[i+1] - window[i], len(window)-1

    pre_d = d[lo:r_idx] if r_idx > lo else np.array([0.0])
    post_d = d[r_idx:hi - 1] if hi - 1 > r_idx else np.array([0.0])
    slope_pre_r = float(np.max(pre_d)) if len(pre_d) else 0.0
    slope_post_r = float(np.min(post_d)) if len(post_d) else 0.0

    local_d = d[lo:hi - 1] if hi - 1 > lo else np.array([0.0])
    peak_abs = float(np.max(np.abs(local_d))) if len(local_d) else 0.0
    threshold = 0.1 * peak_abs

    onset, offset = lo, hi - 1
    if peak_abs > 1e-9:
        for i in range(r_idx - 1, lo - 1, -1):
            if abs(d[i]) < threshold:
                onset = i
                break
        for i in range(r_idx, hi - 1):
            if abs(d[i]) < threshold:
                offset = i
                break
    onset, offset = min(onset, r_idx), max(offset, r_idx)
    qrs_abs_area = float(np.sum(np.abs(window[onset:offset + 1])))

    width_radius_samples = int(round(width_radius_ms / 1000.0 * fs))
    qrs_width_ms = _qrs_width_amplitude_crossing(window, r_idx, fs, fraction=width_fraction,
                                                  width_radius_samples=width_radius_samples)
    return np.array([qrs_width_ms, qrs_abs_area, slope_pre_r, slope_post_r])


def beat_feature_vector(beat: Beat, primary_pre_samples: int,
                         include_timing: bool = False,
                         drop_compensatory_pause: bool = False,
                         timing_only: bool = False,
                         include_r_amp: bool = False,
                         include_qrs_shape: bool = False) -> np.ndarray | None:
    """Step 6 of the filter chain (robust Z-score, per beat window) is
    applied here, right before feature extraction — quality rejection
    upstream (`beats.py`) intentionally runs on the raw window instead, so
    normalization can't mask a genuinely flat/low-amplitude beat.

    `include_timing=False` (default) reproduces the exact 56-dim vector the
    production model (`five_class_xgb.json`) and the live runtime pipeline
    expect -- this function is the single source used by BOTH training
    (`_load_record_beats`/`build_dataset`) and runtime inference
    (`batch_feature_matrix`), so leaving the default unchanged means
    neither path can silently drift out of sync with the other, and
    production is unaffected by anything below. `include_timing=True`
    appends the 7 local-rhythm-context features computed in `segment_beats`
    (final layout: 5 morph + 51 wavelet + 7 timing = 63) -- opt-in only,
    for the timing-features experiment. `drop_compensatory_pause` (only
    meaningful when include_timing=True or timing_only=True) drops just
    that one column. `timing_only=True` returns JUST the timing block (7,
    or 6 if drop_compensatory_pause) with no morphology/wavelet at all --
    for the feature-family ablation (morphology-only vs timing-only vs
    combined) in ABLATION_REPORT.md; note the 56 "morphology" features
    already include 2 raw RR scalars (rr_pre, local_hrv) via
    `_morphological_features`, so "morphology-only" isn't strictly
    zero-timing-information, just missing the 7 richer local-rhythm-context
    features. `include_r_amp=True` appends R-peak amplitude (computed but
    previously discarded in `_morphological_features` -- see
    `_r_amp_feature`'s docstring) as the LAST feature in the vector
    (after wavelet, and after timing if `include_timing` is also set) so
    every previously-established index (0-55, or 0-62 with timing) keeps
    its exact original meaning -- inserting it earlier, between morphology
    and wavelet, was tried first and shifted every wavelet index by one;
    caught by the train/inference parity check. `include_qrs_shape=True`
    appends `_qrs_shape_features` (4: qrs_width_ms, qrs_abs_area,
    slope_pre_r, slope_post_r) at the very end, after r_amp -- same
    append-only reasoning, so turning it on cannot shift any existing
    column (0-55, 0-62 w/ timing, or 63 w/ r_amp) regardless of which other
    flags are combined.
    """
    if beat.primary_window is None or beat.quality_rejected:
        return None
    if timing_only:
        return _timing_features(beat, drop_compensatory_pause=drop_compensatory_pause)
    normalized = robust_zscore(beat.primary_window)
    morph = _morphological_features(normalized, primary_pre_samples, beat.rr_pre_ms, beat.rr_post_ms,
                                     rr_flagged=beat.rr_flagged)
    wavelet = _wavelet_features(normalized)
    parts = [morph, wavelet]
    if include_timing:
        parts.append(_timing_features(beat, drop_compensatory_pause=drop_compensatory_pause))
    if include_r_amp:
        parts.append(_r_amp_feature(normalized, primary_pre_samples))
    if include_qrs_shape:
        parts.append(_qrs_shape_features(normalized, primary_pre_samples))
    return np.concatenate(parts)


def batch_feature_matrix(beats: list[Beat], primary_pre_samples: int,
                          include_timing: bool = False,
                          drop_compensatory_pause: bool = False,
                          timing_only: bool = False,
                          include_r_amp: bool = False,
                          include_qrs_shape: bool = False) -> tuple[np.ndarray, list[int]]:
    """Returns (feature_matrix, indices_into_beats_used) — skipping
    rejected/out-of-bounds beats but preserving which original beat each
    row corresponds to."""
    rows, idxs = [], []
    for i, beat in enumerate(beats):
        vec = beat_feature_vector(beat, primary_pre_samples, include_timing=include_timing,
                                   drop_compensatory_pause=drop_compensatory_pause,
                                   timing_only=timing_only, include_r_amp=include_r_amp,
                                   include_qrs_shape=include_qrs_shape)
        if vec is not None:
            rows.append(vec)
            idxs.append(i)
    if not rows:
        return np.zeros((0, _feature_width(include_timing, drop_compensatory_pause,
                                            timing_only, include_r_amp, include_qrs_shape))), []
    matrix = np.vstack(rows)
    if not timing_only:  # timing_only has no morphology block, so no local_hrv column
        matrix, _ = fill_missing_local_hrv(matrix)
    return matrix, idxs


def fill_missing_local_hrv(matrix: np.ndarray, col: int = LOCAL_HRV_FEATURE_IDX) -> tuple[np.ndarray, dict]:
    """Replaces NaN local_hrv (missed-beat gaps) with the MEDIAN of this segment's
    valid local_hrv values. Median rather than mean because a high-ectopy strip has
    a long-tailed distribution a mean would chase; not 0.0, because that asserts
    "perfectly regular", a real and common reading. Falls back to 0.0 only when the
    whole column is missing, reporting that in stats["all_missing"].

    Mirrors ecg_pipeline/ecg_pipeline_core.py:fill_missing_local_hrv."""
    stats = {"n_missing": 0, "n_total": int(matrix.shape[0]), "fill_value": None,
             "all_missing": False}
    if matrix.size == 0 or col >= matrix.shape[1]:
        return matrix, stats
    missing = np.isnan(matrix[:, col])
    stats["n_missing"] = int(missing.sum())
    if not missing.any():
        return matrix, stats
    valid = matrix[~missing, col]
    if len(valid) == 0:
        stats["all_missing"] = True
        stats["fill_value"] = 0.0
        matrix[missing, col] = 0.0
        return matrix, stats
    fill = float(np.median(valid))
    stats["fill_value"] = fill
    matrix[missing, col] = fill
    return matrix, stats


def recording_level_hrv(beats: list[Beat]) -> dict:
    """SDNN, RMSSD, pNN50, LF/HF power ratio, QRS-width trend — used by
    the stage 8 risk scorer, not the beat classifier."""
    rr = np.array([b.rr_post_ms for b in beats if b.rr_post_ms is not None and not b.rr_flagged])
    if len(rr) < 3:
        # sdnn_ms=None (not 0.0): fewer than 3 valid RR intervals means SDNN
        # is genuinely undefined here, not measured-and-zero. A hardcoded
        # 0.0 sentinel is indistinguishable downstream from a real
        # (physiologically implausible) zero-variability reading and would
        # silently satisfy score_recording()'s "SDNN < threshold" check on
        # segments with too little data to say anything -- see
        # score_recording() and _build_rule_trace() for the None-aware
        # handling this requires. The other fields aren't threshold-checked
        # anywhere downstream, so they're left as 0.0 to avoid unrelated
        # risk to callers that assume a float.
        # Mirrored from ecg_pipeline_core.recording_level_hrv (commit e0da1f7);
        # the two packages carry independent copies of this function.
        return {"sdnn_ms": None, "rmssd_ms": 0.0, "pnn50_pct": 0.0, "lf_hf_ratio": 0.0, "qrs_width_trend": 0.0}

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


