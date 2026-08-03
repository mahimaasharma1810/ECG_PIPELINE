"""ecg_inference.detector -- Stage 5: XQRS R-peak detection and beat
segmentation. Extracted verbatim from ecg_pipeline_core.py's beats.py
section.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocess import BeatWindowConfig, BEATS



@dataclass
class Beat:
    r_peak_idx: int
    r_peak_ms: float
    rr_pre_ms: float | None
    rr_post_ms: float | None
    rr_flagged: bool          # True if RR outside physiological range - kept, not dropped
    primary_window: np.ndarray | None   # 75 samples @125Hz: 200ms pre + R + 400ms post
    wide_window: np.ndarray | None      # 125 samples @125Hz: 500ms pre + R + 500ms post
    quality_rejected: bool
    quality_reject_reason: str | None
    # Local-rhythm-context ("timing") fields, computed once here in
    # segment_beats from the same rr_ms array used for rr_pre_ms/rr_post_ms
    # -- see beat_feature_vector(..., include_timing=True) for how these are
    # consumed. All default to neutral (not-detected) values, never to 0 in
    # a way that would read as "extremely premature"/"extremely irregular"
    # when the true answer is "no history available yet" (see the edge-case
    # notes at each computation site in segment_beats).
    rr_ratio_k8: float = 1.0
    rr_ratio_k16: float = 1.0
    rr_ratio_k32: float = 1.0
    rr_pre_post_ratio: float = 1.0
    rr_cv_local: float = 0.0
    prematurity_score: float = 0.0
    compensatory_pause_flag: float = 0.0


def detect_r_peaks(signal: np.ndarray, fs: float) -> np.ndarray:
    """WFDB XQRS adaptive-threshold R-peak detector.

    CORRECTION (2026-07-31): XQRS's own T-wave-discrimination check
    (`_is_twave`, a slope comparison against the previous beat) only ever
    runs on a candidate peak that falls within `Conf.t_inspect_period` of
    the last accepted beat -- and that defaults to 0, disabling it
    entirely. On this device's morphology (a sharp negative QRS
    immediately followed by an unusually tall/broad T-wave), any peak that
    clears the 200ms hard refractory period and the amplitude threshold
    was being accepted as a second, spurious beat regardless of slope.
    Setting t_inspect_period=0.36 (the conventional Pan-Tompkins-style
    T-wave-inspection window) enables the existing, designed-for-this-
    purpose check. Kept in sync with ecg_pipeline_core.detect_r_peaks --
    see that function's docstring for the full measurement writeup."""
    import wfdb.processing as wp
    if len(signal) < int(fs * 2):
        return np.array([], dtype=int)
    conf = wp.XQRS.Conf(t_inspect_period=0.36)
    xqrs = wp.XQRS(sig=signal, fs=fs, conf=conf)
    xqrs.detect(verbose=False)
    return np.asarray(xqrs.qrs_inds, dtype=int)


def _extract_window(signal: np.ndarray, center_idx: int, pre_samples: int, post_samples: int) -> np.ndarray | None:
    start, end = center_idx - pre_samples, center_idx + post_samples
    if start < 0 or end > len(signal):
        return None
    return signal[start:end]


def _beat_level_sqi(window: np.ndarray, r_local_idx: int, cfg: BeatWindowConfig) -> tuple[bool, str | None]:
    """Reject individual beats with amplitude too low, excess in-window
    baseline drift, or an R-peak that isn't the true local maximum —
    catches detection jitter the window-level gate can't see."""
    if window is None:
        return True, "WINDOW_OUT_OF_BOUNDS"

    amplitude = float(np.ptp(window))
    noise_floor = float(np.median(np.abs(np.diff(window)))) + 1e-9
    if amplitude < cfg.beat_amplitude_min_noise_ratio * noise_floor:
        return True, "LOW_AMPLITUDE"

    drift = float(np.abs(np.median(window[:5]) - np.median(window[-5:])))
    if drift > 3.0 * (np.std(window) + 1e-9):
        return True, "EXCESS_BASELINE_DRIFT"

    search_radius = max(2, len(window) // 20)
    lo = max(0, r_local_idx - search_radius)
    hi = min(len(window), r_local_idx + search_radius)
    local_patch = window[lo:hi]
    if len(local_patch) and np.argmax(np.abs(local_patch)) != (r_local_idx - lo):
        return True, "R_PEAK_NOT_LOCAL_MAX"

    return False, None


def segment_beats(signal: np.ndarray, fs: float, r_peaks: np.ndarray,
                   cfg: BeatWindowConfig = BEATS) -> list[Beat]:
    # Round pre/post independently but derive the total from their sum so the
    # window length is deterministic (e.g. exactly 125 samples for the 1s wide
    # window @ 125Hz) rather than drifting by +/-1 sample from rounding each
    # half separately (Python's round-half-to-even can round both 62.5 -> 62).
    primary_pre = int(round(cfg.primary_pre_ms / 1000.0 * fs))
    primary_total = int(round((cfg.primary_pre_ms + cfg.primary_post_ms) / 1000.0 * fs))
    primary_post = primary_total - primary_pre
    wide_pre = int(round(cfg.wide_pre_ms / 1000.0 * fs))
    wide_total = int(round((cfg.wide_pre_ms + cfg.wide_post_ms) / 1000.0 * fs))
    wide_post = wide_total - wide_pre

    beats: list[Beat] = []
    rr_ms = np.diff(r_peaks) * (1000.0 / fs) if len(r_peaks) > 1 else np.array([])

    for i, r_idx in enumerate(r_peaks):
        rr_pre = float(rr_ms[i - 1]) if i > 0 else None
        rr_post = float(rr_ms[i]) if i < len(rr_ms) else None

        flagged = False
        for rr in (rr_pre, rr_post):
            if rr is not None and not (cfg.rr_min_ms <= rr <= cfg.rr_max_ms):
                flagged = True  # out-of-physiological-range RR: flagged, never dropped

        primary = _extract_window(signal, r_idx, primary_pre, primary_post)
        wide = _extract_window(signal, r_idx, wide_pre, wide_post)
        rejected, reason = _beat_level_sqi(primary, primary_pre, cfg)

        # Local-rhythm-context features. `history` is the up-to-K completed
        # RR intervals strictly BEFORE this beat (rr_ms[i-1] is the interval
        # ending at beat i) -- never includes rr_ms[i] (the POST interval),
        # so nothing here leaks future/lookahead information about this
        # beat's own outcome. Since rr_ms is derived fresh per call from
        # this recording's own r_peaks, a window also never spans two
        # recordings. i < 8/16/32 beats into a recording simply gets a
        # shorter history (rr_ms[0:i]); i == 0 gets none at all, in which
        # case every ratio/score below falls back to its documented neutral
        # default from the Beat dataclass rather than a value that would
        # misread as "very premature" or "very irregular".
        history16 = rr_ms[max(0, i - 16):i]
        mean16 = float(np.mean(history16)) if len(history16) else 0.0

        rr_ratio_k8 = rr_ratio_k16 = rr_ratio_k32 = 1.0
        if rr_pre is not None:
            hist8 = rr_ms[max(0, i - 8):i]
            mean8 = float(np.mean(hist8)) if len(hist8) else 0.0
            rr_ratio_k8 = rr_pre / mean8 if mean8 > 1e-9 else 1.0
            hist32 = rr_ms[max(0, i - 32):i]
            mean32 = float(np.mean(hist32)) if len(hist32) else 0.0
            rr_ratio_k32 = rr_pre / mean32 if mean32 > 1e-9 else 1.0
            rr_ratio_k16 = rr_pre / mean16 if mean16 > 1e-9 else 1.0

        rr_pre_post_ratio = 1.0
        if rr_pre is not None and rr_post is not None and rr_post > 1e-9:
            rr_pre_post_ratio = rr_pre / rr_post

        rr_cv_local = 0.0
        if len(history16) >= 2 and mean16 > 1e-9:
            rr_cv_local = float(np.std(history16) / mean16)

        prematurity_score = 0.0
        if rr_pre is not None and mean16 > 1e-9:
            prematurity_score = (mean16 - rr_pre) / mean16

        compensatory_pause_flag = 0.0
        if rr_post is not None and mean16 > 1e-9 and rr_post > 1.2 * mean16:
            compensatory_pause_flag = 1.0

        beats.append(Beat(
            r_peak_idx=int(r_idx),
            r_peak_ms=float(r_idx * 1000.0 / fs),
            rr_pre_ms=rr_pre,
            rr_post_ms=rr_post,
            rr_flagged=flagged,
            primary_window=primary,
            wide_window=wide,
            quality_rejected=rejected,
            quality_reject_reason=reason,
            rr_ratio_k8=rr_ratio_k8,
            rr_ratio_k16=rr_ratio_k16,
            rr_ratio_k32=rr_ratio_k32,
            rr_pre_post_ratio=rr_pre_post_ratio,
            rr_cv_local=rr_cv_local,
            prematurity_score=prematurity_score,
            compensatory_pause_flag=compensatory_pause_flag,
        ))

    return beats


def _snap_to_local_peak(signal: np.ndarray, r_peaks: np.ndarray, search_radius: int = 8) -> np.ndarray:
    """XQRS-detected indices land close to but not always exactly on the
    true sample-wise |amplitude| local max (empirically ~3 samples off on
    resampled WFDB data -- see the training path's identical helper in
    ecg_pipeline_tools._snap_to_local_peak, which this mirrors). That small
    offset is enough to fail _beat_level_sqi's R_PEAK_NOT_LOCAL_MAX check,
    which looks for the true local max within a much narrower +/-3 sample
    window than this snap's search radius -- confirmed as the dominant
    cause of beat-level over-culling on WFDB (1400-2400 beats/record
    rejected via this one reason before this snap was added). Snapping
    first makes detection agree with what that check expects, rather than
    loosening the check itself.

    search_radius was widened to 15 in an earlier fix, but re-measured here
    across 13 MITDB records (100/101/103/105/111/119/200/203/207/210/213/
    219/223) while diagnosing R-peak over-detection: a wide radius lets the
    snap wander past the true QRS peak onto an unrelated nearby local max
    (a noise spike or an adjacent beat) on noisier records -- record 213
    dropped to 44.6% beat-level retention at radius=15 vs 90.4% at
    radius=8, while every other tested record was flat or improved at 8
    (13-record aggregate retention: 81.0% at radius=15 vs 82.9% at
    radius=8). 8 samples (~64ms @125Hz) still comfortably covers the
    observed ~3-sample jitter with margin.
    """
    # A steepness-based tiebreak for near-tied amplitude candidates (e.g. a
    # negative T-wave close in |amplitude| to the true R-peak) was tried
    # and measured worse (83.3% vs 90.9% peaks-on-true-QRS across 5
    # VitalPatch segments against an independent locator) -- reverted, see
    # ecg_pipeline_core._snap_to_local_peak's docstring for the measurement.
    if len(r_peaks) == 0:
        return r_peaks
    snapped = r_peaks.copy()
    for i, r in enumerate(r_peaks):
        lo, hi = max(0, r - search_radius), min(len(signal), r + search_radius)
        if hi <= lo:
            continue
        snapped[i] = lo + int(np.argmax(np.abs(signal[lo:hi])))
    return snapped


def _apply_refractory_guard(signal: np.ndarray, r_peaks: np.ndarray, fs: float,
                             refractory_s: float = 0.2) -> np.ndarray:
    """Physiological refractory-period guard, applied after detection + snap.
    Kept in sync with ecg_pipeline_core._apply_refractory_guard -- see that
    function's docstring for the full rationale."""
    if len(r_peaks) < 2:
        return r_peaks
    min_gap = int(round(refractory_s * fs))
    kept = [int(r_peaks[0])]
    for r in r_peaks[1:]:
        r = int(r)
        if r - kept[-1] < min_gap:
            if abs(float(signal[r])) > abs(float(signal[kept[-1]])):
                kept[-1] = r
        else:
            kept.append(r)
    return np.asarray(kept, dtype=int)


def detect_and_segment(signal: np.ndarray, fs: float, cfg: BeatWindowConfig = BEATS,
                        detection_signal: np.ndarray | None = None,
                        snap_radius: int = 8) -> list[Beat]:
    """`signal` is what beats are windowed/featurized from (unchanged
    behavior). `detection_signal`, if given, is used only to locate R-peaks
    -- pass the Kalman-skipped variant here (apply_filter_chain(...,
    skip_emg_suppress=True)) to fix R-peak over-detection on small-mV-scale
    sources without changing a single value the classifier ever sees:
    peaks found on `detection_signal` are still snapped to the true local
    max and windowed/featurized from `signal`, exactly as before. Defaults
    to `signal` itself when not given, preserving old callers' behavior.

    `snap_radius` defaults to 8, the value validated via a 13-record MITDB
    (wfdb) sweep -- see _snap_to_local_peak's docstring. That sweep was
    wfdb-only; on real-device sources (vitalpatch/prorhythm) radius=8 was
    later found to cause catastrophic beat-level over-culling on a subset
    of recordings (up to 130/133 beats rejected via R_PEAK_NOT_LOCAL_MAX
    on one, non-monotonically -- radius 3/5/10/15/20 were all fine, only
    8 was not), so callers on those sources should pass the pre-existing
    radius=15 instead. See ECGPipeline.run()'s Stage 5 comment.
    """
    peaks_from = detection_signal if detection_signal is not None else signal
    r_peaks = detect_r_peaks(peaks_from, fs)
    r_peaks = _snap_to_local_peak(signal, r_peaks, search_radius=snap_radius)
    r_peaks = _apply_refractory_guard(signal, r_peaks, fs)
    return segment_beats(signal, fs, r_peaks, cfg)


