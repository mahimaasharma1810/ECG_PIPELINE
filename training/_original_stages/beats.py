"""Stage 5 — R-peak detection and beat segmentation.

Uses WFDB's XQRS rather than Pan-Tompkins: wearable single-lead
electrodes sit at non-standard chest positions and produce morphologically
atypical QRS complexes that XQRS's adaptive threshold handles better.

RR intervals are sanity-checked against a 300-2000 ms physiological range;
outliers are FLAGGED, never silently dropped (recommendation #6 — a fast,
irregular-but-real AFib run must survive to the rhythm classifier instead
of being discarded as a "bad beat").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import BEATS, BeatWindowConfig


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


def detect_r_peaks(signal: np.ndarray, fs: float) -> np.ndarray:
    """WFDB XQRS adaptive-threshold R-peak detector."""
    import wfdb.processing as wp
    if len(signal) < int(fs * 2):
        return np.array([], dtype=int)
    xqrs = wp.XQRS(sig=signal, fs=fs)
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
        ))

    return beats


def detect_and_segment(signal: np.ndarray, fs: float, cfg: BeatWindowConfig = BEATS) -> list[Beat]:
    r_peaks = detect_r_peaks(signal, fs)
    return segment_beats(signal, fs, r_peaks, cfg)
