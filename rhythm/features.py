"""Stages 5 and 6 - RR artefact screen and per-window rhythm features.

Two rules from the brief are load-bearing and implemented explicitly here:

  * 64 beats == 63 intervals (brief 5.6). Never hand-compute this; the
    conversion goes through BEATS_PER_WINDOW/INTERVALS_PER_WINDOW.

  * RMSSD must never span a gap left by a removed interval (brief 5.5).
    Outliers are FLAGGED, the series is SEGMENTED at the flags, and successive
    differences are taken within segments only.

Outliers are counted, never silently dropped. Above MAX_FLAGGED_FRACTION the
window is UNABLE_TO_DETERMINE - a verdict is never forced by deleting
inconvenient values.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

BEATS_PER_WINDOW = 64
INTERVALS_PER_WINDOW = BEATS_PER_WINDOW - 1   # == 63, explicitly

RR_MIN_MS, RR_MAX_MS = 300.0, 2000.0          # physiological bounds (brief 5)
MAX_FLAGGED_FRACTION = 0.20                   # above this -> UNABLE_TO_DETERMINE

UNABLE = "UNABLE_TO_DETERMINE"


@dataclass(frozen=True)
class WindowFeatures:
    n_intervals: int
    n_flagged: int
    flagged_fraction: float
    usable: bool
    reason: str
    mean_rr_ms: float
    hr_bpm: float
    rr_cv: float
    rmssd_ms: float
    sdnn_ms: float
    n_rmssd_pairs: int
    n_segments: int

    def as_dict(self) -> dict:
        return asdict(self)


def screen_rr(rr_ms: np.ndarray) -> np.ndarray:
    """Stage 5: return a boolean mask of PHYSIOLOGICALLY IMPLAUSIBLE intervals."""
    rr = np.asarray(rr_ms, dtype=float)
    return ~np.isfinite(rr) | (rr < RR_MIN_MS) | (rr > RR_MAX_MS)


def segment_aware_rmssd(rr_ms: np.ndarray, flagged: np.ndarray):
    """RMSSD over successive differences that do NOT span a flagged interval.

    Returns (rmssd, n_pairs_used, n_segments).
    """
    rr = np.asarray(rr_ms, dtype=float)
    ok = ~np.asarray(flagged, dtype=bool)
    sq, n_pairs, n_seg = [], 0, 0
    i = 0
    while i < ok.size:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < ok.size and ok[j + 1]:
            j += 1
        seg = rr[i:j + 1]
        n_seg += 1
        if seg.size >= 2:
            d = np.diff(seg)
            sq.append(d ** 2)
            n_pairs += d.size
        i = j + 1
    if n_pairs == 0:
        return float("nan"), 0, n_seg
    return float(np.sqrt(np.concatenate(sq).mean())), n_pairs, n_seg


def window_features(rr_ms: np.ndarray) -> WindowFeatures:
    """Stage 6 features for ONE window of INTERVALS_PER_WINDOW intervals."""
    rr = np.asarray(rr_ms, dtype=float)
    flagged = screen_rr(rr)
    n, nf = rr.size, int(flagged.sum())
    frac = nf / n if n else 1.0

    rmssd, n_pairs, n_seg = segment_aware_rmssd(rr, flagged)
    kept = rr[~flagged]

    usable, reason = True, "ok"
    if n < INTERVALS_PER_WINDOW:
        usable, reason = False, f"{UNABLE}: only {n} intervals, need {INTERVALS_PER_WINDOW}"
    elif frac > MAX_FLAGGED_FRACTION:
        usable, reason = False, (f"{UNABLE}: {nf}/{n} intervals ({frac:.0%}) outside "
                                 f"{RR_MIN_MS:.0f}-{RR_MAX_MS:.0f} ms")
    elif kept.size < 2:
        usable, reason = False, f"{UNABLE}: {kept.size} usable intervals"

    mean_rr = float(kept.mean()) if kept.size else float("nan")
    return WindowFeatures(
        n_intervals=n, n_flagged=nf, flagged_fraction=float(frac),
        usable=usable, reason=reason,
        mean_rr_ms=mean_rr,
        hr_bpm=float(60000.0 / mean_rr) if kept.size and mean_rr > 0 else float("nan"),
        rr_cv=float(kept.std(ddof=1) / kept.mean()) if kept.size >= 2 else float("nan"),
        rmssd_ms=rmssd, 
        sdnn_ms=float(kept.std(ddof=1)) if kept.size >= 2 else float("nan"),
        n_rmssd_pairs=n_pairs, n_segments=n_seg,
    )


def iter_windows(peaks_ms: np.ndarray, step_beats: int = BEATS_PER_WINDOW):
    """Yield (start_beat, rr_slice) for consecutive/overlapping beat windows."""
    peaks = np.asarray(peaks_ms, dtype=float)
    rr = np.diff(peaks)
    for s in range(0, rr.size - INTERVALS_PER_WINDOW + 1, step_beats):
        yield s, rr[s:s + INTERVALS_PER_WINDOW]
