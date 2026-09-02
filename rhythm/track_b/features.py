"""Track B feature set - RATE-NORMALISED by construction.

Every feature is a ratio, so none of them can encode heart rate. This is a
STRUCTURAL guard against a measured confound: heart rate discriminates
irregularity at AUC 0.9221 on these databases, but only because irregular
episodes there happen to run fast (median 112 vs 75 bpm). A model given rate
would learn "fast means irregular" - a property of those cohorts, not of hearts.

Making every feature scale-free means the confound cannot enter even by
accident, rather than relying on someone remembering to exclude it.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = ("rr_cv", "rmssd_norm", "pnn_norm", "iqr_norm",
                 "mad_norm", "range_norm", "turning_point_ratio")


def extract(rr_ms) -> dict:
    """Rate-normalised timing features from one window's RR intervals."""
    rr = np.asarray(rr_ms, dtype=float)
    rr = rr[np.isfinite(rr) & (rr > 0)]
    if rr.size < 8:
        return {k: np.nan for k in FEATURE_NAMES}

    mean = float(rr.mean())
    med = float(np.median(rr))
    d = np.diff(rr)

    # turning points: how often the series changes direction. A random series
    # gives ~2/3; a smooth or alternating one departs from it.
    if rr.size >= 3:
        s = np.sign(d)
        tp = int(np.sum(s[:-1] * s[1:] < 0))
        tpr = tp / (rr.size - 2)
    else:
        tpr = np.nan

    return {
        "rr_cv": float(rr.std(ddof=1) / mean),
        "rmssd_norm": float(np.sqrt(np.mean(d ** 2)) / mean),
        # pNN50 made scale-free: successive changes above 5% of the mean
        "pnn_norm": float(np.mean(np.abs(d) > 0.05 * mean)),
        "iqr_norm": float((np.percentile(rr, 75) - np.percentile(rr, 25)) / med),
        "mad_norm": float(np.median(np.abs(rr - med)) / med),
        "range_norm": float((rr.max() - rr.min()) / med),
        "turning_point_ratio": float(tpr),
    }
