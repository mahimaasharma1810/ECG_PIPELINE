"""Windows for the waveform+timing fusion model (BUILD_RHYTHM_MODEL.md step 6).

WINDOW DESIGN, and why it is 64 BEATS resampled to fixed length rather than a
fixed number of seconds:

  * It keeps the model comparable to everything already validated - Track A's
    threshold, Track B's RR model, and the axes all operate on 64-beat windows,
    so labels and RR features are identical and results can be compared directly.

  * It makes the RATE CONFOUND STRUCTURALLY IMPOSSIBLE. Heart rate discriminates
    irregularity at AUC 0.9221 on these databases, but only because irregular
    episodes there happen to run fast (median 112 vs 75 bpm). A model given rate
    learns "fast means irregular" and collapses on a rate-controlled population.
    Resampling a fixed BEAT COUNT to a fixed SAMPLE COUNT normalises rate away:
    64 beats at 50 bpm and 64 beats at 150 bpm both become the same array
    length, so beat rate is not recoverable from the waveform input.

  * The RR side input is rate-normalised by construction for the same reason
    (see rhythm/track_b/features.py).

So neither branch can use heart rate. Rate stays a separate axis, as measured.

DEVICE-DOMAIN MATCHING. Public data is optionally passed through the device's
signal path: resampled to the native 133.83 Hz, then 2:3 decimated to the 89.70
Hz that the historical captures actually delivered. The decimation is SYSTEMATIC
(every third sample), not random - measured, and the distinction matters:
systematic decimation leaves RR CV unchanged (0.0485 -> 0.0490) whereas random
loss doubles it (0.1008).

NO FILTERING is applied, matching the device path and the device team's own
filter document.
"""
from __future__ import annotations

import numpy as np
import wfdb

from ..degrade import resample_to, DEVICE_FS_NATIVE, DEVICE_FS_DELIVERED
from ..features import window_features, BEATS_PER_WINDOW
from ..track_b.features import extract as rr_features, FEATURE_NAMES

WAVE_LEN = 2048          # samples per window after length normalisation
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")


def clean_aux(a: str) -> str:
    return (a or "").replace("\x00", "").strip()


def device_path(sig, fs_in, decimate=True):
    """Put clean public data through the device's actual signal path."""
    x, fs = resample_to(sig, fs_in, DEVICE_FS_NATIVE)
    if not decimate:
        return x, fs
    keep = np.arange(x.size) % 3 != 2          # systematic 2:3, as measured
    return x[keep], fs * 2.0 / 3.0


def _norm_wave(seg):
    seg = np.asarray(seg, float)
    seg = seg - np.median(seg)
    s = np.percentile(np.abs(seg), 95)         # robust to motion spikes
    return seg / s if s > 0 else seg


def build_record(dbpath, rec, decimate=True, require_mlii=True):
    """Yield (waveform, rr_features, label, record) per 64-beat window."""
    h = wfdb.rdheader(f"{dbpath}/{rec}")
    if require_mlii and "MLII" not in h.sig_name:
        return []
    ch = h.sig_name.index("MLII") if "MLII" in h.sig_name else 0
    try:
        sig = wfdb.rdrecord(f"{dbpath}/{rec}", channels=[ch]).p_signal[:, 0]
    except Exception:
        # A TRUNCATED .dat file. This only affects WAVEFORM work: annotation-only
        # analysis (Track A/B, the threshold) reads .atr and is unaffected, which
        # is why an interrupted download went unnoticed until now. LTAFDB record
        # 110 is 16% of its expected size.
        return []
    fs_in = float(h.fs)
    ann = wfdb.rdann(f"{dbpath}/{rec}", "atr")

    x, fs = device_path(sig, fs_in, decimate)
    keep = np.array([s in BEAT_SYMBOLS for s in ann.symbol])
    samp = np.round(ann.sample[keep] * (fs / fs_in)).astype(int)

    marks = [(s, clean_aux(a)) for s, a in zip(ann.sample, ann.aux_note)
             if clean_aux(a).startswith("(")]
    if not marks:
        return []
    ms = np.array([m[0] for m in marks])
    lb = np.array([m[1] for m in marks], dtype=object)
    idx = np.searchsorted(ms, ann.sample[keep], side="right") - 1
    rl = np.where(idx >= 0, lb[np.clip(idx, 0, None)], "")

    out = []
    for s in range(0, samp.size - BEATS_PER_WINDOW + 1, BEATS_PER_WINDOW):
        u = set(rl[s:s + BEATS_PER_WINDOW])
        if len(u) != 1:
            continue
        y = {"(AFIB": 1, "(N": 0}.get(u.pop())
        if y is None:
            continue
        w = samp[s:s + BEATS_PER_WINDOW]
        a, b = int(w[0]), int(w[-1])
        if a < 0 or b >= x.size or b - a < 64:
            continue
        rr = np.diff(w).astype(float) * 1000.0 / fs
        f = window_features(rr)
        if not f.usable:
            continue
        # resample the beat-span to a fixed length -> rate normalised away
        seg = np.interp(np.linspace(a, b, WAVE_LEN), np.arange(a, b + 1), x[a:b + 1])
        feats = rr_features(rr)
        if not np.all(np.isfinite(list(feats.values()))):
            continue
        out.append((_norm_wave(seg).astype(np.float32),
                    np.array([feats[k] for k in FEATURE_NAMES], np.float32),
                    int(y), rec))
    return out
