"""Resolution matching (brief 4.5) - degrade public data to device resolution.

TWO RATES, AND THEY MEAN DIFFERENT THINGS.

  DEVICE_FS_DELIVERED = 89.70 Hz
      What the 37 historical captures actually contain (measured,
      replay-removed, gap-excluded across 37 captures; CV 0.6%). This is a
      property of the DELIVERY PATH, not the device: the path was dropping
      one sample in three upstream of packet timestamping.

  DEVICE_FS_NATIVE = 133.83 Hz
      Measured on the 2026-09-02 session, inside the vendor's stated 120-140
      Hz band. 89.50 / 133.83 = 0.6688, i.e. 2/3 to within measurement
      scatter - which is what identified the historical loss as a systematic
      2:3 decimation rather than packet loss.

Public data at 360/250/128 Hz must be brought to the device rate before any
comparison, or the separation observed will be optimistic.

WHY THE DEFAULT IS STILL THE DELIVERED RATE. n=1 at 133.83 Hz. Until several
sessions confirm it, the native rate is a documented alternative arm, not the
assumed device property. Callers that care about waveform resolution should
run BOTH (see scripts_rhythm/s03_validate_detector.py).

WHAT THE CORRECTION DOES NOT TOUCH. RR timing is computed as
`delta_index / fs_local`, measured per window, so the decimation cancelled:
89.5 delivered samples still span one real second. Confirmed empirically - the
clock-versus-count residual (`Segment.drop_samples`) has median -1.8 samples
over a 45 s window, i.e. the loss left no timing trace at all. The regularity
threshold is likewise unaffected: projecting the 21,920 LTAFDB derivation
windows from the 11.148 ms grid onto 7.472 ms leaves the Youden point at
0.1275 unchanged, with p95 |CV shift| 0.00064 - 42x smaller than the +/-0.027
refusal band. Quantisation enters RR variance in quadrature and is negligible
at CV ~0.13; it only matters near the floor.

WHAT IT DOES TOUCH: anything depending on WAVEFORM resolution - R-peak
detection (s03) and QRS morphology (s24), where a 100 ms QRS spans ~9.0
samples at 89.70 Hz and ~13.4 at 133.83 Hz.

THE HISTORICAL CAPTURES ARE DEGRADED, NOT WRONG. They carry 89.5 Hz of
information. Re-analysing them at 133.83 Hz would invent resolution that was
never delivered. Do not do it.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

DEVICE_FS_DELIVERED = 89.70   # Hz, measured - see reports_rhythm/s01d_dedup_rate.csv
DEVICE_FS_NATIVE = 133.83     # Hz - session 2026-09-02, n=1, UNCONFIRMED

DEVICE_FS = DEVICE_FS_DELIVERED   # default: what our data actually contains

# Vendor-stated band for the native rate. A session measuring outside this is
# on a delivery path we do not understand.
NATIVE_FS_BAND_HZ = (120.0, 140.0)


def resample_to(sig: np.ndarray, fs_in: float, fs_out: float = DEVICE_FS):
    """Anti-aliased resample. Returns (signal, exact_fs_out_achieved)."""
    frac = Fraction(fs_out / fs_in).limit_denominator(20000)
    up, down = frac.numerator, frac.denominator
    out = resample_poly(sig, up, down)
    return out, fs_in * up / down


def map_sample_indices(idx: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Map annotation sample indices from fs_in to the fs_out grid."""
    return np.round(np.asarray(idx, dtype=float) * (fs_out / fs_in)).astype(np.int64)

