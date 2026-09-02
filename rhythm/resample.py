"""Stage 2 - uniform time grid.

BRIEF DEVIATION, deliberate and load-bearing.

Brief 6 Stage 2 says to resample "using real per-sample timestamps, not an
assumed fixed rate". That is right for a device with genuine per-sample
timing. This device does not have one: `timestamp_ms` is a BLE PACKET-ARRIVAL
stamp shared by a burst of ~6.5 samples, one burst per ~73 ms. Treating those
stamps as sampling instants injects +/-36 ms of transport jitter into every
R-peak, which lands directly in RR.

Measured cost, for a PERFECTLY REGULAR rhythm
(scripts_rhythm/s01e_noise_floor.py):

    reconstruction            CV floor      RMSSD floor
    uniform grid              0.005-0.007   7-9 ms
    per-sample packet stamps  0.046-0.095   ~100 ms

At HR 100 the packet-stamp floor is CV 0.095 - a CV=0.10 threshold would sit
ON the noise floor, and the pipeline would be reporting BLE behaviour as
rhythm irregularity.

ANCHORING (the load-bearing choice, documented because it is):

RR intervals are differences, so an absolute time offset CANCELS. What
survives into RR is the local sample RATE. The grid is therefore built from
SAMPLE INDEX, with the rate measured locally from the clock:

    RR = (index difference) / fs_local

The hardware sample clock is far more stable than BLE arrival stamps, so
index differences carry the timing information and the stamps are used only
to (a) measure fs and (b) detect dropped samples.

A single linear model per segment was tried and REJECTED: over a 1572 s
segment it left residuals of 1710 ms RMS (max 3613 ms), because the rate
drifts. Local windows shorter than ~55 s produced NON-MONOTONE time models.
Measured local-rate stability over 60 s blocks is 0.20-1.06% CV per file,
which is the residual RR error this choice carries.

Concentrated sample drops are the failure this cannot absorb: a window-level
rate silently redistributes them. They are DETECTED (drop_samples below) and
flagged, never smoothed away.

Amplitudes are NOT interpolated. Once the time model says sample i sits at
a + b*i, the series is already uniformly sampled at 1/b; only the time labels
change. Nothing is filtered here (brief 3.2 trap 3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GAP_S = 1.0            # dt above this is a dropout, not a sample interval
MIN_SEGMENT_S = 40.0   # shorter than one 64-beat window is not worth carrying


@dataclass(frozen=True)
class Segment:
    """One continuous stretch. Times come from index + locally measured rate."""
    t_raw_ms: np.ndarray    # original BLE arrival stamps (staircase), diagnostics only
    x: np.ndarray           # amplitude, UNKNOWN units, unfiltered, unscaled
    fs: float               # segment-level rate, samples / elapsed clock
    t0_ms: float
    duration_s: float
    n: int

    def local_fs(self, i0: int, i1: int) -> float:
        """Sample rate measured over [i0, i1] from that span's own clock."""
        i0, i1 = int(max(0, i0)), int(min(self.n - 1, i1))
        el = (self.t_raw_ms[i1] - self.t_raw_ms[i0]) / 1000.0
        return float((i1 - i0) / el) if el > 0 else float("nan")

    def drop_samples(self, i0: int, i1: int) -> float:
        """Samples of clock time unaccounted for by the sample count in [i0,i1].

        Compares elapsed clock against index count at the segment rate. A
        concentrated dropout shows up here as a large positive number; burst
        jitter alone stays near zero.
        """
        i0, i1 = int(max(0, i0)), int(min(self.n - 1, i1))
        el = (self.t_raw_ms[i1] - self.t_raw_ms[i0]) / 1000.0
        return float(el * self.fs - (i1 - i0))


def dedupe_exact(t_ms: np.ndarray, x: np.ndarray):
    """Remove replayed rows: exact duplicate (timestamp, amplitude) pairs.

    28 of 37 captures replay up to 49% of their rows (see
    reports_rhythm/s01d_dedup_rate.csv). Without this, the measured rate reads
    88-178 Hz instead of a consistent ~89.7 Hz.
    """
    pairs = np.stack([t_ms, x], axis=1)
    _, idx = np.unique(pairs, axis=0, return_index=True)
    idx.sort()
    return t_ms[idx], x[idx], int(t_ms.size - idx.size)


def segment(t_ms: np.ndarray, x: np.ndarray, gap_s: float = GAP_S):
    """Split into continuous stretches at dropouts."""
    if t_ms.size == 0:
        return []
    brk = np.flatnonzero(np.diff(t_ms) / 1000.0 > gap_s)
    starts = np.r_[0, brk + 1]
    ends = np.r_[brk, t_ms.size - 1]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]


def build_segment(t_ms: np.ndarray, x: np.ndarray) -> Segment | None:
    """Wrap one continuous stretch, measuring its sample rate from the clock."""
    n = t_ms.size
    if n < 2:
        return None
    el = (t_ms[-1] - t_ms[0]) / 1000.0
    if el <= 0:
        return None
    return Segment(t_raw_ms=t_ms, x=x, fs=float((n - 1) / el),
                   t0_ms=float(t_ms[0]), duration_s=float(el), n=int(n))


def to_uniform_segments(t_ms: np.ndarray, x: np.ndarray,
                        min_segment_s: float = MIN_SEGMENT_S):
    """Full Stage 2: dedupe -> sort -> segment -> uniform grid per segment."""
    t, xx, n_replayed = dedupe_exact(np.asarray(t_ms, float), np.asarray(x, float))
    order = np.argsort(t, kind="stable")     # stable: preserves arrival order in ties
    t, xx = t[order], xx[order]
    segs = []
    for s, e in segment(t, xx):
        seg = build_segment(t[s:e + 1], xx[s:e + 1])
        if seg is not None and seg.duration_s >= min_segment_s:
            segs.append(seg)
    return segs, n_replayed
