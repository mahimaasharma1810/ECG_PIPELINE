"""Real-time rhythm pipeline.

    WebSocket -> rolling buffer -> preprocessing -> R-peak -> RR features
              -> classifier -> severity -> MedGemma (gated)

Transport-agnostic: `RhythmStream.push()` accepts samples from any source, so
the same engine serves the live WebSocket and the file-replay harness.

THREE THINGS THE LIVE PATH MUST HANDLE THAT BATCH DOES NOT:

1. REPLAYED PACKETS. 28 of 37 captures re-deliver a ~13 s block, up to 49% of
   rows. Offline this is fixed by removing exact duplicate (timestamp,
   amplitude) rows. Online the same rule is applied incrementally against a
   bounded set of recently-seen samples.

2. OUT-OF-ORDER ARRIVAL. Timestamps run backwards in 29 of 37 captures, by
   ~13.1 s. The buffer is therefore kept sorted by timestamp on analysis rather
   than assuming arrival order. No extra latency is added for this: the
   analysis window is ~45 s of beats anyway, far longer than the reorder span.

3. VARIABLE SAMPLE RATE AND DROPOUT. The rate is measured per analysis window
   from that window's own clock, and RR comes from index differences at that
   rate - never from per-sample timestamps, which are BLE packet-arrival stamps
   carrying +/-36 ms of transport jitter.

No filtering is applied. The firmware already filters (`ecg_clean`).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .features import window_features, BEATS_PER_WINDOW
from .sqi import assess
from .verdict import decide
from .axes import assess_rate, assess_events, combine_axes
from .axes.rate import rate_trustworthy
from .smoothing import DEFAULT_N_CONSECUTIVE
from .domain_gate import read_gate
from .pipeline import BSQI_MIN, LAG1_MIN, SDNN_GUARD_MS, DETECTOR_B

BUFFER_S = 180.0          # enough for several 64-beat windows plus dropout
SLIDE_BEATS = 8           # emit a verdict every 8 new beats (~6 s at 75 bpm)
GAP_S = 1.0
MIN_ANALYSIS_S = 45.0     # below this, 64 beats cannot fit


@dataclass
class Update:
    t_ms: float
    verdict: object
    features: object
    sqi: object
    rate: object
    events: object
    severity: object
    reported_state: str
    latency_ms: float
    fs_local: float
    n_buffered: int
    n_replayed_dropped: int
    narrative: dict | None = None


@dataclass
class RhythmStream:
    buffer_s: float = BUFFER_S
    slide_beats: int = SLIDE_BEATS
    n_consecutive: int = DEFAULT_N_CONSECUTIVE
    _t: deque = field(default_factory=deque, init=False)
    _x: deque = field(default_factory=deque, init=False)
    _seen: set = field(default_factory=set, init=False)
    _seen_q: deque = field(default_factory=deque, init=False)
    _n_replayed: int = field(default=0, init=False)
    _last_emit_beat_t: float = field(default=-np.inf, init=False)
    _run_val: str | None = field(default=None, init=False)
    _run_len: int = field(default=0, init=False)
    _state: str = field(default="UNABLE_TO_DETERMINE", init=False)
    _gate: object = field(default=None, init=False)

    def __post_init__(self):
        self._gate = read_gate()

    # -- ingest -------------------------------------------------------------
    def push(self, t_ms: float, amplitude: float) -> None:
        """Accept one sample. Drops exact replays; tolerates out-of-order."""
        key = (float(t_ms), float(amplitude))
        if key in self._seen:
            self._n_replayed += 1
            return
        self._seen.add(key); self._seen_q.append(key)
        self._t.append(float(t_ms)); self._x.append(float(amplitude))
        # bound the dedupe memory to roughly the buffer span
        while len(self._seen_q) > 40000:
            self._seen.discard(self._seen_q.popleft())
        newest = max(self._t) if self._t else 0.0
        while self._t and (newest - self._t[0]) / 1000.0 > self.buffer_s:
            self._t.popleft(); self._x.popleft()

    # -- analysis -----------------------------------------------------------
    def _latest_segment(self):
        if len(self._t) < 100:
            return None
        t = np.fromiter(self._t, float); x = np.fromiter(self._x, float)
        o = np.argsort(t, kind="stable"); t, x = t[o], x[o]
        brk = np.flatnonzero(np.diff(t) / 1000.0 > GAP_S)
        s = int(brk[-1] + 1) if brk.size else 0
        if (t[-1] - t[s]) / 1000.0 < MIN_ANALYSIS_S:
            return None
        return t[s:], x[s:]

    def analyse(self, detect_fn) -> Update | None:
        """Run one analysis pass. Returns an Update when a new window completes."""
        t0 = time.perf_counter()
        seg = self._latest_segment()
        if seg is None:
            return None
        t, x = seg
        el = (t[-1] - t[0]) / 1000.0
        fs = (t.size - 1) / el if el > 0 else float("nan")
        if not np.isfinite(fs) or fs <= 0:
            return None
        try:
            pa, pb = detect_fn(x, fs)
        except Exception:
            return None
        if pa.size < BEATS_PER_WINDOW:
            return None

        w = pa[-BEATS_PER_WINDOW:]
        peak_t = t[pa.astype(int)]
        beat_t = float(peak_t[-1])
        # emit only once the newest beat has advanced by slide_beats
        if self._last_emit_beat_t > -np.inf:
            if int(np.sum(peak_t > self._last_emit_beat_t)) < self.slide_beats:
                return None

        i0, i1 = int(w[0]), int(w[-1])
        el_w = (t[i1] - t[i0]) / 1000.0
        fs_loc = (i1 - i0) / el_w if el_w > 0 else fs
        rr = np.diff(w).astype(float) * 1000.0 / fs_loc
        feat = window_features(rr)

        pbw = pb[(pb >= i0) & (pb <= i1)]
        q = assess(x[i0:i1 + 1], fs_loc, w.astype(float) * 1000.0 / fs_loc,
                   pbw.astype(float) * 1000.0 / fs_loc, rr_ms=rr,
                   bsqi_min=BSQI_MIN, lag1_min=LAG1_MIN)
        passed, reason = q.passed, q.reason
        if not passed and np.isfinite(feat.sdnn_ms) and feat.sdnn_ms <= SDNN_GUARD_MS:
            rem = [r for r in reason.split("; ") if "lag-1" not in r]
            passed, reason = (not rem), ("; ".join(rem) or "ok")

        v = decide(feat, passed, reason)

        # hysteresis
        if v.verdict == self._run_val:
            self._run_len += 1
        else:
            self._run_val, self._run_len = v.verdict, 1
        if self._run_len >= self.n_consecutive and self._run_val != self._state:
            self._state = self._run_val

        rate_ok, rate_why = rate_trustworthy(q, feat)
        ra = assess_rate(feat, rate_ok, rate_why)
        ev = assess_events(rr, feat, passed)
        sev = combine_axes(ra, v, ev, self._gate.may_emit_clinical_severity)
        self._last_emit_beat_t = beat_t
        return Update(t_ms=beat_t, verdict=v, features=feat, sqi=q,
                      rate=ra, events=ev, severity=sev,
                      reported_state=self._state,
                      latency_ms=(time.perf_counter() - t0) * 1000.0,
                      fs_local=float(fs_loc), n_buffered=len(self._t),
                      n_replayed_dropped=self._n_replayed)
