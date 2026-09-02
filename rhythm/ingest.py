"""Stage 1 - ingest and normalise ProRhythm / SeNSiO live captures.

Only the WebSocket live-capture format is supported:

    timestamp_ms,amplitude,received_at_utc   (header on row 1)

The device-software "SeNSiO export" format (11 metadata rows, header at row
16) is deliberately NOT supported - it is out of scope for this pipeline.
A file that does not match the live layout raises FormatError rather than
being parsed on a guess.

The effective sample rate is MEASURED as n_samples / time_span. It is never
inferred from median(dt): roughly half the samples in a real capture share a
timestamp with another sample, so median(dt) collapses to 1 ms and reports a
spurious ~1000 Hz.

Amplitude units are UNKNOWN (not mV, probably ADC counts). Nothing in this
module scales them, and no downstream rhythm maths may depend on their value.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

LIVE_COLUMNS = ("timestamp_ms", "amplitude", "received_at_utc")


class FormatError(ValueError):
    """Raised when a file is not a ProRhythm live capture."""


@dataclass(frozen=True)
class Capture:
    """A parsed live capture, with its measured timing facts."""

    path: str
    sha256: str
    subject: str
    t_ms: np.ndarray          # raw timestamps, as recorded (ms)
    amplitude: np.ndarray     # raw amplitude, UNKNOWN units - never scaled
    n_samples: int
    span_s: float
    fs_effective: float       # n_samples / span_s  <- the honest rate
    fs_median_dt: float       # 1000 / median(dt)   <- the misleading one
    n_unique_ts: int
    duplicate_fraction: float
    amp_min: float
    amp_max: float
    monotonic: bool
    n_backward_steps: int
    max_gap_s: float

    def timing_summary(self) -> dict:
        d = asdict(self)
        d.pop("t_ms")
        d.pop("amplitude")
        return d


def detect_format(path: Path) -> str:
    """Return 'live' for a ProRhythm live capture; raise FormatError otherwise."""
    with open(path, "r", errors="replace") as fh:
        first = fh.readline().strip()
    cols = tuple(c.strip() for c in first.split(","))
    if cols[:3] == LIVE_COLUMNS:
        return "live"
    raise FormatError(
        f"{path.name}: not a ProRhythm live capture "
        f"(header row 1 is {first!r}, expected {','.join(LIVE_COLUMNS)})"
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_capture(path: str | Path, subject: str | None = None) -> Capture:
    """Parse a live capture and measure its timing properties.

    No filtering, no resampling, no unit conversion happens here.
    """
    path = Path(path)
    detect_format(path)

    df = pd.read_csv(path, usecols=["timestamp_ms", "amplitude"])
    t_ms = df["timestamp_ms"].to_numpy(dtype=np.float64)
    amp = df["amplitude"].to_numpy(dtype=np.float64)

    finite = np.isfinite(t_ms) & np.isfinite(amp)
    t_ms, amp = t_ms[finite], amp[finite]
    if t_ms.size < 2:
        raise FormatError(f"{path.name}: fewer than 2 usable samples")

    order = np.argsort(t_ms, kind="stable")
    n_backward = int(np.sum(np.diff(t_ms) < 0))
    t_sorted, amp_sorted = t_ms[order], amp[order]

    span_s = float((t_sorted[-1] - t_sorted[0]) / 1000.0)
    n = int(t_sorted.size)
    dt = np.diff(t_sorted)
    med_dt = float(np.median(dt))

    return Capture(
        path=str(path),
        sha256=_sha256(path),
        subject=subject if subject is not None else path.parent.name,
        t_ms=t_sorted,
        amplitude=amp_sorted,
        n_samples=n,
        span_s=span_s,
        fs_effective=float(n / span_s) if span_s > 0 else float("nan"),
        fs_median_dt=float(1000.0 / med_dt) if med_dt > 0 else float("inf"),
        n_unique_ts=int(np.unique(t_sorted).size),
        duplicate_fraction=float(1.0 - np.unique(t_sorted).size / n),
        amp_min=float(amp_sorted.min()),
        amp_max=float(amp_sorted.max()),
        monotonic=bool(n_backward == 0),
        n_backward_steps=n_backward,
        max_gap_s=float(dt.max() / 1000.0) if dt.size else 0.0,
    )
