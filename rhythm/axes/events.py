"""EVENTS axis - pauses and ectopy BURDEN.

BURDEN, NEVER TYPE. A premature beat gives a SHORT interval followed by a
LONGER (compensatory) one, and that pattern is visible in timing alone. WHICH
KIND of premature beat it is - atrial or ventricular - is a waveform-SHAPE
property that timing cannot reach. This axis reports how many, never which.

PAUSE THRESHOLD - DERIVED.
  RR > 2000 ms. Measured on 1,121,159 sinus-labelled RR intervals from 16
  LTAFDB records: such intervals occur in 0.0089% of sinus beats (100 of
  1,121,159), which is the rarity a pause should have. p99 is 1193 ms and
  p99.9 is 1405 ms, so 2000 ms sits well into the tail. It also coincides with
  the physiological upper bound already used by the artefact screen, and with
  the clinical convention of >2 s.

ECTOPY DETECTOR - DERIVED AND VALIDATED against EXPERT BEAT ANNOTATIONS on two
HELD-OUT databases:

  burden >= 1   MITDB Se 0.877 PPV 0.951    SVDB Se 0.899 PPV 0.978
  burden >= 3   MITDB Se 0.759 PPV 0.917    SVDB Se 0.790 PPV 0.959
  burden >= 5   MITDB Se 0.693 PPV 0.893    SVDB Se 0.721 PPV 0.939

  Median detected tracks true closely across bands (true 0 -> 0, 1-2 -> 1,
  3-5 -> 3, 6-10 -> 6, >10 -> 10-11).

  KNOWN LIMITATION: sensitivity falls as the burden threshold rises, so high
  burden is under-called. And an isolated premature beat moves the regularity
  score only 0.06-0.08 - well under the 0.1275 threshold - so isolated ectopy
  is detected by THIS axis while remaining invisible to the regularity axis.
  That is the point of having separate axes.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

PAUSE_MS = 2000.0
SHORT_RATIO = 0.85       # premature interval, relative to the local median
COMP_RATIO = 1.05        # compensatory interval that follows it

BURDEN_NONE = "NONE"
BURDEN_LOW = "LOW"
BURDEN_HIGH = "HIGH"
BURDEN_UNDETERMINED = "UNDETERMINED"

LOW_MIN, HIGH_MIN = 1, 5     # couplets per 64-beat window

PROVENANCE = {
    "pause_threshold_ms": PAUSE_MS,
    "pause_basis": "DERIVED - LTAFDB sinus RR, 0.0089% exceed 2000 ms",
    "ectopy_basis": "DERIVED - validated vs expert beat labels, MITDB + SVDB",
    "ectopy_heldout_se_ppv_burden1": {"mitdb": [0.877, 0.951], "svdb": [0.899, 0.978]},
    "reports_type": False,
    "status": "PROVISIONAL - NOT SHIPPABLE",
}


@dataclass(frozen=True)
class EventsAxis:
    pause_count: int
    longest_rr_ms: float | None
    ectopic_couplets: int
    burden: str
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def count_ectopic_couplets(rr_ms, short_ratio: float = SHORT_RATIO,
                           comp_ratio: float = COMP_RATIO) -> int:
    """Count short-then-compensatory interval pairs against the local median."""
    rr = np.asarray(rr_ms, dtype=float)
    rr = rr[np.isfinite(rr)]
    if rr.size < 4:
        return 0
    med = float(np.median(rr))
    if med <= 0:
        return 0
    n, i = 0, 0
    while i < rr.size - 1:
        if rr[i] < short_ratio * med and rr[i + 1] > comp_ratio * med:
            n += 1
            i += 2                      # consume the couplet
        else:
            i += 1
    return n


def assess_events(rr_ms, features, trustworthy: bool) -> EventsAxis:
    """Pauses and ectopy burden. Refuses independently of the other axes."""
    rr = np.asarray(rr_ms, dtype=float)
    rr = rr[np.isfinite(rr)]
    if not trustworthy or rr.size < 4:
        return EventsAxis(0, None, 0, BURDEN_UNDETERMINED,
                          "beat timing not trustworthy for this window")

    pauses = int(np.sum(rr > PAUSE_MS))
    longest = float(rr.max())
    couplets = count_ectopic_couplets(rr)

    if couplets >= HIGH_MIN:
        burden = BURDEN_HIGH
    elif couplets >= LOW_MIN:
        burden = BURDEN_LOW
    else:
        burden = BURDEN_NONE

    bits = []
    if pauses:
        bits.append(f"{pauses} interval(s) over {PAUSE_MS:.0f} ms "
                    f"(longest {longest:.0f} ms)")
    bits.append(f"{couplets} premature-compensatory couplet(s) -> burden {burden} "
                f"(count only; the TYPE of ectopic beat is not determinable "
                f"from timing)")
    return EventsAxis(pauses, longest, couplets, burden, "; ".join(bits))
