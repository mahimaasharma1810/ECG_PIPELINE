"""Stage 8 - temporal smoothing with hysteresis.

Overlapping windows share most of their beats, so a single borderline
interval flips consecutive verdicts. A display alternating every 10 s
destroys clinical trust faster than being occasionally wrong.

The reported state changes only after N consecutive windows agree. Every
transition is logged with the evidence that caused it.
"""
from __future__ import annotations

from dataclasses import dataclass

from .verdict import UNABLE

DEFAULT_N_CONSECUTIVE = 5   # derived-ish: the flicker curve has NO knee (each
                            # increment buys a near-constant 15-21% reduction
                            # while lag grows linearly), so this is a judgement,
                            # not a measurement. At N=5: 8.43 spurious state
                            # changes/hour, 31.6 s lag, measured on 187
                            # label-stable LTAFDB stretches (86,691 windows).
                            # Largely cosmetic: only REGULAR is actionable, so
                            # display churn on a non-alarming indicator is cheap.


@dataclass(frozen=True)
class Transition:
    at_index: int
    t0_ms: float
    from_state: str
    to_state: str
    n_agreeing: int
    evidence: dict


def smooth(verdicts, t0_ms_list, n_consecutive: int = DEFAULT_N_CONSECUTIVE):
    """Return (reported_states, transitions).

    Starts in UNABLE_TO_DETERMINE: the system asserts nothing until it has
    seen n_consecutive agreeing windows.
    """
    state = UNABLE
    run_val, run_len = None, 0
    states, transitions = [], []

    for i, v in enumerate(verdicts):
        if v.verdict == run_val:
            run_len += 1
        else:
            run_val, run_len = v.verdict, 1

        if run_len >= n_consecutive and run_val != state:
            transitions.append(Transition(
                at_index=i, t0_ms=float(t0_ms_list[i]),
                from_state=state, to_state=run_val,
                n_agreeing=run_len, evidence=dict(v.evidence),
            ))
            state = run_val
        states.append(state)

    return states, transitions
