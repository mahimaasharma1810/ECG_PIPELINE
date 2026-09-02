"""Combination layer - maps the three axes to a severity band.

WHY THIS LAYER EXISTS: it closes a measured gap that a regularity-only view
cannot see.

  Sustained fast atrial rhythms are FAST but EVENLY SPACED, so a regularity
  axis calls them REGULAR. Measured on SVDB: at >16 atrial ectopic beats per
  64-beat window the IRREGULAR flag rate DROPS to 60.0% from 77.6% at 9-16.
  The relationship is not monotonic, and high-burden supraventricular
  tachycardia can therefore read as REGULAR at a dangerous rate.

  Seeing RATE and REGULARITY together catches it. This is also the design
  reason the rhythm verdict must NEVER be displayed without the rate beside it.

ESCALATION POLICY, following from measured predictive values:

  Only REGULAR is trustworthy enough to act on (NPV ~0.998). IRREGULAR has
  PPV ~0.47 at the operating point - fewer than half of such calls are right -
  so it PROMPTS REVIEW and must NEVER raise an alarm on its own.

  Escalation is reserved for RATE EXTREMES and PAUSES, which are the only
  findings here with a defensible severity meaning on their own.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

from .rate import BRADYCARDIC, NORMAL as RATE_NORMAL, TACHYCARDIC, UNDETERMINED as RATE_UNDET
from .events import BURDEN_HIGH, BURDEN_LOW, BURDEN_UNDETERMINED
from ..verdict import REGULAR, IRREGULAR, UNABLE

NORMAL = "NORMAL"
ABNORMAL = "ABNORMAL"
CRITICAL = "CRITICAL"
WITHHELD = "WITHHELD"


@dataclass(frozen=True)
class CombinedAssessment:
    severity: str
    escalate: bool
    reason: str
    rate_state: str
    regularity_state: str
    events_burden: str
    findings: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def combine_axes(rate, verdict, events, clinical_permitted: bool) -> CombinedAssessment:
    """Combine the three axes. Fails closed when clinical output is not permitted."""
    base = dict(rate_state=rate.state, regularity_state=verdict.verdict,
                events_burden=events.burden)

    if not clinical_permitted:
        return CombinedAssessment(
            WITHHELD, False,
            "device-domain validation has not passed; no severity is emitted",
            findings=[], **base)

    # every axis undetermined -> nothing to say
    if (rate.state == RATE_UNDET and verdict.verdict == UNABLE
            and events.burden == BURDEN_UNDETERMINED):
        return CombinedAssessment(
            WITHHELD, False, "no axis produced a trustworthy result",
            findings=[], **base)

    findings, severity, escalate = [], NORMAL, False

    # --- CRITICAL: rate extremes and pauses only -----------------------
    if rate.extreme:
        findings.append(f"rate extreme: {rate.reason}")
        severity, escalate = CRITICAL, True
    if events.pause_count > 0:
        findings.append(f"pause: {events.pause_count} interval(s) over 2000 ms "
                        f"(longest {events.longest_rr_ms:.0f} ms)")
        severity, escalate = CRITICAL, True

    if severity == CRITICAL:
        return CombinedAssessment(severity, escalate, "; ".join(findings),
                                  findings=findings, **base)

    # --- ABNORMAL -------------------------------------------------------
    if rate.state == TACHYCARDIC and verdict.verdict == REGULAR:
        # the gap a regularity-only view misses
        findings.append(f"REGULAR rhythm at a tachycardic rate "
                        f"({rate.hr_bpm:.0f} bpm) - a regularity-only view would "
                        f"call this normal")
        severity = ABNORMAL
    elif rate.state in (BRADYCARDIC, TACHYCARDIC):
        findings.append(f"rate outside the normal band: {rate.reason}")
        severity = ABNORMAL

    if verdict.verdict == IRREGULAR:
        findings.append("rhythm IRREGULAR - PPV at this operating point is "
                        "~0.47, so this prompts review and must not raise an alarm")
        severity = ABNORMAL

    if events.burden == BURDEN_HIGH:
        findings.append(f"high ectopic burden ({events.ectopic_couplets} couplets); "
                        f"the TYPE of ectopic beat is not determinable from timing")
        severity = ABNORMAL
    elif events.burden == BURDEN_LOW:
        findings.append(f"low ectopic burden ({events.ectopic_couplets} couplet(s))")

    # --- partial evidence ------------------------------------------------
    if severity == NORMAL and verdict.verdict == UNABLE:
        return CombinedAssessment(
            WITHHELD, False,
            "rate and events look unremarkable but regularity is undetermined; "
            "NORMAL is not asserted without it",
            findings=findings, **base)

    if severity == NORMAL:
        findings.append(f"rate {rate.hr_bpm:.0f} bpm within the normal band, "
                        f"rhythm REGULAR, no pauses, ectopic burden "
                        f"{events.burden.lower()}")

    return CombinedAssessment(severity, escalate, "; ".join(findings),
                              findings=findings, **base)
