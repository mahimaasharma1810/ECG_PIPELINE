"""TASK 4 - how the five layers combine into one output.

    L1  signal quality      gates everything; may refuse
    L2  rate                catches what rhythm structurally misses
    L3  rhythm regularity   PRIMARY, deterministic, threshold 0.1275
    L4  ectopy burden       timing only; how much, never which type
    L5  ectopy type         morphology; PVC only; PUBLIC DATA ONLY

RULES, in force:

  1. L3 IS PRIMARY AND IS NEVER OVERRIDDEN. If L5 says PVC and L3 says
     REGULAR, both are reported. The verdict does not flip. This keeps the
     deterministic core auditable - a verdict always traces to one threshold
     on one feature.

  2. L2/L4/L5 ATTACH AS ANNOTATIONS. They explain or qualify L3; they do not
     replace it.

  3. EVERY LAYER MAY REFUSE INDEPENDENTLY, and refusals are reported per
     layer, never collapsed. On device data L2 currently works while L3 and L4
     do not - a single collapsed "undetermined" would hide that.

  4. DISAGREEMENT IS REPORTED, NEVER AVERAGED. Track A and Track B already
     disagreed on 17.3% of windows with roughly equal error in both
     directions. More layers means more disagreement; surfacing it is the
     point, not a defect.

  5. THE OUTPUT NAMES WHICH LAYERS DROVE IT.

MEASURED BASIS FOR THE COMBINATIONS (MITDB + SVDB, 4,525 windows):

  L4 SPECIFICITY LIMIT: L4 separates irregular from REGULAR rhythm well
                      (fires on 2.7% of ectopy-free sinus windows) but CANNOT
                      separate ectopy-driven from other irregularity (fires on
                      91.3% of ectopy-free AFIB windows). Once a rhythm is
                      irregular, L4 adds no aetiological information.

  fast + regular      350 windows (7.7%) read REGULAR at HR>100, up to 161
                      bpm. L3 alone calls every one normal. L2 flags 100%.
  slow + regular      339 windows (7.5%) read REGULAR below 60 bpm, median 56.
  alternating         73 true bigeminy windows; L3's lag-1 guard refuses 93.2%,
                      5 reach IRREGULAR, none are called REGULAR.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

from .verdict import REGULAR, IRREGULAR, UNABLE
from .axes.rate import TACHYCARDIC, BRADYCARDIC, NORMAL as RATE_NORMAL, UNDETERMINED as RATE_UNDET
from .axes.events import BURDEN_HIGH, BURDEN_LOW, BURDEN_NONE, BURDEN_UNDETERMINED

# clinically meaningful patterns, enumerated
PATTERN_SVT = "REGULAR_TACHYCARDIA"
PATTERN_BRADY = "REGULAR_BRADYCARDIA"
PATTERN_IRREGULAR = "IRREGULAR_RHYTHM"

# REMOVED, and why. An earlier design distinguished IRREGULARITY_WITH_ECTOPY
# from IRREGULARITY_WITHOUT_ECTOPY. That distinction is NOT SUPPORTED and has
# been withdrawn.
#
# Measured (MITDB): among AFIB windows containing ZERO annotated ectopic beats,
# L4 reports at least one ectopic couplet in 91.3% of them, median 5 couplets -
# statistically indistinguishable from genuine ectopy (97.2%, median 4).
# Against REGULAR windows L4 is specific (fires on 2.7%), which is why its
# earlier validation looked strong: the negatives were almost all regular.
#
# The short-then-long pattern is produced by ANY irregular rhythm. L4 therefore
# measures IRREGULARITY STRUCTURE, not ectopy specifically, once the rhythm is
# already irregular. Claiming otherwise would let the system imply an aetiology
# it cannot establish.
PATTERN_BIGEMINY = "ALTERNATING_PATTERN"
PATTERN_PAUSE = "PAUSE"
PATTERN_UNREMARKABLE = "UNREMARKABLE"
PATTERN_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class LayeredResult:
    pattern: str
    primary_verdict: str          # L3, never overridden
    driven_by: list               # which layers produced the pattern
    refused_layers: list          # which layers declined, and why
    disagreements: list           # conflicts between layers, surfaced
    annotations: list
    escalate: bool

    def as_dict(self) -> dict:
        return asdict(self)


def combine_layers(sqi_passed, rate, verdict, events, morphology=None):
    """Layer L3 stays primary; L2/L4/L5 annotate. Nothing is averaged."""
    refused, notes, disagree, driven = [], [], [], []

    if not sqi_passed:
        refused.append("L1 signal quality: gate not passed")
    if rate.state == RATE_UNDET:
        refused.append(f"L2 rate: {rate.reason}")
    if verdict.verdict == UNABLE:
        refused.append(f"L3 regularity: {verdict.reason}")
    if events.burden == BURDEN_UNDETERMINED:
        refused.append("L4 ectopy burden: not computable for this window")
    if morphology is None:
        refused.append("L5 ectopy type: not computed (public data only; "
                       "UNTESTED-ON-DEVICE)")

    # --- pauses first: the only finding that escalates on its own besides rate
    if events.pause_count > 0:
        driven.append("L4")
        notes.append(f"{events.pause_count} pause(s) over 2000 ms "
                     f"(longest {events.longest_rr_ms:.0f} ms)")
        return LayeredResult(PATTERN_PAUSE, verdict.verdict, driven, refused,
                             disagree, notes, escalate=True)

    # --- L3 primary, L2 qualifying ---------------------------------------
    pattern = PATTERN_INSUFFICIENT
    escalate = bool(rate.extreme)
    if rate.extreme:
        driven.append("L2")
        notes.append(f"rate extreme: {rate.reason}")

    if verdict.verdict == REGULAR:
        driven.append("L3")
        if rate.state == TACHYCARDIC:
            driven.append("L2")
            pattern = PATTERN_SVT
            notes.append(f"REGULAR at {rate.hr_bpm:.0f} bpm - L3 alone would "
                         f"call this unremarkable; 350 such windows (7.7%) occur "
                         f"in public data, up to 161 bpm")
        elif rate.state == BRADYCARDIC:
            driven.append("L2")
            pattern = PATTERN_BRADY
            notes.append(f"REGULAR at {rate.hr_bpm:.0f} bpm")
        elif rate.state == RATE_NORMAL:
            pattern = PATTERN_UNREMARKABLE
            notes.append(f"REGULAR at {rate.hr_bpm:.0f} bpm, no pauses, "
                         f"ectopic burden {events.burden.lower()}")

    elif verdict.verdict == IRREGULAR:
        driven.append("L3")
        notes.append("IRREGULAR - PPV ~0.47 at this operating point, so this "
                     "prompts review and must not raise an alarm")
        pattern = PATTERN_IRREGULAR
        if events.burden in (BURDEN_HIGH, BURDEN_LOW):
            driven.append("L4")
            notes.append(f"{events.ectopic_couplets} short-then-long couplet(s). "
                         f"NOTE: this pattern is produced by any irregular rhythm, "
                         f"not only by ectopy - L4 fires on 91% of AFIB windows "
                         f"containing no ectopic beats. It does NOT establish an "
                         f"ectopic cause")

    # --- L5 annotates; it never flips L3 ---------------------------------
    if morphology is not None:
        driven.append("L5")
        n_pvc = morphology.get("n_pvc", 0)
        notes.append(f"morphology: {n_pvc} beat(s) classified PVC "
                     f"(PVC only; PACs are not identifiable single-lead)")
        if n_pvc > 0 and events.burden == BURDEN_NONE:
            disagree.append("L5 reports PVCs but L4 timing found no ectopic "
                            "couplets - reported, not reconciled")
        if n_pvc == 0 and events.burden == BURDEN_HIGH:
            disagree.append("L4 reports high ectopic burden but L5 morphology "
                            "found no PVCs - consistent with PACs, which L5 "
                            "cannot identify")
        if n_pvc > 0 and verdict.verdict == REGULAR:
            disagree.append("L5 reports PVCs while L3 reports REGULAR - the "
                            "verdict is NOT changed; both are reported")

    return LayeredResult(pattern, verdict.verdict, driven, refused, disagree,
                         notes, escalate)
