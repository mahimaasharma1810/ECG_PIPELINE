# NEWS2 Partial-Coverage Escalation — Decision Needed

**Written:** 2026-07-30
**Status:** Open. This document lays out the options; it does not choose one.
Per `Docs/HANDOFF.md` §8, this is a clinical safety decision, not an
engineering one, and should not be made unilaterally by whoever is writing
code.

## The corrected fact this decision rests on

A prior version of this project's documentation stated that NEWS2 ≥ 7 (the
threshold that triggers this system's CRITICAL escalation) was
**"arithmetically unreachable"** with only 3 of 6 NEWS2 components available
(VitalPatch supplies heart rate, respiratory rate, and temperature; it has no
SpO2 or blood-pressure sensor, and no consciousness/AVPU input).

**That claim was wrong, and this session corrected it with real data.** The
three scoreable components have maximums of HR=3, RR=3, Temp=2
(`ecg_pipeline/agent_bridge.py`'s `_news2_hr_score`/`_news2_rr_score`/
`_news2_temp_score`), which sum to a possible **8**, not below 7. Running the
full multimodal batch on all 3,632 real segments
(`data/reports/multimodal_batch/multimodal_manifest.json`, 2026-07-30) found
**one real segment that scored exactly 7** (`Patch_184635`,
segment `1780050938536_..._seg0`: HR 150.3 bpm → score 3, RR 26.7/min → score
3, Temp 38.3°C → score 1). So reaching the NEWS2 ≥ 7 threshold on 3/6
coverage is **rare, not impossible**: 1 out of 3,632 real segments.

This changes the shape of the decision. The prior framing ("the override can
never fire, so is it dead code or should we lower the bar?") assumed 7 was
unreachable. The real question is now: **is it clinically appropriate to let
this rare event trigger a NEWS2-based CRITICAL escalation when 3 of the 6
standard components (SpO2, blood pressure, consciousness) are entirely
absent and simply not contributing to the score at all** — as opposed to
being present and normal?

Separately, and unaffected by this correction: the **qSOFA** override (≥2)
works as designed and is not arithmetically constrained the same way — it
fired in 107/3,632 segments and actually changed the final risk level in
52/3,632 (1.43%) of them, always as an escalation (see README.md §5).

## The options (not a recommendation)

**Option A — Keep the ≥7 threshold as-is, document the limitation.**
Leave `compute_partial_news2`/the agent-side NEWS2 scoring unchanged. Add a
prominent note (already partially present in code comments) that a 3/6-only
NEWS2 score reaching ≥7 requires two of the three available components to
each be at their own individual maximum simultaneously (e.g., severe
tachycardia *and* severe tachypnea), which is a real, if rare, clinical
presentation on its own merits. Risk: a false sense that this event is
"basically impossible" when it measurably isn't.

**Option B — Lower the device-specific escalation threshold.** Since 3 of 6
components structurally cannot contribute (not "contribute 0 because
normal," but "never measured"), define an explicit lower bound (e.g. ≥5 or
≥6 on the 3-component subscore) as this device's CRITICAL trigger, on the
reasoning that the full 6-component NEWS2 was calibrated assuming all 6 are
observed, and a partial score should be held to a proportionally lower bar.
Risk: no external validation exists for any such device-specific threshold;
this would be a locally invented rule, not a validated one.

**Option C — Disable the NEWS2-based escalation on VitalPatch data entirely**
and rely only on the qSOFA override (which already works and does not have
this ambiguity) plus the ECG-only risk cascade. Risk: discards a real,
if rare, signal (the one segment that did reach 7 combined severe
tachycardia and tachypnea — plausibly a real deteriorating patient, though
unconfirmed since no clinician has reviewed it).

**Option D — Route any 3/6 NEWS2 ≥ 7 case to mandatory clinician review**
rather than an automated CRITICAL label, treating partial-coverage
extreme scores as a flag for a human rather than a fully automated
escalation. This sits between A and C: keeps the signal, but does not let an
under-observed score silently reach the same automated label as a
fully-observed one.

## What this document deliberately does not do

- It does not pick one of the above. That is the clinical decision this
  document exists to route to someone qualified to make it.
- It does not fabricate a clinician's opinion on the one real segment that
  hit NEWS2 = 7. Verified 2026-07-30: this exact segment is **not** among the
  50 segments in `data/reports/clinician_review_sample.csv` (that sample was
  drawn independently, by ECG-only CRITICAL status, before this NEWS2
  correction was made). Its report exists and can be read directly:
  `data/reports/vitalpatch/184635/1780050938536_VC2B008BF_184635_ecg_seg0.md`
  (and matching `.json`). Whoever makes this decision should look at that
  report before choosing an option above.

## Related, not decided here

- `Docs/HANDOFF.md` §8 (original framing, now partially superseded by the
  correction above).
- `Docs/CLINICIAN_REVIEW_INSTRUCTIONS.md` (Task 5 — general CRITICAL segment
  review, of which this NEWS2=7 segment is one instance).
