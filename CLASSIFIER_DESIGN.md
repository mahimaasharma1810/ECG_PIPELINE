# Design: a rhythm-based classifier for cardiac abnormalities

A proposal for extending the current binary regularity indicator into a broader
abnormality detector — structured so that each capability is added only when the
evidence to support it exists.

> Rhythm regularity indicator. Not a diagnosis. Not validated for clinical use.
> Cannot distinguish atrial fibrillation from other causes of irregularity.

---

## 1. The organising principle

**Capability is tied to evidence, not ambition.** Every class the system can
output must have (a) a physical signal that carries it, (b) labelled data to
derive a boundary from, and (c) a held-out test on a different database.

A class that fails any of those three is not shipped — it is listed as
out-of-scope with the reason.

This is not conservatism for its own sake. This project has already produced one
beat classifier whose held-out performance was F1 **0.005** and **0.152** on two
of its five classes, and a regularity distribution on device data that turned out
to be an artefact of detection error. Both looked fine until tested properly.

---

## 2. What the signal can and cannot support

The device gives us **beat timing** and **one waveform channel** (possibly two —
see `PATCH_AND_DATA.md` §7). That draws a hard boundary.

### Tier 1 — supportable from beat TIMING alone (morphology-free)

| Capability | Signal basis | Status |
|---|---|---|
| Rate: brady / normal / tachy | mean RR | trivial, available now |
| **Regularity: regular / irregular / undetermined** | RR variability | **built, provisional** |
| Pause / dropped-beat events | isolated long RR | straightforward |
| Ectopy **burden** (how much, not what type) | short-then-compensatory RR | measured, see §3.2 |
| Rate-regularity **combinations** | the two above | **the highest-value gap** |
| Trend / episode duration | verdict over time | needs the above first |

### Tier 2 — requires waveform SHAPE (morphology), single lead

| Capability | Why timing alone fails |
|---|---|
| PAC vs PVC (which type of extra beat) | both produce a short interval; only shape separates them |
| **Wide- vs narrow-complex tachycardia** | width is a shape property — and this one is life-critical |
| Beat-level classification (N/S/V/F/Q) | shape by definition |

### Tier 3 — requires multiple leads and/or higher fidelity

ST-segment change, ischaemia, electrical axis, bundle branch block, chamber
enlargement. **Out of scope for this device** at 89.7 Hz with unconfirmed
amplitude units.

### Never claimable from this signal

**Atrial fibrillation as a diagnosis.** RR intervals contain no P-wave
information. Irregularity is equally consistent with AF, frequent ectopy,
respiratory sinus arrhythmia, or movement artefact. This is a property of the
physics, not a limitation to be engineered away.

---

## 3. The proposed classifier

### 3.1 Shape: a small set of independent axes, not one flat label

A single flat label ("Normal / AF / PVC / …") forces the system to choose
between things that co-occur and hides which evidence drove the answer. Instead,
**three independent axes**, each with its own confidence and its own refusal:

```
        RATE            REGULARITY          EVENTS
   ┌──────────────┐  ┌────────────────┐  ┌──────────────────┐
   │ bradycardic  │  │ regular        │  │ pause detected   │
   │ normal       │  │ irregular      │  │ ectopy burden:   │
   │ tachycardic  │  │ undetermined   │  │   none/low/high  │
   │ undetermined │  │                │  │ undetermined     │
   └──────────────┘  └────────────────┘  └──────────────────┘
                            │
                            ▼
                   COMBINATION LAYER
        maps the three axes to a clinical severity band
```

Each axis refuses independently. A window can have a trustworthy rate and an
undetermined regularity — which is exactly the current situation on device data,
and the flat-label design cannot express it.

### 3.2 The combination layer closes a real gap

We found a genuine hole in a regularity-only view: **sustained fast atrial
rhythms are fast but evenly spaced, so they read as REGULAR at a dangerous
rate.** At high atrial-ectopy burden the irregular flag rate actually *drops*
(77.6% → 60.0%).

The combination layer catches it because it sees both axes at once:

| Rate | Regularity | Interpretation | Severity |
|---|---|---|---|
| normal | regular | unremarkable | Normal |
| tachycardic | **regular** | **regular tachycardia — the SVT gap** | Abnormal |
| tachycardic | irregular | fast and irregular | Abnormal |
| bradycardic | any | slow | Abnormal / Critical |
| any | any + pause | pause event | Abnormal / Critical |
| < 40 or > 150 bpm | any | rate extreme | **Critical** |
| any | undetermined | insufficient evidence | **withheld** |

This is also why **the rhythm verdict must never be displayed without the heart
rate beside it.**

### 3.3 Two tracks, as now

**Track A — interpretable thresholds (primary, shippable).** Every verdict
traces to a named feature, a derived boundary, and its provenance. A clinician
can audit it. This is what the current pipeline does.

**Track B — supervised model (secondary).** Trained on labelled public data
degraded to device resolution. Used to *pseudo-label* device data and to
*challenge* Track A. **Never used to set Track A's thresholds** — that is
circular and produces a confident, unvalidated system.

Disagreement between them is a finding to investigate, not a number to average.

> Measured argument for keeping Track A first: `rr_cv` alone scores 0.9534 on
> derivation data and **0.9512 on a held-out database** — it barely drops.
> Adding `rmssd` improves derivation to 0.9946 but held-out only to 0.9650.
> The gap between those is the shrinkage a model must survive.

> And a trap to avoid: heart rate looks like a strong predictor of irregularity
> (0.9221) but that is a **cohort artefact** — irregular episodes in those
> databases happen to run fast (median 112 vs 75 bpm). A model given heart rate
> would learn "fast means irregular". **Rate must stay an independent axis, never
> an input to the regularity decision.**

---

## 4. Project structure

Building on the existing package rather than replacing it.

```
rhythm/
  ingest.py            # Stage 1  parse, measure true sample rate      [BUILT]
  resample.py          # Stage 2  dedupe, uniform grid, drop detection [BUILT]
  degrade.py           #          resolution matching to 89.70 Hz      [BUILT]
  features.py          # Stage 5-6 RR screen + window features         [BUILT]
  sqi.py               # Stage 3  quality gate                         [BUILT]
  verdict.py           # Stage 7  regularity axis                      [BUILT]
  smoothing.py         # Stage 8  hysteresis                           [BUILT]
  severity.py          #          combination layer                    [BUILT, provisional]
  narrative.py         # Stage 9  wording + structural validation      [BUILT]
  domain_gate.py       #          blocks clinical output until valid   [BUILT]
  streaming.py         #          real-time engine                     [BUILT]
  ws_schema.py         #          real device packet parser            [BUILT]
  reference.py         #          beat-timing validation harness       [BUILT, unused]
  replay.py            #          capture replayer                     [BUILT]

  axes/                                                              [PROPOSED]
    rate.py            # rate axis + its own refusal
    events.py          # pause detection, ectopy burden
    combine.py         # axis -> severity mapping, with the SVT rule

  morphology/                                            [PROPOSED, TIER 2, GATED]
    qrs_width.py       # narrow vs wide complex
    beat_type.py       # PAC vs PVC — only after Tier 1 is solid

  track_b/                                               [PROPOSED, GATED]
    dataset.py         # windows from labelled public data, degraded
    model.py           # supervised classifier
    pseudo_label.py    # applies to device data, clearly marked

scripts_rhythm/        # derivation + validation scripts, numbered     [BUILT]
tests_rhythm/          # replay, synthetic truth, golden, standing     [BUILT]
reports_rhythm/        # every measurement, as CSV/JSON                [BUILT]
```

---

## 5. Staged roadmap, with entry gates

**Nothing in a stage begins until its gate is met.** The gates are what stop the
same failure recurring.

### Stage 0 — fix beat detection on the device *(BLOCKING EVERYTHING)*

**Gate to exit:** on a controlled recording with a beat-level reference —
sensitivity ≥ 0.95, precision ≥ 0.95, |RR bias| ≤ 10 ms, and **RR CV error p95
≤ 0.027** (the binding one).

That last criterion matters: at today's 4.4% over-detection the device scores
sensitivity 1.0000 and **precision 0.9585 — which passes a "95% agreement"
gate** — while its CV error is **0.1741, 6.4× budget**. Beat-level agreement
alone would wave through a detector that destroys the very quantity the
threshold consumes.

**Needs:** the controlled recording. Nothing else substitutes.

### Stage 1 — rate axis + pause detection

Cheap, and mostly already available. Rate is a direct function of mean RR;
pauses are isolated long intervals.

**Gate:** rate agrees with a reference device within a stated tolerance on the
controlled recording. **Pause thresholds must be derived, not adopted** — the
current HR bands (40/60/100/150) are conventional defaults with **no
provenance**, and are marked as such.

### Stage 2 — combination layer and the SVT rule

**Gate:** demonstrate on public labelled data that regular-tachycardia windows
are caught by the combination layer that a regularity-only view misses.

### Stage 3 — ectopy burden (amount, not type)

Already partly measured: roughly 4+ ventricular or 5+ atrial premature beats per
64 tip a window to irregular; one or two are invisible.

**Gate:** held-out-database sensitivity/specificity for burden bands, reported
with the honest limitation that **isolated ectopy is not detectable** at this
resolution.

### Stage 4 — Track B supervised model

**Gate:** Stage 0 complete. Training on device data before the detector is fixed
would launder detection error into a confident model.

### Stage 5 — morphology (Tier 2)

**Gate:** Stages 0–3 stable, *and* the second lead question resolved
(`PATCH_AND_DATA.md` §7). Wide-complex tachycardia is the highest-value item
here because it is life-critical, and it is a shape property that timing cannot
reach.

---

## 6. Data needed per stage

| Stage | Data | Have it? |
|---|---|---|
| 0 | Controlled recording, patch + per-beat reference | **NO — the blocker** |
| 1 | Same recording; reference rate | NO |
| 2 | Public data with rhythm labels incl. SVT episodes | Partly — LTAFDB, MITDB |
| 3 | Public data with beat-level ectopy labels | **Yes** — SVDB, MITDB |
| 4 | Labelled Lead-II data, degraded to 89.7 Hz | **Yes** |
| 5 | Labelled morphology data + device lead answer | Partly |
| any | Device data with a *known abnormal* rhythm | **NO — all 37 are healthy** |

That last row is a standing gap. Every device recording we hold is one class.
A boundary cannot be derived from one class, which is why public data carries
the other side and why this remains the methodological core of the project.

---

## 7. Evaluation protocol — non-negotiable

- **Held-out DATABASE, never just held-out record.** Cross-database transfer has
  already failed twice here.
- **Never lead with accuracy.** Classes are heavily imbalanced; predicting the
  majority scores ~0.89 on some of these databases. Report sensitivity,
  specificity, PPV, NPV and confusion matrices.
- **Report the operating point and what it costs.** Buying specificity 0.95 on
  the current threshold costs 42–47% of sensitivity.
- **Robustness by leave-one-record-out and cluster bootstrap.** With 6 records
  the threshold moved 0.055 when one patient was dropped; with 16 it moved
  0.0125. Window counts overstate evidence — patients are the unit.
- **A refusal band sized to measured uncertainty**, not to taste.
- **A standing physiological check.** Healthy people should read regular. That
  test exists, fails today, and is the single test that will say when the
  detector is fixed.

---

## 8. What NOT to build

- **Anything that outputs an AF diagnosis.** Physically unsupported.
- **A flat multi-class label.** It hides which evidence drove the answer and
  cannot express partial confidence.
- **A model that takes heart rate as an input to the regularity decision.** It
  will learn a cohort artefact.
- **Any threshold adopted from literature without re-derivation at this
  device's timing resolution.** Two have already been rejected for this.
- **Anything that removes the refusal path to raise yield.** Loosening the
  current quality gate on healthy recordings produces **34 of 39 windows
  labelled IRREGULAR and none REGULAR.**

---

## 9. MEASURED: how the axes behave on real patch data

The axes were derived and validated entirely on public labelled data. Run on
the 37 device captures (1,210 windows, all healthy adults), **one of three
transfers.**

| Axis | Public data | Patch data |
|---|---|---|
| **Rate** | validated | **SURVIVES** |
| Regularity | Se 0.9839, NPV 0.9978 held-out | **BROKEN** |
| Events | Se 0.877-0.899, PPV 0.951-0.978 held-out | **BROKEN - and more so** |

### Rate survives

Median 85.1 bpm, p05 70.9, p95 99.8, max 123.8, **zero windows flagged
extreme**. Physiologically plausible, and consistent with the device's own
reported rate (accurate to ~4%).

### Events fails, and the reason is a design lesson

| Measure | Device (healthy adults) | Credible? |
|---|---|---|
| Windows with >=1 ectopic couplet | **95.8%** | no |
| Median couplets per window | 3.0 | no |
| Windows classed HIGH burden | 29.8% | no |
| Windows containing a pause >2000 ms | 7.5% (138) | no |

**Why it is MORE vulnerable than the regularity axis, not less.**

A spurious R-peak splits one interval into a SHORT one followed by a LONGER
one. That is *precisely* the short-then-compensatory signature the ectopy
detector searches for. The detector is not merely degraded by detection error -
**it is tuned to the exact shape that detection error produces.** A missed beat
does the mirror image: two intervals merge into one long interval, which reads
as a pause.

So an axis validated to Se 0.88-0.90 / PPV 0.95-0.98 against expert beat labels
on two held-out databases is, on this device, close to useless - and would have
been trusted on the strength of those numbers.

### What it would have cost

With the gate open, on healthy adults:

| Severity | Windows |
|---|---|
| WITHHELD | 694 (57.4%) |
| ABNORMAL | 425 (35.1%) |
| **CRITICAL** | **91 (7.5%)** |

**91 false CRITICAL escalations from healthy people**, every one driven by a
manufactured pause. The device-domain gate is currently the only thing
preventing that - a concrete demonstration of its value rather than a
hypothetical one.

### THE RULE THIS ESTABLISHES

**Every axis needs its OWN device-domain validation criterion. Held-out-database
performance does not transfer, and beat-COUNT accuracy is not a sufficient
proxy for any of them.**

A detector could get the beat count exactly right while still placing beats into
short/long pairs. That would satisfy a count-based check and still produce false
ectopy. So when the controlled recording arrives, each axis must be validated
against the beat-level reference **independently**:

| Axis | Its own criterion, against the reference |
|---|---|
| Rate | mean rate agreement within a stated tolerance |
| Regularity | RR CV error p95 <= 0.027 (the refusal-band budget) |
| **Events** | **ectopy-couplet count and pause count agreement - NOT beat count** |

Two standing tests pin the current failures so they cannot be quietly forgotten:
`test_healthy_population_reads_regular` (regularity) and
`test_healthy_population_has_low_ectopy_burden` / `..._has_no_pauses` (events).
All are `xfail(strict=True)`, so the suite goes red the moment any of them
starts passing and someone must confirm the fix was genuine.

---

## 9b. SETTLED BY MEASUREMENT: what this instrument is for

Held-out validation across MITDB + SVDB, 4,457 windows, 124 records.

### It does NOT make "irregular" actionable

On the target Track A was measured against (AFIB label, prevalence 10.6%):

| Output | Se | Sp | PPV | NPV |
|---|---|---|---|---|
| L3 IRREGULAR alone | 0.9194 | 0.8895 | **0.4957** | 0.9894 |
| Layered "not unremarkable" | 1.0000 | **0.5771** | **0.2183** | 1.0000 |

PPV FALLS under layering, because the layers flag rate and ectopy findings that
are not AFIB. That is the expected consequence of adding DIFFERENT findings
rather than better precision on the same one.

Note the Se 1.0000 is at Sp 0.5771 - the layered output flags ~42% of normal
windows, and perfect sensitivity follows substantially from that breadth.

### It DOES reduce false reassurance, by three quarters

Broad target (non-sinus rhythm OR >=5 ectopic beats OR expert-beat HR outside
60-100). **Prevalence 60.2%, so PPV on this target is NOT comparable to the
table above and is not offered as such.**

| | Windows passed as normal | Genuinely significant among them |
|---|---|---|
| L3 alone | 2,159 | **739 (34.2%)** |
| Layered | 1,607 | **187 (11.6%)** |

**552 fewer windows falsely reassured - 75% of L3's misses.** Driven by the
three findings a rhythm-only view structurally cannot see:

| Finding | Windows |
|---|---|
| REGULAR at HR > 100 (up to 161 bpm) | 350 |
| REGULAR at HR < 60 (median 56 bpm) | 339 |
| REGULAR with >= 5 ectopic beats | 415 |

### THE SETTLED FRAMING

**This is a false-reassurance-reduction instrument, not an alerting one.**

  * Operate for high NPV.
  * Surface "appears regular" as the actionable output.
  * NEVER route "irregular" to an alarm - PPV 0.4957 alone, 0.2183 layered.
  * The layered output's job is to WITHHOLD reassurance, not to raise alerts.

Demonstrated by measurement, not asserted.

## 10. Honest position

The most valuable thing this design can do is not add classes — it is to make
the capability boundary explicit, so that what ships is what the evidence
supports.

Today the system has **three** axes. On the target device **one of them works**
(rate); the other two are broken by the same root cause - beat detection wrong
by ~4.4% - and the events axis is broken worse than the regularity axis because
detection error mimics its target pattern exactly. Every stage above is
downstream of fixing that.

The three-axis structure is nevertheless worth having now, for a reason §9
demonstrates: it let us discover that the events axis fails *differently* from
the regularity axis, and would have failed *silently* behind a single flat
label.

The realistic near-term scope is Tier 1: rate, regularity, pauses, ectopy
burden, and the combination layer that catches regular tachycardia. That is a
genuinely useful ambulatory screening tool. It is not an arrhythmia diagnostic,
and this device cannot become one.


---

## 11. Standing analysis practices

Rules adopted after specific failures in this project. Each exists because it
was violated and the violation produced a wrong conclusion that survived review
until it was tested.

### 11.1 No interpretive labels written before the numbers exist

Do not write conclusions into a script's print statements ahead of the result.

**Three occurrences in a single session**, all in the ectopy analysis: text
asserting "FASTER / LESS variable / TIGHTER" printed alongside numbers showing
slower, more variable, wider. Twice the assertion was reported before the
contradiction was noticed.

The frequency indicates a systematic hazard of the workflow, not a lapse:
writing the analysis and its interpretation in one pass invites stating the
expected mechanism as though it were the measured one.

**Practice:** scripts print numbers. Interpretation is written afterwards, from
the output.

### 11.2 Never summarise a possibly-bimodal population with a median

The high-ectopy-burden analysis produced two contradictory summaries before the
population was found to be bimodal: 64 PACs per window occurs at both RR CV
0.045 (fast, tight, reads REGULAR) and RR CV 0.346 (slow, wide, reads
IRREGULAR). A median over both is a number describing neither.

**Practice:** split by outcome or check for bimodality before reporting a
central tendency. Report per band, never pooled.

### 11.3 Check negative-set composition before believing a strong result

A classifier's headline number is only as meaningful as its hardest negatives.

- **L4 ectopy burden** validated at Se 0.88-0.90 / PPV 0.95-0.98. Its negatives
  were almost entirely REGULAR windows. Tested against irregular-but-not-ectopic
  windows it fires on **91.3% of AFIB windows containing zero ectopic beats** -
  so it demonstrated specificity against regular rhythm, not against irregularity
  from another cause. The design element that depended on that distinction was
  withdrawn.
- **L5 PVC morphology**: an initial build used N-only negatives, making the task
  "wide complex vs normal". That is easy and leaves the clinically important
  confusion - PVC vs aberrantly-conducted supraventricular beat - untested. The
  evaluation now breaks performance out against N, S and F separately and
  reports PPV against hard negatives (S+F) alone.

**Practice:** state the negative-set composition alongside every performance
number, and report the metric against hard negatives separately.

### 11.3b The same applies to the TRAINING negative set

A model can only learn to reject hard negatives it was actually shown.

**Measured, L5 PVC morphology.** False-positive rate against supraventricular
beats, held-out database:

| Direction | native | 89.70 Hz | S beats in TRAINING set |
|---|---|---|---|
| MITDB -> SVDB | 0.0424 | **0.1705** | 229 |
| SVDB -> MITDB | 0.1048 | **0.0786** | 2,452 |

The asymmetry is not noise. MITDB contains only 229 supraventricular beats, so
a model trained there barely sees that morphology, never learns to reject it,
and the weakness is exposed on the supraventricular-rich database. Training on
SVDB reverses it.

**Practice:** check BOTH the training and the test negative-class composition
before trusting any number. If L5 proceeds, train on S-rich data; the
SVDB -> MITDB direction is the more honest configuration, and MITDB -> SVDB
shows what happens when the training data lacks hard negatives.

### 11.3c F1 hides what the FP-by-class table shows

Twice now the headline metric concealed a degradation the per-class table
caught. Downsampling L5 to 89.70 Hz moved F1 by -0.009 while quadrupling the
false-positive rate against supraventricular beats (0.0424 -> 0.1705), because
normals dominate the negative set and mask the change.

**Practice:** the FP-rate-by-class table is the primary metric for any
classifier with an imbalanced negative set. F1 and PPV are summaries reported
alongside it, never instead of it.

### 11.3d Metrics that are uninterpretable alone must travel in pairs

Two near-misses in one session, both caught before reporting:

**PPV travels with PREVALENCE.** The layered system scored PPV 0.8761 on a
broad target and Track A scored 0.4697 on an AFIB target. Quoting those side by
side reads as a near-doubling of precision and is meaningless: the prevalences
are 60.2% and 10.7%. Measured on the MATCHED target, layering *lowers* PPV
(0.4957 -> 0.2183), which is the true result.

**SENSITIVITY travels with SPECIFICITY.** The layered output scores Se 1.0000
and NPV 1.0000 - at Sp 0.5771, meaning it flags ~42% of normal windows. Perfect
sensitivity is substantially a consequence of that breadth; anything achieves
Se 1.0 by flagging everything.

**Practice:** never report PPV or NPV without the prevalence they were measured
at, and never report sensitivity without specificity beside it. A comparison
across different targets or prevalences is not a comparison.

### 11.4 Test the claim, do not state it

The L4 finding above cost a designed feature. It was found by testing a claim
rather than asserting it, and is the most useful output of the session that
produced it. Negative results that remove an unsupported capability are worth
more than tuned numbers that keep one.
