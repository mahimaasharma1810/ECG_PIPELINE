# ECG Pipeline — `mod_1` branch

This branch takes the ECG 5-class beat classifier (see [`project_doc.md`](project_doc.md)
for the classifier's own training/eval history) and wires it into a full,
end-to-end **clinical report pipeline**: raw device file → parse → filter →
detect beats → classify → detect rhythm abnormalities → risk cascade →
live MedGemma clinical narrative → saved report. It also fixes three
correctness bugs found while building that pipeline out. Nothing here
retrains, reweights, or reconfigures the classifier, the risk cascade
thresholds, or the AAMI 5-class scheme — every fix below is either in the
signal path *before* the classifier ever sees a beat, or in the narrative
*after* the risk decision has already been made.

> The full plain-English version of everything below — no ML or ECG
> background assumed — is also kept as its own file,
> [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md), if you'd rather read it
> there. It's reproduced in full immediately below so it's visible on
> this page too.

<br>

<details open>
<summary><strong>Plain-English project overview (click to collapse)</strong></summary>

# What this project is (for a reader who knows nothing about ECGs, ML, or this codebase)

**An ECG is just an electrical recording of your heartbeat.** A wearable
chest patch (this project uses two devices — "VitalPatch" and "SeNSiO")
records that electrical signal continuously, the same way a hospital
heart monitor does, but small enough to wear at home. This project reads
that raw recording, automatically finds every single heartbeat in it,
labels each heartbeat as normal or one of a few abnormal types, looks for
dangerous heart-rhythm patterns across many beats, and — for anything
that looks concerning — writes a short clinical report a nurse or doctor
can quickly read.

**The real-world purpose:** a patient recovering at home after surgery
wears the patch for days. Nobody has time to watch days of raw waveform
by eye. This system watches it continuously instead, and only surfaces
what actually needs a human's attention.

Everything below is written so each new term is explained the first time
it shows up, and every number quoted is one this document's author
actually pulled from a file on this machine (source noted under each
table) — not remembered or estimated.

---

## 1. What an ECG "beat" is, and the 5 categories

Your heart's electrical signal isn't smooth — it's a repeating spike-shape
pattern, one spike per heartbeat, called a **beat** or **QRS complex**
(the letters just label three tiny sub-parts of one spike; you don't need
to remember them). Cardiologists sort every individual beat into one of a
small number of standard categories. This project uses the same 5
categories the medical field itself uses (the "AAMI" standard — a
naming convention, not a company or an algorithm):

| Code | Name | Plain-English meaning |
|---|---|---|
| **N** | Normal | A regular heartbeat, electrically starting where it should. |
| **S** | Supraventricular | An early/extra beat that starts in the heart's *upper* chambers (the atria) instead of its normal starting point. Usually harmless on its own, but frequent ones can be a warning sign (e.g. of atrial fibrillation). |
| **V** | Ventricular | An abnormal-shaped beat starting in the heart's *lower* chambers (the ventricles). This is the class doctors care most about — frequent or clustered V beats are the more dangerous pattern. |
| **F** | Fusion | A beat that's a blend of a normal beat and a ventricular beat overlapping. |
| **Q** | Unknown | A beat the system can't confidently place in any of the above — flagged rather than guessed. |

---

## 2. Why this is a hard problem

In any real ECG recording, roughly **9 out of 10 beats are perfectly
Normal**. The beats that actually matter clinically (S, V, F) are rare by
comparison. This is called **class imbalance**, and it causes two
practical problems:

1. A lazy model that just says "Normal" for every single beat, without
   looking at the data at all, would still score around **88.7%
   accuracy** (computed directly from this project's real test data: N
   beats are 40,634 of 45,804 total beats — see Section 5's source table,
   `Docs/archive/ABLATION_REPORT.md`). So "accuracy" as a headline number
   is close to meaningless here — see Section 5 for what to look at
   instead.
2. A model trained the ordinary way naturally gets very good at Normal
   (because it sees millions of examples of it) and comparatively weak at
   the rare classes (because it barely sees any) — exactly the classes a
   clinician actually wants flagged. Most of the work described in this
   document is about honestly measuring and trying to fix that weakness,
   without accidentally making the dangerous classes worse in the
   process.

---

## 3. How the system works, step by step

```
  RAW FILE (VitalPatch CSV, SeNSiO CSV, or a hospital WFDB recording)
        │
        ▼
  STAGE 2 — Signal-quality gate
    Looks at each short window of signal and asks "is this even
    trustworthy?" (flatline? clipped/railing? missing samples? too
    noisy?). Bad windows are marked and excluded rather than guessed at.
        │
        ▼
  STAGE 3 — Resample to a standard rate (125 samples/second)
    Different devices record at different native speeds; everything
    downstream expects one consistent speed.
        │
        ▼
  STAGE 4 — Clean the signal
    Removes baseline drift (the signal slowly floating up/down), mains
    electrical hum, and muscle-noise, without distorting the heartbeat
    shape itself.
        │
        ▼
  STAGE 5 — Find each heartbeat (R-peak detection) and cut it out
    Locates the tallest spike of each beat and extracts a small window
    of signal around it — one little snippet per heartbeat.
        │
        ▼
  STAGE 6 — Describe the shape of each beat as numbers
    Converts each beat snippet into ~56 numbers describing its shape
    (how tall, how wide, how it decomposes at different zoom levels).
        │
        ▼
  STAGE 7 — Classify each beat  (the ML model, XGBoost — see Section 4)
    Turns those 56 numbers into one of the 5 labels: N / S / V / F / Q.
        │
        ▼
  STAGE 8 — Look across many beats for a dangerous RHYTHM
    Not every risk is about one single beat — some are about a *pattern*
    across many beats (e.g. many early upper-chamber beats in a row can
    mean atrial fibrillation). A fixed, human-set set of rules (not a
    model) checks these patterns and picks a final risk level: LOW,
    MEDIUM, HIGH, or CRITICAL. See Section 7 for exactly how.
        │
        ▼
  STAGE 9 — Put it into words (MedGemma, a small local AI language model)
    Turns the numbers and rule outcomes into a short, readable clinical
    paragraph — but never makes the risk decision itself. See Section 7.
        │
        ▼
  SAVED REPORT (one structured file + one readable report per recording)
```

---

## 4. THE RESULTS — how well the beat classifier actually works

**The model:** XGBoost (a well-established, non-deep-learning machine
learning algorithm that builds many small decision rules and combines
them) trained on ~11,000+ real, cardiologist-labeled recordings from the
MIT-BIH Arrhythmia Database (a standard, public hospital ECG dataset used
across this whole research field) plus a second public dataset (SVDB)
chosen specifically because it's rich in S-type beats.

**How it was tested, honestly:** the data was split by *patient* — no
single patient's heartbeats appear in both the training data and the test
data. The final numbers below come from **DS2**, a set of 22 patient
recordings (45,804 individual beats) that were **never used to train or
tune anything** — this is the only number in this whole project meant to
answer "does this actually work on a new patient."

### The real DS2 numbers

*Source: `Docs/archive/ABLATION_REPORT.md`, "Production baseline" section
(exact command shown there: `eval-classifier --model
models/five_class_xgb.json --split-set ds2`). A later retraining pass on
a slightly enlarged validation split (`project_doc.md`, 2026-07-20)
reproduced this within noise (macro-F1 0.3935 vs. 0.3895 here) — both are
real, on-disk numbers for the same production model; the table below is
the one with a full confusion matrix and per-class beat counts on disk.*

| Class | Sensitivity | Precision | F1 | Beats in test set |
|---|---|---|---|---|
| N (Normal) | 0.972 | 0.973 | 0.972 | 40,634 |
| S (Supraventricular) | 0.130 | 0.150 | 0.139 | 1,795 |
| V (Ventricular) | 0.912 | 0.754 | 0.826 | 3,005 |
| F (Fusion) | 0.005 | 0.125 | 0.011 | 363 |
| Q (Unknown) | 0.000 | 0.000 | 0.000 | 7 |

**Macro-F1** (the average of the 5 F1 scores, treating every class
equally regardless of how common it is): **0.3895**. Overall accuracy:
0.9269 — quoted here specifically to demonstrate why it's not the metric
to trust (see Section 2: guessing "Normal" every time gets you most of
the way to this number for free).

**What these three column names actually mean, in one sentence each:**
- **Sensitivity** (also called recall): of all the *real* beats of this
  type, what fraction did the model actually catch? Low sensitivity means
  it misses most of them.
- **Precision**: of all the beats the model *called* this type, what
  fraction were actually right? Low precision means a lot of false
  alarms.
- **F1**: a single number combining sensitivity and precision (their
  harmonic mean) — a simple way to compare classes without having to
  look at two numbers every time. Low F1 means the class isn't working
  well, whichever of the two problems is the cause.

**How to read the confusion matrix** (rows = the beat's true label, columns
= what the model predicted; the numbers are beat counts):

```
              predicted:
              N        S       V      F     Q
true N   [  39478,    960,    187,    9,    0 ]
true S   [    924,    234,    636,    1,    0 ]
true V   [    120,    139,   2742,    4,    0 ]
true F   [     63,    228,     70,    2,    0 ]
true Q   [      4,      1,      2,    0,    0 ]
```

Read any row as "of the beats that were *really* this type, here's what
the model called them." E.g. the true-S row shows 924 real S beats were
called N (missed as ordinary/normal), 636 were called V (confused for
the dangerous class), and only 234 were correctly called S.

**Which classes work, and the honest reason for the ones that don't:**
- **N and V work well** (F1 0.972 and 0.826). V being reliable matters
  most, since it's the clinically dangerous class.
- **S is weak** (F1 0.139). The dominant error (per the confusion matrix
  above, and confirmed again in the pinned-baseline re-run in
  `Docs/archive/ABLATION_REPORT.md`) is **S beats being missed as Normal
  entirely (67.6% of true S beats)** — not being confused with V. The
  honest, tested explanation (Section 6): a single ECG lead often can't
  see the P-wave (the small electrical signature that marks "this beat
  started in the upper chambers"), so the physiological signal that would
  distinguish "early-and-atrial" from "just a normal beat that happens to
  come a bit early" often simply isn't visible on this hardware.
- **F is essentially unsolved** (F1 0.011). MIT-BIH itself only has 363 F
  beats in the entire 22-patient test set, and 403 in all of training —
  there is very little real data of this type to learn from, on any
  device, in this whole research field.

---

## 5. What was tried to fix the weak S class, and why each attempt was rejected

**The non-negotiable safety rule used throughout:** N and V are
"zero-tolerance" classes — any candidate change that measurably hurts V
(the dangerous class) is rejected outright, even if it improves S,
regardless of how good the average score looks.

*All numbers below: `Docs/archive/ABLATION_REPORT.md` and
`Docs/BEAT_CLASSIFICATION_SUMMARY.md`.*

| Attempt | What it did | S F1 | V F1 | Verdict |
|---|---|---|---|---|
| + 7 "timing" features (how early/late a beat is vs. its neighbors) | Added rhythm-context numbers to the same model | 0.182 (better) | 0.775 (**worse**) | **Rejected** — V regressed beyond the safety margin |
| + More training data (LTAFDB/SDDB, 8.5M extra beats) | Threw a lot more real recordings at training | 0.098 (**worse**) | 0.808 (worse) | **Rejected** — the extra data was overwhelmingly Normal-labeled and diluted the rare classes rather than reinforcing them |
| Two-stage classifier (v1–v5: one model flags "abnormal," a second model decides which kind) — 5 versions tried | A completely different architecture, specifically designed around the finding that morphology alone separates S from V well | 0.383 (best version) | 0.789 (**worse**) | **Rejected**, and explicitly closed as a settled result — re-tested on a 2.5x larger validation set (4→10 patients) specifically to rule out "too small a test" as the excuse, and the same trade-off reproduced almost exactly |
| CNN+Transformer (a deep-learning model that looks at the raw waveform shape itself, instead of hand-built numbers) | A fundamentally different, much larger model, tried as the "maybe deep learning just does better" option | 0.121 (**worse than production**) | 0.730 (**much worse**) | **Rejected** — didn't even manage to improve S, the one thing it was built to fix |

**The consolidated, honest conclusion** (stated plainly, not softened):
every different approach tried — extra features, extra data, a
completely different two-stage architecture, and a completely different
deep-learning model — hits the same wall. Getting meaningfully better at
S costs V more than the project is willing to pay. This has been checked
**three genuinely different ways** (feature engineering, architecture,
and model family) and one methodological doubt (was the validation set
too small?) was specifically re-tested and ruled out. The current honest
read is that this is a **real limitation of a single ECG lead and/or the
amount of labeled data available**, not a bug or a lack of trying.
**The production model was never changed as a result of any of these
experiments** — `models/five_class_xgb.json` is exactly the model scored
in Section 4's table.

---

## 6. The clinical report system — turning beat labels into a report

Once every beat in a recording has a label, two more things happen
before a human sees anything:

1. **The risk cascade** (`ecg_pipeline_core.py`) — a fixed set of
   human-written rules, checked in a specific priority order, **not** a
   machine-learning model. The first rule that fires (its condition is
   true) decides the final risk level; if nothing fires, the recording is
   LOW risk. In priority order (thresholds are fixed constants in the
   code, `RiskThresholds`, unchanged by any of this session's work):
   1. PVC burden (% of beats that are ventricular ectopics) > **20%** → CRITICAL
   2. A run of ≥3 consecutive V beats in a row (a "VT run") → CRITICAL
   3. PVC burden > **10%** → HIGH
   4. PAC burden (% of early upper-chamber beats) > **15%**, OR AFib burden > **30%** → HIGH
   5. Sustained low heart-rate variability (SDNN < 20ms) → MEDIUM
   6. None of the above → LOW
2. **MedGemma** (a small, locally-run AI language model, Stage 9) — takes
   the already-decided numbers and risk level and writes them up in plain
   English. **It is a translator, not a decision-maker** — the risk level
   above is fixed by the deterministic rules before MedGemma ever runs;
   MedGemma cannot change it, only describe it.

### One real, saved example, in full

*Source: `data/reports/prorhythm/FD_DC_A1_9C_48_40/ECG_2026-07-07T12-21-26.317821.md`, produced by this pipeline exactly as saved, not edited for this document.*

> **Recording:** SeNSiO device, 92.64 seconds, 81 beats detected / 66 analyzed.
>
> **Beat summary:** N: 55 (67.9%) · S: 7 (8.64%, flagged low-confidence per Section 4's S F1) · V: 4 (4.94%) · F: 0 (0.0%, flagged low-confidence) · Q: 15 (18.52%).
>
> **Rhythm finding:** AFIB_SUSPECTED — beats 0–19, RR-interval coefficient-of-variation 0.398 (over a rule-of-thumb threshold of 0.15).
>
> **Deciding rule:** "PAC burden > HIGH threshold OR AFib burden > HIGH threshold" — PAC burden 8.642% vs. its 15.0% threshold → does **NOT** exceed; AFib burden 100.0% vs. its 30.0% threshold → **EXCEEDS**. Overall: fired = True.
>
> **Final risk level: HIGH.**
>
> **MedGemma's written interpretation:** *"The primary driver for the HIGH risk assessment is the presence of a sustained high rate of atrial fibrillation, as indicated by 100% AFib burden. While there are other factors considered in the decision rule (PAC burden), the significant AFib activity overrides them and contributes to this elevated overall risk level. The classification of S beats with low confidence may be an artifact of the beat classifier and should not be interpreted as a definitive finding."*
>
> *Clinician review suggested — this is decision support, not a diagnosis.*

Notice MedGemma's text only restates numbers that were already computed —
it doesn't invent the 100% figure or decide the risk level; that work was
already done before it ever ran. This is enforced deliberately — see
Section 9.

---

## 7. Real batch results — running this on every recording

*Sources: `data/reports/prorhythm_run_manifest.csv` (complete, exact counts
below computed directly from that file) and a direct count of the JSON
report files under `data/reports/vitalpatch/` as of this writing (that
batch was still running at the time this document was produced — see
caveat below).*

**SeNSiO/prorhythm batch — complete, 18 of 18 recordings:**

| | Count |
|---|---|
| Total recordings | 18 |
| Assessable | 13 |
| NOT_ASSESSABLE | 5 |
| Risk level: LOW | 8 |
| Risk level: HIGH | 3 |
| Risk level: CRITICAL | 2 |
| MedGemma status: ACCEPTED (live narrative) | 11 |
| MedGemma status: SKIPPED_CRITICAL | 2 |
| MedGemma status: SKIPPED_NOT_ASSESSABLE | 5 |

**VitalPatch batch — in progress at the time of writing.** 2,375 raw
files exist under `data/raw/vitalpatch/`; as a live snapshot, 902 segment
reports had been saved so far:

| | Count (partial, growing) |
|---|---|
| Segments processed so far | 902 |
| Assessable | 762 |
| NOT_ASSESSABLE | 140 |
| Risk level: LOW | 521 |
| Risk level: MEDIUM | 58 |
| Risk level: HIGH | 71 |
| Risk level: CRITICAL | 112 |
| MedGemma status: ACCEPTED | 637 |
| MedGemma status: SKIPPED_CRITICAL | 112 |
| MedGemma status: SKIPPED_NOT_ASSESSABLE | 140 |
| MedGemma status: UNAVAILABLE_FALLBACK | 13 |

**What NOT_ASSESSABLE means, and why it's a safety feature, not a
failure:** if too few beats in a recording survive the quality checks to
say anything reliable, the system explicitly refuses to guess — it marks
the recording NOT_ASSESSABLE and skips MedGemma entirely rather than
writing a confident-sounding report from too little real signal. A
system that always produces a risk level, even from garbage input, is
more dangerous than one that sometimes says "I can't tell."

**Honest caveat on the numbers above:** the VitalPatch table is a partial,
in-progress snapshot (the batch is long-running because each MedGemma
call is real and serialized), not the final tally — re-run
`python -m ecg_pipeline.manifest_summary` once it completes for the final
numbers. Separately, 13 of the 902 segments so far show
`UNAVAILABLE_FALLBACK` rather than a live MedGemma narrative — this
happened during a period when the local MedGemma server was transiently
unreachable mid-batch; the structured risk report was still saved
correctly for those segments, just without an LLM-authored paragraph.

---

## 8. Bugs found and fixed this round (why the results above are trustworthy)

Three real bugs were found and fixed in the *signal path before
classification* and the *narrative after the risk decision* — nothing
about the classifier, its thresholds, or the 5-class scheme itself was
touched by any of these.

### Bug 1 — heartbeats were being thrown away right after they were correctly found

The step that locates each heartbeat's peak was landing a few samples off
the true peak on hospital (WFDB) recordings. A later safety check (which
insists the marked peak really is the tallest point nearby) then rejected
almost every real beat as suspicious, even though they were real beats.

*Source: git commit `2420ecc`; before/after retention numbers independently
recomputed here from `check2_after.json`/`check3_after.json` on disk
(scratchpad diagnostic files from the same verification pass):*

| Test recording | Beats kept, before fix | Beats kept, after fix |
|---|---|---|
| 100 | 21.9% of true beats | 89.7% |
| 103 | 69.9% | 131.2%* |
| 105 | 7.7% | 84.6% |
| 111 | 59.2% | 116.9%* |
| 200 | 28.4% | 84.1% |
| 210 | 20.3% | 85.1% |
| 233 | 38.0% | 64.7% |

*\*Over 100% means the detector found more beat candidates than the
official ground-truth count for that recording — a separate, already
known and documented over-detection issue on these two specific
recordings (see Section 9's limitations), not a new problem introduced by
this fix.*

### Bug 2 — hospital recordings were almost entirely rejected as "too noisy"

The signal-quality check scores a recording's own natural baseline
wobble as if it were noise, which is wrong for raw hospital recordings
that haven't been cleaned up yet (that cleanup happens in a later stage,
by design). This drove some clean recordings to near-100% rejection.

*Source: git commit `3add727`; before/after numbers from `check2_after.json`:*

| Test recording | Window-rejection rate, before | After |
|---|---|---|
| 100 | 100.0% | 8.84% |
| 103 | 95.3% | 6.63% |
| 105 | 94.2% | 16.02% |
| 111 | 96.13% | 22.65% |
| 200 | 57.73% | 15.19% |
| 210 | 88.4% | 11.6% |
| 233 | 47.79% | 7.46% |

### Bug 3 — one specific device's recordings returned zero beats entirely

The SeNSiO patch records at 100 samples/second, which puts its
theoretical frequency ceiling exactly on top of the mains-hum filter's
own frequency — a mathematically real edge case (you cannot filter a
frequency at or above that ceiling). The signal-quality check's "how
noisy is this" score ended up dominated by a narrow band the mains filter
literally cannot touch at this device's recording speed, rejecting 100%
of windows on some SeNSiO recordings even when the actual heartbeat shape
underneath was fine.

*Before: as reported when this bug was found — 0 beats detected, 100%
window rejection, NOT_ASSESSABLE (this specific "before" figure was not
independently re-saved to a file for this document; it is the bug report
that prompted the fix, not a re-verified number).*

*After (confirmed live, on disk right now):
`data/reports/prorhythm/FD_DC_A1_9C_48_40/ECG_2026-07-07T12-21-26.317821.json`
— 81 beats detected, 66 analyzed, 68.4% window rejection (still imperfect,
but no longer total), assessable, HIGH risk, live MedGemma narrative
accepted. This is the same recording shown in full in Section 6.*

### Bug 4 — MedGemma's own written narrative sometimes misstated a real number

Separately from the signal-path bugs above: MedGemma (the small local
language model that writes the final paragraph) sometimes wrote things
like *"PAC burden is at 8.642%, which exceeds the HIGH threshold of
15.0%"* — both numbers were real and correctly retrieved, the model just
compared them backwards (8.642 is actually below 15.0). Fixed two ways:
(1) every rule's true/false verdict is now computed in ordinary Python
code and handed to the model as a fact it's told not to re-derive, and
(2) a second check re-reads the model's own finished paragraph afterward
and deletes just that paragraph (keeping the rest of the report, which is
built by code, not by the model) if it still finds a false "exceeds"
claim. This was caught and verified during this round of work, not
previously documented in a committed file — see the reasoning and code in
`ecg_pipeline/agent_bridge.py`.

---

## 9. Honest limitations (stated plainly)

- **S-class detection is weak** (F1 0.139) and, per Section 5, this looks
  like a real limit of single-lead ECG hardware and/or available labeled
  data, not something more feature/model tweaking is likely to fix.
- **F-class is essentially unsolved** (F1 0.011) — there simply isn't much
  real F-labeled data in the public research datasets this field uses.
- **Two specific hospital recordings (103, 111) show real beat
  over-detection** (finding more candidate beats than the official
  ground truth) — investigated and NOT fixed (see git commit `2420ecc`'s
  message for the full investigation); evidence points to the
  beat-detector's automatic rate-tracking getting confused specifically
  on these two noisier recordings, not a units/scaling bug.
- **MedGemma (the local language-model narrator) still occasionally
  writes a wrong comparison** in its own free-text paragraph, observed in
  roughly 3 of 10 generations even after the Bug-4 fix above. This is
  caught and removed by the post-hoc check described in Bug 4 before
  anything is saved, so the *saved* report is never wrong — but the
  underlying model behavior itself was not, and likely cannot be, fully
  fixed by prompting alone.
- **13 of 902 VitalPatch segments processed so far** fell back to a
  non-LLM report because the local MedGemma server was briefly
  unreachable mid-batch (Section 7) — the structured numbers are still
  correct for these, just without the written paragraph.
- **This is decision support, not a diagnosis** — every saved report ends
  with an explicit "clinician review suggested" line, by design, in every
  single report this system produces.

---

## 10. What would make this more accurate — and different from existing off-the-shelf models

Concrete, specific next steps, distinguishing "try harder on what we
have" (already shown in Section 5 not to work) from genuinely new
levers:

1. **A second ECG lead.** Every experiment in Section 5 used one lead.
   The specific physiological signal that would reliably separate S from
   "just an early Normal beat" — the P-wave, the small electrical blip
   marking an upper-chamber origin — is frequently invisible on a single
   lead. A second lead (or a lead placement chosen specifically for
   P-wave visibility) is the most direct fix for the project's single
   biggest known weakness, and is a genuinely different lever, not a
   repeat of what's been tried.
2. **More labeled F-class and S-class data**, ideally from wearable
   devices themselves (VitalPatch/SeNSiO) rather than only decades-old
   hospital recordings — the deep-learning attempt in Section 5 hit a
   "not enough data for a model this size" wall specifically, which more
   hospital data (already tried, made things worse — Section 5) doesn't
   solve, but real device data plausibly could.
3. **Sequence/context-aware models that look at several neighboring
   beats together**, rather than one isolated beat at a time. S and V are
   partly defined by their relationship to the surrounding rhythm (how
   early, how the pause afterward behaves), not shape alone — this
   project's own timing-feature experiments (Section 5) showed that
   signal is real but got tangled with V when added as flat extra columns
   to a single-beat model; a model architected around sequences rather
   than flat feature vectors is a structurally different approach worth
   testing, not a rerun of the same idea.
4. **Resolve the two-record over-detection issue** (Section 9) — a
   focused look at the beat-detector's rate-tracking behavior
   specifically on fast/noisy rhythms, independent of anything else in
   this document.
5. **Fix the remaining MedGemma narrative-accuracy gap at the model
   level**, not just the current after-the-fact filter — e.g. a larger or
   fine-tuned local model, or a stricter structured-output mode, so the
   free-text paragraph itself stops needing to be double-checked and
   silently dropped roughly 30% of the time.
6. **Confirm the VitalPatch batch to completion and re-run the manifest
   summary** (Section 7) for a final, non-partial picture of real-world
   performance across all 6 patients, rather than the in-progress
   snapshot in this document.

---

## 11. Bottom line

The beat classifier reliably tells Normal from dangerous Ventricular
beats (the two classes that matter most for immediate safety), is
honestly and repeatedly weak on the rarer Supraventricular and Fusion
classes despite four genuinely different fix attempts, and that weakness
looks like a real hardware/data limit rather than something more tuning
will solve. On top of that classifier, this round of work fixed three
real bugs that were silently discarding real heartbeats or real
recordings before they ever reached the classifier, and fixed a
narrative-writing bug where the AI language model sometimes described a
correct number incorrectly — with a safety net that now catches and
removes that specific mistake automatically before anything is saved.
Nothing about the classifier itself, its decision thresholds, or the
5-class medical scheme was changed to produce any of the numbers in this
document. What's next isn't more tweaking of the current single-lead,
single-beat approach — Section 5 shows that's been tried multiple honest
ways — it's new data or a structurally different model (Section 10).

</details>

<br>

---

# Engineering reference

*Everything from here down is the technical changelog for people already
familiar with the codebase — what changed, in which file, and why.*

## Pipeline stages

```
raw file (VitalPatch CSV | SeNSiO/prorhythm CSV | WFDB record)
  -> Stage 2: SQI gate (native sample rate)          evaluate_window()
  -> Stage 3: resample to 125Hz                      resample_linear()
  -> Stage 4: filter chain (baseline/highpass/notch/bandpass/Kalman EMG)
  -> Stage 5: R-peak detection + beat segmentation   detect_and_segment()
  -> Stage 6: feature extraction
  -> Stage 7: classification (five_class_xgb.json, AAMI N/S/V/F/Q)
  -> Stage 8: risk cascade (rule_trace, deciding_rule, risk_level)
  -> Stage 9: MedGemma clinical narrative             agent_bridge.py
  -> saved report: <segment_id>.json + <segment_id>.md
```

All of Stages 2–8 live in `ecg_pipeline/ecg_pipeline_core.py`. Stage 9 and
the report assembly live in `ecg_pipeline/agent_bridge.py`.

## What changed on this branch

### 1. R-peak beat-level over-culling (commit `2420ecc`)

`detect_and_segment()` fed XQRS's raw peak indices straight into
`segment_beats()`, but on resampled data those indices land ~3 samples off
the true `|amplitude|` local max. The beat-level SQI check
(`R_PEAK_NOT_LOCAL_MAX`) requires the peak to *be* that local max within a
±3-sample window, so nearly every real beat failed it — 1400–2400 beats
lost per WFDB test record, the dominant cause of beat loss even after the
SQI-gate fix below. Fixed by snapping each XQRS peak to its true local max
before segmentation (mirroring a fix that already existed in the
training/eval path but was never ported to the runtime path).

- Retention (beats analyzed / ground-truth beats) went from 8–70% to
  84–131% across 7 WFDB test records.
- VitalPatch unaffected (no regression: 189 detected, 185 analyzed, up
  from 122).
- **Known, explicitly NOT fixed here** (see commit message for full
  reasoning): true XQRS over-detection on two noisier WFDB records
  (103/111), where XQRS's adaptive rate-tracking appears to "hunt" at its
  hr_max=200bpm floor. No safe low-risk fix identified; flagged as a
  separate follow-up rather than reconfiguring XQRS without validation.

### 2. WFDB SQI over-rejection (commit `3add727`)

`evaluate_window()` scored `baseline_wander_ratio` and `snr_db` directly
on the raw window. That's correct for VitalPatch (already filtered by the
device) but wrong for raw WFDB input, where Stage 4's baseline removal
hasn't run yet — a clean record's own real DC/baseline offset got scored
as "noise," driving some records to 100% window rejection. Fixed by
scoring those two metrics on a locally detrended copy (the same 0.5Hz
highpass Stage 4 already uses) instead of the raw window; flatline/
clipping/missing/kurtosis still score the raw window unchanged.

- WFDB rejection dropped from 48–100% to 7–23% across 7 test records.
- VitalPatch improved too (26.7% → 13.3% rejection) — not a regression.
- Synthetic flatline/white-noise/clipped signals still 100% rejected (the
  gate still gates).

### 3. SeNSiO/prorhythm zero-beat bug (uncommitted on this branch, `ecg_pipeline_core.py`)

A prorhythm/SeNSiO recording (100Hz native rate) returned 0 beats
detected and 100% SQI rejection end-to-end — `NOT_ASSESSABLE`, MedGemma
correctly skipped. Root cause: `snr_db` defines "signal" as 0.5–40Hz
content and "noise" as everything else. At SeNSiO's 100Hz native rate,
Nyquist is exactly 50Hz — the same frequency as the mains notch filter,
and `powerline_notch()` correctly no-ops whenever `notch_hz >= Nyquist`
(you cannot design a notch at/above Nyquist). That leaves a narrow,
un-notch-able sliver right at Nyquist that any near-Nyquist artifact can
dominate, reading as overwhelming "noise" even though the actual QRS
structure is physiologically plausible (verified via an SQI-bypass test:
89.2% RR-plausibility on the target recording).

Fixed by scoring `snr_db` on a copy resampled to `TARGET_FS` (125Hz, the
same resample Stage 3 already performs downstream) *before* the
notch/highpass are applied for scoring purposes — this only activates
when the native rate itself can't support the notch
(`fs / 2.0 <= FILTER.notch_hz`), so WFDB and VitalPatch (both already
≥ 125Hz) are untouched.

- Verified across 3 SeNSiO files with a before/after comparison; no
  VitalPatch regression; synthetic flatline still fully rejected.

### 4. MedGemma narrative quality (uncommitted on this branch, `agent_bridge.py`)

Two defects found in the LLM-authored clinical narrative:

- It ignored `beat_summary` (N/S/V/F/Q counts) entirely.
- It sometimes misread its own numbers — e.g. asserting *"PAC burden
  8.642 exceeds threshold 15.0"* (false; 8.642 < 15.0). Both numbers were
  real, just the comparison was wrong. Observed in roughly 30% of
  generations even after the prompt fix below — a genuine small-model
  (4B) limitation, not something prompting alone can fully close.

Fix, in two parts:

1. **Prompt fix**: every rule's exceeds/does-not-exceed verdict is now
   pre-computed in Python (`_rule_verdict_line`, using the cascade's own
   comparison operators) and handed to MedGemma as an explicit fact,
   alongside explicit `beat_summary` and `rhythm_findings` blocks. The
   prompt forbids MedGemma from inferring its own comparisons.
2. **Deterministic composition + post-hoc guard**: the saved narrative is
   assembled in code with a guaranteed structure — (a) beat summary
   (b) rhythm findings (c) the deciding rule with its real value vs.
   threshold (d) final risk level (e) a clinician-review disclaimer —
   with MedGemma's own prose embedded as a "Clinical interpretation"
   sub-section. A validator (`_narrative_asserts_false_exceed`) scans that
   prose for a false exceed/cross/trigger claim against a rule that did
   **not** fire, and drops just that sub-section (never the structured
   parts) if it finds one. This guarantees the *saved* report is never
   wrong, even though the model itself remains occasionally unreliable.

Verified: 9/9 runs (3 recordings × 3 repeats, including a SeNSiO file and
a VitalPatch file) show complete structure, zero false-exceed claims,
zero invented numbers, stable risk levels.

### 5. Batch pipeline + saved reports (new: `batch_vitalpatch_report.py`,
   `batch_prorhythm_report.py`, `manifest_summary.py`)

- `run_full_report()` (in `agent_bridge.py`) is now run over every raw
  recording in `data/raw/vitalpatch/Patch_*/` and every SeNSiO/prorhythm
  recording in `data/raw/prorhythm/`.
- Every segment gets a saved `<segment_id>.json` (full structured
  `RiskReport`: beats, rhythm findings, rule trace, risk level, narrative)
  and a companion `<segment_id>.md` (human-readable clinical report),
  written under `data/reports/<source>/<patient>/`.
- Each file is wrapped in its own try/except so one bad file never aborts
  the batch — failures are logged to a run manifest (`error` column)
  instead of crashing. This is needed in practice: ~70 VitalPatch files
  contain a literal `'-'` sentinel value in place of a numeric field and
  fail to parse; they're now skipped and logged instead of killing the
  run.
- The `NOT_ASSESSABLE` guard is preserved end-to-end: recordings with too
  few surviving beats stay `NOT_ASSESSABLE`, MedGemma is skipped
  (`medgemma_status = SKIPPED_NOT_ASSESSABLE`), and no narrative is
  fabricated for them.
- `batch_vitalpatch_report.py` is resumable: since parsing + running a
  given raw file is a pure function of that file's bytes and the current
  code, the batch skips any file whose output JSON already exists on disk
  and reconstructs its manifest row from that JSON instead of
  recomputing — important because the batch is long-running and
  serialized MedGemma calls make re-processing expensive.
- `manifest_summary.py` combines both run manifests into one CSV, prints
  summary counts (assessable vs. `NOT_ASSESSABLE`, risk-level
  distribution, MedGemma status distribution), and prints a few full
  "showcase" reports end-to-end (raw file → beat counts → rhythm findings
  → deciding rule → final clinical report).

## Running it

```bash
# one segment, live MedGemma report:
python -m ecg_pipeline.agent_bridge --file <path-to-raw-csv>

# full VitalPatch batch (all patients, resumable):
python -m ecg_pipeline.batch_vitalpatch_report

# full SeNSiO/prorhythm batch:
python -m ecg_pipeline.batch_prorhythm_report

# combined manifest + summary + showcase, once both batches have run:
python -m ecg_pipeline.manifest_summary
```

MedGemma must be served locally via Ollama (`ollama serve`, model
`medgemma:latest`) for Stage 9 to produce a live narrative; without it,
`medgemma_status` falls back to `UNAVAILABLE_FALLBACK` and the structured
report is still saved, just without an LLM-authored interpretation.

## Known limitations / honest caveats

- MedGemma (4B, local) occasionally still misstates a comparison in its
  free-text prose (~30% of generations pre-filter); the post-hoc validator
  catches and strips this before saving, but the model itself was not
  fixed and cannot be fully fixed by prompting alone.
- XGBoost XQRS over-detection on two specific noisy WFDB records
  (103/111) is diagnosed but not fixed — see commit `2420ecc`.
- The VitalPatch batch (2375 files) is serialized on a single local
  Ollama instance (`OLLAMA_NUM_PARALLEL=1`), so a full run takes several
  hours; it is resumable and safe to kill/restart.
- Nothing in this branch changes `five_class_xgb.json`, the risk cascade
  thresholds, or the AAMI N/S/V/F/Q scheme — every fix is upstream (signal
  quality / R-peak detection) or downstream (narrative composition) of
  the classifier itself.
