# Agent Rules — ECG Pipeline

Standing constraints for any agent (or human) working on the ECG 5-class
beat classifier in this repo. These exist because every one of them was
violated, discovered, and fixed at real cost during the 2026-07-16 session
that produced `ABLATION_REPORT.md`. Read them before touching
`ecg_pipeline_core.py`, `ecg_pipeline_tools.py`, or anything under
`models/`.

## 1. DS2 is report-only

MITDB DS2 (`splits.MITDB_DS2`) is the held-out literature test set. Never
train on it, tune a hyperparameter against it, oversample/reweight based on
it, or threshold anything using it. It exists to answer one question
honestly: "does this change actually work?" The moment it's used for any
decision, that answer becomes unusable. If you need a signal to tune
against, use `DS1_VAL` (or a properly-sized validation carve-out — see
rule 2) — never DS2.

## 2. All splits are patient-level, and validation must be big enough to mean something

No patient's beats may appear in more than one of {DS1_TRAIN, DS1_VAL,
DS2}. Beyond that: a validation split has to be large enough to actually
*catch* a regression, not just exist. The concrete anti-pattern from this
session: a docstring once claimed a 4-record `DS1_VAL` was "too small" to
catch a real V-class regression — that claim turned out to be
unverifiable, but when the experiment was actually run, the *real*
4-record split already had 379 S beats and 868 V beats, which is plenty of
statistical power to see an F1 swing. The lesson isn't "always use more
records" — it's **check the actual per-class beat counts in the split
before trusting or distrusting it**. If a class's support in validation is
capped by real data scarcity (e.g. F, which has only 403 beats in all of
DS1), that's a documented limitation, not a bug to "fix" by moving data out
of training.

**F-class validation caveat.** F-class validation support is structurally
near-zero: DS1 contains only ~403 F beats total, ~363 of them concentrated
in record 208, which is deliberately kept in *training*. Therefore F-class
improvements **cannot** be meaningfully validated on DS1_VAL and must be
judged on DS2 with the small-sample caveat stated explicitly each time. Do
not attempt to "fix" thin F validation support by moving record 208 out of
training — that trades a real training signal for a still-underpowered
validation signal. Treat F's ceiling as a documented data-scarcity
limitation, not a bug to engineer around.

## 3. Never overwrite `models/five_class_xgb.json`

That file is production. Every experiment — enrichment, feature change,
architecture variant — writes to a new filename
(`five_class_xgb_<experiment>.json`) and is evaluated independently.
Promotion to production is a separate, explicit, human-approved step, not
something that happens by an experiment script writing over the default
path. If a script's `--out` default ever points at
`five_class_xgb.json`, that's a bug.

## 4. Accuracy is not a success metric

N is ~90%+ of every split. A model that only ever predicts N scores >90%
"accuracy" while being clinically useless. Every report — in conversation,
in a log, in `ABLATION_REPORT.md` — must include, at minimum:
- Per-class sensitivity, precision, F1, support (all 5 AAMI classes: N/S/V/F/Q)
- Macro-F1
- The confusion matrix
- Any specific off-diagonal rate relevant to the experiment (e.g. F→S rate,
  S→V rate) — pick whichever misclassification the experiment is actually
  about and report it explicitly, don't make the reader compute it from the
  confusion matrix themselves.

Accuracy can be mentioned as a footnote. It never determines a verdict.

## 5. Fixed `random_state` everywhere; results must replicate

Splits, samplers (ROS), and XGBoost's `random_state` are all seeded
explicitly (`--seed`, default 42) and never left to library defaults. Two
runs with an identical config and identical seed must produce
byte-identical DS2 metrics. If they don't, that's a bug to fix before
trusting any ablation result built on top of it — a result isn't "better"
if it might just be run-to-run noise.

## 6. No claim without a checkable artifact

**A task is not "done" without one of: a file that exists on disk, a
metric printed from a fresh run executed this session, or a diff someone
can read.** A docstring note, a status comment, a task-list entry marked
"completed," or a reference to a report/model file — none of these are
evidence by themselves. They are claims, and claims get verified before
they get trusted or built on.

This rule exists because three separate phantom artifacts were found in
one session:
1. A docstring in `eval_classifier`'s CLI help referencing
   `models/five_class_xgb_timing_v1.json` — the file didn't exist.
2. A docstring in the features section of `ecg_pipeline_core.py` claiming a
   7-feature timing extension had been "built and evaluated" with specific
   DS1_VAL/DS2 numbers, then reverted — no such code existed anywhere in
   the codebase, and the repo has zero git commits to check against.
3. Both of the above cited `ABLATION_REPORT.md` as the source of the real
   numbers. That file did not exist until this rule was written.

**When you find a phantom artifact:** don't build on it, don't cite it,
and don't leave it in place for the next person to trust. Either produce
the real artifact (run the thing, save the file) or delete/correct the
claim so it stops looking like completed work.

**When a note describes a plausible-sounding failure mode you can't
verify:** you don't have to choose between believing it and ignoring it.
Design the next experiment so the claimed failure mode would be caught if
it's real — that gets you independent confirmation or refutation from
fresh evidence, which is strictly better than trusting or dismissing an
unverifiable claim. (This is exactly how the timing-features V-regression
was actually confirmed — not by trusting the old note, but by rerunning
the experiment properly and watching for that specific failure.)

## 7. One change at a time

Don't stack an untested change on top of another untested change and
expect to be able to attribute the result. If two things need testing (e.g.
a data-enrichment change and a feature-engineering change, or a full
7-feature bundle and a 1-feature ablation of it), run them as separate,
independently-evaluated experiments in `ABLATION_REPORT.md`, even if that's
slower.

## 8. Promotion to production requires all of these

A candidate model replaces `models/five_class_xgb.json` **only** when every
one of the following holds:
- **(a)** No per-class regression beyond a trivial margin on DS2. **V and N
  are safety-critical and effectively zero-tolerance** for regression — a V
  or N drop disqualifies a candidate even if S or macro-F1 improves.
- **(b)** The judgment is made on **DS2, per-class**
  (sensitivity/precision/F1 + confusion matrix + relevant off-diagonal
  rates), never on accuracy or macro-F1 alone.
- **(c)** Explicit **human approval**.
- **(d)** The winning config and its exact reproducing command are recorded
  in `ABLATION_REPORT.md`.

Promotion is always a deliberate, separate step. It never happens as a side
effect of a training script's `--out` path.
