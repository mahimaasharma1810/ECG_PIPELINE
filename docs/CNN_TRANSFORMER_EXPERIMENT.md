# What we're testing right now, in plain English

**Date:** 2026-07-23 (initial run) — **closed 2026-07-24** after a follow-up check.
**Status:** CLOSED — clean no on both variants tried. Not promotable. No
further work planned on this line. Details below.

## The one-sentence version

We're testing whether a bigger, deep-learning model (a CNN+Transformer) can
classify heartbeats better than our current XGBoost model — specifically,
whether it can catch more S-type (supraventricular) beats without getting
worse at the beats we already handle well (Normal and V/ventricular).

## Why we're doing this at all

Our current production classifier is good at Normal (N) and Ventricular (V)
beats, but weak at Supraventricular (S) beats — it catches maybe 1 in 6 of
them. We already tried to fix this with a "two-stage" version (one model
decides normal-vs-abnormal, a second model decides which kind of abnormal),
and with extra hand-built features describing the shape of the QRS spike.
Both helped S a bit, but both broke V worse than the amount we're allowed to
break it (there's a hard rule: V and N are safety-critical, so we're not
allowed to make them meaningfully worse to gain something else).

So the two-stage direction is closed. This is a **different idea**: instead
of hand-designing features and rules, let a deep neural network look at the
raw heartbeat waveform itself and learn its own patterns. This is what
several published research papers on this exact problem do, and it's the
next thing worth trying before we give up on improving S further.

## What a "CNN+Transformer" actually is (no jargon)

Think of it as two specialists looking at the same heartbeat and then a
third specialist combining their opinions:

1. **The shape specialist (CNN):** looks at small pieces of the waveform —
   like "how steep is this bit," "how wide is that bump" — the same way you
   might eyeball a heartbeat shape on a monitor. It looks at a few different
   zoom levels at once (narrow/medium/wide) since some abnormalities are
   subtle and others are broad.
2. **The rhythm specialist (Transformer):** looks at how those shape pieces
   relate to each other across the whole beat, the same way a cardiologist
   doesn't just look at one point on the wave, but how the whole thing flows
   together.
3. **The timing specialist (a small separate branch):** just looks at plain
   numbers — how early or late this beat came compared to the ones before
   it, since a premature beat is a big clue for S/V regardless of its shape.
4. **A final combiner:** takes all three opinions and outputs one of 5
   labels: Normal, Supraventricular, Ventricular, Fusion, or Unknown.

This is genuinely new code (`gate2_cnn_transformer.py`, not yet part of the
production pipeline) — it does **not** touch the production model file.

## What we've done so far, step by step

**Step 0 — Get a GPU working.**
Training a deep model like this on a plain CPU is painfully slow. This
machine has a real GPU (RTX 2080 Ti) but the Python setup on it only had a
CPU-only version of the deep-learning library (PyTorch) installed. We hit a
disk-quota wall trying to install the GPU version in the normal location, so
we installed a fresh, separate copy on a different disk (`/ssd_scratch`,
which has plenty of room) instead of touching your existing setup. Confirmed
working: the GPU now actually runs computations, not just CPU pretending.

**Step 1 — Cheap sanity check before building anything big ("Gate 1").**
Before spending hours building the full deep model, we asked a cheaper
question first: there's already a small pretrained "encoder" sitting unused
in this codebase (something that turns a raw heartbeat shape into 32
numbers) — trained earlier on real, unlabeled VitalPatch/SeNSiO patient
data. We tested whether those 32 numbers already know the difference between
N/S/V/F beats, by training the simplest possible classifier on top of them.

Result: **weak, mostly negative.** It didn't separate the classes well —
worse than our existing baseline. Most likely reason: that encoder learned
from a *different* device's data (VitalPatch), and heartbeat shapes here
come from *hospital* recordings (MITDB) — different noise, different
"normal." So we're building the deep model from scratch, not reusing that
encoder. This was the right call to test cheaply first — it took under an
hour instead of finding out after a multi-hour training run.

**Step 2 — Build the training data ("lean" data set).**
We chose to train on: our normal 12-record MITDB training set, plus SVDB (a
public dataset that's rich in S-type beats), plus SDDB (a public dataset
that's one of our only real sources of F-type beats). Two other public
datasets (LTAFDB, INCART) are deliberately left out for now, to keep this
experiment faster and simpler — they're a "maybe later" ablation, not part
of this run.

**A real bug we found and fixed along the way:** while building this data,
we discovered that 9 out of 12 usable SDDB recordings were silently
contributing **zero** beats to training — not because they lack labels, but
because a tiny number of corrupted/missing samples (signal dropouts) in
those recordings were poisoning the *entire* recording once our filtering
step touched it, causing every single beat in that file to get thrown out.
This is a real bug in the data pipeline, not something we introduced — it's
been silently doing this in every prior experiment too, we just never
happened to look closely at these specific recordings before. We fixed it
for this new deep-model script (by smoothing over those corrupted samples
before filtering, the same trick already used elsewhere in this codebase),
recovering that lost data.

Fixing that bug had a side effect worth flagging: the fixed SDDB recordings
turned out to contain a **huge** number of ordinary Normal beats — about
738,000 of them. If we'd used all of them, SDDB alone would have become
about 95% of the entire training set, which would have quietly turned "add
a bit more S/V/F data" into "mostly train on a totally different dataset,"
risking the model learning Holter-monitor quirks instead of what actually
matters. So we capped SDDB's Normal-beat contribution at a fixed number
while keeping *every single* one of its rare S/V/F beats — getting the
benefit (more rare-class examples) without the downside (drowning out
everything else).

**Step 3 — Train the model (in progress right now).**
The training data is built, and the model is now training on the GPU. Per
plan, we first timed how long 5% of the data takes for one pass, to
estimate the real training time honestly instead of guessing, then launched
the full run. It tunes itself against a held-out "validation" slice of our
own hospital data (never the final report-only test set) and will stop
automatically once it stops improving, to avoid overfitting.

## Step 4 — the final, one-time test ("Gate 3") and the result

Training stopped itself after 13 rounds (it noticed it wasn't improving
anymore and picked its best round, round 8, automatically). Then we ran that
one saved version, exactly once, against our final untouched 22-record test
set (DS2) — the same test every other experiment in this project has been
judged against.

**The honest answer: no, this doesn't work — and it's a clearer "no" than
our last two attempts.**

| Beat type | Old model (today's production) | This new deep model | Change |
|---|---|---|---|
| Normal (N) | 0.966 | 0.901 | worse |
| Supraventricular (S) | 0.160 | 0.121 | **worse** (this was the whole point) |
| Ventricular (V) | 0.866 | 0.730 | worse, and by a lot |
| Fusion (F) | 0.021 | 0.030 | very slightly better |

The two previous attempts (two-stage model, then QRS-shape features) at
least genuinely improved S — they just weren't safe enough on V/N to adopt.
This new deep model didn't even manage that: S got *worse* too, while V
took its biggest hit yet (a 0.136 drop in a single number that's supposed to
barely move). So this isn't "close but not quite" — it's a broader miss
across the board.

**Our best guess at why:** looking at exactly which beats it got wrong, the
model isn't making one specific mistake (like "confusing V for N") the way
our earlier attempts did — it's misfiring in several directions at once,
which usually means the model is confused about what a "normal" beat looks
like in general, not just at the edges. Two likely reasons: (1) even after
capping it, a meaningful chunk of the training data still comes from 24-hour
Holter-monitor recordings (SDDB) whose normal-beat "texture" is a bit
different from the hospital recordings we test against, and (2) this deep
model has a lot more room to learn patterns than our current model, but we
only gave it a relatively modest amount of labeled training data to learn
from — a bigger model with the same-sized answer key can end up learning
noise instead of signal.

**Bottom line (before the follow-up check below):** nothing changes for the
real, live system — this was purely an experiment, saved separately, never
plugged into anything users touch.

## Step 5 — one follow-up check (2026-07-24): was it the SDDB data, or the model?

Before fully closing the book on this, there was one obvious loose end:
the deep model's training data included SDDB, a 24-hour Holter-monitor
dataset that (even after our N-cap fix) still made up a meaningful slice
of "Normal" beats — and those Normal beats look a bit different (different
device, different recording conditions) than the hospital data we test
against. Maybe the model wasn't bad — maybe it just learned a slightly
"wrong" idea of what Normal looks like because of that mixed-in data.

So we re-ran the exact same model, same settings, changing **one thing
only**: we removed SDDB from training completely, keeping just our normal
hospital training set plus the S-boosting dataset (SVDB).

**The honest answer: no, that wasn't it — and it actually got a bit worse.**

| Beat type | Old model (production) | Deep model, with SDDB | Deep model, without SDDB |
|---|---|---|---|
| Normal (N) | 0.966 | 0.901 | 0.889 (slightly worse) |
| Supraventricular (S) | 0.160 | 0.121 | 0.126 (still worse than production) |
| Ventricular (V) | 0.866 | 0.730 | 0.615 (much worse) |
| Fusion (F) | 0.021 | 0.030 | 0.041 (still fine, small class) |

Removing the "suspicious" data didn't fix Normal, and it didn't fix
Supraventricular either — both stayed roughly where they were. Ventricular
actually got noticeably worse. Looking at exactly what the model confused
with what: it started mistakenly labeling a lot more Normal and
Supraventricular beats as Ventricular than before — it's now trigger-happy
about calling things "V" incorrectly, which tanks V's reliability score
even though it still catches most real V beats.

**So the real explanation isn't "bad data" — it's "not enough data for a
model this size."** Removing SDDB left the model with about a third less
training data to learn from, and a bigger model with less data to learn
from tends to get noisier and more easily confused, not cleaner. Adding
more SDDB-style variety back would arguably help more than removing it —
just not enough to be safe on its own with what we have.

## Final bottom line — this line of investigation is now CLOSED

Both "let's replace the current model with something fancier" avenues
(two-stage, and now deep learning — tried two different ways) have come
back negative. The honest place this leaves us: our current model's
weakness on S-type beats looks like a real, structural limit — either of
what a single ECG lead can tell you, or of how much labeled training data
we currently have for a model with this much capacity. Getting past it
would most likely need either a different kind of input (like a second
lead) or meaningfully more labeled training data, not just a smarter model
architecture on the data we already have.

**We're stopping here.** No more variants of this deep-learning experiment
are planned — we tried the obvious data fix (remove the suspicious SDDB
data) and it made things worse, not better, so there isn't a clear next
lever left to pull on "smarter model, same data." The production model
(today's XGBoost classifier) stays exactly as it is. Any future push on the
S-type weakness should start from new data (a second ECG lead, or more
labeled examples) rather than another model-architecture attempt on what
we already have. Full numbers, confusion matrices, and technical detail for
both runs are logged in `docs/archive/ABLATION_REPORT.md`.
