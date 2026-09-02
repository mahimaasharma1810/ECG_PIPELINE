# What happens to your heartbeat data, in order

A step-by-step walk through the rhythm pipeline, following **one real window
from one real recording** all the way through, with the actual numbers at each
stage. No placeholders.

Written so a clinician or a new engineer can follow it without opening a source
file.

> **Rhythm regularity indicator. Not a diagnosis. Not validated for clinical
> use. Cannot distinguish atrial fibrillation from other causes of irregularity.**

**The recording we follow:** `prorithm_ecg/1794/2026-08-09_12.csv`
SHA-256 `674f206a2f41d228…` · 437,054 rows · 2,659 s · subject 1794

**Trust key used throughout:**

| Label | Meaning |
|---|---|
| **VALIDATED** | Tested against expert-labelled data, result reproduced |
| **PROVISIONAL** | Derived from data but not confirmed on the target device |
| **UNTESTED-ON-DEVICE** | Never checked on a real patch. Correctness unknown |

---

## Step 1 — The patch measures a voltage

**IN:** two electrodes on the chest — right arm and left leg.

**WHAT IT DOES:** measures the voltage difference between them, about 90 times
per second. That electrode pair looks along the heart's main electrical axis,
which is the standard definition of **Lead II**, and gives a tall upward R-wave.

**OUT:** a stream of numbers, roughly −270 to +405 in this recording.

**WHY:** Lead II gives the clearest, tallest heartbeat spike, which is what
makes beats findable.

**IF DONE WRONG:** electrodes placed elsewhere give a smaller or inverted
R-wave, and beat detection degrades.

> **Limitation, here at the source: the amplitude units are UNKNOWN.** They are
> not millivolts — most likely raw ADC counts. We never convert them. Nothing in
> the rhythm calculation uses amplitude *values*, only beat *timing*, which is
> why this never becomes a problem. A `/700` conversion found in old code traced
> back to a synthetic data generator, not a device specification.

**Trust: VALIDATED** (lead confirmed from device configuration).

---

## Step 2 — The patch sends packets over Bluetooth

**IN:** the voltage stream.

**WHAT IT DOES:** bundles about 7 readings into a packet and sends it. Each
packet looks like this:

```
pid, sNo, start_time, end_time, duration, status, vitals_status,
vitals: {t, hr, skt, rr, spo2, hrv, sp, dp},
ecg_clean: [{e: -0.63, t: 1786618155941}, {e: -1.89, t: 1786618156075}, …]
```

`e` is the reading, `t` is its timestamp. `sNo` is a packet sequence number.

**OUT:** JSON packets, one every ~73 milliseconds.

**WHY:** Bluetooth is far more efficient sending batches than single readings.

**IF DONE WRONG:** this is where a real bug lived. An earlier client expected
`{"samples": [{"timestamp_ms", "amplitude"}]}`. Fed a real packet, it would
**connect successfully, log no error, and produce zero readings.** The parser
now *refuses* an unrecognised packet rather than returning nothing quietly.

> **The field is named `ecg_clean` because the firmware has already filtered
> it.** Filtering it again destroys it — measured, only 6–7% of the signal
> amplitude survives a second filter chain.

**Trust: UNTESTED-ON-DEVICE.** The schema is correct, but no live patch has been
connected since 2026-08-20. Verified only by replaying recorded files.

---

## Step 3 — Remove re-sent data

**IN:** 437,054 rows.

**WHAT IT DOES:** the device re-sends blocks of about 13 seconds. Rows that are
*exactly* identical — same timestamp and same reading — are duplicates, and are
dropped.

**OUT:** **215,552 rows removed (49.3%)**, leaving 221,502.

**WHY:** without this the recording appears to arrive at **164 readings per
second**. After removal it is **90.4 per second** — the true rate.

**IF DONE WRONG:** every timing calculation downstream is wrong by up to a
factor of two. This affects **28 of the 37** recordings we hold.

> The naive checks both fail here: `rows ÷ duration` says 164 Hz, and the more
> obvious "look at the gap between timestamps" says **infinity**, because more
> than half the readings share a timestamp with the one before.

**Trust: VALIDATED.** After removal, all 37 recordings agree on the sample rate
to within 0.6%.

---

## Step 4 — Build a clean time base

**IN:** 221,502 de-duplicated readings, arriving out of order in places.

**WHAT IT DOES:** sorts by timestamp, splits at dropouts longer than 1 second,
then — crucially — **does not trust the timestamps for timing**.

Instead, beat times come from *counting readings*:

```
time between beats  =  (difference in reading count)  ÷  (measured rate)
```

**OUT:** the longest continuous stretch: **68,648 readings, 759.7 seconds, at a
measured 90.363 readings per second.**

**WHY:** the timestamps are *packet delivery times*, not measurement times.
About 7 readings share one timestamp, so any single reading's true time is
uncertain by **±36 milliseconds**.

**IF DONE WRONG:** this is the most important design decision in the whole
pipeline. Using the timestamps directly, a **perfectly steady** heartbeat would
show a variability score of **0.095**. Using reading counts, it shows **0.006**.
Since the decision threshold is 0.1275, the first approach would put the noise
floor almost on top of the threshold — the system would be measuring Bluetooth
behaviour and calling it heart rhythm.

**Trust: VALIDATED** on the arithmetic and on public data; the counting approach
is **UNTESTED-ON-DEVICE** in the sense that no reference recording has confirmed
the resulting beat times are correct.

---

## Step 5 — Find the heartbeats

**IN:** 68,648 readings at 90.363 Hz.

**WHAT IT DOES:** two independent algorithms look for the sharp R-wave spike.

**OUT:** algorithm A finds **1,091 beats**; algorithm B finds **1,039**.

**WHY:** two algorithms, not one, so we can tell when they disagree — which is
how we detect that a signal is untrustworthy.

**IF DONE WRONG:** everything after this is meaningless. A false beat splits one
gap into two, inventing irregularity that is not there.

**Trust: VALIDATED on public data — F1 0.9890** (sensitivity 0.9915, precision
0.9865) against 105,078 cardiologist-marked beats, tested at this device's exact
sample rate.

> **UNTESTED-ON-DEVICE, and known to be wrong here.** On real patch recordings
> the detector finds about **4.4% too many beats** — marking bumps that are not
> heartbeats, mostly where the wearer moved. This is the central unresolved
> problem, and everything below inherits it.

---

## Step 6 — Turn beats into intervals, and take a window

**IN:** beat positions.

**WHAT IT DOES:** takes **64 consecutive beats**, which give **63 gaps**
between them. (64 beats is 63 intervals, not 64 — an off-by-one that is written
into the code as a named constant so it cannot be mis-typed.)

**OUT:** for our window, starting at sample 3,917 and ending at 7,848,
local rate 90.474 Hz, first ten gaps in milliseconds:

```
674.2  674.2  674.2  519.5  674.2  674.2  663.2  552.6  762.7  773.7
```

Shortest 486.3 ms, longest 829.0 ms, middle value 718.4 ms.

**WHY:** 64 beats (~45 seconds) is long enough for a stable measurement, short
enough to notice a change.

**Trust: VALIDATED** (arithmetic, unit-tested).

---

## Step 7 — Screen out impossible intervals

**IN:** 63 intervals.

**WHAT IT DOES:** flags any gap shorter than 300 ms or longer than 2,000 ms —
physiologically impossible, so almost certainly a detection error.

**OUT for our window: 0 flagged (0.0%).** One continuous segment, 62 usable
beat-to-beat comparisons.

**WHY:** so obviously-broken values do not distort the result.

**IF DONE WRONG:** two rules matter here.
1. Flagged intervals are **shown, never silently deleted** — a reviewer must see
   what was excluded.
2. If more than **20%** are flagged, the window returns **"cannot determine"**
   rather than forcing an answer from what is left.
3. Beat-to-beat comparisons must never *span* a removed interval. If interval 10
   is bad, we do not compare 9 with 11 — that would invent a jump that never
   happened. (Verified: 63 intervals with one flagged gives 62 comparisons, not
   63.)

**Trust: VALIDATED** (unit-tested).

---

## Step 8 — Calculate the features

**IN:** 63 intervals.

**WHAT IT DOES:** computes four numbers.

**OUT for our window:**

| Feature | Value | What it means |
|---|---|---|
| Mean interval | **689.7 ms** | average gap between beats |
| **Heart rate** | **87.0 bpm** | 60,000 ÷ mean interval |
| **Variability (RR CV)** | **0.1159** | **the number the decision uses** |
| RMSSD | 120.5 ms | average beat-to-beat change |
| SDNN | 80.0 ms | overall spread |

**WHY RR CV:** it is variability *relative to rate*, so it means the same thing
at 50 bpm and 100 bpm. Measured discriminative power **0.95** on two independent
databases — the strongest single feature we have.

**IF DONE WRONG — one trap worth naming:** heart rate on its own looks like a
great predictor (0.92), but it is a **trap**. In the databases we used, irregular
episodes happen to run fast (median 112 bpm vs 75). A system using heart rate
would learn *"fast means irregular"*, which is true of those particular patients,
not of hearts. **Heart rate is deliberately never used for the rhythm decision.**

**Trust: VALIDATED** (arithmetic; discriminative power measured on two databases).

---

## Step 9 — Decide whether to trust this window at all

**IN:** the raw signal for the window, plus both algorithms' beat positions.

**WHAT IT DOES:** six checks. The main one compares the two beat-finding
algorithms: if they agree, the signal is probably clean.

**OUT for our window:**

| Check | Value | Pass? |
|---|---|---|
| **Algorithm agreement (bSQI)** | **0.953** | pass (needs ≥0.95) |
| Alternation (lag-1) | −0.137 | pass |
| Peakiness | 16.1 | pass |
| Power in QRS band | 0.522 | pass |
| Flat signal | 1.7% | pass |
| Clipped signal | 0.1% | pass |
| **Overall** | | **PASSED** |

**WHY:** a system that always answers is lying some of the time. This is what
lets it say *"the signal is too poor to judge."*

**IF DONE WRONG — a bug we caught here:** the alternation check, unguarded,
**rejected the steadiest rhythms**. Because the device samples in 11.15 ms steps,
a perfectly regular heartbeat alternates between two rounded values, which looks
like alternation. At 50 bpm it scored −0.56, failing a −0.5 threshold. It now
only applies when there is real variation to look at.

> **Do not lower the 0.95 agreement threshold to make more verdicts appear.**
> Removing this gate on healthy recordings produces **34 of 39 windows labelled
> IRREGULAR and none REGULAR.**

**Trust: PROVISIONAL.** Derived on public data, where it keeps 78% of windows and
cuts detection-induced error sharply. On device data it currently refuses ~82% —
because algorithm B genuinely disagrees, not because the signal is bad.

---

## Step 10 — Compare against the threshold

**IN:** RR CV = **0.1159**.

**WHAT IT DOES:** compares it to **0.1275**, with a **refusal band** of ±0.027
either side — from **0.1005 to 0.1545**. Inside that band, the system declines
to pick a side.

**OUT for our window: 0.1159 falls INSIDE the refusal band → UNABLE TO
DETERMINE.**

**WHY the band:** the measurement itself carries uncertainty — about ±0.027 from
sampling coarseness and residual detection error. A window that close to the line
cannot be called honestly.

**WHY 0.1275:** derived from 16 patient recordings with cardiologist labels,
21,920 windows, timing deliberately coarsened to this device's 89.70 Hz.

**IF DONE WRONG:** two published thresholds were rejected for being derived on
different hardware at different timing resolutions. A threshold only means
something alongside the resolution it was derived at.

**Trust: PROVISIONAL — NOT SHIPPABLE.**

| | |
|---|---|
| Held-out **database** test (MITDB) | Se 0.9839, Sp 0.8660, PPV 0.4692, **NPV 0.9978** |
| Robustness | leave-one-out spread 0.0125; 95% CI [0.1150, 0.1525] |
| Validated on this device | **NO** |

> **Only "REGULAR" is trustworthy enough to act on.** Its negative predictive
> value is ~0.998 — when it says steady, believe it. "IRREGULAR" has a positive
> predictive value of ~0.47: fewer than half of such calls are correct, so it
> **prompts a look and must never raise an alarm on its own.**

> **Limitation — what this step cannot see.** *Isolated ectopy is missed:* one or
> two premature beats in 64 give a score of 0.06–0.08, well under the threshold.
> Roughly 4+ ventricular or 5+ atrial premature beats are needed before a window
> tips over. *And it cannot say **why**.* An IRREGULAR result is equally
> consistent with atrial fibrillation, frequent premature beats, breathing-related
> variation, or movement artefact. The gaps between beats carry no P-wave
> information, so **no atrial fibrillation claim is ever made.**

---

## Step 10b — Two more checks alongside regularity

**IN:** the same 63 intervals.

**WHAT IT DOES:** two further checks run beside the steadiness one.

**Heart rate** — too slow, normal, or too fast. Bands 60-100 bpm normal,
under 40 or over 150 treated as extreme.

**Skipped beats and pauses** — counts short-then-long interval pairs (the
pattern a premature beat makes) and any gap over 2 seconds.

**OUT:** three separate answers instead of one, each able to say "cannot tell"
on its own.

**WHY separate:** a heart beating perfectly evenly at 150 bpm is dangerous, but
the steadiness check alone calls it fine. Seeing rate and steadiness together
catches it — confirmed on reference data, where it flagged about 1 window in 18
that a steadiness-only view passed as normal.

**IF DONE WRONG — a mistake we made and fixed.** We first made all three checks
depend on the same quality test. That made the *rate* check refuse in 97% of
windows, even though rate is reliable on this device. Rate is an average and
tolerates small beat-position errors; steadiness does not. Rate now has its own,
simpler quality test.

> **Limitation 1: the skipped-beat check cannot say WHY a rhythm is uneven.**
> Tested on labelled data: among windows of atrial fibrillation containing
> **zero** skipped beats, the check still reports skipped beats in **91.3%** of
> them. The short-gap-then-long-gap pattern is produced by *any* uneven rhythm,
> not only by skipped beats. So once a rhythm is already uneven, this check
> adds nothing about the cause. It is only meaningful when the rhythm is
> otherwise steady — which is the useful case anyway: spotting new skipped
> beats in a stable patient.
>
> **Limitation 2: it cannot count high numbers.** Below about 6 skipped beats
> per window it is accurate to within 1–2 beats. Above 10 it fails badly —
> average error of 16–24 beats. So "a few" is trustworthy; "a lot" is not a
> number to rely on.
>
> **Limitation 3: it does not work on patch data at all.** On healthy adults it
> reports skipped beats in **95.8%** of windows and pauses in 7.5%. Same
> detection fault: an extra false beat creates a short gap followed by a long
> one, which is exactly what a real skipped beat looks like. A missed beat
> merges two gaps into one, which looks like a pause.
> **Of the three checks, only heart rate survives on real patch data.**

**Trust: rate PROVISIONAL (works on device) · events PROVISIONAL
(UNTESTED-ON-DEVICE, currently failing).**

## Step 10c — Looking at the *shape* of each beat (public data only)

**IN:** the ECG waveform around each detected beat.

**WHAT IT DOES:** all the checks so far use only the *timing* of beats. This one
looks at the *shape*. A beat starting in the heart's lower chambers spreads
differently, producing a wider, differently-shaped complex — so shape can tell
you the beat came from an unusual place.

**OUT:** a count of beats identified as ventricular (PVC) in the window.

**WHY only that one type:** the other common early beat (from the upper
chambers, a PAC) travels the *normal* electrical path, so its shape is almost
identical to an ordinary beat. What distinguishes it is a tiny wave that is
often hidden inside the previous beat's tail and is unreliable to see on a
single lead. An earlier attempt at this scored 0.152 out of 1.0 for that type —
essentially unusable. **So the system counts upper-chamber beats from timing
instead, and never tries to identify them individually.**

**HOW WELL IT WORKS — on clean reference data, at native quality:**

| Measure | Result |
|---|---|
| Overall score (F1), tested on a different database | 0.63–0.73 |
| Previous attempt, same test | 0.513 |
| Precision in realistic use | **0.62** |

Better than the previous attempt. Nowhere near perfect.

**THE PART THAT MATTERS MOST.** How often it wrongly calls a beat a PVC depends
enormously on what the other beat is:

| Wrongly flagged as PVC | Rate |
|---|---|
| ordinary beats | 1.2–4.0% |
| **upper-chamber beats** | **4.2–17.1%** |
| **fusion beats** | **27–39%** |

It is **2–8 times worse** at exactly the comparisons that matter clinically.
Telling a ventricular beat from an ordinary one is easy; telling it from an
unusual upper-chamber beat is the hard part, and it is much weaker there.

**A prediction that was wrong, recorded because it changes the conclusion.** We
expected this check to break at the patch's lower sampling quality — too few
measurements to see the width difference. **It does not.** The score falls by
less than 2 points. So sampling quality is *not* what blocks this from the
patch; the unresolved lead question and the broken beat-finder are.

**Trust: UNTESTED-ON-DEVICE by definition.** This has only ever run on
reference recordings where the lead is known. It cannot run on the patch until
the lead question is answered.

## Step 11 — Don't let the display flicker

**IN:** a verdict every 8 beats (~6 seconds).

**WHAT IT DOES:** the displayed state only changes after **5 consecutive windows
agree**. It starts at "cannot determine" and asserts nothing until it has
evidence.

**OUT:** a stable state, changing at most every ~32 seconds.

**WHY:** consecutive windows share most of their beats, so one borderline
interval can flip the answer back and forth. A display alternating every few
seconds destroys clinical trust faster than being occasionally wrong.

**IF DONE WRONG:** measured on 187 stretches where the true rhythm never changed,
raw output flips **22 times per hour**; with 5-window smoothing, **8.4 times**.

> Honest note: there is **no natural stopping point** on that curve — each extra
> window buys a similar proportional reduction while adding delay. **5 is a
> judgement, not a derivation.**

**Trust: PROVISIONAL** (behaviour measured; the value chosen by judgement).

---

## Step 12 — Package the evidence

**IN:** the verdict and its supporting numbers.

**WHAT IT DOES:** writes one JSON file per window containing the verdict, the
eight evidence numbers, the threshold and refusal band, the quality score, the
provenance, and the scope statement.

**OUT:** for our window —

```json
"verdict": "UNABLE_TO_DETERMINE",
"evidence": { "n_beats": 64, "n_intervals": 63, "mean_rr_ms": 690,
              "heart_rate_bpm": 87, "rr_cv": 0.116, "rmssd_ms": 121,
              "n_intervals_flagged": 0, "pct_intervals_flagged": 0.0 },
"decision": { "threshold_rr_cv": 0.1275, "refusal_band": [0.1005, 0.1545],
              "hysteresis_n": 5 },
"quality":  { "bsqi": 0.95, "sqi_passed": true },
"provenance": { "threshold_status": "PROVISIONAL - NOT SHIPPABLE",
                "validated_on_device": false }
```

**WHY this exact list:** because of what happens next. The language model is
only allowed to use numbers that appear in this file — **so this file defines the
vocabulary of what can be said.** A number not here cannot be spoken. It is
built as an allow-list, not a data dump. Raw signal readings are deliberately
excluded.

**Trust: VALIDATED** (schema-checked; rejects raw-signal keys and oversized
arrays).

---

## Step 13 — The language model writes it up

**IN:** the JSON above, and the already-decided verdict.

**WHAT IT DOES:** MedGemma restates the verdict in two or three sentences.

**IT DOES NOT DECIDE ANYTHING.** The verdict is fixed before the model is
called. Its only job is wording.

**OUT:** a short paragraph — or, if anything is wrong with it, nothing at all.

**Trust: VALIDATED as a mechanism. UNTESTED with the real model** — the MedGemma
weights are not present on this machine and the server is not running, so every
narrative so far has come from the fixed template.

---

## Step 14 — Check what the model wrote

**IN:** the model's text.

**WHAT IT DOES:** four checks, in order.

1. **Every number** in the text must appear in the evidence file.
2. **The verdict word** must be present, correct, and **not negated**.
3. **Unicode is normalised** and non-standard letters rejected.
4. **A banned-phrase list** runs last, as a backstop only.

**IF ANY CHECK FAILS, THE ENTIRE TEXT IS DISCARDED** — no patching, no partial
acceptance — and a fixed template is used instead.

**WHY it is built this way:** an earlier review defeated a simple banned-word
list 9 times out of 9. Our own first version failed too, in ways worth recording:

| Attack | First version | Fix |
|---|---|---|
| "the rhythm is **not** regular" | accepted | reject negation before the verdict word |
| Cyrillic letter that looks like Latin | accepted | **standard Unicode normalisation does not merge these** — reject non-standard letters |
| invented "RR CV 0.0350" | accepted | match numbers at their own precision |
| "consistent with a healthy heart" | accepted | every sentence must contain a real number or the verdict |
| "no quivering of the upper chambers" | accepted | banned-phrase backstop |

**Current status: 16 of 16 attacks rejected.** But this is **an upper bound** —
it measures resistance to attacks written by the same person who wrote the
checker. The disguised-terminology category is closed by listing phrases, which
is inherently incomplete.

**Trust: VALIDATED as tested; incomplete by construction.**

---

## Step 15 — The clinical report

**IN:** verdict, evidence, narrative, and the signal picture.

**WHAT IT DOES:** produces a report with the ECG trace, the detected beats
marked on it, the interval series with any flagged values **shown in red rather
than hidden**, the features, the threshold, and the reason for any refusal.

**OUT — and this is the important part right now:**

```
output_class: ENGINEERING - rhythm verdicts with evidence;
              NO clinical severity, NO MedGemma report
```

**WHY:** a safety gate blocks clinical severity (Normal / Abnormal / Critical)
and the MedGemma report until beat timing has been checked against a reference
recording. The permission is read from a file written by the validation harness,
so it **cannot be switched on by editing a setting**.

> **Known gap in how this is displayed.** Sustained fast atrial rhythms are fast
> but *evenly spaced*, so they can read as REGULAR at an abnormal rate. At high
> premature-beat burden the irregular flag rate actually *drops*. **The rhythm
> verdict must therefore never be shown without the heart rate beside it.**

**Trust: UNTESTED-ON-DEVICE.**

---

## What the whole system is actually for

With five checks running (quality, rate, steadiness, skipped beats, shape), the
obvious question is whether combining them makes the answers better. Tested on
4,457 windows of reference data, the answer has two halves.

**It does NOT make "uneven" more trustworthy.** Measured against the same target
throughout:

| Output | Precision |
|---|---|
| Steadiness check alone | 0.50 |
| All checks combined | 0.22 |

Precision *falls*. That is not a fault — the extra checks find *different*
problems (a fast rate, skipped beats), which are not the same problem the
steadiness check is looking for. **"Uneven" is still not something to act on,
and must never trigger an alarm.**

**It DOES stop the system wrongly saying "all clear".**

| | Windows called normal | Genuinely significant among them |
|---|---|---|
| Steadiness alone | 2,159 | **739 (34.2%)** |
| All checks combined | 1,607 | **187 (11.6%)** |

**552 fewer windows given false reassurance — three quarters of what the
steadiness check alone missed.** Because of three things it structurally cannot
see:

- 350 windows steady but **fast** (up to 161 bpm)
- 339 windows steady but **slow** (median 56 bpm)
- 415 windows steady but containing **5 or more skipped beats**

### So what is this system?

**It is an instrument for withholding false reassurance, not for raising
alarms.**

- "Appears steady" is the trustworthy output — that is what can be acted on.
- "Uneven" prompts a look, never an alert.
- Its job is to stop saying "fine" when something is there.

That was demonstrated by measurement, not assumed.

### One limitation that applies to the whole system

Performance varies enormously **from patient to patient** — so much so that the
average is a poor guide to any individual. The skipped-beat count varies by
about two-thirds of its own average across patients; the shape check varies
almost as much.

This is one property of trying to judge individual beats from a single lead,
not several separate faults. **The practical consequence: no individual patient
can be promised that their abnormal beats will be caught.**

## The honest summary

| Step | Trust |
|---|---|
| 1 Patch measures voltage | VALIDATED (lead confirmed) |
| 2 Bluetooth packets | UNTESTED-ON-DEVICE (no patch since 2026-08-20) |
| 3 Remove re-sent data | VALIDATED |
| 4 Clean time base | VALIDATED (arithmetic) |
| 5 Find heartbeats | VALIDATED on public data · **WRONG on device (~4.4% extra)** |
| 6 Window of 64 beats | VALIDATED |
| 7 Screen impossible intervals | VALIDATED |
| 8 Calculate features | VALIDATED |
| 9 Quality gate | PROVISIONAL |
| 10 Threshold | **PROVISIONAL — NOT SHIPPABLE** |
| 10b Rate check | PROVISIONAL — **works on device** |
| 10b Skipped-beat check | PROVISIONAL — **fails on device**; cannot say *why* a rhythm is uneven; cannot count above ~10 |
| 10c Beat-shape check | **UNTESTED-ON-DEVICE** — public data only; blocked on the lead question |
| 11 Smoothing | PROVISIONAL (value is a judgement) |
| 12 Evidence file | VALIDATED |
| 13 Model narration | Mechanism VALIDATED · model absent |
| 14 Text checking | VALIDATED as tested; incomplete by construction |
| 15 Clinical report | UNTESTED-ON-DEVICE (gated shut) |

**Everything from step 5 onward inherits one unresolved problem.** On real patch
recordings the beat detector finds about 4.4% too many beats. Because each false
beat splits one gap into two, healthy adults currently score **0.19** for
variability where genuinely healthy people in reference data score **0.03** — six
times higher, on a healthier group.

So the system today says **"cannot determine"** for almost every window from a
real patch. **That is the safety gate working correctly, not a fault.** It is
refusing to report numbers it knows are corrupted.

**One thing fixes this: a controlled recording** — someone sitting still for ten
minutes wearing both the patch and a reference chest strap that records
beat-by-beat timings, plus two minutes of deliberate movement. That gives the
ground truth needed to correct the detector and *prove* it was corrected.

Until then, no result from this system on a real patch should be treated as
clinically meaningful.
