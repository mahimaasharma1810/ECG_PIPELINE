# Patch team answers — received 2026-09-02, and what they change

Answers to `PATCH_TEAM_QUESTIONS.md`, with a supporting filter-methods document
from the device side.

> Rhythm regularity indicator. Not a diagnosis. Not validated for clinical use.
> Cannot distinguish atrial fibrillation from other causes of irregularity.

---

## Answer 1 — Lead: **Lead II, both raw and clean. RESOLVED.**

> "Ecg raw and clean both are lead 2"

**What this unblocks:**

- **Our detector validation is now like-for-light.** We validated against
  **MLII** (modified Lead II) in MITDB on the assumption the patch streams Lead
  II. That assumption is confirmed, so the F1 0.9890 result stands as a
  like-for-like comparison rather than a provisional one.
- **Morphology (L5) loses its lead blocker.** Waveform shape is lead-dependent,
  and "which lead reaches us" was one of three reasons L5 could not go near the
  device. That reason is gone.

**What it does NOT resolve:** the second-lead question. The patch has 3
electrodes (RA, LA, LL), which defines a 2-lead ECG, but every file and every
packet carries **one** channel. Whether a second lead exists anywhere in the
stack is still open — and it still matters, because two physically independent
leads would be a far stronger quality check than two algorithms on one waveform.

**L5 now blocks on device beat detection alone**, not on lead or resolution.

---

## Answer 2 — Native sampling rate: **120–140 Hz. CONSISTENT with our measurement.**

> "The sampling rate is around 120-140 hz"

This confirms the 2026-09-02 session measurement of **133.83 Hz**, and confirms
that the historical captures' **89.70 Hz is a DELIVERY rate, not a sampling
rate** — the path was delivering two samples in three.

Independently corroborated here by a spectral test: the QRS-band centroid of
device data sits at 7.17 Hz where real Lead-II ECG resampled to 89.70 Hz sits at
11.45 Hz. The ratio 1.597 implies ~143 Hz — inside the vendor band.

### A wrong conclusion, drawn and retracted

On receiving this answer it was proposed that the missing third of samples was
**the root cause** of the device's inflated RR variability, on the reasoning
that irregularly-spaced survivors treated as uniform would distort the waveform.

**Simulation refuted it.** Real ECG at 133.83 Hz, survivors treated as evenly
spaced:

| Condition | median RR CV |
|---|---|
| No loss | 0.0485 |
| **Systematic 2:3 decimation** | **0.0490** — no meaningful effect |
| Random 1-in-3 loss | 0.1008 — doubles it |
| **Device observed** | **0.1681** |

The device does **systematic decimation**, where survivors remain evenly spaced,
not random loss. The delivery path therefore does **not** explain the inflated
variability. This matches the existing finding in `rhythm/degrade.py`, which the
retracted claim contradicted.

**The root cause remains what it was: R-peak detection error on device data —
motion artefact and low-amplitude false peaks.** Unchanged by these answers.

---

## Answer 3 — Filter methods document

The device-side document specifies standard ECG filtering (IIR high-pass
0.05 Hz; low-pass 35–45 Hz elliptic/equiripple) and then states:

> "the measured SNR of our proposed ECG system, which required P, QRS and T wave
> evaluation at **100 Hz sampling frequency, also needed no additional filters**
> to reject unwanted noises from hardware and the human body"

**This supports the pipeline's existing behaviour.** We apply no filtering, for
reasons measured independently:

| Arm | Se | PPV | F1 |
|---|---|---|---|
| 360 Hz, filtered | 0.9883 | 0.9821 | 0.9852 |
| 89.7 Hz, filtered | 0.9877 | 0.9785 | 0.9831 |
| **89.7 Hz, unfiltered** | **0.9915** | **0.9865** | **0.9890** |

Skipping the chain won on 20 of 46 records. A second chain on `ecg_clean` was
measured to leave 6–7% of the amplitude.

**So the device document and our measurement agree**, and the `BUILD_RHYTHM_MODEL.md`
instruction to "filter noise using light ECG bandpass" should not be followed —
it is contradicted by the device team's own document as well as by our data.

One number worth noting: the document assumes **100 Hz**. The device is stated
at 120–140 Hz and measured at 133.83. The filter specs were designed for a rate
the device does not appear to run at.

---

## Status after these answers

| Question | Status |
|---|---|
| Q1 lead | **RESOLVED — Lead II** |
| Q1b second lead available? | **STILL OPEN** |
| Q2 controlled recording | **STILL OPEN — the blocker** |
| Q3 `sNo` in captures | **STILL OPEN** |
| Q4 sample rate | **RESOLVED — 120–140 Hz native, 89.7 delivered** |
| Q6 firmware filtering | **RESOLVED — no additional filtering needed** |

**Two blockers removed from L5 (lead, resolution). One blocker remains for
everything device-side: beat detection, which needs the controlled recording.**
