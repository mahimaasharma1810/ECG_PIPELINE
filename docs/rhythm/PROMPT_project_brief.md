# ProRhythm Rhythm Regularity — Problem, Data, and Requirements

> **STATUS NOTE (2026-08-30).** Written before the threshold re-derivation
> (now **0.1275** from 16 records) and before the device correction: the
> patch has **3 electrodes (RA, LA, LL) giving a 2-lead ECG**. Which lead we
> receive is OPEN. See `PATCH_AND_DATA.md`.


A self-contained brief. Anyone picking this up cold should be able to read only
this and know what the problem is, what exists, and what is needed next.

> **Scope statement, required verbatim in every output and document:**
> Rhythm regularity indicator. Not a diagnosis. Not validated for clinical use.
> Cannot distinguish atrial fibrillation from other causes of irregularity.

---

## 1. What we are building

A system that reads a single-lead ECG stream from a wearable chest patch
(ProRhythm / SeNSiO, 3 electrodes RA/LA/LL → 2-lead ECG) and classifies the
heart rhythm as **REGULAR**, **IRREGULAR**, or **UNABLE_TO_DETERMINE**, with
every verdict backed by evidence a cardiologist can independently check.

It works from **R-peak timing only** (the gaps between heartbeats). It uses no
waveform morphology. It is an indicator, not a diagnostic.

**Explicitly out of scope:** beat-by-beat morphology classification (V/S/F/N),
and any claim about atrial fibrillation. RR intervals contain no P-wave
information, so AF cannot be distinguished from benign sinus arrhythmia,
ectopy, or artefact. Only *regularity* is defensible.

---

## 2. THE CORE ISSUE

**We cannot currently measure rhythm on the target device, because R-peak
detection on its signal is wrong in a way that manufactures irregularity.**

The evidence, all measured:

- The pipeline finds **~4.4% more beats** than the device itself reports
  (pipeline median RR ÷ device median RR = 0.9578, IQR 0.9495–0.9652, across
  37 captures; pipeline HR exceeds device HR in 33 of 37).
- Each false beat splits one RR interval into two, so ~3 spurious peaks in a
  64-beat window is enough to produce **RR CV ≈ 0.15** from a perfectly steady
  rhythm. Simulated: 57 intervals at 700 ms plus 6 at ~350 ms gives CV 0.156.
- Observed on device data: **median RR CV 0.1464, RMSSD 146 ms, SDNN 100 ms**
  — on healthy, self-recording adults, where RMSSD should be 20–50 ms.
- For comparison, genuinely healthy windows in public labelled data have
  **median RR CV 0.0305**. The device figure is **6× higher on a healthier
  population**. That gap is detection error, not physiology.

**What has been ruled out as the cause:**

| Hypothesis | Test | Result |
|---|---|---|
| Wrong timing reconstruction | index-based vs clock-based vs interpolated | all give CV 0.16–0.23 |
| Wrong detector | 8 different published detectors | all give CV 0.18–0.32 |
| Implausible RR values | physiological screen 300–2000 ms | 0% flagged |
| Wrong mean rate | device HR cross-check | mean rate is correct |

**The actual cause**, confirmed visually: roughly 25% of detections land on
low-amplitude non-R deflections (amplitude ~20–25 against real R-waves at
~80–100), concentrated where signal quality swings within a capture. Motion
artefacts reach ~8× normal amplitude.

**Why this blocks everything:** the decision threshold must be derived from a
known-steady baseline on real hardware. Our only steady-rhythm hardware data is
contaminated by this error, so any threshold derived from it would be measuring
detection failure and calling it physiology.

**Why we cannot simply fix the detector:** we have **no ground truth** for any
patch recording. Without knowing where the beats truly are, we cannot tell a
corrected detector from one overfitted to noise.

---

## 3. WHAT DATA WE HAVE

### 3.1 Device recordings — ProRhythm patch (UNLABELLED)

- **37 captures, 4 subjects, 7.5M samples, 17.9 hours**
- Format: `timestamp_ms,amplitude,received_at_utc`
- **No rhythm labels of any kind.** All healthy adults, self-recorded, protocol
  unknown — i.e. a single class, which cannot define a decision boundary.

Measured properties (all verified, not assumed):

| Property | Value |
|---|---|
| True sample rate | **89.70 Hz** (range 87.96–90.50, CV 0.6%) |
| Amplitude units | **UNKNOWN** — not mV, probably ADC counts |
| Lead | Lead II (confirmed from device configuration) |
| Dropout | 10.2% of recorded span; gaps up to 22 minutes |
| Replayed data | **28 of 37 files** contain exact duplicate rows, up to 49% |
| Timestamps | **BLE packet-arrival stamps, not sampling instants** |

Two traps that must be respected by any new code:

1. **`n_samples / span` gives the wrong rate on 28 of 37 files** because of
   replayed rows. Correct order: remove exact duplicate `(timestamp, amplitude)`
   rows → sort → exclude gaps >1 s → then compute rate.
2. **Do not use per-sample timestamps as sampling times.** ~6.5 samples share
   one packet stamp, one packet per ~73 ms, so a sample's true time is unknown
   to ±36 ms. Using them puts the noise floor at **CV 0.095** — a 0.10
   threshold would sit on top of transport jitter. On a uniform 89.7 Hz grid
   the floor is **CV 0.006**.
3. **Do not apply a filter chain.** The firmware already filters (field name
   `ecg_clean`). A second chain leaves 6–7% of the amplitude, and this was
   confirmed independently: the unfiltered path scored *better* on public data.

### 3.2 Device vitals — paired snapshots (partial reference)

All 37 captures have a paired vitals file. **6,836 snapshots**, cadence ~12.5 s.
`heart_rate` 51–126 bpm; `hrv` is mean RR in ms.

**Limitation:** `hrv` is heavily smoothed (lag-1 autocorrelation +0.656). It
validates **mean rate only** and can never validate beat-to-beat variability —
which is exactly the quantity in dispute. This is why it is not sufficient
ground truth.

### 3.3 Public labelled data (used for the threshold)

| Database | Records | Use |
|---|---|---|
| LTAFDB | 9 (of 84; download ongoing) | threshold derivation, 24-h continuity |
| MITDB | 48 | detector validation + held-out-database test |
| SVDB | 78 | proxy check only — has 1 rhythm annotation total |
| INCARTDB, CUDB | 75, 35 | not used |

---

## 4. WHAT IS ALREADY ESTABLISHED (do not redo)

- **R-peak detection is validated on public data**: Se 0.9915, PPV 0.9865,
  F1 0.9890 over 105,078 expert-annotated beats, at the device's 89.7 Hz.
- **A provisional threshold exists**: RR CV ≥ 0.12, derived from LTAFDB at
  89.7 Hz using expert annotations.
- **It transfers across databases**: held out on MITDB, Se 0.9839, Sp 0.8408,
  PPV 0.4266, NPV 0.9977.
- **But it is imprecise**: 95% CI [0.1025, 0.1777] by record-level bootstrap,
  because the positive class rests on 4 patients, 2 of which supply 76% of it.
  It is marked **PROVISIONAL — NOT SHIPPABLE**.
- **Operating-point design decided**: operate at the Youden point and make
  **only the REGULAR verdict actionable** (NPV ~0.998). IRREGULAR (PPV ~0.40)
  prompts a review and must never fire an alarm.
- Full pipeline exists: ingest → uniform grid → signal-quality gate → detection
  → RR screen → features → verdict → hysteresis → validated narrative →
  clinician review package with figures.

---

## 5. WHAT WE NEED

### 5.1 PRIMARY REQUIREMENT — a controlled recording with beat-level ground truth

This is the single blocking item for the entire project.

**Protocol:**

- One existing subject ID, wearing the **ProRhythm patch** and a **reference
  device that exports per-beat RR intervals** (e.g. Polar H10 chest strap,
  ~1 ms RR resolution) **at the same time**
- 5 minutes rest before recording starts
- **10 minutes seated, still, no talking**, electrode contact verified
- **2 minutes of deliberate movement at the end** — this gives a labelled
  artefact segment for tuning the quality gate
- **Wall-clock time noted at both device start times**, so the two streams can
  be aligned

**What this unblocks:** a true beat-by-beat reference lets us (a) measure how
wrong the detector actually is, (b) fix it and verify the fix rather than
guessing, (c) build the steady-rhythm baseline on real hardware, and
(d) recalibrate the refusal band on the right population.

**Without it, none of the device-side work can proceed honestly.**

### 5.2 SECONDARY — device firmware/transport logging

Add to each transmitted packet: **sequence number, packet counter, and
samples-per-packet.**

This makes replayed data explicit instead of inferred from ~13.1 s backward
timestamp steps, and turns dropped samples from an estimate into a count. It
does not help the existing 37 captures but compounds for every future one.

### 5.3 Useful but not blocking

- More LTAFDB records (target ~20) to narrow the threshold confidence interval
- Confirmation of the **amplitude units** from device documentation
- Any recording from a subject with a **known irregular rhythm** on this
  hardware — currently the entire hardware dataset is one class

---

## 6. NON-NEGOTIABLES

- The system must be able to refuse. `UNABLE_TO_DETERMINE` is a first-class
  output, not a failure.
- Every verdict carries its evidence: the ECG strip, the detected beats, the RR
  series with outliers **shown not hidden**, the features, the threshold, the
  quality score, and the reason for any refusal.
- **No borrowed threshold** without re-derivation at matched timing resolution.
- **No second filter chain** on device data.
- **No atrial fibrillation claim** anywhere — code, output, logs, or docs.
- **The language model never classifies.** The verdict is decided
  deterministically before any model is called; the model only restates it, and
  its output is structurally validated or discarded entirely.
- Report **held-out-database** results, never just held-out-record.
- Never lead with accuracy. Report sensitivity, specificity, PPV, NPV, and
  confusion matrices.
- **Report unmeasured things as unmeasured, in the headline, not a footnote.**
  Accuracy on the target device is currently **UNKNOWN**.
