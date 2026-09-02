# Lead-II Rhythm Regularity Pipeline — Findings Report

> **STATUS NOTE (2026-08-30).** Parts I-IV predate two changes. (1) The
> threshold was re-derived on 16 LTAFDB records and is now **0.1275**, not
> 0.12; held-out MITDB is now Se 0.9839 / Sp 0.8660 / PPV 0.4692 / NPV
> 0.9978. (2) The device is **3 electrodes (RA, LA, LL) giving a 2-lead
> ECG**, not 2 electrodes giving Lead II. Which lead reaches us is still
> OPEN. See `PATCH_AND_DATA.md` and Part VI below.


**Project:** ProRhythm / SeNSiO wearable patch, Lead-II rhythm regularity indicator
**Report date:** 2026-08-27
**Status:** Steps 1–3 complete. Step 4 halted on a validated blocker. Steps 5–8 not reached.

> **Rhythm regularity indicator. Not a diagnosis. Not validated for clinical
> use. Cannot distinguish atrial fibrillation from other causes of irregularity.**

---

## 0. Headline

**Accuracy of this system on the target device is UNKNOWN and currently unmeasurable.**
There are no rhythm labels for any ProRhythm capture, and R-peak detection on device
data has been shown to be wrong in a way that manufactures irregularity. No threshold
has been derived, and none should be until that is fixed.

Three results are established and reusable:

1. The device's true sample rate is **89.70 Hz** (CV 0.6% across 37 captures) — but only
   after removing replayed rows, which 28 of 37 captures contain.
2. `timestamp_ms` is a **BLE packet-arrival stamp**, not a sampling instant. Using it as
   the brief instructs would put the RR noise floor at **CV 0.095**, on top of which no
   regularity threshold can survive.
3. R-peak detection is **validated on MITDB** (F1 0.9890, 105,078 expert beats) and
   **invalid on device data** (~4.4% excess beats, producing CV ≈ 0.15 out of nothing).

---

## 1. Status against brief §12

| Step | Status | Evidence |
|---|---|---|
| 1. Ingest, measure effective `fs`, verify §3.2 | **Done** — §3.2 falsified as n=1 | §3, §4 |
| 2. Validate detector vs MITDB `.atr` (gate) | **PASSED** — Se 0.9915, PPV 0.9865, F1 0.9890 | §7 |
| 3. HR cross-check vs device, median-RR method | **Done** — ratio 0.9578, IQR 0.9495–0.9652 | §10 |
| 4. REGULAR distribution from ProRhythm | **HALTED** — distribution is detection artefact | §11 |
| 5. IRREGULAR distribution from LTAFDB | **BLOCKED** — dataset absent, rerouted | §12.1 |
| 6. Derive and justify threshold | **Done** — RR CV 0.1200 from LTAFDB at 89.70 Hz | §19 |
| 7. Track B, pseudo-labelling | Deferred with re-entry condition | §13 |
| 8. Clinician review package | **Built and generated** | §22 |

### Non-negotiables (§11)

| Requirement | Held? | Note |
|---|---|---|
| System can refuse (`UNABLE_TO_DETERMINE`) | Yes | First-class output, Stage 3 + Stage 5 |
| Every verdict carries evidence | Yes | Per-window SQI, features, reason string |
| No borrowed threshold | Yes | None imported; `cv=0.10` re-refuted quantitatively (§6.4) |
| No second filter chain on device data | Yes | Confirmed *beneficial* independently (§7.3) |
| No AF claim | Yes | Absent from code, output, logs, docs |
| LLM never decides | **Yes** | Verdict decided in Stage 7 before any model call; §21 |
| Held-out-**database** results | Partial | MITDB reported; device transfer failure documented (§11) |
| Report unmeasured as unmeasured | Yes | §0 headline |

---

## 2. Environment and data inventory

### 2.1 Software (pinned for reproducibility)

| Package | Version |
|---|---|
| Python | 3.10 |
| wfdb | 4.3.1 |
| numpy | 1.26.4 |
| scipy | 1.15.3 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| neurokit2 | 0.2.13 |
| matplotlib | 3.10.9 |
| torch | 2.12.1+cpu |

Hardware: NVIDIA GeForce GTX 1080 Ti (11 GB). No GPU work performed to date.

### 2.2 Device data — ProRhythm live captures

`prorithm_ecg/{1049,1789,1793,1794}/YYYY-MM-DD_HH.csv`

- 37 capture files, 4 subjects
- 7,508,399 samples, 64,541 s span (17.9 h)
- Format: `timestamp_ms,amplitude,received_at_utc`, header row 1
- **No rhythm labels.** All healthy adults, self-recorded, protocol unknown.

The SeNSiO device-software export format (11 metadata rows, header at row 16) is
**out of scope** — confirmed with the project owner. Only the live format is parsed;
anything else raises `FormatError` rather than being guessed at.

### 2.3 Device vitals — paired snapshots

`data/raw/cliniaura_live/vitals/{1049,1789,1793,1794}/YYYY-MM-DD_HH.csv`

- 38 files, 6,836 snapshots; **all 37 ECG captures have a paired vitals file**
  (one orphan vitals file, `1794/2026-08-07_11.csv`, has no ECG counterpart)
- Columns: `received_at_utc, snapshot_timestamp, heart_rate, spo2, systolic_bp,
  diastolic_bp, respiration_rate, temperature, hrv`
- Snapshot cadence median 12.47 s
- `heart_rate` 51–126 bpm, integer-valued, 65 distinct values
- **`hrv` is mean RR in ms** (verified, §10.1)
- **`temperature` spans [36.8, 99.8] — mixed °C/°F, unusable.** Not needed.

### 2.4 Public databases

| Database | Status | Records | Notes |
|---|---|---|---|
| mitdb | Available | 48 | 360 Hz; lead verified per record (§7.1) |
| svdb | Available | 78 | Supraventricular arrhythmias |
| incartdb | Available | 75 | |
| cudb | Available | 35 | VF; not useful for regularity |
| challenge2017 | Available | — | Median 30 s — too short for 64 beats |
| **ltafdb** | **PARTIAL** | **6 of 84** | Re-download stopped at 6 complete records (146 h). Headers do NOT name leads (§12.1) |
| **icentia11k** | **MISSING** | 0 | Same cause |
| **sddb** | **MISSING** | 0 | Same cause |

---

## 3. Stage 1 — Ingest and measured timing

### 3.1 Brief §3.2 describes one file, not the device

§3.2's table matches `1789/2026-08-20_10.csv` **exactly**: 140,831 samples, 1,570.0 s,
89.70 Hz, 46.6% duplicate timestamps, amplitude [−220.49, +210.60]. Measured across all
37 captures, none of it generalises:

| Property | §3.2 claim | Measured across 37 captures |
|---|---|---|
| Effective `fs` (raw `n/span`) | 89.7 Hz | median 117.26, **range 51.08–166.53** |
| Duplicate timestamps | 47% | median 50.0%, **range 15.6–74.8%** |
| Amplitude range | −220.5 to +210.6 | **−1439.55 to +998.86** (pooled) |
| Monotonic timestamps | not mentioned | **29 of 37 files are NON-monotonic** |
| Intra-capture gaps | not mentioned | **up to 1,323 s (22 min)** |

`median(dt)` returns 1000.0 Hz (or `inf` where >50% of `dt` are 0). Trap 1 confirmed
emphatically — never infer `fs` this way.

### 3.2 Replay duplication — 28 of 37 captures

28 captures contain **exact duplicate `(timestamp, amplitude)` rows**, up to **49.3%** of
the file. Signature: a **~13,100 ms backward timestamp step** paired with a **~13.2 s
forward gap** — a ~13 s block is re-delivered.

The correlation is perfect: every file with backward steps has replayed rows; every file
without has none. 1,949 backward steps total, max magnitude 14,751 ms.

**Effect on the measured rate:**

| | median | range | CV |
|---|---|---|---|
| `n / active_time` as-recorded | 130.18 Hz | 87.96–178.44 | **24.0%** |
| after exact-duplicate removal | **89.70 Hz** | 87.96–90.50 | **0.6%** |

A CV of 0.6% is what a real fixed-rate device looks like. 24% is not.

**Corrected ingest order (replaces §3.2's `n/span` rule, which is wrong on 28/37 files):**

```
dedupe exact (timestamp, amplitude) rows
  -> sort (stable, repairing reorder)
  -> exclude dropout gaps > 1 s
  -> fs = n / active_time
```

Duplicate *timestamps* are a separate matter from duplicate *rows*: in replay-free files,
**0%** of multi-sample timestamp groups have identical amplitudes — they are genuinely
distinct samples that collided on a coarse clock. In `1794/2026-08-09_12.csv` (74.8% dup),
32.0% of groups are byte-identical — that file carries replayed content.

Active recording time: 57,978 s (16.1 h) of 64,541 s span. **Dropout = 6,563 s (10.2%).**

### 3.3 Timestamps are BLE packet-arrival stamps

Timestamps are genuinely 1 ms resolution (all residues mod 100 present), but:

- ~50% of consecutive `dt` are **0 ms**; ~27% are 1 ms
- mean `dt` = 11.148 ms → 89.70 Hz
- samples arrive in **bursts**: mean 6.55 samples per burst, one burst per **72.97 ms**
  (`1789/2026-08-20_10`); 4.12 samples per 45.78 ms in `1789/2026-08-20_09`

A sample's true sampling instant is therefore unknown to **±36 ms** — half the burst
window. These are packet-arrival times, not sampling instants.

---

## 4. The RR-timing noise floor (research Q2)

Simulated on the **real burst structure** of a real capture, for a **perfectly regular**
rhythm. This is the floor below which no threshold can be defended.

| RR (HR) | (A) uniform 89.7 Hz grid | (B) per-sample packet stamps |
|---|---|---|
| 600 ms (100) | CV 0.0071, RMSSD 6.65 ms, SDNN 4.26 | CV **0.0949**, RMSSD 99.51 ms, SDNN 56.92 |
| 700 ms (86) | CV 0.0065, RMSSD 7.18 ms, SDNN 4.52 | CV 0.0821, RMSSD 100.34 ms, SDNN 57.44 |
| 800 ms (75) | CV 0.0059, RMSSD 7.68 ms, SDNN 4.74 | CV 0.0705, RMSSD 96.37 ms, SDNN 56.43 |
| 1000 ms (60) | CV 0.0051, RMSSD 8.59 ms, SDNN 5.09 | CV 0.0573, RMSSD 101.01 ms, SDNN 57.33 |
| 1200 ms (50) | CV 0.0044, RMSSD 9.41 ms, SDNN 5.34 | CV 0.0459, RMSSD 93.36 ms, SDNN 55.03 |

### 4.1 This re-refutes `cv = 0.10` quantitatively

The brief (§5.1) rejected the literature threshold `cv = 0.10` on **provenance** grounds
(wrong device, population, resolution). The measurement gives a stronger reason:

**Under packet-stamp reconstruction, the noise floor for a perfectly regular rhythm at
HR 100 is CV 0.0949.** A `cv = 0.10` threshold would sit *on* the noise floor. Anyone
adopting it would classify BLE transport jitter as rhythm irregularity and see
plausible-looking positives.

On a uniform grid the floor is 0.006, leaving ~17× headroom — and that 17× is a
**ceiling** on headroom, because the ±36 ms is averaged down, not eliminated.

---

## 5. Stage 2 — Uniform time grid (deliberate brief deviation)

### 5.1 The deviation

Brief §6 Stage 2 instructs resampling "using **real per-sample timestamps**, not an
assumed fixed rate". For this device that is backwards, for the reason in §4: it injects
±36 ms of transport jitter into every R-peak.

**Decision: build the grid from sample INDEX, with the rate measured locally from the
clock.** RR intervals are differences, so an absolute time offset cancels; what survives
into RR is the local sample *rate*.

```
RR = (index difference) / fs_local
```

The hardware sample clock is far more stable than BLE arrival stamps, so index
differences carry the timing and the stamps are used only to (a) measure `fs` and
(b) detect dropped samples.

### 5.2 Anchoring — the load-bearing choice, and what was rejected

**Rejected: one linear time model per segment.** Over a 1,572 s segment it left residuals
of **1,709.5 ms RMS (max 3,612.7 ms)** — the rate drifts, so a single model does not
describe the clock.

**Rejected: short local windows.** Any window ≤ 2,000 samples (~22 s) produced
**non-monotone** time models — time running backwards — which is unusable. Residual RMS
was minimised near W = 1,000 samples (68.7 ms) but monotonicity fails there.

**Measured local-rate stability** over consecutive 60 s blocks — the residual RR error
this choice carries:

| Capture | fs CV @60 s | p1 |
|---|---|---|
| `1789/2026-08-17_08` | 0.27% | 89.08 |
| `1794/2026-08-09_12` | 0.20% | 90.09 |
| `1793/2026-08-08_07` | 0.94% | 86.37 |
| `1789/2026-08-20_10` | 1.06% | 86.84 |

The low p1 values mark blocks where the clock outruns the sample count — **genuine
dropped samples**. A window-level rate silently redistributes those, so they are
**detected** (`Segment.drop_samples`) and flagged, never smoothed away.

### 5.3 No filtering, anywhere, on device data

The firmware field is named `ecg_clean`. Brief §3.2 trap 3 measured that a second chain
left only 6–7% of amplitude. This is confirmed independently in §7.3 — the *unfiltered*
arm scored **best** on MITDB at device resolution.

---

## 6. Stages 5–6 — RR screen and features

Implemented in `rhythm/features.py`.

- **64 beats == 63 intervals** routed through `INTERVALS_PER_WINDOW = BEATS_PER_WINDOW - 1`.
  Never hand-computed (§5.6).
- **Physiological bounds** 300–2000 ms. Outliers are **flagged and counted**, never
  silently dropped.
- **> 20% flagged → `UNABLE_TO_DETERMINE`.** A verdict is never forced by deleting
  inconvenient values.
- **Segment-aware RMSSD** (§5.5): the series is segmented at flags and successive
  differences taken *within* segments only.

**Unit-verified:** 63 intervals with one flagged → 2 segments, **60 RMSSD pairs (not 61)**,
RMSSD exactly 0.0 on a regular series. The hole is not spanned.

Features per window: mean RR, HR, RR CV, RMSSD (segment-aware), SDNN, flagged count and
fraction, segment count.

---

## 7. Step 2 — Detector validation against MITDB (**the gate**)

### 7.1 Lead verification (§4.4) — not assumed

All 48 MITDB headers read. Channel combinations:

| Channels | Records |
|---|---|
| (MLII, V1) | 40 |
| (MLII, V5) | 2 |
| (MLII, V2) | 2 |
| (MLII, V4) | 1 |
| (V5, V2) | 2 |
| (V5, MLII) | 1 |

- **Records 102 and 104 have no MLII channel at all** → **EXCLUDED** from a
  Lead-II-matched analysis.
- **Record 114 carries MLII on channel 1, not 0** → channel selected by name.
- Naive `channel 0` would have silently validated on chest leads.

**Lead-II-equivalent set: 46 records.** All at 360 Hz.

### 7.2 Results — 46 records, 105,078 expert-annotated beats, ±150 ms (ANSI/AAMI EC57)

| Arm | Se | PPV | F1 |
|---|---|---|---|
| A — 360 Hz, cleaned | 0.9883 | 0.9821 | 0.9852 |
| B — 89.7 Hz, cleaned | 0.9877 | 0.9785 | 0.9831 |
| **C — 89.7 Hz, no extra filter (device path)** | **0.9915** | **0.9865** | **0.9890** |

Excluding the 4 paced records (102, 104, 107, 217), 44 records, 100,733 beats:
A 0.9878/0.9813/0.9846 · B 0.9873/0.9776/0.9824 · **C 0.9912/0.9859/0.9885**.

**GATE PASSED.**

Accuracy is deliberately not reported — rhythm classes are heavily imbalanced and the
metric is meaningless here (§5.9).

### 7.3 Two findings from the arm comparison

**Resolution matching is nearly free.** 360 → 89.7 Hz costs F1 0.9852 → 0.9831.

**The second filter chain is not.** Arm C beats arm B on **20 of 46 records** (worse on 8).
Biggest gains from *skipping* the chain: record 113 **+0.1308 F1**, 207 +0.0427,
108 +0.0371, 203 +0.0156. At 89.7 Hz Nyquist is 44.85 Hz, so a standard chain sits on the
edge of the band it is meant to preserve. This is independent confirmation of §3.2 trap 3.

### 7.4 The dominant failure is OVER-detection

Four records below F1 0.95 (arm C):

| Record | Se | PPV | F1 | Failure |
|---|---|---|---|---|
| 113 | 0.9994 | 0.7299 | 0.8436 | **664 false positives** |
| 207 | 0.8032 | 0.9670 | 0.8775 | 366 false negatives |
| 231 | 1.0000 | 0.7967 | 0.8868 | **401 false positives** |
| 108 | 0.9302 | 0.9329 | 0.9316 | both |

**A false R-peak splits one RR interval into two, manufacturing irregularity.** This is the
mechanism that connects detector error directly to false IRREGULAR verdicts.

### 7.5 How much irregularity does detector error manufacture?

Measured per 64-beat window, expert-annotation CV vs detector CV, MITDB at device
resolution, arm C. **1,619 windows across 46 records**, windows defined on expert beats
and paired by identical time span.

| Detector-induced \|CV error\| | Ungated |
|---|---|
| p50 | 0.0017 |
| p90 | 0.0423 |
| p95 | 0.0802 |
| **p99** | **0.3123** |
| max | 0.3630 |
| > 0.05 | 8.6% of windows |

Record 113 goes from expert CV 0.099 to detector CV 0.349 — a near-regular rhythm rendered
wildly irregular by false peaks.

> **Methodological note, recorded because it changed the numbers.** An earlier version of
> this diagnostic paired ref/det windows by *start time within 5 s*. On badly-detected
> records the two window streams desynchronise **because** of over-detection — record 113
> retained only 2 of ~28 windows. That version reported p99 = 0.0914. It was biased toward
> well-detected records and **understated the problem by 3.4×**. The corrected figure is
> p99 = 0.3123. The biased table is superseded and should not be cited.

---

## 8. Stage 3 — Signal Quality Index gate

### 8.1 Design rationale

The failure to catch is over-detection (§7.4). The primary component is therefore **bSQI**
— Clifford's agreement between two independent R-peak detectors, `|matched| / |union|` —
because that is precisely what collapses when peaks are spurious.

Supporting components: kSQI (kurtosis), pSQI (QRS-band power fraction), flatline fraction,
saturation fraction. **None use annotations**, so all are computable on device data.

bSQI unit-verified: 1.0 identical, 1.0 under ±10 ms jitter, 0.8 with 20% missed,
0.6696 with 50% extra false peaks, 0.0 unrelated.

### 8.2 bSQI predicts induced error strongly

| bSQI bin | n | median \|err\| | p95 | max |
|---|---|---|---|---|
| < .5 | 7 | 0.1937 | 0.2495 | 0.2621 |
| .5–.7 | 33 | 0.3000 | 0.3617 | 0.3630 |
| .7–.8 | 26 | 0.1017 | 0.3114 | 0.3143 |
| .8–.9 | 63 | 0.0357 | 0.2620 | 0.2804 |
| .9–.95 | 75 | 0.0174 | 0.1804 | 0.1966 |
| ≥ .95 | 1415 | **0.0014** | 0.0381 | 0.1115 |

Threshold sweep:

| `bsqi_min` | kept | kept % | p95 | p99 | max | > 0.05 |
|---|---|---|---|---|---|---|
| 0.00 | 1,619 | 100.0% | 0.0802 | 0.3123 | 0.3630 | 8.6% |
| 0.80 | 1,556 | 96.1% | 0.0530 | 0.1708 | 0.2804 | 5.7% |
| 0.90 | 1,493 | 92.2% | 0.0450 | 0.0888 | 0.1966 | 4.0% |
| **0.95** | **1,415** | **87.4%** | **0.0381** | **0.0629** | 0.1115 | 2.8% |

### 8.3 A negative result worth keeping: `frac_short` is useless

Fraction of intervals below half the window median: **median 0.0000 on both good and bad
windows.** At 89.7 Hz, over-detection does not produce sub-half-median intervals.
**Dropped from the gate.**

### 8.4 bSQI's blind spot, and the component that covers it

bSQI only detects **uncorrelated** detector disagreement. When both detectors make the
same T-wave error, bSQI stays high. **39 windows across 19 records** pass `bsqi ≥ 0.95`
and are still wrong — record 231 has median bSQI 0.9538 with max error 0.3175.

**RR lag-1 autocorrelation** covers this: inserting a spurious peak produces short/long
alternation.

| | good windows | bad windows |
|---|---|---|
| lag-1 (all) | +0.0113 | −0.0650 |
| lag-1 (among bSQI ≥ .95) | +0.0202 | −0.0821 |

### 8.5 **BUG CAUGHT: unguarded lag-1 rejects the MOST regular rhythms**

On the 11.148 ms grid, a **perfectly regular** rhythm alternates between exactly two
quantisation levels (e.g. 791.53 / 802.68 ms at RR = 800), producing strongly negative
lag-1 — an artefact of quantisation, not of rhythm:

| RR (ms) | SDNN (ms) | lag-1 |
|---|---|---|
| 600 | 4.302 | −0.2220 |
| 800 | 4.781 | −0.3195 |
| 1000 | 5.129 | −0.4266 |
| **1200** | 5.371 | **−0.5619** ← fails a −0.5 gate |

The effect is **worst at low heart rates**. It dissolves once real variability exceeds the
quantisation floor (added SD 10 ms → lag-1 −0.0733; SD 40 ms → −0.0182).

**Fix: a variance guard — apply lag-1 only when SDNN > 15 ms.**

**MITDB structurally cannot test this.** Every MITDB window has SDNN > 10 ms (median 64 ms
good, 207.7 ms bad), because those records are arrhythmia-rich. On MITDB the guard is
**inert** — guarded and unguarded results are identical. The guard is therefore
**validated by simulation, not by MITDB**, and exists to protect *healthy device data*,
which is exactly the population MITDB lacks.

### 8.6 Final gate

```
bSQI >= 0.95  AND  (SDNN <= 15 ms  OR  lag1 >= -0.5)
plus kSQI >= 2.0, pSQI >= 0.10, flatline <= 30%, saturation <= 20%
```

**MITDB performance:** retains 78.4% (1,270 of 1,619); induced \|CV error\| p95 **0.0267**,
p99 **0.0603**, max 0.1115, > 0.05 in 2.1%.

**The gate cannot be made perfect.** A residual max error of 0.1115 survives. That residual
is a stated limitation and should size an `UNABLE_TO_DETERMINE` margin band around any
future threshold — converting a measured weakness into a principled refusal.

### 8.7 Amplitude-unit audit

Rhythm maths is timing-only, so §3.2's unconfirmed-units caveat is a non-issue — but the
SQI components read amplitude. Audited: **every amplitude-touching constant is
scale-relative.** Kurtosis is scale-invariant; pSQI is a power *ratio*; flatline uses
`0.001 × std`; saturation uses `1e-6 × range`; the amplitude-consistency filter uses
`h >= frac × rolling_median(h)`. The only absolute constant is a `1e-12` degenerate-case
floor that fires solely on a constant signal. **No unit assumption entered the pipeline.**

---

## 9. Stage 3 gate on device data — it does not transfer

Running the MITDB-derived gate on all 37 captures (1,210 windows):

| | |
|---|---|
| Passed | **221 (18.3%)** |
| Refused | **989 (81.7%)** |

| Subject | Windows | Passed | Pass rate | median bSQI |
|---|---|---|---|---|
| 1049 | 47 | 4 | 8.5% | 0.922 |
| 1789 | 406 | 102 | 25.1% | 0.923 |
| 1793 | 314 | 33 | 10.5% | 0.783 |
| 1794 | 443 | 82 | 18.5% | 0.922 |

**The refusals are not signal quality.** Refusal bSQI values cluster just below threshold
(0.938, 0.922, 0.906). The cause: on device data `pantompkins1985` finds **1,093 peaks**
where `neurokit` finds **2,517** on the same segment — **35.5 bpm vs 81.8 bpm**. Detector B
misses roughly half the beats, so **bSQI is measuring detector B's failure, not the
signal's**.

This is a **cross-database transfer failure of the gate itself** (§5.8 class). Both halves
of bSQI must be verified to work on the target device before the ratio means anything.

---

## 10. Step 3 — Device HR cross-check

### 10.1 `hrv` is mean RR in ms

Verified across 6,754 paired snapshots. `60000/hrv` tracks `heart_rate`, but the two are
**largely independent estimates**, not one derived from the other:

| Test | Result |
|---|---|
| `round(60000/hrv) == heart_rate` | 12.2% |
| correlation | 0.7237 |
| residual `hr − 60000/hrv` | +1.64 bpm, **sd 7.94** |

So the cross-check is **not circular**. But `hrv` is **heavily smoothed** (lag-1 autocorr
+0.6561; successive median \|diff\| 6.5 ms) — it validates **mean rate only and can never
validate beat-to-beat variability**, which is the quantity in dispute.

### 10.2 Result — median RR durations (sidesteps clock alignment)

| Arm | Captures | RR ratio (pipeline/device) | Excess beats |
|---|---|---|---|
| All windows | 37 | **0.9578**, IQR [0.9495, 0.9652] | **+4.4%**, range −11.8% to +11.7% |
| SQI-passed | 29 | 0.9552, IQR [0.9418, 0.9646] | +4.7%, range −0.8% to +37.2% |

Pipeline HR > device HR in **33 of 37** captures. \|HR difference\| > 5 bpm in only 4 of 37.

**The mean rate is essentially correct; the beat count is ~4.4% too high.**

### 10.3 The excess is systematic, not motion-driven

Tested per-window (1,191 windows paired to a device snapshot within 30 s). Median
pipeline − device = **+2.47 bpm**, sd 7.63.

Correlation of \|HR difference\| with quality is weak (Spearman: bSQI −0.123,
CV +0.176, saturation +0.112, drop +0.015). Decisively:

| | n | median bias | \|diff\| p90 |
|---|---|---|---|
| SQI passed | 218 | **+2.66 bpm** | 12.1 |
| SQI refused | 973 | **+2.42 bpm** | 13.4 |

If the device under-counted during motion, refused windows would show a *larger* gap.
They do not. **The excess is a systematic offset, so 4.4% is a point estimate rather than
a floor.** Caveat: window-level agreement is loose regardless (\|HR diff\| p90 ≈ 12 bpm
even in the top bSQI quartile), so 4.4% holds at corpus level only.

---

## 11. Step 4 — REGULAR distribution: **HALTED**

### 11.1 The distribution as computed

Device, SQI-passed, n = 221 windows:

| Feature | median | p05 | p95 | p99 | max |
|---|---|---|---|---|---|
| HR (bpm) | 88.57 | 76.67 | 100.72 | 114.41 | 119.13 |
| mean RR (ms) | 677.41 | 595.70 | 782.56 | 819.64 | 827.68 |
| **RR CV** | **0.1464** | 0.1147 | 0.2237 | 0.2410 | 0.2669 |
| **RMSSD (ms)** | **145.53** | 116.59 | 207.06 | 239.91 | 243.45 |
| SDNN (ms) | 99.63 | 75.71 | 148.65 | 169.48 | 199.47 |

RR artefact screen: **0.0% of windows contain any interval outside 300–2000 ms.**

**These values are not physiological.** Healthy adults at rest have RMSSD ≈ 20–50 ms.
An RMSSD of 146 ms and CV of 0.146 on healthy self-recording adults is implausible.

### 11.2 The arithmetic that explains it

~4.4% excess beats (§10.2) means ~3 spurious peaks per 64-beat window. Each splits one
interval in two. Simulating 57 intervals at 700 ms plus 6 at ~350 ms gives:

**mean 666 ms, SD 103.6 ms, CV 0.156** — against the **0.146–0.166 observed**.

**The entire apparent irregularity is manufactured by over-detection.**

### 11.3 What was ruled out

| Hypothesis | Test | Result |
|---|---|---|
| Timing reconstruction | index vs clock vs interpolated-uniform-grid | **All the same**, CV 0.16–0.23 |
| Detector choice | 8 neurokit2 detectors | **All bad**, CV 0.18–0.32 |
| RR artefact screen | physiological bounds | 0% flagged |
| Mean rate wrong | device cross-check | Mean is correct (§10.2) |

**Detector bake-off** (median RR ratio vs device; 1.0 = agreement):

| Method | RR ratio | CV | RMSSD (ms) |
|---|---|---|---|
| rodrigues2021 | 0.9950 | 0.2541 | 267.6 |
| hamilton2002 | 0.9814 | 0.2600 | 267.7 |
| neurokit | 0.9773 | **0.1802** | **184.6** |
| kalidas2017 | 0.9748 | 0.2716 | 274.0 |
| pantompkins1985 | 0.9748 | 0.2506 | 247.9 |
| engzeemod2012 | 0.9712 | 0.3080 | 297.9 |
| elgendi2010 | 0.9535 | 0.2421 | 234.4 |
| christov2004 | 0.8162 | 0.3153 | 212.6 |

**Dropped samples do corrupt individual intervals**, but fixing that does not move the
aggregate. Short index-based intervals carry **+10.6 unaccounted samples** vs −3.5 for
normal ones; their clock RR is 620 ms against an index RR of 503 ms. But clock-based RR
gives the same window CV (0.1588 vs 0.1655) because it carries ±36 ms burst jitter
instead. The two independent estimates disagree per-interval with **SD ≈ 90 ms**.
**Both timing paths are broken in different ways.**

### 11.4 The actual cause — visual evidence

`reports_rhythm/s09_device_strip.png` (`1789/2026-08-17_08`, 2,517 peaks, median RR
725.5 ms, 12.5% of intervals below 0.8 × median):

- **The detector marks deflections at amplitude ~20–25 alongside genuine R-waves at
  ~80–100.** A true R-wave does not vary 4× beat-to-beat.
- **Motion artefacts reach ~500** against a normal p95 of 45.6 — roughly 8× — and the
  detector marks small bumps in the low-amplitude stretches around them.
- Clean stretches show correct detection.

**Signal quality swings hard *within* a single capture.**

### 11.5 Amplitude filtering is not the fix

Rejecting peaks below a fraction of the local median R height moves median-RR agreement
toward 1.00 but makes CV **worse** — because removing a peak fuses two intervals into one:

| Variant | RR ratio | window CV | window RMSSD |
|---|---|---|---|
| raw | 0.9854 | 0.1878 | 197.4 |
| amp ≥ 0.4 × med | 1.0006 | 0.2050 | 217.8 |
| amp ≥ 0.5 × med | 1.0006 | 0.2203 | 229.5 |
| amp ≥ 0.6 × med | 1.0006 | 0.2486 | 259.8 |

**Removal alone is insufficient; re-detection is required.**

### 11.6 Decision

**No threshold derived. No REGULAR distribution published.** Deriving a boundary from
§11.1 would be measuring detection error and calling it physiology — the same category
of error as the imported `cv = 0.10`, but self-inflicted.

This is brief §12's gate operating as designed: *"If the detector is wrong, every number
after it is meaningless."* The detector passed MITDB at F1 0.9890 and is still wrong here
— **cross-database transfer failure for the detector**, the same class as §5.8.

---

## 12. Blockers

### 12.1 LTAFDB partial; icentia11k / sddb absent

`/ssd_scratch/mahimakopalley/` was purged; all three symlinks dangled. A PhysioNet
re-download was started and **stopped after 6 of 84 records** (331 MB). icentia11k and
sddb remain absent.

**The 6 recovered LTAFDB records are complete and usable:**

| Record | Duration | Beats | `(AFIB` episodes |
|---|---|---|---|
| 00 | 21.0 h | 106,337 | 44 |
| 03 | 24.3 h | 84,266 | 22 |
| 05 | 24.3 h | 112,963 | 3 |
| 06 | 25.2 h | 104,939 | 19 |
| 07 | 25.5 h | 109,625 | 8 |
| 08 | 25.8 h | 108,367 | 4 |

Pooled: **146 h, 626,497 beats, 100 `(AFIB` episodes, 2,234 `(N` episodes**, plus `(SBR`
1,588, `(B` 419, `(AB` 105, `(SVTA` 70, `(VT` 42. Sampled at **128 Hz** — §4.5 degradation
to 89.70 Hz applies.

This is enough for a provisional IRREGULAR distribution **with genuine 24-hour
continuity**, which 30-minute records cannot provide.

> **Lead caveat (§4.4).** LTAFDB headers name both channels simply `ECG`. Unlike MITDB —
> where MLII was verified per record and 102/104 excluded — **the actual lead cannot be
> established from LTAFDB headers.** Any LTAFDB-derived distribution therefore carries an
> unverified-lead caveat and is not a like-for-like Lead-II comparison. This must be stated
> wherever such a distribution is reported.

**Complementary reroute:** MITDB AFIB-annotated records (201, 202, 203, 210, 219, 221, 222
among others) plus SVDB supraventricular arrhythmias, where the lead *is* header-verified.
Using both lets the unverified-lead LTAFDB result be cross-checked against a
lead-verified one. CUDB (VF) and challenge2017 (median 30 s, too short) do not help.

Resuming the remaining 78 LTAFDB records is optional, not blocking.

### 12.2 No beat-level reference for the device

`hrv` is a smoothed mean RR per ~12 s snapshot. It validates mean rate and **cannot**
validate variability (§10.1). There is no per-beat ground truth for any ProRhythm capture.

**Highest-value action: a controlled recording with a simultaneous per-beat RR reference**
(Polar H10 or equivalent, ~1 ms RR export). Minimum useful protocol: 10 min seated still
after 5 min rest, verified electrode contact, no talking, both devices concurrently, one
existing subject ID; plus ~2 min of deliberate movement to yield a **labelled artefact
segment** for SQI calibration.

Without it, building a device-specific detector means tuning against an unverifiable
target, with no way to distinguish a fixed detector from an overfitted one.

### 12.3 No packet sequence numbers

No WebSocket capture server exists in this repository. Recommended per-packet logging:
**device sequence number, packet counter, samples-per-packet.** This makes replay explicit
rather than inferred from ~13.1 s backward steps, and turns dropped samples from an
estimate into a count. Does not help the existing 37 captures; compounds for all future ones.

---

## 13. Deferred work and re-entry conditions

**Track B (supervised model, pseudo-labelling) — deliberately deferred, not omitted.**
Pseudo-labels would inherit exactly the detection error that invalidates Track A's REGULAR
distribution, laundering it into a model that looks confident.

> **Re-entry condition: device R-peak detection reaches ≥ 95% agreement with a beat-level
> reference on a held-out capture.**

**Track A threshold derivation** — blocked on the same condition.

**Stage 7 (two-track cross-check), Stage 8 (hysteresis smoothing), Stage 9 (MedGemma
narrative + structural validation), clinician review package** — not reached.

---

## 14. Exit criteria if detection cannot be fixed

Decision rule on per-window RR CV measured on **verified-regular** signal:

| Measured CV floor | Action |
|---|---|
| **< 0.03** | Ship rhythm regularity as specified. Adequate headroom below any plausible boundary. |
| **0.03 – 0.08** | Ship with an `UNABLE_TO_DETERMINE` band sized to the measured floor. Materially reduced yield. |
| **> 0.08** | **Do not ship a rhythm verdict.** It would sit on the noise floor — the same error as the imported `cv = 0.10`, self-inflicted. Fall back to HR-only with no regularity claim. |

Current measured value is **0.146** (third band), but contaminated by detection error.
The controlled recording (§12.2) is the measurement that determines the true band.

---

## 15. Limitations (brief §10.7)

1. **No rhythm labels exist for the target device.** Accuracy on ProRhythm is **UNKNOWN**
   and is stated as such in the headline, not a footnote.
2. **All local data is from healthy adults** — a single class. A decision boundary cannot
   be derived from one class. Public labelled data supplies the other side; this is the
   central methodological dependency of the project.
3. **Cross-database transfer has now failed twice in this project.** Previously a beat
   classifier (V-class F1 0.830 MITDB → 0.513 SVDB). Now the **detector** (F1 0.9890 MITDB
   → ~4.4% excess beats on device) and the **SQI gate** (78.4% retention MITDB → 18.3% on
   device). Every cross-dataset claim must be treated as guilty until proven otherwise.
4. **Amplitude units are unconfirmed** (not mV, probably ADC counts). No unit assumption
   entered the pipeline — audited in §8.7.
5. **Aetiology cannot be determined.** RR intervals contain no P-wave information, so
   atrial fibrillation cannot be distinguished from respiratory sinus arrhythmia, ectopy,
   or artefact. Only *regularity* is defensible.
6. **The SQI gate cannot be made perfect.** A residual induced-CV error of max 0.1115
   survives gating on MITDB.
7. **The lag-1 variance guard is simulation-validated only** — MITDB structurally cannot
   exercise the low-SDNN case it protects.
8. **The uniform-grid anchoring choice is load-bearing.** The ±36 ms burst uncertainty is
   *averaged*, not eliminated, so the 17× headroom figure is a ceiling.
9. **Device dropout is 10.2% of recorded span**, with intra-capture gaps up to 22 minutes.
10. **`temperature` in the vitals export mixes °C and °F** and is unusable (not required).

---

## 16. Reproducibility

Same input → identical output. All randomness seeded (`SEED = 20260827` in the noise-floor
simulation; no other stochastic step). Versions pinned in §2.1. Every capture is
SHA-256 hashed at ingest (`Capture.sha256`).

Every reported metric states database, lead, resampled rate, held-out basis, and N.

### Code

| Path | Purpose |
|---|---|
| `rhythm/ingest.py` | Stage 1 — format detection, parsing, timing measurement |
| `rhythm/resample.py` | Stage 2 — dedupe, segment, uniform grid, drop detection |
| `rhythm/degrade.py` | §4.5 resolution matching to 89.70 Hz |
| `rhythm/features.py` | Stages 5–6 — RR screen, segment-aware RMSSD, window features |
| `rhythm/sqi.py` | Stage 3 — bSQI, lag-1, kSQI, pSQI, flatline, saturation |
| `rhythm/pipeline.py` | Stages 1–6 end-to-end on device captures |

### Scripts and outputs

| Script | Output |
|---|---|
| `s01_measure_fs.py` | `s01_capture_timing.csv` |
| `s01b_diagnose_timing.py` | `s01b_segment_timing.csv` |
| `s01c_duplicates.py` | (stdout) |
| `s01d_dedup_rate.py` | `s01d_dedup_rate.csv` |
| `s01e_noise_floor.py` | (stdout) — research Q2 |
| `s02_survey_vitals.py` | `s02_vitals_survey.csv` |
| `s03_validate_detector.py` | `s03_detector_mitdb.csv` — **the gate** |
| `s04_detector_rr_impact.py` | `s04_rr_impact*.csv` — **superseded, biased** |
| `s05_sqi_derive.py` | `s05_sqi_windows.csv` |
| `s05b_sqi_rr_components.py` | `s05b_sqi_rr.csv` |
| `s06_device_pipeline.py` | `s06_device_windows.csv` |
| `s07_hr_crosscheck.py` | `s07_hr_crosscheck.csv` |
| `s08_detector_bakeoff.py` | `s08_detector_bakeoff.csv` |
| `s09_strip.py` | `s09_device_strip.png` |

---

## 17. Research questions (brief §9) — status

1. **Published RR regularity measures and their derivation resolutions** — partially
   answered. Quantified that `cv = 0.10` is invalid under packet-stamp reconstruction at
   ~90 Hz because it sits on the noise floor (§4.1). A full literature survey with
   citations is outstanding.
2. **How RR-timing resolution affects CV and RMSSD** — **ANSWERED** (§4). Uniform-grid
   floor CV 0.0044–0.0071, RMSSD 6.65–9.41 ms; packet-stamp floor CV 0.0459–0.0949,
   RMSSD ~93–101 ms.
3. **Which databases contain genuine Lead-II signals** — **ANSWERED for MITDB** (§7.1),
   verified from headers: 46 of 48 records, with 102/104 excluded and 114 requiring
   channel 1. SVDB and INCARTDB not yet header-verified.
4. **Reported AF-vs-sinus RR separability, and how much survives degradation to ~90 Hz** —
   **NOT ANSWERED.** Blocked on the IRREGULAR distribution.
5. **Clinical validation requirements for a rhythm indicator** — **NOT ANSWERED.**

---

## 18. Immediate next actions, in order

1. **Obtain the controlled recording with a beat-level RR reference** (§12.2). Highest
   value; may dissolve the blocker rather than require engineering around it.
2. **Enable packet sequence-number logging** (§12.3). Cheap, compounding, independent of 1.
3. **Only then:** build and validate a device-specific R-peak detector against that
   reference.
4. Build the provisional IRREGULAR distribution from the **6 recovered LTAFDB records**
   (146 h, 100 AFIB episodes, unverified lead) cross-checked against **MITDB AFIB records +
   SVDB** (lead-verified) — §12.1. This can proceed in parallel, as it does not depend on
   device data.
5. Re-derive the SQI gate on device data once detector B is chosen by device-side
   agreement rather than inherited from MITDB.

---

> **Rhythm regularity indicator. Not a diagnosis. Not validated for clinical
> use. Cannot distinguish atrial fibrillation from other causes of irregularity.**


---

# PART II — Verdict, narrative and report (added 2026-08-28)

## 19. Threshold derivation (brief step 6)

Derived from the 6 recovered LTAFDB records, degraded to the device's measured
**89.70 Hz** grid. RR comes from **expert beat annotations, not detector output**, so
the boundary reflects physiology rather than detection error.

Labels per brief: `(AFIB` -> IRREGULAR, `(N` -> REGULAR. All other rhythm annotations
(SBR, B, AB, SVTA, VT, T) are **excluded**, not mapped - neither label is defensible for
them. Windows spanning an episode boundary are discarded.

**8,528 labelled windows: 7,784 REGULAR, 744 IRREGULAR.**

### Distributions at 89.70 Hz (64 beats = 63 intervals)

| Feature | Class | p05 | p25 | median | p75 | p95 |
|---|---|---|---|---|---|---|
| RR CV | REGULAR | 0.0131 | 0.0191 | 0.0305 | 0.0744 | 0.1951 |
| RR CV | IRREGULAR | 0.1280 | 0.1730 | 0.2085 | 0.2564 | 0.2985 |
| RMSSD (ms) | REGULAR | 10.50 | 13.05 | 17.46 | 80.38 | 235.57 |
| RMSSD (ms) | IRREGULAR | 66.48 | 95.22 | 161.63 | 237.86 | 300.47 |
| SDNN (ms) | REGULAR | 10.03 | 15.27 | 24.29 | 59.73 | 150.08 |
| SDNN (ms) | IRREGULAR | 47.53 | 67.83 | 114.11 | 173.09 | 219.33 |
| HR (bpm) | REGULAR | 63.66 | 70.00 | 75.05 | 81.02 | 91.07 |
| HR (bpm) | IRREGULAR | 77.75 | 86.67 | 112.46 | 154.94 | 162.74 |

### The overlap, reported honestly

**REGULAR p95 = 0.1951 vs IRREGULAR p05 = 0.1280 - the classes OVERLAP.** The boundary
is a trade-off, not a separation.

| Threshold | Se | Sp | PPV | NPV | Youden |
|---|---|---|---|---|---|
| 0.075 | 0.9973 | 0.7515 | 0.2773 | 0.9997 | 0.7489 |
| 0.100 | 0.9946 | 0.8183 | 0.3435 | 0.9994 | 0.8130 |
| **0.1200 (chosen)** | **0.9718** | **0.8592** | **0.3975** | **0.9969** | **0.8236** |
| 0.150 | 0.8683 | 0.9062 | 0.4695 | 0.9863 | 0.7745 |
| 0.200 | 0.5497 | 0.9549 | 0.5382 | 0.9569 | 0.5046 |

**PPV is only 0.3975** because of ~10:1 class imbalance. NPV is 0.9969 - the threshold is
far better at ruling irregularity OUT than ruling it IN. This must be stated wherever the
verdict is shown.

The threshold sits **20x above** the 0.006 uniform-grid noise floor (§4), which is the
headroom the packet-stamp reconstruction would not have provided.

### Caveats attached to this threshold

- **Lead UNVERIFIED.** LTAFDB headers name both channels `ECG`. Unlike MITDB, where MLII
  was verified per record, the lead cannot be established. **This is not a like-for-like
  Lead-II result.**
- **No held-out split.** Derived on the full set of 6 records; not cross-validated.
- **NOT VALIDATED ON THE TARGET DEVICE.** Applying it to ProRhythm is an untested
  cross-database transfer, and transfer has already failed twice here (§15.3).

## 20. Stages 7-8 - Verdict and hysteresis

**Stage 7** (`rhythm/verdict.py`) decides deterministically, before any model is called.

**Refusal band.** CV carries measurement uncertainty from the quantisation floor (~0.006)
and residual detector-induced error surviving the SQI gate (p95 0.0267 on MITDB). A window
whose CV falls within **+/-0.027** of the boundary - i.e. **0.0930-0.1470** - is reported
`UNABLE_TO_DETERMINE` rather than forced to a side. This converts a measured weakness into
a principled refusal.

**Stage 8** (`rhythm/smoothing.py`) reports a state only after **3 consecutive windows
agree**, starting from `UNABLE_TO_DETERMINE` - the system asserts nothing until it has
evidence. Every transition is logged with the evidence that caused it.

## 21. Stage 9 - MedGemma narrative and structural validation

The verdict is decided in Stage 7. The model's only job is to restate it.

### MedGemma runtime status: NOT AVAILABLE

| Component | State |
|---|---|
| `ollama` binary | Present at `/home2/mahimakopalley/bin/ollama` |
| Ollama server | **Not running** - nothing on `:11434` |
| Model weights | **Missing** - `medgemma-4-bit-model/` absent, no `.gguf` on disk |

Per brief §7 the deterministic template must always be available so the system never
depends on the model being reachable. It is, and every narrative generated so far has come
from it, logged explicitly as `model unreachable: URLError`.

### Validation is positive and structural

1. Every **number** in the prose must be grounded in the evidence JSON.
2. The **verdict word** must be present, correct, and **not negated**.
3. **NFKC-normalise** first; reject any non-ASCII letter.
4. Content **denylist last**, as a backstop only.

Any single failure discards the **entire** narrative - no patching, no partial acceptance.

### Three validator bugs found by adversarial self-test, and fixed

The first implementation passed 3 of 8 attacks. All three failures were the exact classes
the brief warned about:

| Attack | First implementation | Cause | Fix |
|---|---|---|---|
| `"The rhythm is not regular"` | **ACCEPTED** | Word-presence check ignores negation | Reject a negation token within 40 chars before the verdict word |
| Cyrillic homoglyph | **ACCEPTED** | **NFKC does NOT fold Cyrillic to Latin**; the homoglyph string still contains the substring `regular` | Reject any non-ASCII alphabetic character |
| `"RR CV 0.0350"` (invented) | **ACCEPTED** | Numbers matched at *any* rounding, so `0.0350` matched an evidence `0` from `n_flagged=0` | Match each token **at its own precision** |

Final result - **9 of 9 attacks rejected, 3 of 3 faithful narratives accepted**, including
a one-digit CV drift (0.0422 against a true 0.0421).

## 22. Clinician review package (brief §8, deliverable 6)

`scripts_rhythm/s12_generate_report.py` emits, per capture: `report.json`, `report.md`, and
a PNG per window carrying the ECG strip with R-peaks overlaid, the RR series with **flagged
outliers shown not hidden**, the features, the threshold and refusal band, the SQI value and
the reason for any refusal.

### First generated report - `1789/2026-08-17_08.csv`

| | |
|---|---|
| Windows (64 beats) | 39 |
| `UNABLE_TO_DETERMINE` | **38 (97.4%)** |
| `IRREGULAR` | 1 |
| `REGULAR` | 0 |
| Hysteresis transitions | 0 |
| Narrative source | template (MedGemma unreachable) |

Dominant refusal reason: **bSQI below 0.95** (values 0.882-0.938).

**This near-total refusal is the correct behaviour, not a defect.** Given §11's finding
that device R-peak detection manufactures CV ~ 0.15, a system returning confident
REGULAR/IRREGULAR verdicts here would be lying. The refusal path is doing exactly what
brief §11 specifies.

### Detector B re-selected for the device

The MITDB-inherited pair (`neurokit` + `pantompkins1985`) was invalid on device data (§9).
B was re-chosen by agreement with A **on device data**:

| Candidate B | n_B/n_A | median bSQI | windows >=0.95 |
|---|---|---|---|
| **rodrigues2021** | 0.949 | **0.9297** | **34.0%** |
| pantompkins1985 | 0.919 | 0.9173 | 19.7% |
| hamilton2002 | 0.958 | 0.9147 | 15.6% |
| kalidas2017 | 0.881 | 0.8741 | 9.2% |
| christov2004 | 1.093 | 0.7871 | 7.4% |
| elgendi2010 | 1.033 | 0.7834 | 0.0% |

`bsqi_min` was **deliberately kept at 0.95** rather than relaxed to raise yield. Lowering it
would admit precisely the high-detection-error windows the gate exists to catch.

## 23. Archived legacy work

The prior beat-classification system was moved to
`archive/legacy_beat_classification_2026-08-28/` (via `git mv`, history preserved):
`ecg_inference/`, `ecg_pipeline/`, `run_inference.py`, `models/`, `training/`,
`validation/`, `examples/`, `scripts/`, `tests/`, old `README.md`, and old report outputs.

Reason: it is a five-class beat-morphology classifier (brief §2: **out of scope, on hold**),
its `parse_prorhythm_ecg` reads only the SeNSiO export format (out of scope), it has no
rhythm verdict, and its own reported DS2 held-out F1 is V 0.830 / **S 0.152 / F 0.005**.
None of it is reusable for rhythm regularity on live captures.

`MedGemma-Agent/` is a git submodule and was left in place.

## 24. What is now ready, and what is not

| Component | Status |
|---|---|
| Stages 1-6 (ingest to features) | Built, validated |
| Stage 7 verdict + refusal band | **Built**, threshold derived from LTAFDB |
| Stage 8 hysteresis | **Built**, unit-verified |
| Stage 9 narrative + validation | **Built**, 9/9 attacks rejected |
| Clinician review package | **Built**, generating |
| MedGemma model itself | **NOT AVAILABLE** - weights missing, server down |
| Real-time / streaming | **NOT BUILT** - no ingestion server in this repo |
| Device R-peak detection | **STILL BROKEN** - §11 blocker unchanged |
| Threshold validated on device | **NO** - no device labels exist |

The report pipeline is complete and honest. What it currently reports, on device data, is
overwhelmingly *"I cannot tell"* - which is the truthful answer until the detection blocker
in §11 and §12.2 is resolved.


---

# PART III — Cross-database validation and robustness (2026-08-28)

## 25. TASK 1 — Held-out DATABASE validation: **the threshold TRANSFERS**

Derived on LTAFDB, evaluated on MITDB, which the threshold has never seen.
Held-out **database**, not held-out record.

| | LTAFDB (derivation) | **MITDB (held-out database)** |
|---|---|---|
| Se | 0.9718 | **0.9839** |
| Sp | 0.8592 | **0.8408** |
| PPV | 0.3975 | **0.4266** |
| NPV | 0.9969 | **0.9977** |
| median RR CV, REGULAR | 0.0305 | 0.0559 |
| median RR CV, IRREGULAR | 0.2085 | 0.2175 |

MITDB confusion matrix at 0.12: **TP 122, FN 2, FP 164, TN 866**; 1,154 windows,
43 records, prevalence 10.7%. Six records contribute IRREGULAR (201, 202, 203, 210,
219, 221). Paced-excluded figures are identical - no paced record contributed a
labelled window.

**This is a positive transfer result**, and notable because cross-database transfer had
already failed twice in this project (§15.3). Performance is essentially unchanged, with
sensitivity marginally better and specificity marginally worse on the unseen database.

### A silent-failure bug found while building this

MITDB `aux_note` strings carry **trailing NUL bytes** (`'(AFIB\x00'`). `str.strip()` does
not remove `\x00`. LTAFDB has no NULs, so the original extractor worked there and would
have silently produced **zero** MITDB rhythm labels. Stripping explicitly recovered
**6 AFIB-contributing records instead of 3**.

### SVDB cannot serve as an expert-labelled held-out set

SVDB contains **one** rhythm annotation in the entire database and **zero** `(AFIB`. Its
supraventricular content is at the BEAT level, not rhythm episodes. The continuation brief
assumed otherwise.

A **proxy** analysis was run instead, clearly marked and never pooled with the primary
result: windows labelled IRREGULAR by S-beat density >= 10%, ventricular-heavy windows
excluded. **Se 0.6847, Sp 0.9220, PPV 0.7818, NPV 0.8775** (1,083 windows, 69 records).

Sensitivity drops sharply: the threshold catches sustained AF-like irregularity but misses
~32% of intermittent ectopy-driven irregularity. This is a **proxy label, not ground
truth**, and is weaker evidence than the MITDB result.

## 26. TASK 2 — Leave-one-record-out: **FAILS the stability criterion**

| Dropped | n REGULAR | n IRREGULAR | Threshold | Se | Sp |
|---|---|---|---|---|---|
| none (full set) | 7,784 | 744 | 0.1175 | 0.9785 | 0.8550 |
| **record 00** | 6,467 | 450 | **0.1675** | 0.9867 | 0.9099 |
| record 03 | 7,322 | 642 | 0.1175 | 0.9751 | 0.8604 |
| record 05 | 6,033 | 744 | 0.1125 | 0.9866 | 0.8450 |
| record 06 | 6,268 | 670 | 0.1175 | 0.9761 | 0.8752 |
| record 07 | 6,719 | 470 | 0.1125 | 0.9787 | 0.8669 |
| record 08 | 6,111 | 744 | 0.1175 | 0.9785 | 0.8400 |

Five of six land in 0.1125-0.1175. **Dropping record 00 moves the threshold to 0.1675 -
outside the 0.110-0.125 stability band, a shift of 0.055.** Record 00 alone supplies 294 of
744 IRREGULAR windows (40%).

**By the stated criterion this is a fit to individual patients, not a threshold.** The
value is dominated by one patient's rhythm. This is the strongest argument for Task 7
(more records) and it is why 0.12 must stay PROVISIONAL regardless of the successful
cross-database transfer in §25 - transfer being good does not make the derivation robust.

(The full-set optimum reads 0.1175 rather than 0.1200 because this sweep used a finer
0.0025 grid than the original 0.005 grid. The difference is immaterial.)

## 27. TASK 3 — Operating points

| Database | Operating point | thr | Se | Sp | PPV | NPV | TP | FP | FN | Se cost |
|---|---|---|---|---|---|---|---|---|---|---|
| LTAFDB | Youden-optimal | 0.1175 | 0.9785 | 0.8550 | 0.3920 | 0.9976 | 728 | 1,129 | 16 | — |
| LTAFDB | Sp >= 0.95 | 0.1975 | 0.5659 | 0.9523 | 0.5316 | 0.9582 | 421 | 371 | 323 | −0.4126 |
| LTAFDB | Sp >= 0.99 | 0.2550 | 0.2567 | 0.9906 | 0.7235 | 0.9331 | 191 | 73 | 553 | −0.7218 |
| MITDB | Youden-optimal | 0.1350 | 0.9839 | 0.8786 | 0.4939 | 0.9978 | 122 | 125 | 2 | — |
| MITDB | Sp >= 0.95 | 0.2150 | 0.5161 | 0.9515 | 0.5614 | 0.9423 | 64 | 50 | 60 | −0.4677 |
| MITDB | Sp >= 0.99 | 0.2650 | 0.2823 | 0.9913 | 0.7955 | 0.9198 | 35 | 9 | 89 | −0.7016 |

**Recommendation (not a selection).** This instrument is a **rule-out** device at every
operating point: NPV stays >= 0.92 throughout, PPV never exceeds 0.80. The shape of the
trade is unusually steep - buying Sp 0.95 costs **~42-47% of sensitivity**, and Sp 0.99
costs **~70%**.

- If the output feeds a **human reviewer** who can dismiss false flags cheaply, the
  Youden point is defensible: it misses almost nothing (FN 2 of 124 on MITDB).
- If it feeds an **escalation desk or alarm**, Youden is not defensible - 1,129 false
  positives against 728 true positives on LTAFDB would manufacture alarm fatigue. Sp>=0.95
  roughly halves sensitivity to buy a 3x PPV improvement.
- **Sp >= 0.99 is not recommended in any framing**: it discards ~70% of true irregularity
  to reach PPV 0.72-0.80, which is a poor trade for a screening indicator.

The decision depends on what consumes the output, which is not mine to choose.

## 28. TASK 4 — Hysteresis N: **the curve does not flatten**

The task assumed a knee exists to read N off. Measured, it does not.

First pass over all sliding windows (8-beat slide, 87.5% overlap) conflated **real
paroxysmal rhythm changes** with spurious flicker. Re-measured on **187 label-stable
stretches** (single rhythm throughout, >= 30 windows, 86,691 windows total), which isolates
genuine flicker:

| N | spurious state changes/hour | lag (s) | marginal reduction |
|---|---|---|---|
| 1 (none) | 21.96 | 6.3 | — |
| 2 | 15.69 | 12.6 | 28.6% |
| **3 (current)** | **12.40** | **19.0** | 21.0% |
| 4 | 10.12 | 25.3 | 18.4% |
| 5 | 8.43 | 31.6 | 16.7% |
| 6 | 7.11 | 37.9 | 15.7% |
| 7 | 6.05 | 44.2 | 14.9% |
| 8 | 4.98 | 50.6 | 17.7% |

**Each increment buys a roughly constant 15-21% proportional reduction while lag grows
linearly.** There is no flattening point, so **N cannot be derived from this curve** - it
is a continuous trade, and the choice is a clinical-tolerance judgement rather than a
measurement.

Two further observations:

- **Hysteresis alone does not eliminate flicker.** Even at N=8, with 50.6 s of lag, 4.98
  spurious state changes per hour remain. The residual comes from windows genuinely sitting
  near the boundary, which is what the refusal band - not hysteresis - exists to absorb.
- N=3 gives a state change every ~4.8 minutes at 19 s lag; N=5 gives every ~7.1 minutes at
  32 s lag.

**Recommendation:** N=5 as a balance, explicitly as a judgement, not a derivation. N=3 is
retained in code until that judgement is made.

## 29. TASK 5 — Validator gaps

**(a) Remaining passes: NONE.** All 16 adversarial cases are now rejected, including the
aetiology euphemism that survived the previous round. Faithful narratives for all three
verdicts still pass, and the deterministic template passes its own validator.

**(b) Scope-statement exclusion is an EXACT string match.** The code is
`body = norm.replace(scope_norm, " ")` - a literal `str.replace`, no regex, no fuzzy or
partial matching. Verified adversarially: wrapping banned content inside a **near-miss** of
the scope statement (one clause altered) is REJECTED, because the altered text no longer
matches the literal string and is then caught by sentence-anchoring.

**(c) The euphemism class is closed by DENYLIST, i.e. structurally incomplete.**
"quivering of the upper chambers" was blocked by adding `upper chamber`, `atri`, `ventric`,
`quiver`, `fibrillat`, `healthy heart` and related terms. This is enumeration, which is
exactly the approach the brief calls insufficient - a paraphrase not on the list will pass.
The structural checks (number grounding, verdict matching, sentence anchoring) are
principled and complete; the aetiology check is not.

**The current 16/16 pass rate is an UPPER BOUND.** It measures resistance to attacks I
conceived while holding the implementation in mind. An independent adversary should be
assumed to do better, and the euphemism class is where they would start.

## 30. TASK 6 — One lead standard, applied to both databases

**Decision:** lead verification is required **wherever the waveform is processed**, and is
immaterial **where only expert annotation beat times are used**.

Rationale: lead determines R-wave morphology and amplitude, so it governs whether a
detector finds the beat - that is a signal-processing question. An expert annotation is a
**timestamp**; the lead the annotator viewed does not change when the beat occurred, and RR
features are computed purely from those timestamps.

Applied consistently:

| Context | Waveform processed? | Lead requirement | Effect |
|---|---|---|---|
| §7 detector validation (MITDB) | Yes - detector runs on signal | **Verified MLII required** | 102/104 excluded - correct |
| §9 device pipeline (ProRhythm) | Yes - detector runs on signal | Lead II, confirmed from device config | unchanged |
| §19 threshold derivation (LTAFDB) | No - annotations only | **Not required** | UNVERIFIED accepted - now justified |
| §25 held-out evaluation (MITDB) | No - annotations only | **Not required** | re-run without the MLII filter |

Re-running §25 under the annotation-based standard, without the MLII filter, gives
**identical results** - 43 records, 1,154 windows, Se 0.9839, Sp 0.8408, PPV 0.4266,
NPV 0.9977. The non-MLII records contribute no labelled windows, so the standard change is
a no-op here. The inconsistency is resolved in principle and makes no difference in
practice.

## 31. TASK 7 — LTAFDB download (running)

Targeted fetch of ~20 records (not all 84), running unattended. **8 complete records** at
time of writing (00, 01, 03, 05, 06, 07, 08, 10), up from 6. This directly addresses the
effective-n problem in §26.

## 32. Status after Part III

| Item | Before | After |
|---|---|---|
| Held-out-database result | absent | **Se 0.9839 / Sp 0.8408 on MITDB - transfers** |
| Threshold robustness | unknown | **FAILS LORO - one patient moves it 0.055** |
| Operating-point framing | Youden only | **three points, rule-out recommended** |
| Hysteresis N | arbitrary | **measured; no knee exists; N=5 recommended as judgement** |
| Validator | 5 of 15 attacks passing | **16/16 rejected; euphemism class remains denylist-bound** |
| Lead standard | inconsistent | **one standard, documented, applied** |

**0.12 remains PROVISIONAL and NOT SHIPPABLE.** The successful cross-database transfer
(§25) is encouraging but does not change that: §26 shows the value itself is dominated by
a single patient, and the device-side blocker (§11) is untouched.


---

# PART IV — Operating-point design and threshold precision (2026-08-28)

## 33. The Task 3 recommendation was internally contradictory — corrected

Part III concluded "rule-out instrument at every operating point" and then recommended
Sp>=0.95 for alarm contexts. Those are incompatible, and the measurement shows why.

**NPV falls MONOTONICALLY as the threshold tightens** (verified across the sweep):

| Threshold | Se | Sp | PPV | NPV |
|---|---|---|---|---|
| 0.100 | 0.9946 | 0.8183 | 0.3435 | **0.9994** |
| 0.120 | 0.9718 | 0.8592 | 0.3975 | 0.9969 |
| 0.180 | 0.6962 | 0.9379 | 0.5175 | 0.9700 |
| 0.240 | 0.3253 | 0.9823 | 0.6368 | 0.9384 |
| 0.300 | 0.0417 | 0.9992 | 0.8378 | **0.9160** |

Tightening degrades the single property this instrument is genuinely good at, to buy one
it never achieves usefully. PPV does exceed 0.80, but only above threshold 0.28, where
sensitivity has collapsed to 0.13 — so the earlier claim "PPV never exceeds 0.80" was
loosely stated: it holds only jointly with usable sensitivity.

### The claim survives prevalence

Se/Sp fixed at the Youden point, NPV and PPV recomputed at other prevalences:

| Prevalence | NPV | PPV |
|---|---|---|
| 1% | 0.9997 | 0.0638 |
| 5% | 0.9987 | 0.2620 |
| 8.7% (LTAFDB) | 0.9976 | 0.3913 |
| 30% | 0.9893 | 0.7430 |
| 50% | 0.9755 | 0.8709 |

A REGULAR call stays trustworthy at every plausible prevalence. An IRREGULAR call only
becomes trustworthy at prevalences a screening indicator will never see.

## 34. ADOPTED DESIGN — operate at Youden, make only REGULAR actionable

```
REGULAR    -> ACTIONABLE.     "regularity established"     NPV ~0.998
IRREGULAR  -> NOT actionable. "regularity not established" PPV ~0.40
UNABLE     -> NOT actionable. "regularity not established"
```

IRREGULAR prompts a look. It must **never** fire an alarm or reach an escalation desk on
its own. This keeps full sensitivity (FN 2 of 124 on the held-out database) while
preventing PPV 0.40 from ever reaching a consumer that would act on it — sidestepping alarm
fatigue without paying the 42-47% sensitivity cost that Sp>=0.95 demanded.

Implemented in `rhythm/verdict.py`: every `Verdict` now carries `actionable: bool` and
`action_state` (`REGULARITY_ESTABLISHED` / `REGULARITY_NOT_ESTABLISHED`), and both appear in
every `report.json` alongside the policy and its rationale.

**Side effect: the hysteresis N choice becomes largely cosmetic.** Display churn on a
non-alarming indicator costs little, so N is now a readability preference rather than a
clinical parameter. **N = 5 adopted** (8.43 spurious state changes/hour, 31.6 s lag),
recorded in code as a judgement, not a derivation.

## 35. Task 2 restated — the threshold is IMPRECISE, not biased

Part III called the LORO result "a fit to individual patients". That overstates it.
Dropping record 00 removes **40% of the entire positive class**, and a Youden optimum
shifting under that is expected from an imprecise estimate, not evidence of bias.

Quantified with a **record-level (cluster) bootstrap** — resampling the 6 records, not the
windows, because windows within a record are not independent. 2,000 resamples:

| | |
|---|---|
| Point estimate (full set) | 0.1175 |
| **Bootstrap median** | **0.1175** — centred, no bias |
| **95% CI** | **[0.1025, 0.1777]** |
| 80% CI | [0.1125, 0.1775] |
| MITDB independent estimate | 0.1350 — **inside the 95% CI** |
| Resamples inside 0.110-0.135 | 55.9% |

**Correct statement:** the threshold's *region* (~0.11-0.135) is supported across two
independent databases, but its *point value* has a wide confidence interval because the
positive class rests on 4 records, of which 2 supply 76% of IRREGULAR windows.

The fix is therefore **more patients**, not a different approach — which is precisely what
Task 7 addresses. The CI and its interpretation now travel in
`THRESHOLD_PROVENANCE["threshold_ci95_cluster_bootstrap"]` and `["ci_note"]`.

## 36. Re-derivation plan when the download completes

Not just a re-derive. The success criterion is a **narrower LORO spread, not the same
number** — the value is expected to move, and movement is not regression.

1. Threshold on the full expanded record set.
2. **LORO across all records**, reporting the spread and comparing it to the current
   0.1125-0.1675.
3. **Cluster-bootstrap CI**, compared to the current [0.1025, 0.1777]. Narrowing is the
   test that more patients fixed it.
4. Whether the IRREGULAR class is **still dominated by one or two records** (currently
   76% from two).
5. **Re-run the MITDB held-out check at the new threshold**, so the transfer result is
   re-established rather than assumed to carry over.

Download status at time of writing: **9 complete records**, up from 6.

## 37. Unchanged

None of Part IV touches the device blocker. All threshold work is on public data. The
controlled recording with a beat-level RR reference (§12.2) remains the only thing that
unblocks the device side, and it is **still not scheduled**. Tasks 8+ remain untouched.


---

# PART V — Real-time implementation (2026-08-28)

## 38. Architecture

```
ProRhythm WebSocket -> rolling buffer -> preprocessing -> R-peak -> RR features
                    -> classifier -> severity -> MedGemma   [GATED]
```

Implemented in `rhythm/streaming.py` (engine), `rhythm/severity.py` (severity
layer), `rhythm/domain_gate.py` (gate), `scripts_rhythm/s17_realtime.py` (runner).
The engine is transport-agnostic: the same code serves the live WebSocket and a
CSV replay harness, so the live path is testable without hardware.

### Three things the live path handles that the batch path did not

1. **Replayed packets, online.** Exact `(timestamp, amplitude)` duplicates are
   dropped incrementally against a bounded set of recently-seen samples.
   **Verified on `1794/2026-08-09_12.csv`: 18,971 rows dropped, 33.1% of
   arrivals**, giving a measured rate of ~90.5 Hz instead of the ~178 Hz that
   file yields undeduplicated.
2. **Out-of-order arrival.** The buffer is sorted by timestamp at analysis time
   rather than trusting arrival order. This adds no latency: the analysis
   window spans ~45 s of beats, far longer than the ~13.1 s reorder span.
3. **Variable rate and dropout.** `fs` is measured per analysis window from that
   window's own clock; RR comes from index differences at that rate, never from
   per-sample timestamps.

### Measured performance

| | |
|---|---|
| Analysis latency | **~30 ms** per pass (first pass ~0.7-1.9 s, warm-up) |
| Emit cadence | every 8 new beats (~6 s at 75 bpm) |
| Buffer | 180 s rolling |
| Hysteresis | N=5 (~32 s to a state change) |

## 39. Severity layer (separate, as chosen)

The rhythm verdict is **not overwritten**. It feeds a severity engine alongside
HR, SQI and flagged-interval count.

| Condition | Severity | Escalate? |
|---|---|---|
| REGULAR, HR 60-100, SQI ok | NORMAL | no |
| REGULAR, HR outside 60-100 | ABNORMAL | no |
| IRREGULAR | ABNORMAL | **no** |
| HR < 40 or > 150 | **CRITICAL** | **yes** |
| UNABLE / SQI fail / gate closed | WITHHELD | no |

**IRREGULAR never escalates to CRITICAL.** Its PPV at the operating point is
~0.40, so it prompts review and must not raise an alarm — consistent with the
actionable-REGULAR design in §34.

**HR limit provenance: NONE.** Unlike the rhythm threshold, the HR bands
(40/60/100/150 bpm) are conventional clinical defaults. They were **not derived
from or validated on any dataset in this project**. Marked
`PROVISIONAL - NOT SHIPPABLE` in `HR_LIMITS_PROVENANCE` and carried in output.

## 40. Device-domain gate — clinical output fails closed

Clinical severity and MedGemma are blocked until device R-peak timing is
validated against a controlled reference recording.

| State | rhythm verdict | clinical severity | MedGemma |
|---|---|---|---|
| **UNVALIDATED** (current) | permitted | **BLOCKED** | **BLOCKED** |
| FAILED | permitted | BLOCKED | BLOCKED |
| VALIDATED | permitted | permitted | permitted |

Permission is read from an artifact written by the validation harness, so it
cannot be granted by editing a flag. Reports are stamped
`output_class: ENGINEERING` and the block appears in `report.json` and
`report.md`. Live output currently shows every window as `WITHHELD`.

## 41. Beat-timing validation harness (`rhythm/reference.py`)

Ready to consume a paired ProRhythm + reference-RR recording. Solves clock
alignment by maximising beat-train agreement over a search range — **verified to
recover a 7.2 s offset to within 1 ms** on synthetic pairs.

### The re-entry condition needed strengthening

The previously stated condition was ">= 95% agreement with a beat-level
reference". Simulating the device's *current* behaviour (4.4% spurious peaks)
against a reference:

| Metric | Value | Verdict |
|---|---|---|
| Sensitivity | 1.0000 | passes |
| **PPV** | **0.9585** | **passes a >=95% gate** |
| **RR CV error p95** | **0.1741** | **6.4x the budget** |

**Beat-level agreement alone would have passed a detector that corrupts RR CV
by 0.17.** A detector can find every beat and still misplace it enough to
destroy the quantity the threshold consumes. The binding criterion is therefore
CV error, not beat agreement:

```
PASS requires ALL of:
  sensitivity      >= 0.95
  PPV              >= 0.95
  |RR bias|        <= 10 ms
  RR CV error p95  <= 0.027   <- the refusal-band budget; binding criterion
```

## 42. Open items on the real-time path

- **The WebSocket message schema is assumed**, not confirmed: samples carrying
  `timestamp_ms` and `amplitude`, singly or batched under `samples`. No server
  exists in this repository. The schema must be confirmed against the real
  endpoint before the live path is trusted.
- Reconnection, backpressure and session handling are not implemented.
- Sequence-number logging (§12.3) is still absent, so replay is still inferred
  rather than reported by the device.
- **Everything downstream of R-peak detection remains unvalidated on the device
  domain.** The real-time path is correct engineering on a signal whose beat
  timing is known to be wrong.


---

# PART VI — Three-axis classifier, wired and measured (2026-08-30)

## 43. Structure: three independent axes, not one flat label

| Axis | Output | Trust criterion |
|---|---|---|
| **Rate** | bradycardic / normal / tachycardic / undetermined | **signal-level SQI only** |
| **Regularity** | regular / irregular / undetermined | full SQI gate (bSQI >= 0.95) |
| **Events** | pauses, ectopy BURDEN (never type) | full SQI gate |

A combination layer maps the three to Normal / Abnormal / Critical.

## 44. Thresholds: derived where possible, marked where adopted

| Threshold | Basis |
|---|---|
| Normal band 60-100 bpm | **ADOPTED convention, SUPPORTED by measurement** — 15,016 LTAFDB sinus windows run p05 63.7, median 75.0, p95 91.1 |
| Extreme bands 40 / 150 bpm | **ADOPTED, NOT derived, NOT validated here** |
| Pause > 2000 ms | **DERIVED** — 0.0089% of 1,121,159 sinus RR intervals exceed it (p99 = 1193 ms) |
| Ectopy detector | **DERIVED and validated vs expert beat labels** |

Ectopy burden, held-out against cardiologist-marked beats:

| Burden >= | MITDB Se / PPV | SVDB Se / PPV |
|---|---|---|
| 1 | 0.877 / 0.951 | 0.899 / 0.978 |
| 5 | 0.693 / 0.893 | 0.721 / 0.939 |

Sensitivity falls as burden rises: **high burden is under-called.**

## 45. The SVT gap, confirmed held-out

Sustained fast atrial rhythms are fast but *evenly spaced*, so a regularity-only
view calls them normal.

| | MITDB | SVDB |
|---|---|---|
| REGULAR + TACHYCARDIC windows | 90 (5.6%) | 162 (5.7%) |
| Median HR of those | 110 bpm | 130 bpm (max 161) |
| Regularity-only verdict | NORMAL | NORMAL |
| Combination layer | **all ABNORMAL** | 150 ABNORMAL, 12 CRITICAL |
| Extra windows flagged overall | 220 (24.3%) | 351 (28.0%) |

Escalation stays rare and rate-driven: 2.65% and 0.49%. **IRREGULAR never
escalates** (PPV ~0.47).

## 46. On device data, ONE axis of three survives

1,210 windows, 37 captures, all healthy adults.

| Axis | Public data | Patch data |
|---|---|---|
| Rate | validated | **SURVIVES** — median 85.1 bpm, p05 70.9, p95 99.8, **0 extremes** |
| Regularity | Se 0.9839, NPV 0.9978 | **BROKEN** — RR CV median 0.1681 vs 0.0305 healthy |
| Events | Se 0.88-0.90, PPV 0.95-0.98 | **BROKEN, and more so** |

Events on healthy adults: **95.8%** of windows show >=1 ectopic couplet, median
3.0, **29.8%** classed HIGH burden, **7.5%** contain a "pause" over 2000 ms.

### Why the events axis fails hardest — a design lesson

A spurious R-peak splits one interval into a **short** one followed by a
**longer** one. That is *precisely* the short-then-compensatory signature the
ectopy detector searches for. The detector is not merely degraded by detection
error — **it is tuned to the shape that detection error produces.** A missed
beat does the mirror image: two intervals merge into one long interval, read as
a pause.

An axis validated to Se 0.88-0.90 / PPV 0.95-0.98 against expert labels on two
held-out databases is, on this device, close to useless — and would have been
trusted on the strength of those numbers.

### What it would have cost

With the gate open, on healthy adults: **91 windows (7.5%) would escalate to
CRITICAL**, every one driven by a manufactured pause. The device-domain gate is
currently the only thing preventing that.

### THE RULE

**Every axis needs its OWN device-domain validation criterion. Held-out-database
performance does not transfer, and beat-COUNT accuracy is not a sufficient proxy
for any of them.** A detector could get the count exactly right while still
placing beats into short/long pairs — passing a count-based check and still
producing false ectopy.

## 47. A bug the wiring exposed

The first wiring passed the SQI gate result as the trust flag for **all three**
axes. Rate then came out UNDETERMINED in **97%** of device windows — directly
contradicting §46.

The error was conceptual. The bSQI gate protects **beat-position precision**,
which regularity needs because it measures interval-to-interval variation. Rate
is a **mean over 64 beats** and tolerates small position errors — the device
cross-check shows mean rate accurate to ~4% while regularity is unusable.
Gating rate on bSQI refuses a trustworthy measurement for a reason that does not
apply to it.

Rate now uses **signal-level quality only** (kurtosis, QRS-band power, flatline,
saturation), explicitly excluding bSQI and lag-1. After the fix: **rate NORMAL
87%, UNDETERMINED 13%** on the same capture.

Events remains gated on bSQI, correctly — it hunts short/long patterns, which do
need accurate beat positions.

## 48. Standing tests

| Test | Watches | State |
|---|---|---|
| `test_healthy_population_reads_regular` | regularity | **XFAIL by design** |
| `test_healthy_population_has_low_ectopy_burden` | ectopy | **XFAIL by design** |
| `test_healthy_population_has_no_pauses` | pauses | **XFAIL by design** |

Ectopy and pauses are split deliberately: they fail through *opposite* errors
(extra beats vs missed beats), so a partial fix could cure one and not the other.
All `xfail(strict=True)` — the suite goes red the moment any starts passing.

Suite: **76 passed, 3 xfailed, golden manifest unchanged.**


---

# PART VII — Layered classification: L4 ectopy, L5 morphology, combined (2026-08-30)

## 49. Three design elements removed by measurement

This part is mostly negative results. Each removed a capability that looked
supported until it was tested.

### 49.1 L4 cannot separate ectopy-driven from other irregularity — WITHDRAWN

A designed distinction between `IRREGULARITY_WITH_ECTOPY` and
`IRREGULARITY_WITHOUT_ECTOPY` is **not supported**.

| Group | n | median couplets | % with >=1 |
|---|---|---|---|
| **AFIB, zero true ectopic beats** | 23 | **5.0** | **91.3%** |
| sinus, zero true ectopic beats | 565 | 0.0 | 2.7% |
| sinus, >=3 true ectopic beats | 246 | 4.0 | 97.2% |

The short-then-long couplet is produced by ANY irregular rhythm. L4 is specific
against REGULAR rhythm (2.7%) but not against irregularity from another cause
(91.3%).

**This also reframes L4's earlier validation** (Se 0.877-0.899, PPV
0.951-0.978): its negatives were almost entirely regular windows, so it
demonstrated specificity against regular rhythm, not against irregular rhythm.
A validation-set composition failure.

**Consequence:** burden is interpretable ONLY when L3 says REGULAR. The pattern
was replaced with a single `IRREGULAR_RHYTHM`, couplet count reported and
explicitly annotated as not establishing an ectopic cause.

### 49.2 The learned burden estimator loses to a counting rule — DISCARDED

Ectopic-count MAE on held-out databases, L3-REGULAR windows only:

| | MITDB -> SVDB | SVDB -> MITDB |
|---|---|---|
| Model, 7 RR features | 2.824 | 1.480 |
| Baseline: RR CV alone | 2.764 | 2.875 |
| **Baseline: deterministic couplet counter** | **2.301** | **0.823** |
| Baseline: predict-the-mean | 3.489 | 3.261 |

The stop condition was "must beat RR CV alone". The model fails it in one
direction. More decisively, **the deterministic counter already in the codebase
beats the model in both directions**. A model that loses to a counting rule is
discarded.

High burden is not estimable at all: the >10 band gives MAE **15.9-23.9 beats**
(predicted mean 15.6 against true values to 64). Below 6 beats it is usable
(MAE 1.0-1.7).

### 49.3 L5 morphology survives downsampling — a prediction refuted

It was predicted that QRS-width separation would collapse at 89.70 Hz, making
L5 device-infeasible for physical reasons. **It does not.**

| Database | Native Cohen's d | 89.70 Hz Cohen's d |
|---|---|---|
| MITDB | 2.61 | 2.00 |
| SVDB | 1.47 | 1.33 |

| Direction | Native F1 | 89.70 Hz F1 | delta |
|---|---|---|---|
| MITDB -> SVDB | 0.6338 | 0.6248 | -0.0090 |
| SVDB -> MITDB | 0.7290 | 0.7117 | -0.0173 |

**Downsampling costs under 2 F1 points.** The prediction reasoned from true QRS
duration (80-100 ms vs >120 ms, a 1.4:1 ratio) but the feature measures width
at 50% of the beat's own peak (22-33 ms vs 45-58 ms, a 2:1 ratio) - a
peak-sharpness measure, not a duration. Proportionally wider separation, which
survives coarser sampling.

**Resolution is not an L5 blocker. The open lead and the broken detector are.**

## 50. L5 PVC morphology — full result

Built fresh; the archived classifier was not imported (V-class F1 0.830 on
MITDB, 0.513 held-out, predating current standards).

**PVC only.** A PAC conducts normally so its QRS is near-identical to a sinus
beat; the discriminator is the P wave, small and often buried in the preceding
T, unreliable single-lead. The prior attempt scored S-class F1 0.152 for this
reason.

| Direction | Grouped-CV F1 | Se | PPV | F1 |
|---|---|---|---|---|
| MITDB -> SVDB | 0.6952 +/- 0.207 | 0.5894 | 0.6854 | 0.6338 |
| SVDB -> MITDB | 0.6627 +/- 0.118 | 0.8813 | 0.6215 | 0.7290 |

Beats the archived 0.513 held-out; does not approach 0.830 within-database.

### False-positive rate by negative class — THE PRIMARY METRIC

| Direction / res | vs N | vs S | vs F |
|---|---|---|---|
| MITDB->SVDB native | 0.0122 (n=34,538) | 0.0424 (n=2,452) | n=4, too few |
| MITDB->SVDB 89.7 Hz | 0.0223 | **0.1705** | n=4, too few |
| SVDB->MITDB native | 0.0395 (n=16,164) | 0.1048 (n=229) | **0.3852** (n=122) |
| SVDB->MITDB 89.7 Hz | 0.0217 | 0.0786 | 0.2705 |

**F1 hid a 4x degradation.** Downsampling MITDB->SVDB moved F1 by -0.009 while
quadrupling the false-positive rate against supraventricular beats. Normals
dominate the negative set and mask the change.

**The asymmetry is a TRAINING-set composition effect.** MITDB contains only 229
supraventricular beats, so a model trained there never learns to reject that
morphology. Training on SVDB (2,452 S beats) reverses it. If L5 proceeds, train
on S-rich data.

**Deployment PPV is 0.62** (native), 0.54-0.70 at 89.7 Hz. A hard-negatives-only
PPV of 0.94 is a **base-rate artefact** - ~90% of false positives come from
normals purely because there are so many - and must not be quoted.

Fusion confusion is **MITDB-only** (n=122; SVDB has 4). It is the least-wrong
error available, since fusion beats are partly ventricular, and is reported
separately from S-class confusion.

## 51. Combined validation — what this instrument is for

See CLASSIFIER_DESIGN.md §9b for the full tables. Summary:

| Property | Result |
|---|---|
| Makes "irregular" actionable | **No** - PPV 0.4957 alone, 0.2183 layered, matched target |
| Reduces false reassurance | **Yes** - 739 -> 187 missed significant windows |
| False-reassurance rate | **34.2% -> 11.6%** |

**Settled framing: a false-reassurance-reduction instrument, not an alerting
one.** Operate for high NPV, surface "appears regular" as the actionable
output, never route "irregular" to an alarm.

## 52. Per-patient unpredictability is ONE systemic property

| Layer | Grouped-CV spread |
|---|---|
| L4 burden | MAE 3.211 +/- 2.100 (SD = 2/3 of mean) |
| L5 PVC morphology | F1 SD 0.12-0.21 |

Not two independent limitations: a systemic property of per-beat inference on
single-lead data. **No individual patient can be promised a detection rate.**

## 53. Standing analysis practices adopted

Recorded in CLASSIFIER_DESIGN.md §11, each after a specific failure:

1. No interpretive labels written into scripts before the numbers exist
   (3 occurrences in one session).
2. Never summarise a possibly-bimodal population with a median
   (2 contradictory summaries before bimodality was found).
3. Check negative-set composition - TEST and TRAINING - before believing any
   number.
4. The FP-rate-by-class table is primary for imbalanced negatives; F1 and PPV
   are summaries reported alongside, never instead.
5. Metrics that are uninterpretable alone travel in pairs: PPV with prevalence,
   sensitivity with specificity.
6. Test the claim, do not state it.

## 54. Unchanged

**Everything in Part VII is public data. None of it applies to the device.**
The lead configuration is OPEN, device beat detection is wrong by ~4.4%, and
the controlled recording is unscheduled. The classifier design is in good
shape; the device is exactly where it was.


---

# PART VIII — Assessment of BUILD_RHYTHM_MODEL.md (2026-09-01)

A build brief was supplied at `prorithm_ecg/1789/BUILD_RHYTHM_MODEL.md`. Full
assessment in `BUILD_BRIEF_GAP_ANALYSIS.md`. Summary:

**10 of its 12 steps are already built and validated.** Its "what to avoid" list
is already enforced in code.

**One instruction must NOT be followed.** Step 3 says "filter noise using light
ECG bandpass". The firmware already filters (`ecg_clean`); a second chain leaves
6-7% of the amplitude, and on public data at device resolution the UNFILTERED
arm scored best (F1 0.9890 vs 0.9831). The pipeline correctly applies none.

**One step is genuinely new: the waveform CNN (step 6).** Not built, and blocked
behind (a) device beat detection wrong by ~4.4%, (b) the OPEN lead
configuration, since waveform shape is lead-dependent, and (c) torch being
CPU-only on this host despite a GTX 1080 Ti. Hand-built lead-robust morphology
features already reach held-out F1 0.63-0.73; a CNN would plausibly beat that on
clean public data without removing either blocker, and would be harder to audit.
Recommendation: the deterministic verdict stays primary; a CNN annotates.

**One step is half done and worth doing: device-artefact simulation (step 8).**
Public data is already resampled to 89.70 Hz, but the measured device artefacts
- +/-36 ms BLE jitter, up to 49.3% replay, 10.2% dropout, ~8x motion spikes -
are characterised and not injected. Doing so would BOUND the expected device
degradation without needing the controlled recording. It does not replace that
recording, since simulated artefacts are not ground truth.

**Datasets:** PTB-XL (named in the brief) is not on this host and, being
10-second records, cannot produce a 64-beat window - same limitation as
challenge2017. INCARTDB (75 records, present, unused) is the one genuinely
useful addition, as a third held-out database.

**Windowing:** the brief suggests fixed 5/10/30 s windows; the system uses 64
beats. A fixed-second window holds a variable number of intervals, so RR CV over
5 s at 50 bpm (4 intervals) is not comparable to 5 s at 150 bpm (12 intervals).
Fixed-beat keeps the statistic comparable across rates, which matters because
rate and regularity are separate axes here.


---

# PART IX — Waveform + timing fusion: the waveform adds nothing (2026-09-02)

`BUILD_RHYTHM_MODEL.md` step 6 recommends a waveform + timing fusion model
(1D CNN on the waveform, RR features as side input). Built and tested on GPU.
**The recommendation is not supported by measurement.**

## 55. The ablation

Trained on LTAFDB (20,331 windows, 15 records), tested on MITDB (1,154 windows,
43 records) — held-out DATABASE. Identical windows and splits across arms.

| Arm | Held-out MITDB AUC |
|---|---|
| RR features only | **0.9865** |
| Waveform only | 0.9206 |
| **Fusion (waveform + RR)** | **0.9633** |
| Track A threshold (RR CV >= 0.1275) | 0.9511 |
| **Track B gradient boosting, RR only** | **0.9957** |

**Fusion is WORSE than RR alone** (-0.0232). The waveform branch does not merely
fail to contribute - it drags the fused model below the RR-only arm, which is
what an uninformative input that the model overfits looks like.

And the neural RR arm (0.9865) still loses to plain gradient boosting (0.9957).

**Interpretation:** rhythm IS timing. For a regularity target the information
lives entirely in the RR intervals, and the waveform contributes noise. This is
consistent with the project's founding premise - a morphology-free rhythm
indicator - and it is now measured rather than assumed.

**Decision: no CNN. The deterministic threshold plus Track B stands.**

## 56. Design choices that made the test fair

**The rate confound was made structurally impossible.** Windows are 64 BEATS
resampled to a fixed 2048 samples, so 64 beats at 50 bpm and at 150 bpm produce
identical array lengths - beat rate is not recoverable from the waveform input.
The RR side input is rate-normalised by construction. Neither branch could learn
"fast means irregular" (the cohort artefact measured at AUC 0.9221).

**The device signal path was simulated**, not just the sample rate: public data
was resampled to the native 133.83 Hz and then 2:3 decimated to the delivered
89.70 Hz. The decimation is SYSTEMATIC, as measured - the distinction matters,
since systematic decimation leaves RR CV essentially unchanged (0.0485 ->
0.0490) whereas random loss doubles it (0.1008).

**No filtering**, matching the device path and the device team's own filter
document.

## 57. Lead caveat on this result

Our standard requires lead verification wherever the WAVEFORM is processed.
LTAFDB headers name both channels `ECG` and do not identify the lead, so the
`wave` branch trained on an UNVERIFIED lead. The `rr` arm has no lead dependence
and is the control.

Because the waveform added nothing, the caveat is moot for this conclusion - but
it would have to be resolved before any positive waveform result could be
believed. The TEST side (MITDB) is verified MLII, and the device is confirmed
Lead II.

## 58. Data integrity issue found

LTAFDB record **110** has a truncated `.dat` (7.0 MB of an expected 44 MB, 16%).
Annotation-only work is unaffected - Track A, Track B and every RR result read
`.atr`, which is complete. The truncation only surfaced when waveforms were
first read. The earlier "16 complete records" check verified file EXISTENCE, not
completeness. 15 of 16 records have usable waveforms.

## 59. GPU environment

torch 2.5.1+cu121 on `/ssd_scratch/mahimakopalley/venv-gpu`, GTX 1080 Ti
(sm_61), 8,281 GFLOP/s measured. See `GPU_ENVIRONMENT.md`. Note that sm_61 is
absent from the build's arch list yet kernels run - verified by execution, not
by `torch.cuda.is_available()`, which returns True before any kernel has run.

## 60. Unchanged

**Everything in Part IX is public data.** The device blocker - beat detection
wrong by ~4.4%, needing the controlled recording - is untouched. Confirming
Lead II removed the lead blocker from morphology work; it did not remove that one.
