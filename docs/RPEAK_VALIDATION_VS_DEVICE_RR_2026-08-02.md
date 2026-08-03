# R-peak detection validated against device-reported RR intervals

**Date:** 2026-08-02 · **Branch:** `mod_1` · **Measurement only — no pipeline code was changed.**

Independent validation of pipeline R-peak detection on VitalPatch, using the
device firmware's own per-beat RR intervals (vitals CSV **column 6**) as
reference. **No clinician required.**

Comparison is between **medians of RR-interval durations**, which sidesteps
clock alignment entirely: an interval is a duration whether or not the two
devices agree on absolute time. Classification (stage 7) was skipped — only
detection is under test.

---

## Headline

| Criterion (pre-registered) | Threshold | Measured | Verdict |
|---|---|---|---|
| Disagreement > 50 ms | fail if > 10% of segments | **1.09%** | **PASS** |
| Disagreement < 20 ms | validated if > 90% of segments | **93.29%** | **PASS** |

> **Validation statement.** Across **2,651 real VitalPatch ECG segments**, pipeline-detected
> median RR agrees with the device's own firmware-reported RR to within **20 ms on
> 93.3% of segments** (mean absolute difference **6.47 ms**, median **4.0 ms**).
> Expressed as heart rate — the quantity a clinician reads — the two agree within
> **5 bpm on 98.9% of segments**, mean absolute difference **0.68 bpm**.
> Reference: **4,364,472** device-reported RR intervals, 99.9% physiological.

---

## Step 1 — Device RR census (vitals column 6)

Filtered to the physiological range 300–2000 ms.

| Patient | Files | col 6 non-empty | In range | < 300 ms | > 2000 ms | Median RR | Median HR |
|---|---|---|---|---|---|---|---|
| 183594 | 412 | 866,610 | 864,971 | 1,247 | 392 | 656 ms | 91.5 |
| 1844AC | 502 | 774,840 | 774,501 | 211 | 128 | 784 ms | 76.5 |
| 184635 | 480 | 776,998 | 776,490 | 499 | 9 | 704 ms | 85.2 |
| 1849DF | 504 | 805,592 | 804,916 | 665 | 11 | 672 ms | 89.3 |
| 184B27 | 189 | 304,254 | 302,181 | 1,435 | 638 | 600 ms | 100.0 |
| 184B2F | 425 | 841,730 | 841,413 | 264 | 53 | 688 ms | 87.2 |
| **TOTAL** | **2,512** | **4,370,024** | **4,364,472** | 4,321 | 1,231 | **688 ms** | **87.2** |

**99.87% of device readings fall inside the physiological range** — the column is
a clean signal, not noise. Per-patient median HR spans 76.5–100.0 bpm, all
plausible for post-operative monitoring.

## Step 2 — Per-segment comparison

For each ECG segment, device RR readings whose own timestamps fall inside the
segment's `[t_start, t_end]` window were selected, and their median compared
against the median of the pipeline's non-flagged RR intervals.

Requirements: ≥10 device RR readings and ≥10 pipeline RR intervals in-window.

### Coverage

| | Segments | % |
|---|---|---|
| **Compared** | **2,651** | **73.0%** |
| Skipped — too few device RR in window | 841 | 23.2% |
| Skipped — too few pipeline RR | 99 | 2.7% |
| Skipped — insufficient signal | 41 | 1.1% |
| **Attempted** | **3,632** | 100% |

The 23.2% skipped for sparse device RR is a **device logging gap, not a pipeline
failure** — the firmware does not emit RR continuously for every window.

### Agreement

| Metric | Value |
|---|---|
| Segments compared | 2,651 |
| **Mean absolute difference** | **6.47 ms** |
| Median absolute difference | 4.0 ms |
| Mean *signed* difference | **−1.35 ms** (negligible bias) |
| Within 10 ms | 2,231 (84.2%) |
| **Within 20 ms** | **2,473 (93.29%)** |
| Within 50 ms | 2,622 (98.91%) |
| **Over 50 ms** | **29 (1.09%)** |
| p90 / p99 / max | 16 ms / 54 ms / 208 ms |

The signed mean of −1.35 ms (0.2% of a 688 ms interval) means the pipeline runs
a hair *fast* relative to the device, with no meaningful systematic bias.

### In heart-rate terms

| Metric | Value |
|---|---|
| Mean absolute HR difference | **0.68 bpm** |
| Median absolute HR difference | 0.20 bpm |
| **Within 5 bpm** | **98.87%** |
| Over 5 bpm | 1.13% |

This is the directly meaningful number, and it independently corroborates the
`heart_rate_bpm` field added in v3 (median of non-flagged RR) — the method choice
is now validated at corpus scale, not just on one segment.

## Step 3 — Where the disagreements are

### By patient — the corpus-level pass is not uniform

| Patient | n | Mean abs diff | Median | Within 20 ms | > 50 ms |
|---|---|---|---|---|---|
| 183594 | 369 | 2.49 ms | **0.0 ms** | **98.64%** | 0.27% |
| 184B2F | 557 | 4.20 ms | **0.0 ms** | **97.67%** | 0.18% |
| 1844AC | 514 | 5.21 ms | **0.0 ms** | **95.72%** | 0.39% |
| 1849DF | 517 | 7.80 ms | 4.0 ms | 90.72% | 1.74% |
| **184B27** | 186 | 11.25 ms | 8.0 ms | **87.63%** | 2.69% |
| **184635** | 508 | 10.02 ms | 8.0 ms | **86.81%** | 2.17% |

**Flagged honestly: 2 of 6 patients fall below the 90% bar** (184635 at 86.8%,
184B27 at 87.6%). The corpus-wide 93.3% passes, but the claim "≥90% on every
patient" would be false. Three patients have a **median absolute difference of
exactly 0 ms** — bit-identical medians.

### Disagreement tracks signal quality, monotonically

| SQI rejection rate | n | Mean abs diff | Within 20 ms | > 50 ms |
|---|---|---|---|---|
| < 10% | 1,302 | **4.60 ms** | 96.5% | 0.5% |
| 10–30% | 1,120 | 7.85 ms | 90.8% | 1.6% |
| 30–50% | 138 | 9.25 ms | 89.9% | 0.7% |
| > 50% | 91 | **11.91 ms** | 83.5% | 3.3% |

Clean monotonic degradation as signal quality worsens — the expected and benign
pattern. On the cleanest half of the corpus, agreement is 96.5% within 20 ms.
Correlation is real but weak (r = 0.19 with SQI rejection, r = 0.23 with flagged
fraction), so signal quality explains the *trend* but not most of the variance.

### The 29 outliers

**26 of 29 have the pipeline reporting a SHORTER median RR than the device** —
i.e. residual **over-detection** (extra beats), not missed beats. Only 3 are
misses. This is the same failure mode the `t_inspect_period` fix addressed, now
down to roughly 1% of segments rather than the corpus-wide problem it was.

Worst cases:

| Patient | Device RR | Pipeline RR | Diff | SQI rej | Device n |
|---|---|---|---|---|---|
| 184B27 | 1032 | 824 | −208 ms | 0.13 | 129 |
| 184B27 | 1096 | 904 | −192 ms | 0.04 | 120 |
| 1849DF | 1016 | 888 | −128 ms | 0.17 | 123 |
| 183594 | 592 | 492 | −100 ms | 0.59 | 58 |
| 184635 | 944 | 856 | −88 ms | 0.24 | 116 |

Note the two worst have **low SQI rejection (0.13, 0.04)** — clean signal, yet
still ~200 ms apart. These are not explained by noise and deserve individual
review. A few other outliers have very low device counts (15, 34, 41 readings),
where the device's own median is itself unstable.

---

## What this does and does not establish

**Does establish:**
- Pipeline R-peak detection agrees with an independent, device-internal reference
  across 2,651 real segments and 4.36M reference intervals.
- The `heart_rate_bpm` method (median of non-flagged RR) is corroborated at scale.
- Residual detection error is small, quality-correlated, and predominantly
  over-detection.

**Does not establish:**
- **Beat classification accuracy.** This validates *detection* — where beats are —
  not *labels*. Whether a beat is N/S/V/F is untouched by this and still has no
  VitalPatch ground truth.
- **That the device is correct.** VitalPatch firmware RR is an independent
  reference, not a gold standard; it has its own undisclosed detection algorithm.
  Agreement means the two concur, which is strong evidence but not proof both are
  right. A shared failure mode would be invisible here.
- **Anything about the 23.2% of segments** the device did not log enough RR for.
- **Per-patient uniformity** — 2 of 6 patients sit below the 90% bar.

## Why this matters for the classifier

The SHAP work showed the classifier leans on **timing** features (`rr_pre_ms`,
`local_hrv_ms`) far more than morphology. Those features are computed directly
from the RR series validated here. So this result substantiates the *dominant
input* to beat classification, even though it says nothing about the labels
themselves — the largest single piece of VitalPatch validation obtainable without
clinician annotation.
