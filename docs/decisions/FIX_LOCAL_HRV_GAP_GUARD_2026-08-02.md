# Fix: missed-beat gaps leaking into the beat-classifier feature vector

**Date:** 2026-08-02 · **Branch:** `mod_1` · **Scope:** `_morphological_features()` guard + median fill

Found via the SHAP attribution work in `DIAGNOSTIC_REPORT_184B27_seg1_v3.md`.
This is a **classifier-input correctness bug**, not a threshold or reporting bug.

---

## The bug

`_morphological_features()` computed `local_hrv = rr_post_ms - rr_pre_ms` with no
`rr_flagged` guard. When an adjacent RR interval was a missed-beat gap across an
SQI-rejected stretch, that gap duration entered the classifier as if it were
physiology.

`recording_level_hrv()` and `RhythmContextEngine._afib_suspected()` both already
excluded flagged intervals. The feature extractor was the one place that did not.

**Concrete instance** — VitalPatch `184B27/seg1`, beat 27:

```
rr_pre  =  728 ms   (normal)
rr_post = 5832 ms   (missed-beat gap, rr_flagged=True)
local_hrv fed to classifier = 5104 ms
```

TreeSHAP ranked that value the **single largest attribution in the entire
segment**: `+2.93 toward class V`. The model called the beat ventricular because
the device lost signal for 5.1 seconds.

---

## The fix

Exactly as scoped — the guard goes in `_morphological_features()`. **`rr_flagged`
logic itself was not touched.**

1. `_morphological_features(..., rr_flagged: bool = False)` — emits
   `local_hrv = np.nan` when the beat's RR is flagged.
2. `beat_feature_vector()` passes `beat.rr_flagged` through.
3. `fill_missing_local_hrv()` (new) replaces NaN with the **median of that
   segment's valid `local_hrv` values**, called from `batch_feature_matrix()`.
4. Mirrored into `ecg_inference/features.py` so the deployment copy does not
   drift — the exact failure mode that left the SDNN 0.0 sentinel live in that
   package after it was fixed in `ecg_pipeline_core.py`.

**Why NaN and not 0.0:** `0.0` is a real and common `local_hrv` (perfectly regular
rhythm) and would be indistinguishable from "unknown" — the same sentinel trap as
the SDNN bug.

**Why median and not mean:** a high-ectopy strip has a long-tailed `local_hrv`
distribution that a mean would chase. The median of the segment's own valid beats
is a physiologically real value for this patient at this time.

**Edge cases verified:** all-NaN column → falls back to `0.0` and reports
`all_missing: True` rather than emitting NaN silently; empty matrix → no-op;
`timing_only=True` → skipped (no morphology block). On any single-vector path
that bypasses the median fill, **XGBoost handles NaN natively as `missing`** —
verified `predict_proba` returns finite probabilities on a NaN input.

**Deliberately NOT guarded:** `rr_pre` (feature 0) carries the same gap value when
the gap *precedes* the beat. That is outside the scope given and is flagged below
as an open item, not silently expanded into.

---

## 1. SHAP re-run — beat 27 (requested check)

**The gap is no longer the top attribution. Beat 27 is no longer in the five
lowest-confidence beats at all.**

```
BEFORE:  beat 27  pred=V  conf=0.4809
           1. local_hrv_ms   shap +2.9265   value 5104.0   <-- the gap
           2. rr_pre_ms      shap -1.2738   value  728.0

AFTER:   beat 27  pred=N  conf=1.0000
         (dropped out of the low-confidence set entirely)
```

The model went from an uncertain **V (0.48)** to a confident **N (1.0000)** once
the gap was removed. The new fifth-lowest beat is beat 41 (S, 0.4941).

Post-fix top attributions across the five lowest-confidence beats — no gap values
remain; `local_hrv` now ranges −712…+688 ms, all physiological:

| Beat | Pred | Conf | Top feature | SHAP | Value |
|---|---|---|---|---|---|
| 52 | V | 0.3760 | `rr_pre_ms` | +1.0266 | 328.0 |
| 79 | S | 0.3879 | `wavelet_d4_5` | +0.6162 | 1.7992 |
| 51 | S | 0.3973 | `local_hrv_ms` | −0.9466 | −80.0 |
| 68 | N | 0.3983 | `rr_pre_ms` | −2.6620 | 424.0 |
| 41 | S | 0.4941 | `rr_pre_ms` | +0.8777 | 416.0 |

## 2. Synthetic ground-truth suite (requested check)

**5/5 pass** — NORMAL, PVC_BURDEN, VT_RUN, AFIB_LIKE, NOISY. Unchanged.

## 3. New PVC burden for 184B27 seg1 (requested)

| | Pre-fix | Post-fix |
|---|---|---|
| **PVC burden** | **29.921%** | **27.559%** |
| V beats | 38 | 35 |
| Beat counts | N 72, S 9, V 38, F 0, Q 8 | N 77, S 7, V 35, F 0, Q 8 |
| **Risk level** | **CRITICAL** | **CRITICAL** (unchanged) |
| Deciding rule | PVC burden > 20.0% | PVC burden > 20.0% |
| Heart rate | 118.1 bpm | 118.1 bpm (unchanged) |

**Only 5 of 127 beats changed label, and every one was `rr_flagged`:**

| Beat | Time | Change | rr_pre | rr_post |
|---|---|---|---|---|
| 27 | 29.680 s | V → N | 728 | **5832** |
| 45 | 44.616 s | V → N | 640 | **5728** |
| 101 | 84.736 s | S → N | 456 | **7784** |
| 110 | 95.984 s | S → N | **280** | 584 |
| 117 | 99.888 s | V → N | 584 | **5328** |

Four of the five sat next to a multi-second detection gap. No unflagged beat
moved.

## 4. Corpus-wide verdict impact (requested)

Full VitalPatch corpus scored **twice in one pass** — identical detection,
identical beats, identical everything except the one changed feature — so the
comparison isolates this fix.

**3,043 segments · 350,540 beats · 0 errors · 2,275 s**

**Harness validated:** reproduces the known 184B27 seg1 result exactly
(V 38→35, PVC 29.9213→27.5591). **0 segments had more label changes than flagged
beats**, confirming only flagged beats moved.

### Verdict changes: 20 of 3,043 (0.66%)

| Transition | Count |
|---|---|
| HIGH → LOW | 11 |
| **CRITICAL → HIGH** | **5** |
| HIGH → MEDIUM | 3 |
| **LOW → HIGH** | **1** |

| Level | Pre | Post | Δ |
|---|---|---|---|
| CRITICAL | 15 | **10** | **−5** |
| HIGH | 979 | 971 | −8 |
| MEDIUM | 142 | 145 | +3 |
| LOW | 1,716 | 1,726 | +10 |
| NOT_ASSESSABLE | 191 | 191 | 0 |

The five CRITICAL→HIGH segments:

| Segment | Beats | Flagged | V | PVC % |
|---|---|---|---|---|
| `1778995259863_..._183594_seg0` | 152 | 47 | 26→14 | 17.11→9.21 |
| `1778300912197_..._1844AC_seg0` | 31 | 7 | 7→5 | 22.58→16.13 |
| `1780068602592_..._184B27_seg1` | 11 | 2 | 3→2 | 27.27→18.18 |
| `1778392310788_..._184B2F_seg0` | 11 | 6 | 3→2 | 27.27→18.18 |
| `1778926311535_..._184B2F_seg0` | 19 | 4 | 4→3 | 21.05→15.79 |

**Not one-directional — reported, not hidden.** One segment went **LOW → HIGH**
(`1778646373133_..._1844AC_seg0`, V 1→2, PVC 7.14→14.29). Removing a gap value
can move a beat *into* V as well as out. Three HIGH→LOW/MEDIUM transitions had
**no V change at all** (V 0→0 or 1→1) — those moved because label changes in
other classes shifted the AFib/PAC or HRV terms, not PVC burden.

### The scale of the contamination

| Measure | Value |
|---|---|
| `rr_flagged` beats | 10,928 (**3.12%** of corpus) |
| V beats pre-fix | 3,526 (1.01% of beats) |
| V beats post-fix | **1,187** (0.34% of beats) |
| **V beats removed** | **2,339 — 66.3% of all pre-fix V calls** |
| P(called V \| **gap** beat) | **21.40%** |
| P(called V \| normal beat) | **0.35%** |
| **Enrichment** | **a gap beat was 61× more likely to be called V** |

Gap beats were 3.12% of the corpus but accounted for **66.3% of all pre-fix PVC
calls**. That 61× enrichment is the signature of the bug, not a tuning
preference: two-thirds of every PVC this system has ever reported on VitalPatch
data were at least partly attributable to signal loss rather than ectopy.

**Why the verdict impact is so much smaller than the label impact:** PVC burden
thresholds are 10% / 20%, and most segments sit far from them. 1,606 segments had
at least one label change but only 20 crossed a threshold.

## 5. Clean-signal regression (MITDB, not requested — added as a control)

The frozen model was trained on MITDB using this same `beat_feature_vector`, so
changing it risks train/inference skew. Measured, rather than assumed:

**MITDB flagged-beat rate: 414 / 16,299 = 2.54%**, vs 15.7% on VitalPatch
`184B27/seg1`. The training domain is far cleaner, so the model was largely
trained *without* gap values in `local_hrv` and was extrapolating when fed them at
inference. **The guard moves inference toward the training distribution, not away
from it.**

8 MITDB records (100, 101, 103, 106, 111, 200, 208, 219), 16,299 beats:

| Record | Beats | V pre | V post | Level pre | Level post |
|---|---|---|---|---|---|
| 100 | 2,061 | 3 | **0** | LOW | LOW |
| 101 | 1,747 | 9 | **0** | LOW | LOW |
| 103 | 1,957 | 12 | **0** | LOW | LOW |
| 106 | 1,854 | 329 | 323 | HIGH | HIGH |
| 111 | 1,908 | 101 | 84 | CRITICAL | CRITICAL |
| 200 | 2,204 | 613 | 605 | CRITICAL | CRITICAL |
| 208 | 2,682 | 623 | 611 | CRITICAL | CRITICAL |
| 219 | 1,886 | 32 | 22 | HIGH | HIGH |

**0 verdict changes. 104 / 16,299 labels changed (0.64%).** Records 100/101/103
are predominantly normal sinus and went to **0 spurious V beats**, while the
genuinely ectopic records (106, 200, 208) kept the overwhelming majority of theirs
(323/329, 605/613, 611/623). The fix removes false positives without gutting true
ones.

### Scored against MITDB ground-truth annotations

The above only shows counts moving. This scores predictions against the actual
`.atr` beat annotations (AAMI V class = symbols `V`/`E`), matched to detected
beats within 150 ms — i.e. **does the fix make the classifier more or less
correct?**

| Record | True V | TP pre | FP pre | FN pre | TP post | **FP post** | FN post |
|---|---|---|---|---|---|---|---|
| 100 | 0 | 0 | 3 | 0 | 0 | **0** | 0 |
| 101 | 0 | 0 | 9 | 0 | 0 | **0** | 0 |
| 103 | 0 | 0 | 12 | 0 | 0 | **0** | 0 |
| 106 | 468 | 328 | 1 | 140 | 323 | **0** | 145 |
| 111 | 1 | 1 | 96 | 0 | 1 | **79** | 0 |
| 200 | 717 | 610 | 2 | 107 | 604 | **0** | 113 |
| 208 | 888 | 598 | 23 | 290 | 596 | **14** | 292 |
| 219 | 53 | 21 | 11 | 32 | 21 | **1** | 32 |

| | Pre-fix | Post-fix | Change |
|---|---|---|---|
| True positives | 1,558 | 1,545 | −13 |
| **False positives** | **157** | **94** | **−63 (−40.1%)** |
| False negatives | 569 | 582 | +13 |
| **Precision** | 0.9085 | **0.9426** | **+0.0341** |
| Recall | 0.7325 | 0.7264 | −0.0061 |
| **F1** | 0.8110 | **0.8205** | **+0.0095** |

**The fix is a net accuracy improvement against ground truth**, not just a
count reduction. It eliminates **63 false-positive PVCs at the cost of 13 true
ones — a ~5:1 favourable trade** — and F1 rises. On the three records with **zero
true PVCs**, it removed **100% of the false positives** (3→0, 9→0, 12→0). On
record 219 it removed 10 of 11 false positives while keeping every true positive.

This is the strongest evidence available: unlike the VitalPatch corpus numbers,
these are scored against real clinician annotations.

(For orientation: the frozen model's published DS2 V-class F1 is 0.826; this
8-record subset is not DS2, so the 0.811/0.821 figures are not directly
comparable to it — they are only comparable to each other, which is the point.)

---

## What is NOT resolved

1. **`rr_pre` (feature 0) has the same contamination** and was left unguarded per
   the stated scope. When a gap *precedes* a beat, `rr_pre` carries the gap
   duration directly. It is visibly load-bearing — `rr_pre_ms` is the top
   attribution for 3 of the 5 lowest-confidence beats post-fix. Same guard would
   apply; it needs a decision, not an assumption.

2. **The frozen model was trained with the unguarded feature.** The MITDB control
   shows the skew is small and in the right direction (F1 improves), but the
   principled fix is to retrain with the guard active. Not done here — retraining
   is out of scope and would invalidate the published DS2 F1 figures until
   re-measured.

3. **No clinician-annotated VitalPatch labels exist.** The MITDB result shows the
   fix improves accuracy on clean annotated data from a *different device*. It
   does not establish that the remaining 1,187 V beats on VitalPatch are correct.
   Record 111 still carries 79 false positives after the fix, so this bug was
   never the only source of V-class error.

4. **The 2,339 removed V beats have not been individually reviewed.** The 61×
   enrichment is strong statistical evidence they were artifacts, not proof that
   every one was.

5. **Saved reports on disk are now doubly stale** — pre-`t_inspect_period` *and*
   pre-this-fix. Anything under `data/reports/vitalpatch/` should be regenerated
   before it is read as current.
