# Beat-by-Beat ECG Classification — Summary

Scope of this document: **only the beat-level classifier** (labeling each
individual heartbeat as N / S / V / F / Q). MedGemma integration, AFib-burden,
and student-distillation work are tracked separately and are out of scope here.

## What the system does

Classifies every individual heartbeat in an ECG recording into one of 5
standard AAMI clinical categories:

| Class | Meaning |
|---|---|
| N | Normal beat |
| S | Supraventricular ectopic (irregular, upper chambers) |
| V | Ventricular ectopic (dangerous, lower chambers) |
| F | Fusion beat (mixed normal + ventricular) |
| Q | Unknown / unclassifiable |

## Block diagram — processing flow

```
Raw ECG signal (real patient recordings, MIT-BIH database)
        │
        ▼
Preprocessing (filter noise, resample to standard rate)
        │
        ▼
Beat detection (find each individual heartbeat / R-peak)
        │
        ▼
Feature extraction (shape + timing features for that one beat)
        │
        ▼
XGBoost classifier  →  one label per beat:
        │                N = Normal
        │                S = Supraventricular
        │                V = Ventricular
        │                F = Fusion
        │                Q = Unknown
        ▼
Per-beat labeled output (used by everything downstream)
```

## Data / evaluation setup

- Source: MIT-BIH Arrhythmia Database (standard clinical benchmark, per-beat
  annotated by cardiologists).
- **Patient-level split** — no patient appears in both train and test, so the
  model is always scored on patients it has never seen:
  - `DS1_TRAIN` — used to fit the model.
  - `DS1_VAL` — used to make model-selection decisions (10 records, enlarged
    from an original 4 — see "Validation-split question," below).
  - `DS2` — held-out, report-only, 22 records. Never used to pick anything;
    only used to report final numbers.
- **Safety rule (zero-tolerance):** any candidate model that regresses V-class
  (ventricular) detection is disqualified, even if it improves other classes.
  V beats are the clinically dangerous ones, so this is treated as
  non-negotiable, not a tunable trade-off.

## Metrics achieved so far (DS2, held-out)

> **Read the two blocks below separately — they were measured under different
> code.** Every row in the comparison table was produced by the 2026-07-16/23
> runs and is internally consistent *with the other rows*, so the A/B verdicts
> ("discarded", "rejected") still stand. The production model's *absolute*
> numbers have since shifted slightly because the shared feature path changed
> (`segment_beats`, `to_target_rate`, `robust_zscore` — commits `2420ecc`,
> `e6bf01f`), not because the model was retrained. Quote the current-code
> figures, not the table, when stating what production does today.

**Current production figures (recomputed 2026-08-03, current code)** — full
per-class table and confusion matrix in
[`docs/CLASSIFIER_EVAL_MITDB_SVDB_2026-08-03.md`](../CLASSIFIER_EVAL_MITDB_SVDB_2026-08-03.md):

| Model | Macro-F1 | S F1 | V F1 | F F1 | Measured |
|---|---|---|---|---|---|
| **Production (`five_class_xgb.json`)** | **0.3906** | **0.152** | **0.830** | **0.005** | 2026-08-03, current code |

**Comparison table (all rows measured 2026-07-16/23, older feature path):**

| Model | Macro-F1 | S F1 | V F1 | F F1 | Status |
|---|---|---|---|---|---|
| Production (`five_class_xgb.json`) — as measured then | 0.3895 | 0.139 | 0.826 | 0.011 | baseline all rows below compare against |
| + LTAFDB/SDDB data enrichment | 0.3779 | 0.098 | 0.808 | 0.011 | discarded (worse) |
| + 7 timing features | 0.3883 | 0.182 | 0.775 | 0.010 | discarded (V regressed) |
| + 6 timing features (no compensatory-pause flag) | 0.3759 | 0.125 | 0.763 | 0.016 | discarded (S and V both worse) |
| Morphology-only features (56-dim) | 0.4026 | 0.160 | **0.866** | 0.021 | motivated the two-stage attempt |
| Timing-only features (7-dim) | 0.2895 | 0.054 | 0.446 | 0.000 | confirms timing alone is weak |
| Flat model, retrained on enlarged 12-record DS1_TRAIN | 0.3935 | 0.154 | 0.847 | 0.000 | ~equivalent to production |
| **Two-stage v5** (Stage 1 gate + Stage 2 S/V/F, enlarged split) | 0.4343 | **0.383** | 0.789 | 0.043 | **rejected — V regression** |

**Two-stage validation-split check** (the question we specifically had to close
before trusting any two-stage result):

| Threshold | Old DS1_VAL (4 pts) S-recall | New DS1_VAL (10 pts) S-recall | DS2 (22 pts) S-recall |
|---|---|---|---|
| 0.05 | 0.781 | 0.778 | 0.942–0.948 |
| 0.50 | 0.644 | 0.618 | 0.875–0.899 |
| 0.95 | 0.525 | 0.488 | 0.802–0.803 |

Enlarging the validation set 4→10 patients reproduced the same steep S-recall
drop-off, ruling out "small sample size" as the explanation — confirmed as a
real DS1-population property, not a sampling artifact.

## Fresh full 5-class two-stage run (verified 2026-07-23)

Re-ran the two-stage classifier **from scratch** today, training freshly on
real data, to confirm the earlier verdict wasn't stale and to get true,
complete per-class scores for all 5 AAMI classes (including N and Q, which
the earlier summary table above didn't show).

**Command:**
```
python -m ecg_pipeline.ecg_pipeline_tools train-twostage \
    --out-prefix models/five_class_xgb_twostage_beatonly
```

**Data used:** DS1_TRAIN (12 MITDB records) + all 78 SVDB records for
training, DS1_VAL (10 records) for threshold tuning, DS2 (22 held-out
records, 40,634+ N beats / 1,795 S / 3,005 V / 363 F / 7 Q) for the reported
scores below. Same patient-level splits described above — DS2 patients are
never touched during training or tuning.

**Architecture:** Stage 1 = Normal-vs-Abnormal gate (63-dim: 56
morphology/wavelet + 7 timing), threshold picked on DS1_VAL via Youden's J
→ selected threshold 0.05. Stage 2 = S/V/F discriminator (56-dim,
morphology-only), trained only on true-abnormal beats.

**True per-class scores on DS2 (held-out, never used for tuning):**

| Class | Sensitivity | Precision | F1 | Support |
|---|---|---|---|---|
| N | 0.921 | 0.994 | 0.956 | 40,634 |
| S | 0.624 | 0.276 | 0.383 | 1,795 |
| V | 0.922 | 0.690 | 0.789 | 3,005 |
| F | 0.025 | 0.173 | 0.043 | 363 |
| Q | 0.000 | 0.000 | 0.000 | 7 |

Macro-F1: **0.4343**. Overall accuracy: 0.9027 (not the metric that matters
here — accuracy is dominated by the 40k N beats).

**Confusion matrix** (rows = true, cols = predicted; order N, S, V, F, Q):

```
        pred_N  pred_S  pred_V  pred_F  pred_Q
true_N   37445    2521     634      34       0
true_S     105    1120     569       1       0
true_V      65     160    2772       8       0
true_F      51     259      44       9       0
true_Q       5       1       1       0       0
```

**Vs. the flat production baseline (same DS2 set):**

| Class | Flat baseline F1 | Two-stage F1 | Δ | Verdict |
|---|---|---|---|---|
| N | 0.966 | 0.956 | −0.010 | within tolerance |
| S | 0.160 | 0.383 | **+0.223** | large improvement |
| V | 0.866 | 0.789 | **−0.077** | **regression — exceeds zero-tolerance margin** |
| F | 0.021 | 0.043 | +0.022 | small improvement |
| Q | 0.000 | 0.000 | 0.000 | unchanged (unlearnable — only 7 DS2 examples, dropped from training entirely) |
| Macro-F1 | 0.4026 | 0.4343 | +0.0317 | improves on paper only |

**Verdict (per `AGENT_RULES.md` Rule 8 — V/N are zero-tolerance):** **NOT
promotable.** V's F1 drop (−0.077) and precision drop (0.828 → 0.690) exceed
the allowed regression margin, even though S nearly doubled and macro-F1
went up. This exactly reproduces the earlier v5 result (macro-F1 0.4343, S
F1 0.383, V F1 0.789) — confirming today, on an independently retrained
model, that the trade-off is real and repeatable, not a one-off artifact.

Model artifacts from this run: `five_class_xgb_twostage_beatonly_{stage1,stage2,threshold}.json`
(saved to `ecg_pipeline/models/`, **not committed, not promoted**).

## What we've achieved

1. Built and froze a working 5-class beat classifier (production baseline),
   patient-split, evaluated on a held-out 22-patient set.
2. Ran a structured search for improvements: data enrichment, extra timing
   features, morphology-only vs. timing-only feature ablations.
3. Identified that morphology-only features push V F1 to 0.866 (best V result
   found) — this is what motivated trying a two-stage architecture.
4. Designed and built a two-stage classifier (Stage 1: abnormal/normal gate,
   Stage 2: S vs V vs F discrimination) — 5 iterations (v1–v5).
5. Caught a real methodological risk before trusting the result: the first
   negative verdict came from only a 4-patient validation set. Rather than
   accept or reject on that alone, enlarged it to 10 patients and reran the
   full comparison from scratch on independently retrained models.
6. Got a confirmed, reproducible answer: two-stage genuinely trades V-class
   accuracy for S-class accuracy, on both the small and the enlarged
   validation set — closed as a real limitation, not unresolved.
7. Enforced the zero-tolerance safety rule throughout: every candidate that
   improved S/macro-F1 at V's expense was rejected, with no exceptions made.
8. Kept an audit trail (`docs/archive/ABLATION_REPORT.md`) of every
   experiment tried, including the ones that failed, so decisions are
   traceable rather than just "we tried some things."

## What's stuck / open

- **The core trade-off**: every experiment so far — feature engineering,
  data enrichment, two-stage architecture — hits the same wall: getting
  better at rare classes (S, F) costs V-class accuracy beyond what's
  allowed. No approach tried yet escapes this.
- **F-class remains essentially unsolved** (F1 ≈ 0.01–0.04 across every
  variant) — there's very little F-class data in MIT-BIH to begin with.
- No experiment run so far is promotable as a production replacement;
  `five_class_xgb.json` (the original flat model) remains production,
  unchanged.

## What to do more (next steps, not yet started)

1. Decide, as a team, whether current per-class performance is "good enough
   to freeze" — i.e., accept the flat model's S/F weakness as a known,
   documented limitation and move downstream — or keep searching for a
   fundamentally different approach (not just more feature tweaks on the
   same architecture, since that family has now been tried multiple ways).
2. If continuing to search: look at approaches that don't share the
   "single discriminative feature space" limitation the two-stage attempts
   all had — e.g. sequence/context-aware models that use neighboring beats,
   not just single-beat features, since S and V beats are partly defined by
   their relationship to surrounding rhythm, not shape alone.
3. If freezing now: formally document the frozen model's per-class
   limitations (this table) as a known constraint for anything built on top
   of it (rhythm-level analysis, MedGemma integration), so downstream
   consumers don't assume uniform reliability across all 5 classes.
4. Either way, this decision is the gating step — the rest of the pipeline
   (rhythm analysis, MedGemma integration) consumes whatever classifier is
   frozen here, so it's blocked on this call being made explicitly, not
   left implicit.
