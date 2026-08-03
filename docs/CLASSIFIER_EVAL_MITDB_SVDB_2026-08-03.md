# Beat-classifier evaluation on MIT-BIH (MITDB) and SVDB

**Date:** 2026-08-03 · **Branch:** `mod_1` · **Report-only — no model was trained,
retrained, or written. Nothing under `models/` was modified.**

Scores the beat classifier against expert `.atr` ground-truth annotations on
the two labeled databases on disk. This is the gold-standard counterpart to
`RPEAK_VALIDATION_VS_DEVICE_RR_2026-08-02.md`, which could only validate
*detection* (VitalPatch is unlabeled). Here the *classification* decision is
scored directly against cardiologist beat labels.

Per `AGENT_RULES.md` rule 4, accuracy is reported but is not the metric —
N is 89% of DS2 and 89% of SVDB, so a predict-N-always model scores ~0.89.

---

## Headline

| Finding | Evidence |
|---|---|
| V-class holds up on MITDB DS2 | V F1 **0.830**, sensitivity 0.907 |
| V-class **does not transfer** to an unseen database | V F1 **0.513** on SVDB, precision collapses 0.765 → **0.377** |
| S-class is unsolved on both | S F1 0.152 (DS2), 0.130 (SVDB) |
| F-class is unsolved | F1 0.005 (DS2); SVDB has only 22 F beats, not measurable |
| SVDB in training is earning its keep | Removing it costs DS2 macro-F1 0.398 → 0.379, S F1 0.162 → 0.036 |
| **Production must never be scored on SVDB** | It was trained on all 78 SVDB records; doing so yields a fake macro-F1 of 0.761 |

---

## Setup

| | |
|---|---|
| Production model | `ecg_pipeline/models/five_class_xgb.json` (identical bytes to `ecg_inference/models/`, md5 `9d0e7810…`) |
| Architecture | XGBoost, 56-dim morphology-only features, 4 output classes `["F","N","S","V"]` |
| Ablation artifacts | `training/model_artifacts/five_class_xgb_{with,no}_svdb.json` |
| MITDB DS2 | 22 records, 45,881 annotated beats — held-out, report-only (rule 1) |
| SVDB | 78 records, 172,139 annotated beats |
| Feature path | `_load_record_beats` — annotation positions, not runtime detection, so this isolates classification from R-peak detection |
| Runner | shipped `eval-classifier` CLI for DS2; a direct `evaluate()` call for SVDB (the CLI hardcodes `data_root/mitdb`) |

**Q class:** the production model has no Q output — `train-classifiers`
defaults to `--drop-q` because Q is too rare to learn (7 beats in all of DS2,
68 in SVDB). Q therefore scores 0.000 by construction, not by failure. It is
excluded from the interpretation below but left in the tables as-is.

---

## 1. Production model on MITDB DS2 (the honest headline number)

45,881 beats, 22 patients never seen in training.

| class | sensitivity | precision | F1 | support |
|---|---|---|---|---|
| N | 0.966 | 0.965 | 0.966 | 40,711 |
| S | 0.140 | 0.165 | **0.152** | 1,795 |
| V | 0.907 | 0.765 | **0.830** | 3,005 |
| F | 0.003 | 0.083 | **0.005** | 363 |
| Q | 0.000 | 0.000 | 0.000 | 7 |

**Macro-F1 0.3906** · accuracy 0.9224 (not the metric)

Confusion matrix, rows = true, cols = pred, order `[N,S,V,F,Q]`:

```
[[39342   960   401     8     0]
 [ 1169   252   374     0     0]
 [  156   121  2725     3     0]
 [  112   191    59     1     0]
 [    5     1     1     0     0]]
```

Dominant error modes:

- **`S→N` 65.1%** (1,169/1,795) — two thirds of supraventricular beats are
  called normal. This is the single largest clinical gap: missed SVEs.
- **`S→V` 20.8%** (374/1,795) — most of the remaining S beats get called
  ventricular, i.e. over-escalated.
- **`F→S` 52.6%** (191/363), with 1 of 363 F beats classified correctly.
- **`V→S` 4.0%** (121/3,005) — V is rarely lost, which is the behaviour the
  zero-tolerance V rule in `BEAT_CLASSIFICATION_SUMMARY.md` is protecting.

### Drift vs. the documented table

`docs/thesis/BEAT_CLASSIFICATION_SUMMARY.md` records macro-F1 0.3895 / S 0.139 /
V 0.826 / F 0.011 for this same model file. Measured today: 0.3906 / 0.152 /
0.830 / 0.005.

This is **not** a model change — the `.json` is untouched. That table was last
written at `16a04d2` (2026-07-24) and `ecg_pipeline/ecg_pipeline_core.py` has
changed four times since. Two of those commits edit functions that sit
directly in the eval feature path used by `_load_record_beats`:

- `2420ecc` — `segment_beats` (snap XQRS peaks to true local max)
- `e6bf01f` — `to_target_rate`, `robust_zscore`, `segment_beats`

So the same weights now see slightly different features. The direction is
mixed (S and V up, F down) and the magnitudes are small, but the previously
published numbers should be treated as **stale relative to current code**, not
as a second opinion on it.

**Resolved 2026-08-03.** The figures in this document are now the ones quoted
by every current-state surface: `docs/thesis/BEAT_CLASSIFICATION_SUMMARY.md`
(which now separates current-code figures from the older internally-consistent
A/B table), `README.md`, `docs/PROJECT_STATUS.md`,
`docs/thesis/PIPELINE_METHODS_AND_RESULTS.md`,
`docs/thesis/PROJECT_OVERVIEW.md`, `docs/HANDOFF.md`,
`ecg_inference/INFERENCE_README.md`, and the clinician-facing strings in
`ecg_inference/report.py`, `ecg_pipeline/agent_bridge.py`,
`scripts/generate_clinician_report.py`, `scripts/generate_validation_pdf.py`.

Two places were **deliberately left on the old numbers**, because both are
frozen A/B references whose partner values were measured under the same older
code — updating one side alone would compare across code versions:
`training/ecg_pipeline_tools.py`'s `baseline_ds2`/`timing_v1_ds2` pair (which
sets the V-regression disqualification boundary) and the dated diagnostic /
audit / experiment-history documents, which are point-in-time records.

---

## 2. Cross-database generalization: SVDB as a genuinely unseen database

The production model was trained on **all 78 SVDB records**
(`train-classifiers --include-svdb` defaults to on), so it cannot be scored
there. The clean substitute is `five_class_xgb_no_svdb.json`, for which SVDB
is an entirely unseen database *and* an unseen patient population.

172,139 beats, 78 records.

| class | sensitivity | precision | F1 | support |
|---|---|---|---|---|
| N | 0.930 | 0.945 | 0.937 | 152,581 |
| S | 0.077 | 0.416 | **0.130** | 10,531 |
| V | 0.803 | 0.377 | **0.513** | 8,937 |
| F | 0.045 | 0.001 | 0.002 | 22 |
| Q | 0.000 | 0.000 | 0.000 | 68 |

**Macro-F1 0.3163** · accuracy 0.8705 (not the metric)

```
[[141868   1074   8840    799      0]
 [  6574    810   3003    144      0]
 [  1614     60   7173     90      0]
 [    21      0      0      1      0]
 [    32      1     35      0      0]]
```

### The finding: V precision, not V sensitivity, is what breaks

Comparing the same `no_svdb` model on both databases:

| | MITDB DS2 | SVDB (unseen DB) |
|---|---|---|
| V sensitivity | 0.870 | 0.803 |
| V **precision** | 0.686 | **0.377** |
| V F1 | 0.767 | **0.513** |
| N→V false-positive rate | 419/40,711 = **1.03%** | 8,840/152,581 = **5.79%** |

V *sensitivity* transfers acceptably (0.870 → 0.803). V *precision* does not:
on an unseen database the model calls 5.79% of normal beats ventricular, a
5.6× increase in the normal-beat false-alarm rate. Because N is ~89% of any
recording, that small-looking percentage is what destroys precision — 8,840
false V's against only 7,173 true ones.

This matters for the deployment claim. "V F1 0.83, V is the reliable class"
is a **MITDB DS2** statement. On a database the model has not seen, the same
architecture produces roughly one false ventricular alarm for every true one.
VitalPatch is a third, different, unlabeled distribution, so the honest
expectation for real device data is nearer the SVDB figure than the DS2
figure — and there is no way to confirm that without labels.

F on SVDB has 22 beats total; F numbers there are not measurable and should
not be quoted.

---

## 3. Does SVDB in the training set help? Yes.

Both artifacts scored on the same held-out MITDB DS2, 45,881 beats.

| | macro-F1 | S F1 | V F1 | F F1 | N F1 |
|---|---|---|---|---|---|
| `no_svdb` | 0.3788 | 0.036 | 0.767 | 0.120 | 0.971 |
| `with_svdb` | **0.3979** | **0.162** | **0.847** | 0.015 | 0.965 |
| production (shipped) | 0.3906 | 0.152 | 0.830 | 0.005 | 0.966 |

Adding SVDB to training improves DS2 S F1 4.5× (0.036 → 0.162) and V F1
0.767 → 0.847, at the cost of F (0.120 → 0.015, on 363 beats). The
`--include-svdb` default is justified by held-out evidence.

`with_svdb` and the shipped production model follow the same recipe but are
separate training runs and are not byte-identical (16.3MB vs 15.7MB); the
~0.007 macro-F1 spread between them is run-to-run variance, not a recipe
difference. **This is not a promotion recommendation** — rule 3 promotion is a
separate, human-approved step, and `with_svdb`'s apparent edge is inside that
variance.

`no_svdb`'s S collapse is worth naming: S sensitivity 0.021 with `S→V` at
42.2%. Without SVDB's supraventricular beats the model does not merely miss
S, it actively routes S into V.

---

## 4. The contamination trap

For contrast, the production model scored on SVDB — a database it was
**trained on**:

| class | sensitivity | precision | F1 | support |
|---|---|---|---|---|
| N | 0.994 | 0.999 | 0.996 | 152,581 |
| S | 0.970 | 0.963 | **0.966** | 10,531 |
| V | 0.991 | 0.912 | **0.950** | 8,937 |
| F | 0.955 | 0.840 | 0.894 | 22 |

**Macro-F1 0.7611** · accuracy 0.9914

These numbers are **meaningless as performance** — they measure memorization.
The gap is the point: the same model file scores macro-F1 **0.761** on data it
memorized and **0.391** on held-out data, and S F1 **0.966** vs **0.152**.

Recorded here because it is an easy mistake to make: SVDB sits in
`data/raw/public/` beside MITDB, looks like a natural test set, and the
`eval-classifier` CLI will not stop you — you have to know that
`--include-svdb` defaults to `True` in `train-classifiers`. Any future SVDB
score for a production-lineage model must be treated as invalid unless the
model was trained with `--no-include-svdb`.

---

## What this does and does not establish

**Establishes:**
- Held-out MITDB DS2 numbers for the shipped model, recomputed under current code.
- V-class reliability on DS2 is real (F1 0.830, sensitivity 0.907).
- V-class precision degrades sharply on an unseen database (0.765 → 0.377).
- S and F are unsolved on both databases; S's failure mode is `S→N` (~62–65%).
- SVDB's presence in training is supported by held-out evidence.

**Does not establish:**
- Anything about VitalPatch classification accuracy. Both databases are
  clinical-grade multi-lead recordings at 360Hz/128Hz; VitalPatch is a
  single-lead 125Hz wearable in raw ADC counts with no labels. Section 2 is
  the reason to expect degradation, not a measurement of it.
- Anything about end-to-end pipeline accuracy. This scores classification
  from **ground-truth annotation positions**; runtime prepends R-peak
  detection, whose errors compound with these.
- Any F-class conclusion from SVDB (22 beats).

## Reproducing

```bash
# Section 1
python3 -m training.ecg_pipeline_tools eval-classifier \
    --model ecg_pipeline/models/five_class_xgb.json --split-set ds2

# Sections 2-4 call evaluate() directly; the CLI hardcodes data_root/mitdb
# and cannot target SVDB.
```

Environment: system `python3` (3.10) — wfdb 4.3.1, xgboost 3.2.0,
scikit-learn 1.7.2. `MedGemma-Agent/venv` does not have `wfdb` installed and
cannot run this.
