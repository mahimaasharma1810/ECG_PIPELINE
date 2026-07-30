# Cliniaura ECG Pipeline — Methods, Approaches, and Results

**Scope:** every stage from raw device file to multimodal risk output, with the
method used, the approach chosen (and rejected alternatives), and the measured
result. Every number below was computed from a file on this machine during the
writing of this document, or cited to the artifact that contains it. Anything
not measured is marked `[UNVERIFIED]`.

**Date:** 2026-07-29
**Branch:** `mod_1` (8 commits ahead of origin, unpushed — push blocked on credential helper)

---

## 0. Executive summary

| Question | Answer | Source |
|---|---|---|
| Is the ECG pipeline complete end-to-end? | **Yes**, 9 stages, run on 3,628 real segments (re-run 2026-07-30) | `data/reports/vitalpatch_run_manifest.csv` |
| How often is real VitalPatch ECG assessable? | **79.6%** (2,889 / 3,628 segments) | ibid. |
| How often are vitals pairable to an ECG segment? | **100.0%** (2,375/2,375) after the 2026-07-30 fix; was 60.3% | measured, §7.2 |
| How many NEWS2 components can VitalPatch supply? | **3 of 6** (HR, RR, Temp). SpO2 and BP have no sensor. | §7.1 |
| Do vitals change the ECG risk level? | **0 / 14** in the only multimodal run that exists | `sample14_manifest_preserved.json` |
| Is the multimodal batch complete? | **No.** 14 of ~3,570 segments. Killed by SLURM time limit. | §8 |

The single most important open item is **not** the SpO2/BP hardware gap. The
vitals-pairing bug that was discarding ~40% of usable data is fixed as of
2026-07-30 (§7.2). What remains is that the multimodal batch has never
completed (§8) — that determines whether vitals actually change any risk
levels at scale, and is still unanswered.

---

## 1. The problem and the data

### 1.1 Clinical problem

A post-operative patient recovering at home wears a single-lead chest patch for
days. Nobody can watch days of raw waveform. The system must watch continuously
and surface only what needs a clinician's attention.

### 1.2 Target device data (unlabeled, real)

`data/raw/vitalpatch/` — 6 patients, **2,375 raw ECG CSV files**:

| Patient folder | ECG files | Vitals files |
|---|---|---|
| Patch_183594 | 289 | 412 |
| Patch_1844AC | 501 | 502 |
| Patch_184635 | 479 | 480 |
| Patch_1849DF | 502 | 504 |
| Patch_184B27 | 180 | 189 |
| Patch_184B2F | 424 | 425 |
| **Total** | **2,375** | **2,512** |

*Measured by direct file count, 2026-07-29.*

These files carry **no ground-truth labels**. They cannot be used to measure
classifier accuracy — only to measure pipeline behaviour (assessability rates,
risk distribution, failure modes).

A second device is also present: `data/raw/prorhythm/` (SeNSiO), 18 recordings.

### 1.3 Labeled training/evaluation data

*Source: `Docs/DATASETS.md`.*

| Dataset | Records | Role |
|---|---|---|
| MITDB (MIT-BIH Arrhythmia) | 48 | Primary train + eval, per-beat cardiologist annotations |
| SVDB (Supraventricular) | 78 | Train-only S-class enrichment, never touches DS2 |
| INCARTDB | 75 | Auxiliary |
| LTAFDB (Long-Term AF) | 84 | AFib rule ground truth + extra training beats |
| SDDB (Sudden Cardiac Death) | 23 | F-class enrichment attempt |
| CUDB, CHALLENGE2017, ICENTIA11K | 35 / — / — | Auxiliary, largely unused |

**Split method — patient-level, inter-patient (de Chazal convention):**
- `DS1_TRAIN` — 12 records: 101,106,108,109,112,115,119,122,203,208,209,230
- `DS1_VAL` — 10 records (enlarged from an original 4, deliberately, to rule out
  "validation set too small" as an explanation for an ablation result)
- `DS2` — 22 records, **held out, report-only, never used to select anything**

No patient appears in both train and test. This is the honest split; the easier
intra-patient split (mixing beats from the same patient across train/test) was
not used, because it inflates results and does not reflect deployment.

---

## 2. Stage-by-stage method

The pipeline is 9 stages, implemented in `ecg_pipeline/ecg_pipeline_core.py`
(development) and extracted verbatim into `ecg_inference/` (deployment — see
`Docs/inference_ready.md` for the byte-for-byte equivalence proof).

### Stage 1 — Parse

Reads three input formats: VitalPatch CSV, SeNSiO CSV, and hospital WFDB.
Normalizes to an internal `Recording` object with a sample rate and a segment ID.

A VitalPatch file may contain **multiple recording segments**, which is why
2,375 files produced 3,570 segments — roughly 1.5 segments per file.

**Result:** 70 of 3,570 segments failed to parse
(`PARSE_ERROR: ValueError: could not convert...`) — a `-` sentinel value in the
sample column. This is a **real, open bug**, 2.0% of segments.

### Stage 2 — Signal-quality gate (SQI)

Each short window is checked for flatline, clipping/railing, missing samples,
and excessive noise. Bad windows are **excluded, not repaired or guessed**.

**Design decision:** if too few beats survive, the recording is marked
`NOT_ASSESSABLE` and the downstream narrative is skipped entirely. A system that
always emits a risk level, even from noise, is more dangerous than one that
sometimes says "I can't tell."

**Result:** median quality score **0.889** across 3,570 real segments; 775
segments (21.7%) fell below the assessability floor.

### Stage 3 — Resample

All devices resampled to a common **125 Hz**. Devices record at different native
rates; every downstream stage assumes one rate.

### Stage 4 — Filter

Baseline-wander removal, mains-hum notch, and muscle-noise suppression, chosen
to avoid distorting QRS morphology (the thing the classifier reads).

### Stage 5 — R-peak detection and beat segmentation

XQRS detector locates each R-peak; a fixed window around each peak is cut out as
one beat snippet.

**Fixed defect:** R-peak over-detection was a P1 deployment blocker — see
`Docs/EDGE_DEPLOYMENT_FIX_REPORT.md`.

**Result:** 357,909 beats analyzed across the real VitalPatch corpus; median 106
beats per segment; 81.9 hours of signal total.

### Stage 6 — Feature extraction

Each beat snippet → **56 numbers**: 5 morphological features (amplitude, width,
etc.) + 51 wavelet coefficients (Daubechies-4 decomposition). Handcrafted, not
learned.

**Approach chosen over the alternative:** learned representations (a
self-supervised encoder, and a CNN+Transformer on raw waveform) were both built
and both rejected — see §4.

### Stage 7 — Beat classification (XGBoost)

56 features → one AAMI label per beat: N / S / V / F / Q.
Production weights: `ecg_pipeline/models/five_class_xgb.json` (15 MB).

### Stage 8 — Rhythm detection and the risk cascade

Some risks are not about one beat but a **pattern across many**. A fixed,
human-written rule set — **not** a model — checks these and assigns the final
risk level. First rule that fires wins.

*Thresholds: `RiskThresholds`, `ecg_pipeline_core.py:141-149`.
Cascade: `score_recording()`, `ecg_pipeline_core.py:2062-2091`.*

| Priority | Condition | Level |
|---|---|---|
| 1 | PVC burden > 20% | CRITICAL |
| 2 | ≥1 VT run (≥3 consecutive V beats) | CRITICAL |
| 3 | PVC burden > 10% | HIGH |
| 4 | PAC burden > 15% **or** AFib burden > 30% | HIGH |
| 5 | SDNN < 20 ms (HRV suppression) | MEDIUM |
| 6 | nothing fired | LOW |
| **override A** | NEWS2 ≥ 7 | escalate to CRITICAL |
| **override B** | qSOFA ≥ 2 | escalate to HIGH |

Both overrides are **escalate-only** — they can raise a level, never lower one.
The guard `RISK_LEVELS.index(level) < RISK_LEVELS.index(...)` at lines 2084 and
2088 enforces this.

### Stage 9 — Narrative (MedGemma)

A locally-run MedGemma model turns the already-decided numbers into a readable
clinical paragraph. **It is a translator, not a decision-maker** — the risk level
is fixed by the deterministic cascade before MedGemma runs, and MedGemma cannot
change it.

**Result:** on the real batch — 2,413 ACCEPTED live narratives, 705 skipped as
NOT_ASSESSABLE, 277 skipped as CRITICAL (deliberately: a CRITICAL result must not
wait on an LLM), 105 UNAVAILABLE_FALLBACK (deterministic text when the model was
down).

---

## 3. Classifier results — the honest numbers

**Production model, DS2 held-out, 45,804 beats, never re-tuned against this split.**
*Source: `Docs/archive/ABLATION_REPORT.md`, "Production baseline".*

| Class | Sensitivity | Precision | F1 | Support | Trust |
|---|---|---|---|---|---|
| N (Normal) | 0.972 | 0.973 | **0.972** | 40,634 | High — reliable |
| S (Supraventricular) | 0.130 | 0.150 | **0.139** | 1,795 | Low — screening signal only |
| V (Ventricular) | 0.912 | 0.754 | **0.826** | 3,005 | Moderate-high — the clinically critical class |
| F (Fusion) | 0.005 | 0.125 | **0.011** | 363 | Essentially unsolved — treat as noise |
| Q (Unknown) | 0.000 | 0.000 | 0.000 | 7 | N/A — near-zero support |

**Macro-F1: 0.3895.**

**Why accuracy is not reported as a success metric:** N is 40,634 of 45,804 beats
(88.7%). A model that only ever predicts "N" scores 88.7% accuracy while being
clinically useless. The overall 0.927 accuracy figure is therefore not evidence
of anything. This is codified as rule 4 in `Docs/AGENT_RULES.md`.

**What is actually good here:** V — the class that matters most clinically — is
at F1 0.826 with 0.912 sensitivity. That is the number the system's clinical
value rests on.

---

## 4. What was tried to fix S, and why every attempt was rejected

**The non-negotiable rule:** N and V are zero-tolerance. Any change that
measurably hurts V is rejected outright, however much it helps S.

*Sources: `Docs/archive/ABLATION_REPORT.md`, `Docs/CNN_TRANSFORMER_EXPERIMENT.md`.*

| Attempt | Approach | S F1 | V F1 | Verdict |
|---|---|---|---|---|
| +7 timing features | Add rhythm-context (how early/late vs neighbours) | 0.182 ↑ | 0.775 ↓ | **Rejected** — V regressed past the safety margin |
| +8.5M beats (LTAFDB/SDDB) | More real training data | 0.098 ↓ | 0.808 ↓ | **Rejected** — extra data was overwhelmingly N-labeled, diluting rare classes |
| Two-stage classifier, v1–v5 | Gate "abnormal", then classify which kind | 0.383 ↑↑ | 0.789 ↓ | **Rejected and closed** — re-tested on a 2.5× larger validation set (4→10 patients) to rule out "test too small"; trade-off reproduced |
| CNN+Transformer, 2 variants | Deep model on raw waveform, no handcrafted features | 0.121 ↓ | 0.730 ↓↓ | **Rejected** — failed to improve the one thing it was built for |

**Conclusion, stated plainly:** four genuinely different approaches — feature
engineering, more data, a different architecture, and a different model family —
hit the same wall. The honest read is that this is a **real limitation of
single-lead hardware and/or available labeled data**, not a bug and not a lack of
effort. Physiological explanation: a single lead often cannot see the P-wave, the
electrical signature that distinguishes "early atrial beat" from "normal beat
that came early." 67.6% of true S beats are missed **entirely** — not confused
with V, simply not seen.

**The production model was never modified by any of these experiments.**

---

## 5. AFib rule validation (2026-07-28)

The rhythm-level AFib-suspected rule (RR coefficient-of-variation over rolling
20-beat windows) had never been checked against real AFib ground truth — flagged
in `Docs/archive/RESEARCH_AUDIT.md` as "the single biggest unvalidated clinical
claim in the pipeline."

**Method:** LTAFDB, 84 records, **449,749 windows**, ground truth from real
rhythm-change annotations. Threshold swept.

| Threshold | Sensitivity | Specificity | F1 |
|---|---|---|---|
| 0.15 (old default) | 0.808 | 0.800 | 0.829 |
| **0.10 (current default)** | **0.971** | **0.716** | **0.893** |

0.10 is Youden's-J-optimal and has the best F1 of any threshold tested. For a
*suspected* screening flag that feeds a clinician review step, missing ~3% of
true AFib windows instead of ~19% was judged worth the specificity cost. Changed
in both `ecg_pipeline_core.py` and `ecg_inference/classifier.py`.

**Open limitation:** only the *threshold* was swept. `window=20` was never swept.

---

## 6. ECG-only results on real device data

### 6.1 VitalPatch — complete batch, 3,628 segments

*Source: `data/reports/vitalpatch_run_manifest.csv`, regenerated 2026-07-30 via
`python -m ecg_pipeline.batch_vitalpatch_report` after the `-` sentinel parser
fix (the previous manifest, aggregated 2026-07-29, predated that fix and had 70
`PARSE_ERROR` rows out of 3,570 — those numbers are superseded below). Note:
`Docs/PROJECT_OVERVIEW.md` §7 still quotes a 902-segment partial snapshot —
that batch has since completed and the numbers below supersede it too.*

| Metric | Value |
|---|---|
| Segments processed | 3,628 |
| Total signal | **84.3 hours** |
| Median segment duration | 116.9 s |
| Total beats analyzed | 368,072 |
| Median quality score | 0.88 |
| **Assessable** | **2,889 (79.6%)** |
| NOT_ASSESSABLE | 739 (20.4%) |
| Parse errors | 0 |

**Risk distribution:**

| Level | Count | % of all | % of assessable |
|---|---|---|---|
| LOW | 1,881 | 51.8% | 65.1% |
| MEDIUM | 166 | 4.6% | 5.7% |
| HIGH | 552 | 15.2% | 19.1% |
| CRITICAL | 290 | 8.0% | 10.0% |
| NOT_ASSESSABLE | 739 | 20.4% | — |
| (parse error) | 0 | 0.0% | — |

**Rhythm findings present (`n_rhythm_findings` > 0):** 1,220 of 3,628 segments
(33.6%). Segments with an AFIB_SUSPECTED/VT/PVC finding specifically
(`has_afib_vt_pvc_finding`): 1,385 of 3,628 (38.2%).

**This is the current headline ECG-only result: 79.6% assessability on real,
uncontrolled, at-home wearable data** — up from a previously-reported 78.3%
that was computed on the stale, pre-parser-fix manifest.

**A caution that must not be lost:** 10.0% of assessable segments are CRITICAL.
That is a very high rate for a recovering post-op cohort and is **not
independently validated** — there is no ground truth on this corpus. It is at
least as likely to reflect classifier false positives on noisy single-lead data
(V-class precision is 0.754, so ~1 in 4 V calls is wrong, and PVC burden drives
the two CRITICAL rules) as it is to reflect real events. `[UNVERIFIED — requires
clinician adjudication of a sample]`

### 6.2 SeNSiO/prorhythm — complete, 18 recordings

*Source: `data/reports/prorhythm_run_manifest.csv`.*

| Metric | Count |
|---|---|
| Total | 18 |
| Assessable | 13 |
| NOT_ASSESSABLE | 5 |
| LOW | 8 |
| HIGH | 3 |
| CRITICAL | 2 |
| MedGemma ACCEPTED | 11 |

Sample size too small for rate estimates; useful as a second-device smoke test
only.

---

## 7. The multimodal integration

### 7.1 What VitalPatch can and cannot supply

NEWS2 (Royal College of Physicians 2017) requires 6 components. VitalPatch
vitals CSVs supply 3.

| NEWS2 component | VitalPatch column | Available? |
|---|---|---|
| Heart rate | col 1 | **Yes** |
| Respiratory rate | col 2 | **Yes** |
| Temperature | col 3 | **Yes** |
| SpO2 | — | **No sensor** |
| Systolic BP | — | **No sensor** |
| Consciousness (AVPU) | — | Not recorded |
| *(posture, non-NEWS2)* | col 5 | Yes |

**A spec error caught and corrected:** the integration spec claimed column 7 was
SpO2. It is not. Column 7 is constant within each file and non-physiological
across files (observed values 2, 3, 7, 16, 17, 20, 32, 47, 75). Confirmed against
`parse_vitalpatch_vitals()`'s own docstring. **No SpO2 value was ever fabricated.**

Consequence, enforced in code: SpO2, SBP and DBP are always emitted with
`real: False` and a `None` value. Only sensor-derived values are marked
`real: True`.

**Structural consequence:** with a maximum of 3 of 6 components, the highest
attainable NEWS2 score is bounded well below the **NEWS2 ≥ 7 CRITICAL override
threshold**. The observed maximum across the sample was 3. *The NEWS2 override is
therefore currently unreachable on VitalPatch hardware, by construction — not by
chance.*

### 7.2 Vitals pairing — the finding that changes the picture

ECG and vitals files are separate, with independent timestamps. The spec assumed
exact filename matching; **0 of 424 files matched**. The implemented rule pairs
each ECG file to the vitals file whose *filename timestamp* is nearest, within a
**30-second** cutoff.

That cutoff was calibrated on **one patient** (Patch_184B2F, 98.3% coverage).
Measured across all 6 patients today, it does not generalize:

| Patient | ECG files | Paired ≤30 s | Coverage |
|---|---|---|---|
| Patch_183594 | 289 | 286 | **99.0%** |
| Patch_1844AC | 501 | 76 | **15.2%** |
| Patch_184635 | 479 | 427 | 89.1% |
| Patch_1849DF | 502 | 91 | **18.1%** |
| Patch_184B27 | 180 | 135 | 75.0% |
| Patch_184B2F | 424 | 417 | 98.3% |
| **Overall** | **2,375** | **1,432** | **60.3%** |

**Root cause, measured:** vitals files are written on a **20-minute** cadence
(median inter-file spacing 1,200,258 ms) and each file's rows span a median of
**20.0 minutes** of internal per-row timestamps. On some patients the ECG capture
is co-triggered with the vitals file boundary (99% coverage); on others it is
free-running, so a valid ECG lands mid-file and the *filename* timestamp is
minutes away even though the ECG is squarely **inside** that file's covered time
range.

The correct rule is interval containment — pair when the ECG timestamp falls
within `[first_row_ts, last_row_ts]` of a vitals file. Measured:

| Patient | Coverage, filename ±30 s | Coverage, interval containment |
|---|---|---|
| Patch_183594 | 99.0% | 95.8% |
| Patch_1844AC | 15.2% | **100.0%** |
| Patch_184635 | 89.1% | 92.3% |
| Patch_1849DF | 18.1% | **100.0%** |
| Patch_184B27 | 75.0% | **100.0%** |
| Patch_184B2F | 98.3% | 85.1% |
| **Overall** | **60.3%** | **95.3%** |

*(Two patients drop slightly under containment alone — 183594 and 184B2F —
because a few of their ECGs fall in gaps between vitals files.)*

**Fixed 2026-07-30** (`ecg_pipeline/agent_bridge.py`, commit `709e225`):
`load_real_vitals()` now takes the union — interval containment first, the
original nearest-filename-within-30s rule as fallback for gap cases. Re-measured
on all 2,375 real ECG files (`ecg_pipeline/verify_vitals_pairing.py`):

| Patient | Containment only | + 30s fallback (shipped) |
|---|---|---|
| Patch_183594 | 95.2% (275/289) | 100.0% (289/289) |
| Patch_1844AC | 100.0% (501/501) | 100.0% (501/501) |
| Patch_184635 | 92.1% (441/479) | 100.0% (479/479) |
| Patch_1849DF | 100.0% (502/502) | 100.0% (502/502) |
| Patch_184B27 | 100.0% (180/180) | 100.0% (180/180) |
| Patch_184B2F | 84.9% (360/424) | 100.0% (424/424) |
| **Overall** | **95.1% (2,259/2,375)** | **100.0% (2,375/2,375)** |

The containment-only column matches the ~95.3% figure predicted in the table
above (within ~0.6 points per patient), confirming the predicted fix works as
expected. The combined figure is higher than that ~95.3% prediction: almost
every ECG that misses containment turns out to be within seconds of a vitals
file's boundary (median 2.0s, since vitals files for these patients chain
together nearly back-to-back), so the 30s fallback — which the pairing rule
was always meant to keep — catches nearly all of them. **Real, measured
coverage after the fix is 100.0% (2,375/2,375), not an estimate.** This
switches vitals from 60.3% to 100.0% coverage — 943 additional ECG segments
gain real vitals. This was a bug in the pairing heuristic, not a data
limitation.

### 7.3 Changes made to make partial vitals work

Six layers had to change. Each was individually "fixed" and still produced wrong
output until the next was found:

1. **`vitals/schemas.py`** — `systolic_bp` / `diastolic_bp` / `spo2` made
   `Optional[float] = None`; `respiratory_rate` / `temperature` / `posture`
   added; None-guards on every validator. `heart_rate` stays required.
2. **`guardrails/clinical_rules.py`** — None-guards on `_score_sbp` / `_score_dbp`
   / `_score_spo2`; new `_score_respiratory_rate` / `_score_temperature`.
   Parameters were **appended, not reordered** — all 18 existing tests call these
   functions positionally and a reorder would have silently scrambled every one.
3. **`guardrails/thresholds.yaml`** — RR and temperature bands added under
   `news2:`; `rr_threshold: 22` under `qsofa:`.
4. **`db/database.py:75-78`** — BP/SpO2 columns made `nullable=True`, plus a
   table-rebuild migration of 2,588 existing rows. Without this the API returned
   HTTP 500 `NOT NULL constraint failed`.
5. **`guardrails/input_guardrails.py:47-70`** — the subtle one. This function
   precomputes NEWS2/qSOFA, `input_validator.py:12` stores the result in state,
   and `risk_scorer.py:28` does `state.get("news2") or calculate_news2(...)` — a
   Pydantic instance is always truthy, so **the risk scorer's own call never
   executes.** RR and temperature had to be threaded in *here*, not only there.
   Until this was found, pushes succeeded but returned
   `respiratory_rate_value: null`.
6. **`ecg_pipeline/agent_bridge.py`** — send RR and temperature in the payload;
   read `news2.total_score` / `news2.coverage` / `qsofa.score` back out; re-run
   `ECGPipeline.run(recording, news2_score=..., qsofa_score=...)` to produce the
   combined risk, tagging the source `agent_live`.

All changes to `MedGemma-Agent/` (a third-party submodule) comment out the
original code rather than deleting it. Test suite: **25/25 passing** (18 original
+ 7 new).

### 7.4 Multimodal results — 14 segments

*Source: `data/reports/multimodal_batch/sample14_manifest_preserved.json`.
This is the **only** multimodal result set that exists. n=14 out of ~3,570.*

| Metric | Result |
|---|---|
| Segments | 14 (2–3 per patient, all 6 patients) |
| Agent push success | **14 / 14** |
| Vitals file found | 14 / 14 |
| Pairing offset | 78 ms – 6,371 ms |
| HR present | 14 / 14 |
| RR present | 11 / 14 |
| Temperature present | 6 / 14 |
| **3/6 NEWS2 coverage** | **6 / 14 (42.9%)** |
| 2/6 coverage | 5 / 14 |
| 1/6 coverage | 3 / 14 |
| NEWS2 scores observed | 0 (×6), 1 (×5), 2 (×1), 3 (×2) |
| qSOFA scores observed | 0 (×9), 1 (×4), **2 (×1)** |
| **Risk level changed by vitals** | **0 / 14** |

Per-segment:

| Segment | Beats | ECG-only | Combined | NEWS2 | qSOFA | Coverage | HR | RR | Temp |
|---|---|---|---|---|---|---|---|---|---|
| 183594 …267337 seg0 | 185 | HIGH | HIGH | 0 | 0 | 1/6 | 84.1 | — | — |
| 183594 …267337 seg1 | 35 | MEDIUM | MEDIUM | 0 | 0 | 1/6 | 84.1 | — | — |
| 183594 …553369 seg0 | 258 | LOW | LOW | 0 | 0 | 3/6 | 85.4 | 16.9 | 37.0 |
| 1844AC …448928 seg0 | 170 | HIGH | HIGH | 1 | 0 | 3/6 | 82.7 | 18.0 | 36.0 |
| 1844AC …649141 seg0 | 247 | CRITICAL | CRITICAL | 1 | 0 | 3/6 | 75.4 | 17.1 | 35.9 |
| 184635 …877851 seg0 | 34 | HIGH | HIGH | 3 | **2** | 2/6 | 102.8 | 22.2 | — |
| 184635 …991724 seg0 | 190 | LOW | LOW | 1 | 1 | 2/6 | 92.5 | 16.3 | — |
| 1849DF …151158 seg0 | 157 | LOW | LOW | 1 | 0 | 3/6 | 89.8 | 14.7 | 36.0 |
| 1849DF …352675 seg0 | 157 | LOW | LOW | 0 | 0 | 3/6 | 87.8 | 15.0 | 36.1 |
| 184B27 …134136 seg0 | 76 | HIGH | HIGH | 2 | 1 | 1/6 | 115.2 | — | — |
| 184B27 …764380 seg0 | 166 | HIGH | HIGH | 3 | 1 | 3/6 | 118.0 | 18.6 | 36.0 |
| 184B2F …595043 seg0 | 49 | CRITICAL | CRITICAL | 0 | 0 | 2/6 | 85.5 | 18.0 | — |
| 184B2F …595043 seg1 | 0 | LOW | LOW | 0 | 0 | 2/6 | 85.5 | 18.0 | — |
| 184B2F …796701 seg0 | 180 | LOW | LOW | 1 | 1 | 2/6 | 100.3 | 19.5 | — |

**Three findings from this sample, stated precisely:**

**(a) RR and temperature are not redundant with HR.** Temperature was the *sole*
contributor to the NEWS2 score in 3 of 14 segments; RR supplied 2 of the 3 points
in another. The two components added in this work carry independent signal — the
3/6 extension was not cosmetic.

**(b) The qSOFA override fired its condition once, and was masked.** Segment
`184635…877851` scored qSOFA = 2, which meets the `qsofa_high_threshold`. It
produced no change because the ECG cascade had *already* assigned HIGH, and the
override is escalate-only (`ecg_pipeline_core.py:2088`). This corrects an earlier
claim in this project's notes that the qSOFA proxy "caps at 1" — it does not.
**The override machinery works; it simply had nothing to raise.**

**(c) "Vitals changed nothing in 14/14" is a real observation but a weak one.**
n=14, and the NEWS2 override is structurally unreachable at 3/6 coverage (§7.1).
An override that cannot fire, not firing, is not evidence about clinical value.

**On the ECG↔NEWS2 relationship:** mean NEWS2 by ECG level is CRITICAL 0.50,
HIGH 1.80, MEDIUM 0.00, LOW 0.50 — non-monotonic, with the highest mean at HIGH
rather than CRITICAL. At n=14 with 1–3 segments per cell this is **not
interpretable**. No correlation claim, positive or negative, is supported.

**On the two CRITICAL segments:** neither had deranged vitals. This does *not*
imply they were false alarms — a run of ventricular beats is a real electrical
event that need not move HR, RR, or temperature at all. Read the other way: it
shows ECG and vitals are **complementary channels**, which is the argument *for*
multimodal fusion, not against it.

---

## 8. Current state of the full multimodal batch

**The batch has not run.** Status as of 2026-07-29:

- SLURM job **2660129** exceeded its wall-clock time limit; the allocation was
  revoked at **15:37:17** and the job was killed. This was an **infrastructure
  limit, not a code defect** — the runner had processed only the first patient's
  file listing.
- The MedGemma-Agent server (`gnode079`) died with the allocation and has not
  been restarted.
- `data/reports/multimodal_batch/multimodal_manifest.json` (15:36) is the
  **14-segment validation run**, not a full batch. It is preserved separately as
  `sample14_manifest_preserved.json`.
- A new allocation exists (job **2660412**, gnode079) but nothing has been
  launched on it.

**Nothing was lost.** All 8 commits are local and intact; the enriched batch
runner and the preserved sample manifest are on disk.

**To resume, three things are required, in order:**
1. A SLURM allocation with a wall-clock limit exceeding the batch runtime.
   Runtime estimate: the 14-segment sample took 15.5 s including 14 live LLM
   round-trips (~1.1 s/segment). Extrapolated to 3,570 segments: **~65 minutes**,
   assuming Agent latency stays flat. `[UNVERIFIED at scale]`
2. The MedGemma-Agent FastAPI server restarted on the allocated node.
3. Fix the pairing rule first (§7.2) — otherwise the run will discard vitals for
   ~40% of segments and produce a coverage figure that is an artifact of the
   heuristic rather than a property of the data.

---

## 9. Artifacts produced

**Code (8 commits on `mod_1`, unpushed):**

| Commit | Content |
|---|---|
| `d1b94e8` | Replace simulated vitals with real VitalPatch CSV reader |
| `3e6aa71` | Load real HR/RR/Temp/posture, compute partial NEWS2 |
| `8a9b9d8` | Wire partial NEWS2 into the ECG risk cascade |
| `98e77ae` | Harden `push_main` against an unreachable Agent |
| `0052b8a` | Send RR+Temp in payload; wire Agent NEWS2 into combined-risk loopback |
| `5139e00` | Submodule: nullable BP/SpO2 DB migration (2,588 rows) |
| `7564d17` | Submodule: `input_guardrails` fix |
| *(earlier)* | `4d37be6` NEXT_STEPS.md, `487ba7f` docs consolidation |

**Key functions added to `ecg_pipeline/agent_bridge.py`:**
`load_real_vitals()`, `_extract_numeric_column()`, `_numeric_stat_dict()`,
`compute_partial_news2()` (+ `_news2_hr_score` / `_news2_rr_score` /
`_news2_temp_score`), `compute_qsofa_proxy()`.

**Data artifacts:**
- `data/reports/vitalpatch_run_manifest.csv` — 3,570 segments, complete
- `data/reports/prorhythm_run_manifest.csv` — 18 recordings, complete
- `data/reports/multimodal_batch/sample14_manifest_preserved.json` — n=14
- `data/reports/vitalpatch/`, `data/reports/prorhythm/` — per-segment JSON + MD

---

## 10. Failure modes found and resolved

| # | Symptom | Root cause | Resolution |
|---|---|---|---|
| 1 | Spec said col 7 = SpO2 | False — constant within file, non-physiological across files | Never fabricated; SpO2 permanently `real: False` |
| 2 | Exact filename pairing matched 0/424 | ECG and vitals have independent timestamps | Nearest-timestamp with 30 s cutoff (now known insufficient — §7.2) |
| 3 | Reported temperature always empty | My own error — checked one file | Re-checked 20 files: 18/20 have real temperature. Self-corrected |
| 4 | `score_with_vitals()` in spec | Function does not exist | Used real `ECGPipeline.run(recording, news2_score=, qsofa_score=)` |
| 5 | Proposed parameter reorder | All 18 tests call positionally — would have silently scrambled every one | Appended parameters instead |
| 6 | Spec tests used dict-subscript | Targets are Pydantic models | Attribute access |
| 7 | Referenced a nonexistent ECG file | I had fabricated the filename by pairing a vitals timestamp with an `_ecg.csv` suffix | Located a real replacement; flagged the fabrication |
| 8 | HTTP 500 `NOT NULL constraint failed` | DB schema never updated alongside Pydantic schema | Table-rebuild migration, 2,588 rows, row count verified unchanged |
| 9 | `respiratory_rate_value: null` on successful push | `input_guardrails` precomputes and `risk_scorer`'s `or` short-circuits | Threaded RR/temp into `input_guardrails.py` |
| 10 | Windows venv (`Include/Lib/Scripts`) | Cross-platform submodule | Rebuilt natively |
| 11 | pip disk quota exceeded | `nvidia_cudnn_cu13` download | Purged 4.5 GB cache, installed CPU-only torch |
| 12 | `start.sh` → `bash\r: not found` | CRLF line endings | `sed -i 's/\r$//'` |
| 13 | Batch table would have been entirely empty | Proposed script used `--input`/`--output` flags that don't exist on `push_main` | In-process runner; classifier loaded once instead of 3,570 times |
| 14 | Batch killed mid-run | SLURM wall-clock limit | **Open** — needs a longer allocation |
| 15 | `git push origin mod_1` fails | VS Code credential helper socket ECONNREFUSED | **Open** |
| 16 | 70 segments fail to parse | `-` sentinel in sample column | **Open** — 2.0% of corpus |

---

## 11. Honest limitations

1. **S-class recall is 13%; F-class is unsolved (F1 0.011).** Four independent
   fix attempts failed identically. Treat any F finding as noise and any S
   finding as a screening signal only.
2. **The AFib rule's `window=20` was never swept** — only the threshold was.
3. **The NEWS2 ≥ 7 CRITICAL override is unreachable on VitalPatch hardware** at
   3/6 component coverage. It is dead code on this device, by construction.
4. **The 9.9% CRITICAL rate on real data is unvalidated.** No ground truth
   exists on this corpus, and V-class precision of 0.754 means roughly 1 in 4
   ventricular calls is wrong. `[UNVERIFIED]`
5. **Confidence tiers in the JSON report are a heuristic, not calibrated
   probabilities** — no conformal calibration set is loaded by default.
6. **MedGemma occasionally misstates a numeric value** in its free text. The
   risk level itself is never affected — it is fixed before MedGemma runs.
7. **The multimodal result set is n=14.** Every multimodal statement in this
   document is provisional until the full batch completes.
8. **Vitals pairing currently loses 39.7% of pairable segments** (§7.2).

---

## 12. What the evidence actually supports doing next

Ordered by evidence strength, not by effort.

**1. Fix the vitals pairing rule before re-running the batch.**
Measured: 60.3% → 95.3% coverage, 831 additional segments. This is the
highest-value change available and it is a heuristic bug, not new research. Use
interval containment with nearest-within-30 s as fallback, and select the vitals
*rows* nearest the ECG timestamp rather than averaging the whole 20-minute file.
Running the batch before this fix bakes a heuristic artifact into the headline
coverage number.

**2. Re-run the full multimodal batch on a sufficient allocation.**
~65 min estimated. This is the only way to replace n=14 with n≈3,570 and is
prerequisite to every strategic question currently open.

**3. Get clinician adjudication on a sample of the 277 CRITICAL segments.**
This is the largest unquantified risk in the system. Everything downstream —
alert burden, deployment readiness, whether the 9.9% rate is signal or
false-positive noise — depends on an answer that no amount of further engineering
can produce. A stratified sample of ~50 segments would be enough to bound it.

**What the evidence does *not* currently support:** concluding anything about
whether the SpO2/BP gap matters clinically. The 0/14 "no risk change" result is
consistent with the gap being unimportant *and* with the gap being critical, and
cannot distinguish them, because the override that the missing components would
drive is mathematically unreachable at 3/6 coverage. That question stays open
until §12.2 completes.
