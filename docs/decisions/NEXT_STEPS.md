# Next Steps — Cliniaura ECG Pipeline

Compiled from `docs/decisions/AGENT_RULES.md`, `docs/archive/ABLATION_REPORT.md`,
`docs/archive/RESEARCH_AUDIT.md`, `docs/PROJECT_STATUS.md`, and
`docs/EDGE_DEPLOYMENT_FIX_REPORT.md`. Every item below is traceable to one
of those files or to a live check against the current codebase run while
writing this document (2026-07-28). Nothing here is invented; anything I
could not fully verify is marked **[UNVERIFIED]**.

**One correction to the brief this document was written against:** the
"missing morphology features" item (QRS width, absolute QRS area, pre-R/
post-R slopes) was originally described as "not yet added to features.py."
That's not accurate — these features already exist in the code
(`_qrs_shape_features()`, `_qrs_width_amplitude_crossing()`), were added
and tested in `ABLATION_REPORT.md`'s two-stage experiments, and were found
insufficient on their own. Section 2 below describes the real, still-open
next step accurately.

---

## 1. BUGS STILL OPEN

### 1.1 — NEWS2/qSOFA dead code

**What it is:** `score_recording()` accepts `news2_score`/`qsofa_score`
parameters and `RiskReport` carries them as fields, and the two safety
overrides (NEWS2 ≥ 7 → force CRITICAL, qSOFA ≥ 2 → force HIGH) are real,
implemented cascade rules — but no caller anywhere in the codebase
(`report_ui.py`, `demo_stream.py`, `batch_vitalpatch_report.py`,
`batch_prorhythm_report.py`) ever passes a real value in. The rules are
permanently inert.

**Where:** `ecg_pipeline/ecg_pipeline_core.py` lines 2035-2036 (`RiskReport`
fields), 2042 / 2481 (`score_recording`/`ECGPipeline.run` signatures),
2083-2091 (the two override checks). Self-disclosed in
`ecg_pipeline/agent_bridge.py` line 415's `known_limitations` text.

**Symptom:** every real report's `safety_overrides` block shows
`applied: false` for both NEWS2 and qSOFA, always. Not a silent-wrong-output
bug (it's disclosed in the report itself, not fabricated) — but any claim
of "NEWS2/qSOFA-aware risk scoring" is false as currently shipped.

**Exact fix:** either (a) build a real vitals-ingestion path that supplies
`news2_score`/`qsofa_score` to `ECGPipeline.run()` — this is a genuinely
separate integration effort (a vitals source doesn't exist in this
ECG-only repo at all, see Section 4.4), or (b) if no such integration is
planned near-term, explicitly document NEWS2/qSOFA as out of scope in the
schema/README rather than leaving it silently unresolved (per
`docs/PROJECT_STATUS.md` Part 4 item 4's own recommendation).

*Source: `docs/PROJECT_STATUS.md` Part 2 #4, Part 4 #4.*

---

### 1.2 — `_afib_suspected` index-mapping bug (latent)

**What it is:** inside `_afib_suspected`, RR intervals are filtered to drop
flagged/`None` entries (`clean_rr = [r for r in rr_ms if r is not None]`),
and each `AFIB_SUSPECTED` finding's `start_beat_idx`/`end_beat_idx` is set
to a position **within `clean_rr`** (the filtered list). Downstream code
consumes those indices as if they were positions in the **original,
unfiltered `beats` list**.

**Where:** written at `ecg_pipeline/ecg_pipeline_core.py` lines 1990-2000
(`_afib_suspected`, specifically `clean_rr = [...]` on line 1992 and the
`RhythmFinding("AFIB_SUSPECTED", start, start + window - 1, ...)` call on
line 2000). Consumed (mismatched) at `ecg_pipeline/agent_bridge.py` lines
154-155: `beats[f.start_beat_idx]` / `beats[f.end_beat_idx]`.

**Symptom:** latent — only triggers when at least one RR interval upstream
of a flagged AFib window was itself flagged (`Beat.rr_flagged=True`, e.g.
from a missed R-peak across a noisy stretch). When it does trigger, the
reported `start_time_s`/`end_time_s` for an AFIB_SUSPECTED finding would
point at the wrong beat(s) in the recording. Not yet reproduced — the one
recording it was checked against had `n_rr_flagged=0`, so this is confirmed
*not currently observed*, not confirmed *safe in general*.

**Exact fix:** build a parallel index list alongside the filter, then map
back through it before constructing the finding:
```python
clean_rr = []
clean_indices = []
for i, r in enumerate(rr_ms):
    if r is not None:
        clean_rr.append(r)
        clean_indices.append(i)
...
findings.append(RhythmFinding("AFIB_SUSPECTED",
                               clean_indices[start], clean_indices[start + window - 1],
                               {"rr_cv": cv}))
```
Needs a targeted test with a recording that has real RR-flagged beats
upstream of a flagged AFib window before it can be called fixed and
verified (per `PROJECT_STATUS.md` Part 4 item 6).

*Source: `docs/PROJECT_STATUS.md` Part 2 #6, Part 4 #6; verified live
against current `ecg_pipeline_core.py`/`agent_bridge.py` while writing this
document (line numbers current as of 2026-07-28, post-AFib-threshold-fix).*

---

### 1.3 — NOT_ASSESSABLE rate attribution unresolved (~20% VitalPatch)

**What it is:** 705 of 3,570 real VitalPatch segments (19.7%) come out
`NOT_ASSESSABLE` (fewer than `MIN_BEATS_FOR_ASSESSMENT=5` beats survived
quality gating). The floor itself is a verified, working safety property,
not a bug — but **what fraction of that 19.7% is genuine low-quality
real-world signal vs. any residual symptom of pre-fix beat loss** was never
established.

**Where:** the floor logic itself is at `ecg_pipeline/agent_bridge.py` line
64 (`MIN_BEATS_FOR_ASSESSMENT = 5`) and lines 384-385 (the assessable
check). This is not a code bug to point at — it's an unresolved
*measurement* question about the manifest data.

**Symptom:** none currently known to be wrong (NOT_ASSESSABLE is an honest
"can't tell" state, not a false LOW) — but the ~20% rate can't yet be
trusted as "this is the expected real-world operating rate" vs. "part of
this is still fixable."

**Exact fix:** no code fix — needs a dedicated investigation: sample N of
the 705 NOT_ASSESSABLE segments, inspect their SQI-gate reject codes and
raw signal directly (short segment? disconnection? motion artifact?), and
report the composition. **[UNVERIFIED]** — this is explicitly marked
unverified in the source document, not a settled number.

*Source: `docs/PROJECT_STATUS.md` Part 2 #2, Part 4 #3.*

---

### 1.4 — scipy `RuntimeWarning: divide by zero` in Lomb-Scargle/HRV

**What it is:** a benign `RuntimeWarning: divide by zero encountered in
divide` / `invalid value encountered in multiply` appears during batch
runs. It does not stop processing (the full 2,375-file VitalPatch batch
completed with 0 hard errors).

**Where:** not root-caused in the source report — flagged only as "likely
HRV-related." The most likely site, based on the only Lomb-Scargle call in
the codebase: `_lf_hf_ratio()` at `ecg_pipeline/ecg_pipeline_core.py` line
1535, which calls `scipy.signal.lombscargle` at line 1538 as part of
`recording_level_hrv`'s LF/HF computation. **[UNVERIFIED]** — this location
is inferred from being the only matching scipy call in the pipeline, not
confirmed by a traceback in the source report.

**Symptom:** log noise only; does not corrupt output or halt the batch.

**Exact fix:** not yet written — needs a repro first (most likely trigger:
a short or near-constant RR sequence producing a zero-variance or
zero-frequency divide inside the periodogram calculation) before a targeted
fix (e.g. a minimum-length/variance guard before calling `lombscargle`) can
be written and verified.

*Source: `docs/EDGE_DEPLOYMENT_FIX_REPORT.md`, "Open, non-blocking item"
section.*

---

### 1.5 — `MedGemma-Agent` submodule dirty/diverged

**What it is:** the `MedGemma-Agent` git submodule shows 65 files and a
~7,900-line diff against its own tracked `dev` branch, uncommitted.

**Where:** `MedGemma-Agent/` (a separate git repository,
`saranambiar/MedGemma-Agent`, checked out as a submodule of this repo).

**Symptom:** **[UNVERIFIED]** — flagged only because it surfaces as `m
MedGemma-Agent` in this repo's top-level `git status`; not investigated
further in the source audit, and not investigated further in writing this
document either (it's a separate component/repo, out of this ECG pipeline's
own scope).

**Exact fix:** needs its own audit inside the submodule (what changed, is
it intentional in-progress work, does it need committing/discarding)
before any action is taken here.

*Source: `docs/PROJECT_STATUS.md` Part 2 #8, Part 4 #9.*

---

## 2. IMPROVEMENTS WITH A CLEAR NEXT STEP

### 2.1 — Connect the encoder embedding to the classifier

**What to do:** concatenate the self-supervised encoder's 32-dim embedding
to the classifier's existing 56-dim handcrafted feature vector (88-dim
total), retrain, and re-evaluate on DS2 against the pinned baseline.

**Why:** `docs/archive/RESEARCH_AUDIT.md` Section 4 #2 and Section 6 #1
both flag this as a real, cheap, currently-unrealized gap — the encoder's
embedding is computed on every pipeline run but the classifier
(`FiveClassBeatClassifier.fit()`/`predict_one()`) only ever receives the
56-dim handcrafted vector; the two are never concatenated.

**Artifact that exists:** `ecg_encoder.pt` (36KB, 15 training epochs,
6,498 real unlabeled VitalPatch/SeNSiO windows, per its own
`ecg_encoder.meta.json`). **State note:** this file was deleted from
`training/model_artifacts/` in a 2026-07-28 cleanup pass (commit
`8124726`) as a non-required experimental artifact. It's recoverable from
git history — `git show 803c461:training/model_artifacts/ecg_encoder.pt >
training/model_artifacts/ecg_encoder.pt` (and the same for
`ecg_encoder.meta.json`) — or can be retrained fresh via `python -m
training.ecg_pipeline_tools train-encoder`.

**First concrete step:** no CLI flag for this exists yet (confirmed by
grep — `train-classifiers` has no `--encoder-path` or equivalent). The
actual first step is writing a small script following the same
append-only, train/inference-parity-checked pattern already used for
`r_amp`/timing/`qrs_shape` in `ABLATION_REPORT.md`: load the encoder via
`load_encoder()`, compute `embed_windows()` on each beat's wide window,
concatenate the 32-dim result to the existing 56-dim vector, and retrain
with `FiveClassBeatClassifier.fit()` — then evaluate on DS2 and report
per-class F1/confusion matrix, same protocol as every other experiment in
`ABLATION_REPORT.md`.

---

### 2.2 — Test QRS-shape features on the flat (single-stage) architecture

**What to do (corrected from the original brief — see the note at the top
of this document):** `_qrs_shape_features()` (QRS width, absolute QRS
area, pre-R/post-R slopes) already exists in `ecg_pipeline_core.py`, gated
behind `include_qrs_shape` (default `False`), and was already tested — but
**only inside the two-stage classifier architecture**
(`five_class_xgb_twostage_qrsfeat_*`/`_qrsfix_*`), where it gave a real,
measured improvement (V precision 0.690→0.728, S F1 0.383→0.428) that was
still not enough to clear the two-stage architecture's own V-regression
problem. **It has never been tested on the flat, single-stage,
production-style classifier** — so it's unknown whether the two-stage
architecture itself (not the QRS features) was the reason V still
regressed.

**Why:** isolates a real confound. `ABLATION_REPORT.md`'s own "Feature-family
ablation" section (lines 458-594) already showed morphology-only features
work well on the flat model (V F1 0.866, better than production);
QRS-shape is a 4-feature morphology extension that was validated to help
S specifically, just never tried outside the two-stage design that has its
own independent, already-diagnosed weakness (beats leaking past Stage 1's
gate get forced into a narrower Stage 2 label space).

**Artifact that exists:** the feature functions themselves (current, live
code, `ecg_pipeline_core.py`'s features.py section), and the fixed QRS-width
delineator (`_qrs_width_amplitude_crossing()`, which resolved an earlier
8ms-floor degeneracy — see `ABLATION_REPORT.md`'s "QRS-width detector fix"
section).

**First concrete step:** `include_qrs_shape` exists as a parameter on the
internal `build_dataset`/`_load_record_beats` functions (verified by grep,
`training/ecg_pipeline_tools.py` lines 566/605/620/636), but — unlike
`--include-r-amp` or `--timing-features` — it is **not wired to any
`train-classifiers` CLI flag**. `ABLATION_REPORT.md`'s own qrsfeat/qrsfix
runs used a one-off, not-checked-in script
(`/scratchpad/retrain_stage2_qrsfeat.py`) because they needed to load a
frozen Stage 1 model, which no CLI path supports. The real first step is
adding a `--include-qrs-shape` `argparse.BooleanOptionalAction` flag to
`main_train_classifiers`'s parser (same pattern as `--include-r-amp`
at the relevant `add_argument` call) so the flat model can be trained with
it directly:
```
python -m training.ecg_pipeline_tools train-classifiers \
    --train-split ds1_train --seed 42 --include-qrs-shape \
    --out ecg_pipeline/models/five_class_xgb_flat_qrsshape.json
```
(command shown is the *target* state once the flag is added — it will
error today, `unrecognized arguments: --include-qrs-shape`, until that
wiring is done).

---

### 2.3 — INCART cross-database generalization test

**What to do:** train on MITDB+SVDB only (as today), then evaluate the
resulting model on INCART as a **held-out test set only** — never as
training data — to measure the hospital-to-different-hospital (and by
extension, hospital-to-wearable) domain gap.

**Why:** `docs/archive/RESEARCH_AUDIT.md` Section 6 finding #7: "All
reported metrics rest on one fixed, small test set — MITDB DS2 ... No
cross-database generalization test exists yet (e.g. train on MITDB+SVDB,
test on INCART — INCART is already downloaded and sitting unused for
exactly this purpose)."

**Nuance not in the original brief:** INCART is not purely unintegrated —
`training/ecg_pipeline_tools.py` line 763 has a real `--include-incart`
flag, and it was used once, bundled with `--include-ltafdb --include-sddb`
in the (discarded) "LTAFDB+SDDB enrichment" experiment — but INCART's own
isolated effect was never separated out from LTAFDB's, and it has never
been used purely as a held-out *evaluation* set (the specific idea this
item is about).

**Artifact that exists:** `data/raw/public/incartdb/` (75 records, ~796MB,
confirmed present on disk), `INCART_RECORDS` list already defined in
`training/ecg_pipeline_tools.py` line 131.

**First concrete step:** confirmed by grep (`training/ecg_pipeline_tools.py`
line 1219) — `eval-classifier`'s `--split-set` only accepts `ds1_train`,
`ds1_val`, `ds2`; no INCART option exists. The first step is adding a
fourth choice (e.g. `incart`) that builds a dataset from `INCART_RECORDS`
via the existing `build_dataset()` function and evaluates the current
production `five_class_xgb.json` against it, reporting the same per-class
sensitivity/precision/F1/confusion matrix as every DS2 evaluation.

---

### 2.4 — Integrate CinC Challenge 2017 and CUDB

**What to do:** these are two different datasets serving two different
purposes — treat them separately, not as one task.

- **CinC Challenge 2017** (8,528 recordings, ~306MB, whole-recording AFib
  labels — not per-beat AAMI labels): per `RESEARCH_AUDIT.md`'s own
  framing (Section 0.3, Section 2), its realistic integration path is
  **training a real, rhythm-level AFib classifier** to replace the current
  unvalidated `cv_threshold`-based heuristic (see also
  `docs/archive/ABLATION_REPORT.md`'s "AFib rule validation" section,
  which validated the *existing* heuristic against LTAFDB but did not
  replace it with a trained model). No training code path for CinC2017
  exists yet — confirmed by grep, `training/ecg_pipeline_tools.py` only
  has a `download_cinc2017()` function (line 228), no `build_dataset`/CLI
  flag equivalent.
- **CUDB** (35 records, ~6.8MB): a smaller supplementary AAMI-style
  dataset, same general integration shape as INCART (Section 2.3) —
  currently has **zero** integration code (confirmed by grep: appears
  only in the `download-datasets` name mapping, `training/ecg_pipeline_tools.py`
  line 203; no `CUDB_RECORDS` list, no `--include-cudb` flag, unlike
  INCART which at least has a flag).

**Artifact that exists:** both already downloaded and confirmed present on
disk — `data/raw/public/challenge2017/` (306MB), `data/raw/public/cudb/`
(6.8MB).

**First concrete step:** for CUDB, the smaller/simpler of the two — add a
`CUDB_RECORDS` list and an `--include-cudb` flag to `train-classifiers`,
following the exact same pattern as `--include-incart`, then run the same
enrichment-experiment protocol as `ABLATION_REPORT.md`'s LTAFDB/SDDB
experiment (train, eval on DS2, compare per-class F1 against the pinned
baseline, do not promote without a clean win per `AGENT_RULES.md` Rule 8).
For CinC2017, the first step is scoping a new rhythm-level classifier
design (per `RESEARCH_AUDIT.md` Section 2's recommended order: this is
explicitly ranked *after* connecting the encoder embedding, item 2.1
above) — this is a larger, separate piece of design work, not a one-line
CLI flag addition like CUDB.

---

### 2.5 — Conformal prediction calibration on real outcome data

**What to do:** replace the current synthetic-proxy calibration set for
`ConformalRiskPredictor` with real recording-level clinical outcome
labels, once such labels exist.

**Why:** `docs/archive/RESEARCH_AUDIT.md` Section 4 #5 and Section 6 #3:
the conformal predictor's "statistically guaranteed" coverage is currently
calibrated against a made-up proxy (`_label_to_risk_idx`/
`_beat_proba_to_risk_scores` in `training/ecg_pipeline_tools.py` lines
1123-1129), not real clinical outcomes — presenting the 90%-coverage
number without that caveat would be a real methodological overstatement.
There's also a structural gap: the proxy mapping can never produce a
CRITICAL calibration example, even though CRITICAL is a real alert level
the deterministic rules do emit.

**Artifact that exists:** the conformal predictor implementation itself
(`ConformalRiskPredictor`, `ecg_pipeline/ecg_pipeline_core.py` line 2129)
is real, working, standard split-conformal code — the gap is purely the
calibration *data*, not the method.

**First concrete step:** this is explicitly an open research question, not
a quick fix (per `RESEARCH_AUDIT.md` Section 4 #5's own "Difficulty: High
— Open research question" rating) — no real clinical-outcome-labeled
recordings exist yet anywhere in this project. The first real step is the
same one Section 4.1 below names for the S/F ceiling: hand-labeling a
batch of real device recordings with actual clinical outcomes, not a code
change.

---

### 2.6 — Validate edge deployment latency on Jetson hardware

**What to do:** measure real inference latency/throughput/memory footprint
on actual Jetson-class target hardware.

**Why:** `docs/PROJECT_STATUS.md` Part 3 states this plainly: **[UNVERIFIED
— no evidence found]** anywhere in this repo that the pipeline has run on
target hardware; no profiling script, no timing manifest exists.
`demo_stream.py`'s `--speed` flag is a simulated-real-time replay
multiplier for the demo UI, not a hardware benchmark.

**Flag clearly — this is an external dependency, not this repo's scope:**
per `docs/PROJECT_STATUS.md` Part 3, this matches the project's own
documented scope split — Jetson/quantization/distillation work belongs to
a teammate, not this ECG-pipeline codebase. Nothing in this repo should
claim edge-deployment readiness until that separate work happens and
reports back real numbers here.

**Artifact that exists (for sizing context only, not a benchmark):**
production classifier `five_class_xgb.json` is 15MB on disk (the only
model file `ECGPipeline`'s default inference path loads); MedGemma's local
model blobs total 3.5GB (`~/.ollama/models`), almost certainly too large
for a Jetson Nano's typical budget without the quantization/distillation
work that's out of this repo's scope.

**First concrete step (once that teammate's work is ready to test):**
run the existing `ecg_inference/` package's `run_inference.py` on target
hardware with real timing instrumentation (e.g. wrap the `ECGPipeline.run()`
call in a timer) — no such instrumentation exists yet in this repo.

---

## 3. CLOSED — DO NOT RETRY

Every experiment below has a documented, evidence-based negative verdict
in `docs/archive/ABLATION_REPORT.md`. Re-attempting any of these without
new data or a fundamentally different approach would re-derive the same
documented failure.

| Experiment | Result | Verdict | Source (ABLATION_REPORT.md) |
|---|---|---|---|
| **Two-stage classifier** (v1, v2, v3, v4, v5, qrsfeat, qrsfix, beatonly — 8 variants total) | Every version that improved S regressed V beyond the pre-registered zero-tolerance safety margin (`AGENT_RULES.md` Rule 8). Best variant (v5) reproduced the same V regression on an independently 2.5x-enlarged validation split, ruling out a sampling artifact. Even after fixing a real QRS-width-detector bug (qrsfix), V still regressed (0.866→0.810 F1). | **CLOSED.** Production `five_class_xgb.json` never replaced. | "Two-stage classifier" section (lines 619-1265), "Summary table" (986-997) |
| **CNN+Transformer** (v1 with SDDB, v2_nosddb without) | Both variants failed Rule 8 on **both** V and N (V F1 dropped to 0.730/0.615; even N dropped to 0.901/0.889). Removing SDDB made V and N *worse*, ruling out "bad training-domain data" as the sole explanation — points instead to training-set-size/model-capacity mismatch. Didn't even manage to improve S, the one class it was built to fix (S F1 0.121/0.126, both worse than the flat baseline's 0.139). | **CLOSED**, explicitly: "no further SDDB-composition ablation is planned... no further 'replace the classifier' architecture experiments are planned on this data." | "CNN+Transformer investigation: CLOSED" section (lines 1595-1630) |
| **LTAFDB/SDDB train-only enrichment** | LTAFDB alone contributed 8.5M beats (40-80x DS1's size), overwhelmingly N-labeled, with no down-weighting mechanism — drowned the real 749-example F pool rather than reinforcing it (277/363, 76% of true F beats collapsed into N). Macro-F1 0.3895→0.3779, worse on every rare class. | **DISCARDED.** | "Experiment: LTAFDB + SDDB train-only enrichment — DISCARDED" section (lines 203-246) |
| **`r_amp` feature addition** | Noise-to-mildly-negative: S F1 −0.022, V F1 −0.013, S→V confusion got measurably worse (+0.043). The dead-code bug that motivated finding this (r_amp computed but discarded) was fixed and kept; the feature itself was not promoted. | **DISCARDED** (feature not added to production; the underlying dead-code bug fix was kept separately). | "Prerequisite 2: wire in the dead `r_amp` feature" section (lines 147-199) |
| **timing-v1** (7 rolling-RR/prematurity features, 63-dim) | S F1 improved (+0.043, real) but V F1 regressed −0.051 (0.826→0.775) — disqualified per the pre-registered rule (a V regression disqualifies a run even when S improves). Mechanism confirmed via confusion matrix: true-S-predicted-as-V rose from 35.4%→55.8%, crushing V precision. | **DISCARDED, do not promote.** | "Experiment: +7 timing/local-rhythm features — DISCARDED" section (lines 250-368) |
| **timing-v2** (6 features, `compensatory_pause_flag` removed) | Hypothesis (that one feature was uniquely responsible) falsified: S F1 fell *below* the original baseline (0.125 vs 0.139) and V did not recover (0.763, worse than timing-v1's 0.775). Confirms the entanglement is general to any strong prematurity/timing signal, not one isolated feature — S and V share the same "premature beat" timing signature by definition; the distinguishing information between them is morphological, not timing. | **DISCARDED, hypothesis falsified — fundamental, not fixable by more feature tweaking within a flat model.** | "Micro-experiment: timing features minus `compensatory_pause_flag`" section (lines 371-452) |

**Do not re-attempt any of the above as "smarter model, same data."** Per
`ABLATION_REPORT.md`'s own closing verdict on the CNN+Transformer
investigation: any future S-class improvement attempt should be scoped as
either (1) a genuinely new data source (a second lead, or more S-beat
volume from a matching single-lead wearable domain), or (2) accepted as a
documented data/hardware limitation — not a third retry of "same data,
different architecture."

---

## 4. HONEST LIMITATIONS THAT CANNOT BE FIXED WITHOUT NEW DATA

### 4.1 — S class (F1 = 0.139): a detection problem, not a discrimination problem

The dominant S failure mode is **67.6% of true S beats being missed as
Normal entirely** — not confused with V. S beats look almost identical in
*shape* to Normal beats (they originate just above the ventricles, so the
electrical signal still travels the normal pathway); the actual clinical
signature is that they arrive *early* (premature timing), which requires
P-wave/PR-interval information the current 56-dim feature set only
partially captures (2 of 56 dimensions are timing-based) and which is
frequently not visible at all on a single-lead wearable device. Section 3's
"CLOSED" experiments confirm this isn't fixable by feature-tweaking within
the current flat-model, single-lead approach — every attempt that gave
timing signal more weight blurred the S/V boundary instead of sharpening
S/N. Real fix requires either a second ECG lead (P-wave visibility) or a
sequence/context-aware model architecture — both are new-data or
new-architecture questions, not tuning questions.

*Source: `docs/PROJECT_STATUS.md` §1.1, `docs/archive/ABLATION_REPORT.md`
Section 4/6 feature-family ablation.*

### 4.2 — F class (F1 = 0.011): a hard data ceiling

F (fusion) beats have only ~403 real training examples in all of DS1 (363
of them concentrated in one MITDB record, deliberately kept in training
rather than validation). No available dataset adds independent F
examples: Icentia11k has **zero** F-labeled beats at all (confirmed both
by direct scan and PhysioNet's own official class list, per
`RESEARCH_AUDIT.md` Section 1 #5/#11); SVDB adds only 22 more F beats.
62.8% of true F beats get predicted as S — not random scatter, but
systematic crowding-out by the much larger, shape-adjacent S class in the
same ambiguous, in-between region of feature space F beats naturally
occupy. This is a genuine data-scarcity ceiling, not a modeling failure —
confirmed by ruling out label-ambiguity (F uses the same cardiologist
annotation protocol as every other class) and bad-features (the same
56-dim vectors support strong N/V performance) as explanations.

*Source: `docs/archive/RESEARCH_AUDIT.md` Section 6/Q6 (evidence-based,
not guessed), `docs/PROJECT_STATUS.md` §1.1.*

### 4.3 — Domain gap: all labeled training data is hospital-recorded, deployment is wearable

Every classifier metric in this document (N/S/V/F/Q sensitivity/precision/
F1) is measured on MITDB/SVDB — cardiologist-labeled, in-hospital,
multi-lead-capable Holter recordings. The actual deployment devices
(VitalPatch, SeNSiO) are single-lead wearables, and **zero hand-labeled
examples from the project's own real devices exist anywhere** — every
"trained model is more realistic than the old rule-based logic" claim
rests on eyeballing 1-2 real recordings' plausibility, not measured
accuracy on the actual deployment domain. No public dataset substitutes
for this: Icentia11k is the closest available proxy (continuous,
single-lead, wearable-style) but is silver-standard (technologist-reviewed,
not board-certified-cardiologist-reviewed) and has no F labels at all.
This is flagged in `RESEARCH_AUDIT.md` as "the most likely thing an expert
reviewer would push back on," and the fix has no shortcut: hand-labeling a
batch of real VitalPatch/SeNSiO heartbeats is the one thing no public
dataset can substitute for.

*Source: `docs/archive/RESEARCH_AUDIT.md` Section 0.9, Section 4 #6,
Section 6 #4.*

### 4.4 — NEWS2/qSOFA: this ECG-only bridge cannot supply vitals

As documented in Section 1.1 above, the NEWS2/qSOFA safety overrides are
real, implemented cascade rules with nothing feeding them — because this
repository's pipeline is ECG-only by design. NEWS2 (systolic BP,
respiratory rate, SpO2, temperature, heart rate, consciousness) and qSOFA
(respiratory rate, systolic BP, mental status) both require vitals this
codebase has no sensor path for. `MedGemma-Agent`'s own `Vitals Monitor`
tool (a separate system, per its README) already computes NEWS2+qSOFA from
real vitals — closing this gap means integrating *with* that system (or an
equivalent real vitals source), not adding code inside this ECG pipeline
alone.

*Source: `docs/PROJECT_STATUS.md` Part 2 #4 (self-disclosed in
`agent_bridge.py`'s own `known_limitations` text); `MedGemma-Agent/README.md`
for what a real vitals source would look like.*
