# Cliniaura ECG Pipeline

A wearable single-lead ECG pipeline for post-operative patient monitoring. A raw
device recording goes through a signal-quality gate, filtering, R-peak detection,
per-beat feature extraction, AAMI 5-class (N/S/V/F/Q) beat classification,
rhythm-pattern detection, and a deterministic risk cascade, producing a
structured JSON report with an optional MedGemma-authored clinical narrative. A
second path pushes the ECG risk summary plus real VitalPatch vitals (heart rate,
respiratory rate, temperature) to the `MedGemma-Agent` service for NEWS2/qSOFA
scoring, and feeds the returned scores back into the risk cascade as
escalation-only safety overrides.

**Authoritative technical reference:**
[`Docs/PIPELINE_METHODS_AND_RESULTS.md`](Docs/PIPELINE_METHODS_AND_RESULTS.md) —
full stage-by-stage methods, rejected alternatives, and every measured result,
with the artifact each number came from. This file is the map; that file is the
evidence.

Plain-English walkthrough with no ML/ECG background assumed:
[`Docs/PROJECT_OVERVIEW.md`](Docs/PROJECT_OVERVIEW.md).
Classifier training/eval handover history: [`Docs/project_doc.md`](Docs/project_doc.md).

---

## Repository structure

*Verified against the working tree, 2026-07-30.*

```
run_inference.py             single CLI entry point for ECG-only inference
requirements_inference.txt   pinned runtime dependencies for ecg_inference/ (no torch)

ecg_inference/               DEPLOYMENT: inference-only package, no training code
  preprocess.py                Stages 1-4: parse device streams, SQI gate, resample, filter chain
  detector.py                  Stage 5: XQRS R-peak detection + beat segmentation
  features.py                  Stage 6: 56-dim handcrafted per-beat features + HRV
  classifier.py                Stages 7-8: beat/rhythm classification + risk scoring
  report.py                    Stage 9: structured RiskReport JSON + deterministic narrative
  pipeline.py                  orchestrator: ECGPipeline wires stages 1-9 together
  __init__.py                  re-exports the public API
  models/                      five_class_xgb.json + .classes.json (production weights)
  INFERENCE_README.md          every flag, every JSON field, every threshold

ecg_pipeline/                DEVELOPMENT pipeline: same logic as ecg_inference/, plus the
                              live MedGemma integration, batch runners, and demo tooling
  ecg_pipeline_core.py          the 9-stage pipeline; source of truth ecg_inference/ was
                                 extracted from verbatim (see Docs/inference_ready.md)
  agent_bridge.py               two CLI modes: `push` (ECG + real vitals -> MedGemma-Agent,
                                 NEWS2/qSOFA loopback) and `report` (single-file report)
  batch_vitalpatch_report.py    resumable batch runner over data/raw/vitalpatch/
  batch_prorhythm_report.py     resumable batch runner over data/raw/prorhythm/ (SeNSiO)
  manifest_summary.py           combines both batch manifests, prints summary + showcase reports
  demo_stream.py, report_ui.py, synthetic_ecg.py, test_pipeline_synthetic.py
                                 demo/replay UI and a synthetic-ground-truth self-test suite
  verify_vitals_pairing.py      measures real vitals-ECG pairing coverage against all files on
                                 disk -- how Task 1's fix (interval containment) was verified
  README.md, DEMO_UI_README.md  code-level guide and demo-UI guide
  models/                       production model only (five_class_xgb.json + .classes.json)
  example_reports/              8 saved JSON reports (7 WFDB + 1 VitalPatch)

training/                    TRAINING-ONLY: never imported by ecg_inference/ or the
                              production report path (see Docs/inference_ready.md, item 4)
  ecg_pipeline_tools.py          download-datasets / train-classifiers / eval-classifier /
                                 train-twostage / analyze-twostage CLI
  model_artifacts/               4 files: the SVDB train-enrichment comparison pair
                                 (five_class_xgb_with_svdb / _no_svdb, + .classes.json each).
                                 The 46 earlier experimental artifacts were removed in
                                 commit 8124726 -- their results live in
                                 Docs/archive/ABLATION_REPORT.md, not in this repo.
  _original_stages/              pre-consolidation per-stage source files, kept for diffing only

Docs/                        documentation (16 tracked files)
  PIPELINE_METHODS_AND_RESULTS.md  ** the authoritative methods + measured results reference **
  HANDOFF.md                     teammate handoff -- what's done, broken, and the task order
                                 (source of truth for the 2026-07-30 session below)
  AGENT_RULES.md                standing research-integrity rules -- read before touching the
                                 classifier or training code
  DATASETS.md                    every dataset used, where it lives, what it's for
  BEAT_CLASSIFICATION_SUMMARY.md concise classifier results summary
  CNN_TRANSFORMER_EXPERIMENT.md  the closed deep-learning experiment, summarized
  NEXT_STEPS.md                  open bugs, improvements, closed experiments, limitations
  PROJECT_OVERVIEW.md            the full plain-English project writeup
  PROJECT_STATUS.md              point-in-time project status snapshot
  EDGE_DEPLOYMENT_FIX_REPORT.md  P1-P4 deployment-blocker fixes
  inference_ready.md             the ecg_inference/ packaging pass and equivalence proof
  project_doc.md                 classifier training/eval handover doc
  CLINICIAN_REVIEW_INSTRUCTIONS.md  how to review the 50-segment CRITICAL sample (Task 5)
  NEWS2_PARTIAL_COVERAGE_DECISION.md  options for the NEWS2 3/6-coverage escalation policy --
                                 not decided, a clinical call (item 8 above)
  TODAY_IMPROVEMENTS_REPORT.md   2026-07-30 session: before/after table, PASS/FAIL/BLOCKED per
                                 task, production-readiness assessment
  SESSION_LOG_TODAY.md           2026-07-30 session: commits, blockers, git state, next task
  archive/                       GITIGNORED, local only -- full ablation history
                                 (ABLATION_REPORT.md) and open research items
                                 (RESEARCH_AUDIT.md). Cited throughout this file; not
                                 available to anyone cloning this repo.

data/                        GITIGNORED -- raw datasets, generated reports, batch manifests,
                              UI caches. Every measured result quoted below was computed from
                              files here; they are not in git (icentia11k alone is ~257GB).
                              See Docs/DATASETS.md for how to re-download each source.

MedGemma-Agent/              git submodule (separate repo: saranambiar/MedGemma-Agent) -- the
                              vitals + ABG post-op monitoring agent this pipeline pushes to.
                              See "Submodule state" below before cloning.
```

---

## Quickstart

### ECG-only inference

No training dependencies, no `torch`:

```bash
pip install -r requirements_inference.txt

python run_inference.py \
    --input  data/raw/vitalpatch/Patch_1844AC/1778423422284_VC2B008BF_1844AC_ecg.csv \
    --output output.json \
    --source vitalpatch \
    --narrative
```

Flags: `--input`, `--output` (both required), `--source {vitalpatch,sensio,wfdb}`,
`--classifier`, `--segment-index`, `--narrative`.
Full reference: [`ecg_inference/INFERENCE_README.md`](ecg_inference/INFERENCE_README.md).

### Multimodal push (ECG + real vitals → MedGemma-Agent)

Requires the `MedGemma-Agent` FastAPI service running and reachable.

```bash
python -m ecg_pipeline.agent_bridge push \
    --vitalpatch-root data/raw/vitalpatch \
    --vitals-root     data/vitals_downloads \
    --limit 5
```

`push` is the default mode, so the `push` keyword may be omitted.
Flags: `--vitalpatch-root`, `--vitals-root`, `--classifier`, `--agent-url`,
`--api-key`, `--limit`, `--seed`.

**There is no `--input`/`--output` on `push`.** `--vitalpatch-root` must be the
*parent* of one or more `Patch_*` directories, not a `Patch_*` directory itself;
files are discovered by glob in sorted order and taken up to `--limit`.

### Single-file report with a live MedGemma narrative

Needs `ollama serve` running the `medgemma:latest` model.

```bash
python -m ecg_pipeline.agent_bridge report \
    --file <path-to-raw-csv> \
    --source vitalpatch

python -m ecg_pipeline.batch_vitalpatch_report   # full resumable batch
python -m ecg_pipeline.manifest_summary          # combined summary + showcase reports
```

`report` flags: `--file`, `--source {vitalpatch,wfdb}` (both required),
`--classifier`, `--limit`, `--out-dir`.

### Training / evaluation

Real datasets required (see [`Docs/DATASETS.md`](Docs/DATASETS.md)); governed by
[`Docs/AGENT_RULES.md`](Docs/AGENT_RULES.md) — read it first.

```bash
python -m training.ecg_pipeline_tools download-datasets --all
python -m training.ecg_pipeline_tools train-classifiers --dataset mitdb
python -m training.ecg_pipeline_tools eval-classifier \
    --model ecg_pipeline/models/five_class_xgb.json --split-set ds2
```

---

## What is verified working

Every number in this section was computed from a file on disk. Sources are named.

### 1. Beat classifier — DS2 held-out, 45,804 beats

*Source: `Docs/archive/ABLATION_REPORT.md`, "Production baseline" (gitignored,
local only). Patient-level inter-patient split; never re-tuned against DS2.*

| Class | Sensitivity | Precision | F1 | Support | Trust |
|---|---|---|---|---|---|
| N (Normal) | 0.972 | 0.973 | **0.972** | 40,634 | High — reliable |
| S (Supraventricular) | 0.130 | 0.150 | **0.139** | 1,795 | Low — screening signal only |
| V (Ventricular) | 0.912 | 0.754 | **0.826** | 3,005 | Moderate-high — the clinically critical class |
| F (Fusion) | 0.005 | 0.125 | **0.011** | 363 | Essentially unsolved — treat as noise |
| Q (Unknown) | 0.000 | 0.000 | 0.000 | 7 | N/A — near-zero support |

**Macro-F1: 0.3895.** N is 40,634 of 45,804 beats (88.7%), so a model that only
ever predicts N scores 88.7% "accuracy" while being clinically useless — never
use accuracy as the success metric here (`Docs/AGENT_RULES.md`, rule 4).

### 2. AFib rule validation — LTAFDB, 84 records, 449,749 windows

*Source: `Docs/archive/ABLATION_REPORT.md`, "AFib rule validation". Ground truth
from real rhythm-change annotations. Validated 2026-07-28.*

| RR-CoV threshold | Sensitivity | Specificity | F1 |
|---|---|---|---|
| 0.15 (old default) | 0.808 | 0.800 | 0.829 |
| **0.10 (current default)** | **0.971** | **0.716** | **0.893** |

0.10 is Youden's-J-optimal on this data and has the best F1 of any threshold
tested. Changed in both `ecg_pipeline_core.py` and `ecg_inference/classifier.py`.

### 3. ECG-only batch on real device data — complete, re-run 2026-07-30

*Source: `data/reports/vitalpatch_run_manifest.csv` (gitignored, local only),
regenerated 2026-07-30 via `python -m ecg_pipeline.batch_vitalpatch_report`
after the parser fix (previous manifest predated it and had 70 `PARSE_ERROR`
rows — see prior "Known and unfixed" entry, now resolved); aggregated in
`Docs/PIPELINE_METHODS_AND_RESULTS.md` §6.1.*

| Metric | Value |
|---|---|
| Raw ECG files, 6 patients | 2,375 |
| Segments processed | 3,628 |
| Total signal | 84.3 hours |
| Total beats analyzed | 368,072 |
| Median quality score | 0.88 |
| **Assessable** | **2,889 (79.6%)** |
| NOT_ASSESSABLE | 739 (20.4%) |
| Parse errors | 0 |

Risk distribution: LOW 1,881 (51.8%) · MEDIUM 166 (4.6%) · HIGH 552 (15.2%) ·
CRITICAL 290 (8.0%) · NOT_ASSESSABLE 739 (20.4%). CRITICAL is 10.0% of
assessable segments (290/2,889).

**79.6% assessability on real, uncontrolled, at-home wearable data is the
current headline ECG-only result**, up from a previously-reported 78.3%
computed on a stale, pre-parser-fix manifest (70 of 3,570 rows were
`PARSE_ERROR`; re-parsing added 128 net segments with 0 failures). The
10.0%-of-assessable CRITICAL rate is reported as observed, not endorsed — see
"Known and unfixed" below.

A second device batch (SeNSiO/prorhythm, `data/reports/prorhythm_run_manifest.csv`)
is complete at 18 recordings, 13 assessable. Too small for rate estimates; useful
as a second-device smoke test only.

### 4. MedGemma-Agent partial-vitals integration

The Agent schema now accepts partial vitals: `heart_rate` is required;
`spo2`, `systolic_bp`, `diastolic_bp`, `respiratory_rate` and `temperature` are
`Optional`. NEWS2 scores respiratory rate and temperature; qSOFA scores
respiratory rate. Missing components are scored as *missing*, never imputed.

**Test suite: 25/25 passing** (18 pre-existing + 7 new) — re-verified 2026-07-29
via `MedGemma-Agent/venv/bin/python -m pytest tests/test_guardrails.py`.

Getting this working required changes at six layers (Pydantic schema, scoring
rules, threshold config, DB nullability + a 2,588-row migration, input
guardrails, and the ECG-side payload/loopback). The subtlest: `input_guardrails`
precomputes NEWS2 and `risk_scorer` does `state.get("news2") or calculate_news2(...)`
— a Pydantic instance is always truthy, so the risk scorer's own call never runs.
Full detail in `Docs/PIPELINE_METHODS_AND_RESULTS.md` §7.3.

### 5. End-to-end multimodal push — full batch complete 2026-07-30

*Source: `data/reports/multimodal_batch/multimodal_manifest.json` (gitignored,
local only), produced by `python ecg_pipeline/multimodal_batch.py`, elapsed
2800.6s (~46.7 min). Supersedes the earlier 14-segment plumbing check
(`sample14_manifest_preserved.json`) and the small-scale 25-segment check
(`sample25_manifest_smallcheck.json`) done immediately before this run.*

| Metric | Result |
|---|---|
| Segments | 3,632 (all 6 patients) |
| `agent_push_status: SUCCESS` | 3,619 / 3,632 (99.6%) |
| `agent_push_status: SKIPPED_MISSING_REQUIRED_FIELDS` | 13 / 3,632 — all 13 have a matched vitals file but no valid HR reading in it (`hr: None`, correctly, not defaulted) |
| Vitals file found | **3,632 / 3,632 (100.0%)** |
| Assessable (ECG-only) | 2,891 / 3,632 (79.6%) |
| NEWS2 coverage 3/6 | 3,301 / 3,632 |
| NEWS2 coverage 2/6 | 286 / 3,632 |
| NEWS2 coverage 1/6 | 32 / 3,632 |
| **Risk level changed by vitals (`combined_risk != ecg_only_risk`)** | **52 / 3,632 (1.43%)** |

**All 52 changes were escalations (LOW/MEDIUM → HIGH/CRITICAL), never
downgrades** — consistent with the design rule that vitals can only push risk
up. All 52 were triggered by qSOFA reaching 2 (both the HR and RR flags true
simultaneously). One of those 52 segments (`Patch_184635`,
`1780050938536_..._seg0`) also independently reached **NEWS2 = 7** (HR score 3
+ RR score 3 + Temp score 1, from real HR 150.3 bpm, RR 26.7/min, Temp 38.3°C)
— **this corrects a prior assumption in this project** (see "Known and
unfixed" and `Docs/NEWS2_PARTIAL_COVERAGE_DECISION.md`): NEWS2 ≥ 7 is not
arithmetically unreachable with only 3 components (HR max 3 + RR max 3 + Temp
max 2 = 8, so 7 is reachable), it is just rare — 1 real segment out of 3,632
reached it.

**Relationship between NEWS2 score and ECG-only risk level:** mean
agent-returned NEWS2 score by ECG-only risk level (all real, from the same
manifest): LOW 1.05 (n=2,051), MEDIUM 0.93 (n=205), HIGH 1.07 (n=1,066),
CRITICAL 0.69 (n=297). **No visible monotonic relationship** — this is a
directly measured null result, not an absence of testing.

**This confirms the pipeline works end-to-end at full scale, with the vitals
integration now doing something measurable** (52 real escalations) — a
material change from the earlier 0/14 plumbing-only result. It is still not a
clinical validation: no ground truth exists to say whether any of these 52
escalations, or the 290 CRITICAL calls overall, are correct.

---

## Known and unfixed

Listed explicitly so nobody builds on a number that isn't settled.

- **Vitals-pairing bug — FIXED 2026-07-30.** `load_real_vitals()` now matches
  primarily by interval containment (does a vitals file's own
  [first_row_ts, last_row_ts] contain the ECG's timestamp?), falling back to
  the original nearest-filename-within-30s rule only for genuine gaps. Measured
  on all 2,375 real ECG files (`ecg_pipeline/verify_vitals_pairing.py`):
  interval containment alone gets ~95.1% (matches the previously-estimated
  ~95.3%); combined with the 30s fallback, coverage is **100.0%** (2,375/2,375)
  — nearly every containment miss turns out to be within seconds of a
  vitals-file boundary, since files chain together almost back-to-back. Was
  60.3% before the fix. See commit `709e225`.

- **The full 6-patient multimodal batch — COMPLETE 2026-07-30.** An earlier
  attempt (SLURM job 2660129) was killed by a wall-clock limit before finishing
  the first patient; this session's run (job 2660558's allocation) completed
  all 3,632 segments in 2800.6s with 0 errors. See §5 above for the real
  measured results: 52/3,632 (1.43%) segments had their risk level changed by
  vitals, all escalations. The claim that NEWS2 ≥ 7 is "arithmetically
  unreachable" at 3/6 coverage was wrong — HR(max 3) + RR(max 3) + Temp(max 2)
  sums to a possible 8, and one real segment measured exactly 7. It is rare,
  not impossible: 1 segment out of 3,632.

- **S class (F1 0.139) and F class (F1 0.011) — closed research problem.** Four
  genuinely different approaches were tried and all failed identically: extra
  timing features, +8.5M extra training beats (LTAFDB/SDDB), a two-stage
  gate-then-classify architecture in 5 variants, and a CNN+Transformer in 2 data
  variants. Every one that improved S regressed V past the project's
  zero-tolerance safety margin. **Do not re-attempt without new data.** The
  production model was never changed by any of these experiments. See
  `Docs/archive/ABLATION_REPORT.md` and `Docs/CNN_TRANSFORMER_EXPERIMENT.md`.

- **SpO2 and blood pressure — hardware gap.** VitalPatch has no sensor for
  either. They are always emitted as `None` with `real: False`, never imputed and
  never defaulted to a "normal" value. Only 3 of the 6 NEWS2 components (heart
  rate, respiratory rate, temperature) can be sourced from this device.

- **The 10.0%-of-assessable CRITICAL rate is unvalidated.** `[UNVERIFIED]` No
  ground truth exists for this corpus, and V-class precision is 0.754 — roughly
  1 in 4 ventricular calls is wrong, and PVC burden drives both CRITICAL rules.
  Clinician adjudication of a sample is the only way to settle whether this rate
  is signal or false-positive noise. A 50-segment stratified sample is prepared
  at `data/reports/clinician_review_sample.csv` (see
  `Docs/CLINICIAN_REVIEW_INSTRUCTIONS.md`); no clinician has reviewed it yet.

- **The ECG-only batch manifest was regenerated 2026-07-30** after the parser
  fix (`ecg_pipeline_core.py:292-307`, coerce-to-NaN with pairwise drop). Re-run
  via `python -m ecg_pipeline.batch_vitalpatch_report`: **0 PARSE_ERROR rows**,
  3,628 segments (was 3,570 with 70 `PARSE_ERROR`). Assessability moved from
  78.3% to **79.6%**. Numbers above are current as of this re-run.

- **The AFib rule's `window=20` was never swept** — only its threshold was.

- **EHR RAG (in the `MedGemma-Agent` submodule) is unaudited.** Not reviewed in
  the work that produced the current state; no claim is made about its behaviour.

- **Edge / Jetson deployment has not been started** from this repo.
  Distillation, quantization, and Jetson benchmarking are a separate, ongoing
  scope owned by a teammate.

- **Confidence tiers in the JSON report are a heuristic, not calibrated
  probabilities** — no conformal calibration set is loaded by default.

- **MedGemma's free-text narrative occasionally misstates a numeric comparison.**
  A post-hoc validator catches and strips this before saving, so the *saved*
  report is never wrong, but the underlying 4B model remains imperfect. The risk
  level itself is never affected — it is fixed by the deterministic cascade
  before MedGemma runs.

- **MITDB records 103 and 111 show real R-peak over-detection** — investigated,
  not fixed. See commit `2420ecc`.

- This is **decision support, not a diagnosis.** Every report ends with an
  explicit clinician-review disclaimer.

---

## What to fix next

Ordered by evidence strength and value, not by effort. Each item names the
measurement that justifies it.

### Done as of 2026-07-30 (were previously blocking)

1. ~~Fix the vitals-pairing rule.~~ **DONE**, commit `709e225`. Interval
   containment + 30s fallback. Measured coverage: 60.3% → **100.0%**
   (2,375/2,375 real ECG files).

2. ~~Complete the full 6-patient multimodal batch.~~ **DONE**, 2026-07-30.
   3,632 segments, 0 errors, 2800.6s. 52/3,632 (1.43%) segments had their risk
   level changed by vitals (all escalations). See §5 above.

3. ~~Re-run the ECG-only batch to refresh the manifest.~~ **DONE**. 3,628
   segments, 0 parse errors, 79.6% assessable (was 78.3%).

4. ~~Select vitals rows near the ECG timestamp instead of averaging the whole
   file.~~ **DONE**, commit `8286455`. `±2 min` window around the ECG
   timestamp, falling back to whole-file only if the window is empty. Example:
   one real segment's HR average narrowed from 81.9 (n=300, whole file) to
   80.2 (n=42, windowed).

5. ~~Repo hygiene — fix `git gc` refusing to run.~~ **DONE**, 2026-07-30. Root
   cause was 10,020 old, unreferenced `git stash` loose objects (confirmed via
   empty `git stash list` before pruning); `git prune` + `git gc` resolved it,
   `.git/gc.log` is gone, `git fsck --full` is clean.

### Blocking — needs someone other than this session

6. **Push the `MedGemma-Agent` submodule commits upstream.** `37f4432`,
   `1f36dda` and `087f261` exist only on a local `dev` branch, so the submodule
   pointer recorded here cannot be resolved by a fresh clone. Needs coordination
   with the submodule's owner — it is a separate repository. Not attempted this
   session; see `Docs/SESSION_LOG_TODAY.md`.

### High value — quantifies risk that engineering alone cannot resolve

7. **Get clinician adjudication on a sample of the 290 CRITICAL segments.** This
   is the largest unquantified risk in the system. 10.0% of assessable segments
   returning CRITICAL is high for a recovering post-op cohort, V-class precision
   is 0.754 (~1 in 4 ventricular calls wrong), and PVC burden drives both
   CRITICAL rules — but there is no ground truth on this corpus, so the rate
   cannot be shown to be either signal or noise from inside the pipeline. A
   stratified sample of 50 segments is prepared at
   `data/reports/clinician_review_sample.csv` (see
   `Docs/CLINICIAN_REVIEW_INSTRUCTIONS.md`; not yet reviewed). Alert burden,
   deployment readiness, and threshold tuning all depend on the answer.

8. **Decide what the NEWS2 override should do on 3/6 coverage.** Corrected
   2026-07-30: the earlier claim that NEWS2 ≥ 7 is "arithmetically unreachable"
   at 3/6 coverage was wrong — HR(max 3) + RR(max 3) + Temp(max 2) sums to a
   possible 8, and one real segment (of 3,632) measured exactly 7. It is rare,
   not impossible. The remaining question — whether a rare-but-real ≥7
   escalation on 3/6 coverage is clinically sound to act on — is a clinical
   safety decision, not an engineering one; four options are written up in
   `Docs/NEWS2_PARTIAL_COVERAGE_DECISION.md` (the write-up is done, the
   decision itself is not). *Note: the qSOFA ≥2 override works and drove all
   52 real escalations measured in the full batch.*

### Improvements — real gains, none blocking

9. **Sweep the AFib rule's `window=20`.** Only the threshold was ever swept
   (against LTAFDB, 449,749 windows). Window size remains an open validation
   question and the same harness can answer it. **Do not attempt without
   sign-off** — untested statistical claims territory (see `Docs/HANDOFF.md`).

10. **Calibrate the confidence tiers.** They are currently a heuristic; no
    conformal calibration set is loaded by default. `ConformalConfig` already
    exists in `ecg_pipeline_core.py` but is unused in the default path.

11. **Add a consciousness/AVPU input path** to reach 4 of 6 NEWS2 components.
    This is the only remaining component obtainable without new hardware — SpO2
    and BP require sensors VitalPatch does not have.

### Explicitly not to be re-attempted

- **S-class and F-class improvement on the current data.** Four independent
  approaches failed identically, each trading V-class regression for S-class
  gain. Closed until genuinely new labeled data or multi-lead hardware exists —
  not a tuning problem. See `Docs/archive/ABLATION_REPORT.md`.

---

## Submodule state

`MedGemma-Agent/` is a submodule of a **separate repository**
(`github.com/saranambiar/MedGemma-Agent`). Three commits carrying the
partial-vitals work (`37f4432`, `1f36dda`, `087f261`) exist **only on the local
`dev` branch** and have not been pushed upstream, so the submodule pointer
recorded here **cannot currently be resolved by a fresh clone**. Its working tree
also carries uncommitted changes: 56 files differ, of which 50 are line-ending
(CRLF) churn only and 6 contain real pre-existing ECG-integration edits not
authored as part of this pipeline's work.

---

## Standing rules

Before touching `ecg_pipeline_core.py`, `training/ecg_pipeline_tools.py`, or
anything under a `models/` directory, read
[`Docs/AGENT_RULES.md`](Docs/AGENT_RULES.md) — every rule in it exists because it
was violated once at real cost.
