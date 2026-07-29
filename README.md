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

*Verified against the working tree, 2026-07-29.*

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

Docs/                        documentation (10 tracked files)
  PIPELINE_METHODS_AND_RESULTS.md  ** the authoritative methods + measured results reference **
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

### 3. ECG-only batch on real device data — complete

*Source: `data/reports/vitalpatch_run_manifest.csv` (gitignored, local only);
aggregated in `Docs/PIPELINE_METHODS_AND_RESULTS.md` §6.1.*

| Metric | Value |
|---|---|
| Raw ECG files, 6 patients | 2,375 |
| Segments processed | 3,570 |
| Total signal | 81.9 hours |
| Total beats analyzed | 357,909 |
| Median quality score | 0.889 |
| **Assessable** | **2,795 (78.3%)** |
| NOT_ASSESSABLE | 775 (21.7%) |

Risk distribution: LOW 1,832 · MEDIUM 159 · HIGH 527 · CRITICAL 277 ·
NOT_ASSESSABLE 705 · parse error 70.

**78.3% assessability on real, uncontrolled, at-home wearable data is the
headline ECG-only result.** The 9.9%-of-assessable CRITICAL rate is reported as
observed, not endorsed — see "Known and unfixed" below.

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

### 5. End-to-end multimodal push — validated on 14 segments

*Source: `data/reports/multimodal_batch/sample14_manifest_preserved.json`
(gitignored, local only).*

| Metric | Result |
|---|---|
| Segments | 14 (2–3 per patient, all 6 patients) |
| `agent_push_status: SUCCESS` | **14 / 14** |
| Vitals file found | 14 / 14 |
| NEWS2 coverage 3/6 | 6 / 14 |
| NEWS2 coverage 2/6 | 5 / 14 |
| NEWS2 coverage 1/6 | 3 / 14 |
| Risk level changed by vitals | 0 / 14 |

**This confirms the pipeline works end-to-end.** It is a plumbing validation, not
a clinical result — see the next section for why the 0/14 figure must not be read
as evidence about vitals' clinical value.

---

## Known and unfixed

Listed explicitly so nobody builds on a number that isn't settled.

- **Vitals-pairing bug — coverage numbers are NOT final.** The current
  nearest-filename-timestamp rule with a 30-second tolerance achieves **60.3%**
  coverage across the 6-patient set (1,432 of 2,375 ECG files), ranging from
  15.2% (Patch_1844AC) to 99.0% (Patch_183594) — the 30 s cutoff was calibrated
  on a single patient and does not generalize. Root cause: vitals files span
  ~20 minutes of internal per-row timestamps, so a valid ECG can land mid-file
  while the *filename* timestamp is minutes away. Interval-containment matching
  would recover to **95.3%**, but **is not implemented**. Do not quote any vitals
  coverage figure as final until this is fixed. Measurement:
  `Docs/PIPELINE_METHODS_AND_RESULTS.md` §7.2.

- **The full 6-patient multimodal batch has NOT completed.** The only attempt
  (SLURM job 2660129) was killed by a wall-clock time limit before finishing the
  first patient. **No override, escalation, or risk-change statistics exist from
  a completed run**, and none should be reported as fact. The 14-segment sample
  above is the only multimodal data on disk. Note also that at 3/6 NEWS2
  coverage the NEWS2 ≥ 7 CRITICAL override is arithmetically unreachable, so
  "vitals changed nothing" cannot currently distinguish "vitals don't matter"
  from "the override can't fire."

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

- **The 9.9%-of-assessable CRITICAL rate is unvalidated.** `[UNVERIFIED]` No
  ground truth exists for this corpus, and V-class precision is 0.754 — roughly
  1 in 4 ventricular calls is wrong, and PVC burden drives both CRITICAL rules.
  Clinician adjudication of a sample is the only way to settle whether this rate
  is signal or false-positive noise.

- **The ECG-only batch manifest predates the parser fix.** 70 of its 3,570 rows
  are `PARSE_ERROR: could not convert string to float: '-'`. That `-` sentinel
  bug has since been fixed (`ecg_pipeline_core.py:292-307`, coerce-to-NaN with
  pairwise drop) — re-parsing all 2,375 raw files with the current parser gives
  **0 failures in 27 s** (verified 2026-07-29). The batch has not been re-run, so
  the 78.3% assessability figure above is computed from the pre-fix manifest and
  will shift slightly once it is. Treat it as a close lower bound, not final.

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

### Blocking — do these before quoting any new multimodal number

1. **Fix the vitals-pairing rule.** Replace nearest-filename-timestamp (±30 s)
   with interval containment against each vitals file's internal first/last row
   timestamps, keeping nearest-within-30 s as a fallback for ECGs that fall in
   gaps between files. *Measured payoff: 60.3% → 95.3% coverage, +831 segments.*
   Two patients (Patch_1844AC, Patch_1849DF) go from 15.2%/18.1% to 100%. This is
   a heuristic bug, not a data limitation, and it is contained to
   `load_real_vitals()` in `ecg_pipeline/agent_bridge.py`. Running the full batch
   before this fix bakes a pairing artifact into the headline coverage figure.

2. **Complete the full 6-patient multimodal batch.** The only attempt was killed
   by a SLURM wall-clock limit. Requires: an allocation longer than the run, the
   `MedGemma-Agent` service restarted on the allocated node, and item 1 done
   first. *Runtime estimate: the 14-segment sample took 15.5 s including 14 live
   Agent round-trips (~1.1 s/segment), extrapolating to ~65 min for ~3,570
   segments — `[UNVERIFIED at scale]`, Agent latency has not been measured under
   sustained load.* Until this completes, no override or risk-change statistic
   exists to report.

3. **Push the `MedGemma-Agent` submodule commits upstream.** `37f4432`,
   `1f36dda` and `087f261` exist only on a local `dev` branch, so the submodule
   pointer recorded here cannot be resolved by a fresh clone. Needs coordination
   with the submodule's owner — it is a separate repository.

### High value — quantifies risk that engineering alone cannot resolve

4. **Get clinician adjudication on a sample of the 277 CRITICAL segments.** This
   is the largest unquantified risk in the system. 9.9% of assessable segments
   returning CRITICAL is high for a recovering post-op cohort, V-class precision
   is 0.754 (~1 in 4 ventricular calls wrong), and PVC burden drives both
   CRITICAL rules — but there is no ground truth on this corpus, so the rate
   cannot be shown to be either signal or noise from inside the pipeline. A
   stratified sample of ~50 segments would bound it. Alert burden, deployment
   readiness, and threshold tuning all depend on the answer.

5. **Re-run the ECG-only batch to refresh the manifest.** The parser fix for the
   `-` sentinel already landed and recovers all 70 previously-lost segments
   (verified: 2,375/2,375 files parse, 0 errors, 27 s), but
   `data/reports/vitalpatch_run_manifest.csv` still holds the pre-fix run, so
   every headline ECG-only number is computed from a manifest missing 2.0% of the
   corpus. Cheap to fix and it makes the numbers final.

### Improvements — real gains, none blocking

6. **Select vitals *rows* near the ECG timestamp** rather than aggregating the
   whole file. Each vitals file spans ~20 minutes; a single averaged HR/RR/Temp
   over that window is coarser than the data supports, and coarser than NEWS2
   assumes. Naturally follows from item 1.

7. **Sweep the AFib rule's `window=20`.** Only the threshold was ever swept
   (against LTAFDB, 449,749 windows). Window size remains an open validation
   question and the same harness can answer it.

8. **Decide what the NEWS2 override should do on 3/6 coverage.** At three
   components the maximum attainable NEWS2 is far below the ≥7 CRITICAL
   threshold, so the override is arithmetically unreachable on VitalPatch
   hardware — it is currently dead code on this device. Either adopt a documented
   device-specific threshold, or state explicitly that the override is inactive
   for this hardware. Silently shipping an override that cannot fire is the worst
   of the three options. *Note: the qSOFA ≥2 override is not affected — it
   reached 2 in 1 of the 14 sampled segments and works.*

9. **Calibrate the confidence tiers.** They are currently a heuristic; no
   conformal calibration set is loaded by default. `ConformalConfig` already
   exists in `ecg_pipeline_core.py` but is unused in the default path.

10. **Add a consciousness/AVPU input path** to reach 4 of 6 NEWS2 components.
    This is the only remaining component obtainable without new hardware — SpO2
    and BP require sensors VitalPatch does not have.

11. **Repo hygiene.** `git gc` currently reports too many unreachable loose
    objects and refuses to run automatically; `.git/gc.log` needs clearing after
    the root cause is addressed.

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
