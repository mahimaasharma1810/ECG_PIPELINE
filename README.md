# Cliniaura ECG Pipeline

A wearable single-lead ECG pipeline for post-operative patient monitoring:
raw device recording → signal-quality gate → filter → R-peak detection →
per-beat feature extraction → AAMI 5-class (N/S/V/F/Q) beat classification
→ rhythm-pattern detection → deterministic risk cascade → structured JSON
report, with an optional MedGemma-authored clinical narrative on top.

For the full plain-English walkthrough (no ML/ECG background assumed),
see [`Docs/PROJECT_OVERVIEW.md`](Docs/PROJECT_OVERVIEW.md). For the
classifier's own training/eval handover history, see
[`Docs/project_doc.md`](Docs/project_doc.md). This file is the map: what's
in the repo, how to run it, and the honest current numbers.

---

## Repository structure

```
ecg_inference/              # DEPLOYMENT: inference-only package, no training code
  __init__.py                  re-exports the public API
  preprocess.py                Stages 1-4: parse device streams, SQI gate, resample, filter chain
  detector.py                  Stage 5: XQRS R-peak detection + beat segmentation
  features.py                  Stage 6: 56-dim handcrafted per-beat features + HRV
  classifier.py                Stages 7-8: beat/rhythm classification + risk scoring
  report.py                    Stage 9: structured RiskReport JSON + deterministic narrative
  pipeline.py                  orchestrator: ECGPipeline wires stages 1-9 together
  models/
    five_class_xgb.json           production XGBoost classifier weights
    five_class_xgb.classes.json   its label-encoder classes
  INFERENCE_README.md          full reference: every flag, every JSON field, every threshold

run_inference.py             single CLI entry point (see Quickstart below)
requirements_inference.txt   pinned runtime dependencies for ecg_inference/ (no torch)

ecg_pipeline/                DEVELOPMENT pipeline: same logic as ecg_inference/, plus the
                              live MedGemma integration, batch runners, and demo tooling
  ecg_pipeline_core.py          the 9-stage pipeline, source of truth ecg_inference/ was
                                 extracted from (verbatim, see Docs/inference_ready.md)
  agent_bridge.py               Stage 9 MedGemma report builder + CLI (--file/--source)
  batch_vitalpatch_report.py    resumable batch runner over data/raw/vitalpatch/
  batch_prorhythm_report.py     resumable batch runner over data/raw/prorhythm/ (SeNSiO)
  manifest_summary.py           combines both batch manifests, prints summary + showcase reports
  demo_stream.py, report_ui.py, synthetic_ecg.py, test_pipeline_synthetic.py
                                 demo/replay UI and a synthetic-ground-truth self-test suite
                                 (exercises the real frozen pipeline; not training/eval code)
  README.md                     code-level guide to ecg_pipeline_core.py / (historically)
                                 ecg_pipeline_tools.py's internal sections
  DEMO_UI_README.md             what the demo/replay UI is and how to run it
  models/                       production model only (five_class_xgb.json + .classes.json)
  example_reports/              sample saved JSON reports (WFDB + VitalPatch)

training/                    TRAINING-ONLY: never imported by ecg_inference/ or the
                              production report path (see Docs/inference_ready.md, item 4)
  ecg_pipeline_tools.py          download-datasets / train-classifiers / eval-classifier /
                                 train-twostage / analyze-twostage CLI
  model_artifacts/               46 non-production experimental model files: the self-supervised
                                 encoder, 2 CNN+Transformer checkpoints, 7 two-stage classifier
                                 variants (v1-v5, qrsfeat, qrsfix, beatonly) — all "discarded" or
                                 "closed" per Docs/archive/ABLATION_REPORT.md, kept for reference
  _original_stages/              pre-consolidation per-stage source files, kept for diffing only

Docs/                        documentation
  AGENT_RULES.md                standing research-integrity rules (read before touching the
                                 classifier or training code — every rule exists because it was
                                 violated once)
  DATASETS.md                    every dataset used, where it lives, what it's for
  BEAT_CLASSIFICATION_SUMMARY.md concise classifier results summary
  CNN_TRANSFORMER_EXPERIMENT.md  the closed deep-learning experiment, summarized
  project_doc.md                 classifier training/eval handover doc (paths corrected 2026-07-28)
  PROJECT_OVERVIEW.md            the full plain-English project writeup
  PROJECT_STATUS.md              point-in-time project status snapshot
  EDGE_DEPLOYMENT_FIX_REPORT.md  P1-P4 deployment-blocker fixes (R-peak over-detection, SQI,
                                 parser robustness, report integrity)
  inference_ready.md             the ecg_inference/ packaging pass: what moved, what was
                                 verified, byte-for-byte-identical proof
  archive/                       detailed historical session logs and the full ablation history
                                 (gitignored — local reference only, not pushed; see
                                 Docs/archive/ABLATION_REPORT.md for every real experiment run
                                 against this classifier, and RESEARCH_AUDIT.md for open items)

data/                         gitignored — raw datasets, generated reports, UI caches. Not in
                              git (icentia11k alone is ~257GB); see Docs/DATASETS.md for how to
                              re-download each source.

MedGemma-Agent/              git submodule (separate repo: saranambiar/MedGemma-Agent) — the
                              vitals + ABG post-op monitoring agent this ECG pipeline's risk
                              summaries are designed to feed into. Not modified by this repo.
```

---

## Quickstart

**Inference only** (no training dependencies, no `torch`):

```bash
pip install -r requirements_inference.txt

python run_inference.py \
    --input  data/raw/vitalpatch/Patch_1844AC/1778423422284_VC2B008BF_1844AC_ecg.csv \
    --output output.json \
    --source vitalpatch \
    --narrative
```

See [`ecg_inference/INFERENCE_README.md`](ecg_inference/INFERENCE_README.md)
for every flag, every output field, and the full risk cascade.

**Full pipeline with live MedGemma narrative** (needs `ollama serve` running
the `medgemma:latest` model):

```bash
python -m ecg_pipeline.agent_bridge --file <path-to-raw-csv> --source vitalpatch
python -m ecg_pipeline.batch_vitalpatch_report   # full resumable batch
python -m ecg_pipeline.manifest_summary          # combined summary + showcase reports
```

**Training / evaluation** (real datasets required, see `Docs/DATASETS.md`;
governed by `Docs/AGENT_RULES.md` — read it first):

```bash
python -m training.ecg_pipeline_tools download-datasets --all
python -m training.ecg_pipeline_tools train-classifiers --dataset mitdb
python -m training.ecg_pipeline_tools eval-classifier --model ecg_pipeline/models/five_class_xgb.json --split-set ds2
```

---

## Honest performance (production model, DS2 held-out, 45,804 beats)

*Source: `Docs/archive/ABLATION_REPORT.md`, "Production baseline" section —
exact command shown there, never re-tuned against this split (see
`Docs/AGENT_RULES.md` rule 1). Accuracy (0.927) is not a success metric on
this data — see rule 4 and the note below the table.*

| Class | Sensitivity | Precision | F1 | Support | Trust |
|---|---|---|---|---|---|
| N (Normal) | 0.972 | 0.973 | **0.972** | 40,634 | High — reliable |
| S (Supraventricular) | 0.130 | 0.150 | **0.139** | 1,795 | Low — screening signal only |
| V (Ventricular) | 0.912 | 0.754 | **0.826** | 3,005 | Moderate-high — the clinically critical class, and the most reliable of the rare ones |
| F (Fusion) | 0.005 | 0.125 | **0.011** | 363 | Essentially unsolved — treat any F finding as noise |
| Q (Unknown) | 0.000 | 0.000 | 0.000 | 7 | N/A — near-zero support |

Macro-F1: 0.3895. N is ~89% of every real recording, so a model that only
ever predicts N would score >0.89 "accuracy" while being clinically
useless — never use accuracy as the success metric here.

**Why S and F are weak, and what was tried:** four genuinely different
fixes were attempted (extra timing features, extra training data from
LTAFDB/SDDB, a two-stage gate-then-classify architecture in 5 variants, and
a CNN+Transformer deep model in 2 data variants) — all four hit the same
wall: any meaningful S gain cost V more than the project's zero-tolerance
safety margin allows. This is now a closed research direction (see
`Docs/archive/ABLATION_REPORT.md`'s summary table and CNN+Transformer
closure section) — not something more tuning on the current data is
expected to fix. Production `five_class_xgb.json` was never changed by any
of these experiments.

---

## AFib rule validation (2026-07-28)

The rhythm-level AFib-suspected rule (RR coefficient-of-variation over
rolling 20-beat windows) had never been checked against real AFib ground
truth — flagged in `Docs/archive/RESEARCH_AUDIT.md` as "the single biggest
unvalidated clinical claim in the pipeline." Validated for the first time
against LTAFDB (84 records, 449,749 windows, ground truth from real
rhythm-change annotations):

| Threshold | Sensitivity | Specificity | F1 |
|---|---|---|---|
| 0.15 (old default) | 0.808 | 0.800 | 0.829 |
| **0.10 (current default)** | **0.971** | **0.716** | **0.893** |

0.10 is Youden's-J-optimal on this data and has the best F1 of any
threshold tested — for a "suspected" screening flag feeding a clinician
review step, the sensitivity gain (missing ~3% of true AFib windows
instead of ~19%) was judged worth the specificity cost. Changed in both
`ecg_pipeline_core.py` and `ecg_inference/classifier.py` (kept in sync).
Full method and sweep table: `Docs/archive/ABLATION_REPORT.md`, "AFib rule
validation" section.

---

## Known limitations

- **S-class recall is weak (13%)** and **F-class is essentially unsolved
  (F1 0.011)** — a documented, four-times-validated limitation of
  single-lead hardware and/or available labeled data, not a bug. See
  "Honest performance" above.
- **The AFib rule's `window=20` was not swept**, only its threshold — window
  size remains an open validation question.
- **NEWS2/qSOFA safety overrides exist in the risk cascade schema but are
  permanently unevaluated** in the ECG-only path — no vitals source is
  wired in here (that pairing lives in `MedGemma-Agent`, a separate
  system).
- **Confidence tiers in the JSON report are a heuristic, not a calibrated
  probability** — no conformal calibration set is loaded by default.
- **MedGemma's free-text narrative occasionally misstates a numeric
  comparison** (~30% of generations pre-filter); a post-hoc validator
  catches and strips this before saving, so the *saved* report is never
  wrong, but the underlying small (4B) model itself remains imperfect.
- **Two specific hospital recordings (MITDB 103/111) show real R-peak
  over-detection**, investigated and not fixed — see git commit `2420ecc`.
- **Edge/Jetson deployment has not been benchmarked** from this repo —
  distillation/quantization/Jetson work is a separate, ongoing scope.
- This is **decision support, not a diagnosis** — every report ends with an
  explicit clinician-review disclaimer.

---

## Standing rules

Before touching `ecg_pipeline_core.py`, `training/ecg_pipeline_tools.py`,
or anything under a `models/` directory, read
[`Docs/AGENT_RULES.md`](Docs/AGENT_RULES.md) — every rule in it exists
because it was violated once at real cost.
