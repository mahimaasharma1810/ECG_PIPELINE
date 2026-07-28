# Inference-Ready Packaging — Report

**Scope:** package the ECG pipeline's runtime-prediction path into a standalone,
training-free `ecg_inference/` package plus a single `run_inference.py` entry
point. **Strictly a packaging/refactoring task** — no classifier weights,
thresholds, feature extraction, or AAMI mapping were changed. See
"Freeze verification" below for how that's proven, not just asserted.

**Addendum (2026-07-28, later same day):** the freeze above describes this
packaging pass itself. In a *separate*, explicitly user-authorized
follow-up ("rule/threshold tuning"), the AFib-suspected rule's
`cv_threshold` was changed 0.15 -> 0.10 after being validated for the first
time against real LTAFDB ground truth — see `Docs/archive/ABLATION_REPORT.md`'s
"AFib rule validation" section for the full method and results, and §2's
freeze-verification note below for what that means for this document's
"byte-for-byte identical" claim. No classifier weights, feature extraction,
or AAMI mapping were touched by that change either — it's a deterministic
rhythm-rule constant, not the frozen XGBoost model.

---

## 1. What was built

```
ecg_inference/
    __init__.py       # re-exports the public API (ECGPipeline, load_classifier, ...)
    preprocess.py      # Stages 1-4: parse device streams, SQI gate, resample, filter chain
    detector.py         # Stage 5: XQRS R-peak detection + beat segmentation
    features.py         # Stage 6: 56-dim handcrafted per-beat features + HRV
    classifier.py       # Stages 7-8: beat/rhythm classification + risk scoring
    report.py           # Stage 9: structured RiskReport JSON + deterministic narrative
    pipeline.py          # Orchestrator: ECGPipeline wires stages 1-9 together
    models/
        five_class_xgb.json           # production XGBoost classifier weights
        five_class_xgb.classes.json   # its label encoder classes

run_inference.py       # CLI entry point (repo root)
requirements_inference.txt
```

Every line of pipeline logic in these files is a **verbatim extraction** from
the existing frozen pipeline (`ecg_pipeline/ecg_pipeline_core.py`'s internal
stage sections, plus `ecg_pipeline/agent_bridge.py`'s report-builder
functions) — copied via exact line-range slicing, not retyped, to rule out
transcription drift. The only content changes are two removals of code that
was already **dead in every production path**, documented in each affected
file's module docstring:

1. `ECGPipeline.__init__`'s optional `encoder` parameter, and the `if
   self.encoder is not None` branch in `pipeline.py`'s Stage 6 — every real
   `ECGPipeline(...)` construction across the codebase (`agent_bridge.py`,
   `demo_stream.py`, `report_ui.py`, `synthetic_ecg.py`) always omits
   `encoder`, so `self.encoder` was always `None` and this branch never ran.
   Removing it drops the `torch` dependency entirely.
2. `BinaryGateCNN` (an unused two-pass-triage CNN, `classify.py` section) —
   never instantiated anywhere. Excluded from `classifier.py`.

No other code was rewritten, reordered, or renamed **during this packaging
pass**. A later, separate, user-authorized change (2026-07-28, see
addendum above) updated `_afib_suspected`'s `cv_threshold` 0.15 -> 0.10 in
both `ecg_pipeline_core.py` and `ecg_inference/classifier.py` identically
— the two copies remain in sync with each other, but neither matches this
document's original byte-for-byte comparison against the pre-2026-07-28
pipeline anymore for recordings where that rule fires differently at the
two thresholds. Re-verified: the two live copies still produce identical
JSON output to each other post-change (only `generated_at` differs) — see
`Docs/archive/ABLATION_REPORT.md` for why the value changed.

---

## 2. Freeze verification

Requirement: "Do not modify classifier weights / retrain / change thresholds
/ change feature extraction / modify the AAMI mapping."

- **Weights**: `ecg_inference/models/five_class_xgb.json` is a byte-copy of
  `ecg_pipeline/models/five_class_xgb.json` (same file, `cp`, not
  regenerated).
- **Thresholds, features, AAMI mapping**: `config.py`'s `AAMI_CLASSES`,
  `RiskThresholds`, `SQIThresholds`, `BeatWindowConfig`, `ConformalConfig`,
  `TemporalTrackingConfig` and every feature-extraction function in
  `features.py` were extracted verbatim (see §1) — none were edited.
- **Proof, not just design**: ran the *old* pipeline
  (`ecg_pipeline_core.ECGPipeline` + `agent_bridge.build_risk_report_json`)
  and the *new* `ecg_inference` pipeline on the same three real recordings
  (two VitalPatch CSVs, one MITDB WFDB record — the WFDB case specifically
  exercises the `source == "wfdb"` skip-Kalman detection branch) and diffed
  the resulting JSON reports field-by-field. **Byte-for-byte identical**
  except the `generated_at` wall-clock timestamp (a live `datetime.now()`
  call in the report builder, not a pipeline output). See §5 for the exact
  commands.

---

## 3. Startup instructions

```bash
pip install -r requirements_inference.txt

python run_inference.py \
    --input  data/raw/vitalpatch/Patch_1844AC/1778423422284_VC2B008BF_1844AC_ecg.csv \
    --output output.json \
    --source vitalpatch \
    --narrative
```

Flags:
- `--input` (required): raw ECG file.
- `--output` (required): where to write the structured JSON report.
- `--source {vitalpatch,sensio,wfdb}` (default `vitalpatch`): raw device
  format of `--input`. VitalPatch CSVs can split into multiple gap-separated
  segments; use `--segment-index` (default `0`) to pick a different one.
- `--classifier` (default `ecg_inference/models/five_class_xgb.json`):
  override the pretrained model path.
- `--narrative`: also compute the deterministic (network-independent)
  narrative and include it in the output JSON under `"narrative"`. This
  calls `ecg_inference.report._deterministic_narrative`, **not** the live
  MedGemma path — no network call.

**Known network-dependent side effect (pre-existing, not added by this
packaging pass):** `ECGPipeline.run()`'s internal Stage 9 always calls
`generate_report(...)`, which — unless the deterministic `risk_level` is
already `CRITICAL` — attempts one HTTP POST to `http://localhost:11434/api/generate`
(Ollama/MedGemma) with a 10s timeout. This is frozen behavior from the
original pipeline; it fails gracefully (returns `None`, logged to the audit
trail) if Ollama isn't reachable, and **does not affect** the JSON report
`run_inference.py` writes (`build_risk_report_json` never reads that
result). No code in this path was changed to add or remove this call.

---

## 4. Removed / isolated training components

Everything below was `git mv`'d out of the inference-adjacent tree into a
new top-level `training/` directory (history preserved, nothing deleted).
`ecg_inference/` imports none of it — confirmed by grep (see §5).

| Moved from | Moved to | Why |
|---|---|---|
| `ecg_pipeline/ecg_pipeline_tools.py` | `training/ecg_pipeline_tools.py` | All training/download/eval/ablation code: `download_datasets`, `download_icentia11k`, `train_encoder`, `train_classifiers`, `evaluate`/`eval_classifier`, `train_twostage`, `analyze_twostage`. |
| `_original_stages/` (repo root) | `training/_original_stages/` | Pre-consolidation original per-stage files, including `train_classifiers.py`, `train_encoder.py`, `eval_classifier.py`, `download_datasets.py`, `download_icentia11k_full.py`, `splits.py`. Kept for reference/diffing only — was never imported by any live code before or after this move. |
| `ecg_pipeline/models/ecg_encoder.pt`, `ecg_encoder.meta.json` | `training/model_artifacts/` | Self-supervised encoder training artifacts. Encoder is dead code in production (§1). |
| `ecg_pipeline/models/five_class_cnntx_v1.pt`, `five_class_cnntx_v2_nosddb.pt` | `training/model_artifacts/` | CNN-transformer experiments — never loaded by `ECGPipeline`. |
| `ecg_pipeline/models/five_class_xgb_twostage_{v1..v5,qrsfeat,qrsfix,beatonly}_{stage1,stage2,threshold}.json(.classes.json)` (36 files) | `training/model_artifacts/` | Two-stage experimental classifier variants — the production path only ever loads plain `five_class_xgb.json`. |
| `ecg_pipeline/models/five_class_xgb_newsplit_flat.json`, `.classes.json` | `training/model_artifacts/` | An alternate-split ablation model, never referenced by any default arg or call site. |

`ecg_pipeline/models/` now contains **only** the two production model files
(`five_class_xgb.json`, `five_class_xgb.classes.json`).

One reference had to be fixed to keep the still-active `ecg_pipeline/`
package working after the move: `agent_bridge.py`'s
`_verify_against_annotations` (a debug-only ground-truth cross-check, not
part of the report) imported `AAMI_SYMBOL_MAP` from `ecg_pipeline_tools` —
updated to import from its new location, `training.ecg_pipeline_tools`. No
other code changed as a result of the move (confirmed by grep, §5).

**Not moved / no notebooks found:** a repo-wide search (`find -iname
"*.ipynb"`) found no notebooks outside `MedGemma-Agent/` (a separate git
submodule, out of scope). Markdown docs under `Docs/` (including
`ABLATION_REPORT.md`, `CNN_TRANSFORMER_EXPERIMENT.md`) were left in place —
pure documentation, not executable, zero import risk. `ecg_pipeline/`'s
demo/UI files (`demo_stream.py`, `report_ui.py`, `synthetic_ecg.py`,
`test_pipeline_synthetic.py`) were also left in place: they were already
explicitly out of scope per `EDGE_DEPLOYMENT_FIX_REPORT.md` ("Out of
scope: demo UI"), and none of them are training/ablation/eval code — they
exercise the real frozen pipeline, they don't train or evaluate a model.

---

## 5. Final verification checklist

| Check | Result |
|---|---|
| Model loads successfully | ✅ `load_classifier(Path("ecg_inference/models/five_class_xgb.json"))` → `is_trained=True` |
| `ecg_inference` imports with no missing imports | ✅ `python3 -c "import ecg_inference"` succeeds |
| `ecg_inference` never imports training code | ✅ `grep -rn "ecg_pipeline_tools\|_original_stages" ecg_inference/*.py` — only appears inside comments/docstrings explaining what was *excluded*, never in an `import` statement |
| `torch` not a runtime dependency | ✅ confirmed via `sys.modules` diff before/after `import ecg_inference` — no `torch*` module loaded |
| One CSV processed start to finish, no runtime errors | ✅ `python run_inference.py --input <vitalpatch csv> --output out.json --narrative` completes and writes valid JSON |
| JSON output matches current pipeline | ✅ `build_risk_report_json` output diffed field-by-field against the old `ecg_pipeline_core` + `agent_bridge` pipeline on 3 real recordings (2 VitalPatch, 1 WFDB/MITDB) — byte-for-byte identical except the live `generated_at` timestamp |
| Existing `ecg_pipeline/` package still works after isolating training code | ✅ `import ecg_pipeline.ecg_pipeline_core`, `import ecg_pipeline.agent_bridge`, and `training.ecg_pipeline_tools.AAMI_SYMBOL_MAP` (the one cross-reference) all still import cleanly post-move |
| No classifier weights / thresholds / features / AAMI mapping changed | ✅ verbatim-extraction methodology (§1) + byte-for-byte output match (§2) |

### Commands used

```bash
# import + model load
python3 -c "import ecg_inference; from pathlib import Path; \
  c = ecg_inference.load_classifier(Path('ecg_inference/models/five_class_xgb.json')); \
  print(c.is_trained)"

# end to end
python run_inference.py --input data/raw/vitalpatch/Patch_1844AC/1778423422284_VC2B008BF_1844AC_ecg.csv \
  --output output.json --narrative

# no training imports
grep -rn "ecg_pipeline_tools\|_original_stages" ecg_inference/*.py run_inference.py
```
