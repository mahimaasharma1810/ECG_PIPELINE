# `ecg_inference` — Reference

Deployment-only package for turning one raw ECG recording into a
structured risk report. No training code, no `torch`. Every number this
package produces comes from the same frozen logic as
`ecg_pipeline/ecg_pipeline_core.py` — see `../docs/inference_ready.md` for
the byte-for-byte-identical proof. For the project-level story, see the
root `README.md`.

---

## Install

```bash
pip install -r ../requirements_inference.txt
```

Dependencies: `numpy`, `pandas`, `scipy`, `xgboost`, `scikit-learn`,
`PyWavelets`, `requests`, `wfdb`. Deliberately excludes `torch` — the only
code that needed it (the self-supervised encoder) is dead in every
production path (see `pipeline.py`'s `ECGPipeline.__init__` docstring).

---

## Running it

```bash
python run_inference.py \
    --input  path/to/ecg.csv \
    --output output.json \
    --source vitalpatch \
    --narrative
```

(run from the repo root, one directory up from this file)

### CLI flags

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--input` | yes | — | Raw ECG file: a VitalPatch/SeNSiO CSV, or a WFDB record path (no extension, e.g. `data/raw/public/mitdb/100`) |
| `--output` | yes | — | Where to write the structured JSON report |
| `--source` | no | `vitalpatch` | One of `vitalpatch`, `sensio`, `wfdb` — which parser to use for `--input` |
| `--classifier` | no | `models/production/five_class_xgb.json` | Path to the pretrained classifier weights |
| `--segment-index` | no | `0` | A VitalPatch file can split into multiple gap-separated segments (see `Recording` in `preprocess.py`); which one to report on |
| `--narrative` | no | off | Also compute the deterministic (network-independent) narrative and attach it as `narrative` / `narrative_source` in the output JSON. Does **not** call MedGemma. |

**Known side effect, not something the CLI adds:** `ECGPipeline.run()`
always attempts one internal Stage 9 call to a local MedGemma/Ollama
server (`http://localhost:11434`), regardless of `--narrative`, unless the
deterministic risk level is already `CRITICAL`. It fails gracefully if
unreachable (10s timeout, logged, never crashes the run) and its result is
**not** used anywhere in the output JSON — `build_risk_report_json` never
reads it. The output JSON's `pipeline_stage9_llm_status` field tells you
what happened there (see below), independent of whether `--narrative` was
passed.

---

## Every file in the package

| File | Stage(s) | What it contains |
|---|---|---|
| `__init__.py` | — | Re-exports the public API: `ECGPipeline`, `PipelineResult`, `load_classifier`, `build_risk_report_json`, `render_narrative`, `save_report`, `Recording`, parser functions, `Beat`, `FiveClassBeatClassifier`, `RiskReport` |
| `preprocess.py` | 1-4 | Device parsers (`parse_vitalpatch_ecg`, `parse_sensio_ecg`, `parse_wfdb_record`), the SQI gate (`run_sqi_gate`), resampling (`to_target_rate`), the 5-step filter chain (`apply_filter_chain`), and every config constant/threshold dataclass (`RiskThresholds`, `SQIThresholds`, `BeatWindowConfig`, `ConformalConfig`, `TemporalTrackingConfig`) |
| `detector.py` | 5 | XQRS R-peak detection + beat segmentation (`detect_and_segment`), the `Beat` dataclass |
| `features.py` | 6 | The 56-dim handcrafted per-beat feature vector (5 morphological + 51 wavelet) and recording-level HRV (`recording_level_hrv`) |
| `classifier.py` | 7-8 | `FiveClassBeatClassifier` (`.load()`/`.predict_one()` only — `.fit()` exists on the class but is never called by this package), `RhythmContextEngine` (bigeminy/trigeminy/VT-run/AFib-suspected rules), `score_recording` (the risk cascade) |
| `report.py` | 9 | `SimilarCaseIndex`, `load_classifier`, `build_risk_report_json` (the JSON report builder), `_deterministic_narrative` / `render_narrative` (narrative layer), `save_report` |
| `pipeline.py` | orchestrator | `ECGPipeline` — wires stages 1-9 together, returns a `PipelineResult` |
| `models/five_class_xgb.json` | — | Production XGBoost classifier weights — frozen, never overwritten by this package |
| `models/five_class_xgb.classes.json` | — | The classifier's label-encoder classes |

---

## The risk cascade — exact thresholds, evaluated in order

The **first** rule that fires sets `risk_level`; if none fire, `LOW`.
Source: `preprocess.py`'s `RiskThresholds`, applied in `classifier.py`'s
`score_recording`.

| # | Condition | Threshold | Sets |
|---|---|---|---|
| 1 | PVC burden (% of analyzed beats classified V) | `> 20.0%` | CRITICAL |
| 2 | VT run (≥N consecutive V beats) | `>= 3` consecutive | CRITICAL |
| 3 | PVC burden | `> 10.0%` | HIGH |
| 4 | PAC burden (% classified S) `>` threshold, OR AFib burden `>` threshold | PAC `> 15.0%` / AFib `> 30.0%` | HIGH |
| 5 | Sustained HRV suppression | SDNN `< 20 ms` | MEDIUM |
| 6 | NEWS2 safety override | `>= 7` (never evaluated in this ECG-only path — no vitals source wired in) | CRITICAL |
| 7 | qSOFA safety override | `>= 2` (never evaluated, same reason) | HIGH |
| floor | Too few beats survive quality gating | `< 5` beats analyzed | NOT_ASSESSABLE |

**AFib burden** comes from `RhythmContextEngine`'s AFib-suspected rule:
non-overlapping 20-beat windows of RR intervals, flagged if
`std(RR)/mean(RR) > 0.10` (validated against LTAFDB 2026-07-28 — see root
`README.md`'s "AFib rule validation" section; changed from an unvalidated
default of 0.15).

---

## Every output JSON field

```jsonc
{
  "generated_at": "...",          // ISO timestamp, the ONLY field that differs run-to-run
                                   // for identical input (live wall-clock, not a pipeline output)
  "recording": {
    "source": "vitalpatch|sensio|wfdb",
    "patient_id": "...", "segment_id": "...",
    "duration_s": 118.26,
    "n_raw_samples": 14650,
    "fs_nominal_hz": 125.0, "fs_processed_hz": 125.0,
    "quality_score": 0.9167,           // 1 - sqi_window_rejection_rate
    "sqi_window_rejection_rate": 0.0833,
    "n_beats_detected": 305, "n_beats_analyzed": 236, "n_beats_flagged_low_quality": 69
  },
  "beat_summary": {                 // one entry per AAMI class actually present
    "N": {"count": 168, "pct_of_analyzed_beats": 55.08,
          "confidence": "HIGH", "note": "..."},
    // ...S, V, F, Q — confidence tier + note are static, keyed off each
    // class's real DS2 F1 from the "Honest performance" table in the root README
  },
  "rhythm_findings": [              // sequence-level findings, one per detected episode
    {"kind": "BIGEMINY|TRIGEMINY|VT_RUN|AFIB_SUSPECTED",
     "start_beat_idx": 4, "end_beat_idx": 11,
     "start_time_s": 3.072, "duration_s": 2.464,
     "evidence": {...}, "evidence_text": "human-readable restatement"}
  ],
  "assessable": true,               // false <=> risk_level == "NOT_ASSESSABLE"
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL|NOT_ASSESSABLE",
  "rule_trace": [                   // EVERY cascade rule, not just the one that fired
    {"condition": "...", "measured_value": ..., "threshold": ...,
     "fired": true, "would_set_level": "...",
     "depends_on_low_confidence_class": false, "is_deciding_rule": true}
    // exactly one entry has is_deciding_rule=true
  ],
  "deciding_rule": {...},           // the one rule_trace entry that actually set risk_level
  "confidence": {
    "tier": "HIGH|MODERATE-HIGH|LOW|...",
    "statement": "...",             // ties confidence to the deciding rule's dependent class's real F1
    "caveat": "heuristic, not a calibrated probability"   // always present, always true
  },
  "safety_overrides": [             // NEWS2 + qSOFA, always "applied": false in this ECG-only path
    {"override": "NEWS2", "applied": false, "note": "..."},
    {"override": "qSOFA", "applied": false, "note": "..."}
  ],
  "known_limitations": ["...", "..."],   // static, class-F1-driven caveats — always present
  "pipeline_stage9_llm_status": {   // added by run_inference.py, not build_risk_report_json itself
    "bypassed_llm": true,              // did ECGPipeline.run()'s OWN internal MedGemma call fire?
    "llm_rejected_reason": null        // why not, if it didn't
  },
  "narrative": "...",               // only present if --narrative was passed
  "narrative_source": "deterministic_template (network-independent, not MedGemma)"
                                     // only present if --narrative was passed; makes explicit
                                     // that this text did NOT come from the live LLM call above
}
```

---

## What to trust vs. not trust in this output

| Trust it | Treat as a screening signal, not a finding | Don't trust it as calibrated |
|---|---|---|
| `N` and `V` beat counts (F1 0.966 / 0.830) | `S` beat counts (F1 0.152 — 65.1% of true S beats get called N) | `confidence.tier` / `confidence.statement` — a heuristic tied to which class the deciding rule depends on, **not** a statistically calibrated probability (no conformal set loaded by default) |
| `BIGEMINY` / `TRIGEMINY` / `VT_RUN` findings (pure RR/label pattern rules, not classifier-dependent beyond N/V which are reliable) | `F` beat counts (F1 0.005 — essentially unsolved, treat any F as noise) | `safety_overrides` — schema exists but is structurally inert; no vitals source is wired into this ECG-only path |
| `AFIB_SUSPECTED` findings (validated 2026-07-28: sensitivity 0.971, F1 0.893 against real LTAFDB ground truth at the current threshold) | `pipeline_stage9_llm_status` when `bypassed_llm=false` — a live LLM call, occasionally misstates a numeric comparison in free text (caught/stripped before saving in the full `agent_bridge.py` path, but this package's own `--narrative` output is the pure deterministic template, not LLM text at all) | Any single segment in isolation for a NOT_ASSESSABLE-adjacent recording — the `< 5 beats analyzed` floor exists specifically because too little surviving signal should never produce a confident-sounding number |

`risk_level`/`rule_trace`/`deciding_rule` themselves are deterministic
Python comparisons against fixed thresholds (the cascade table above) —
trustworthy as "the rules fired exactly as coded," not as a clinically
validated cutoff choice beyond what's documented (PVC/PAC/VT thresholds
are clinical rule-of-thumb values, not statistically calibrated to this
model or device; the AFib threshold is the one exception that has been
validated against real data, see above).

This is decision support, not a diagnosis.
