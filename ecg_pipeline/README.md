# ECG Pipeline — Code Guide

This README documents what's actually inside the two runnable files in
this directory: **`ecg_pipeline_core.py`** (the 9-stage ECG processing
pipeline) and **`ecg_pipeline_tools.py`** (everything needed to acquire
data and train the models the pipeline uses). Each was assembled by
merging several smaller modules into one file; this doc walks through
them section by section so you don't have to open every function to
find your way around. For the project-level story (why this exists,
what changed vs. the old baseline, known limitations) see
`Docs/README.md`.

Both files are plain Python modules inside the `ecg_pipeline` package —
run them with `python -m ecg_pipeline.<module_name> ...` from the
`projects/` directory (see "Running it" at the bottom of each section).

---

## `ecg_pipeline_core.py` — the runtime pipeline

Everything needed to take one raw ECG recording and produce a clinical
risk report lives here, in 15 sections stitched into one file (in this
order). Every section corresponds to one of the original per-stage
files (`config.py`, `audit.py`, `ingest.py`, ... — still preserved
verbatim under `../_original_stages/` if you want the un-merged
version).

### 1. `config` — shared constants and thresholds

Nothing but dataclasses of tunable numbers, each documented with *why*
that number was chosen:

- `TARGET_FS = 125.0` — the common sample rate everything gets resampled to.
- `AAMI_CLASSES = ["N","S","V","F","Q"]` — the 5 beat classes the classifier predicts.
- `RISK_LEVELS = ["LOW","MEDIUM","HIGH","CRITICAL"]` — the 4 alert levels.
- `SQIThresholds` — flatline/clipping/missing-sample/kurtosis/baseline-wander/SNR cutoffs for the stage-2 quality gate.
- `FilterChainConfig` — cutoff frequencies for the stage-4 filter chain.
- `BeatWindowConfig` — beat window sizes (200ms pre/400ms post primary, 500/500 wide) and RR sanity range (300–2000ms).
- `RiskThresholds` — PVC/PAC burden percentages and VT-run length that trigger HIGH/CRITICAL.
- `ConformalConfig` — alpha (0.1 → 90% coverage) and minimum calibration set size for conformal prediction.
- `TemporalTrackingConfig` — rolling window length and slope threshold for "is this patient trending worse".

Singletons (`SQI`, `FILTER`, `BEATS`, `RISK`, `CONFORMAL`, `TEMPORAL`) are
instantiated once here and passed around as default arguments everywhere
else in the file — one place to retune a threshold.

### 2. `audit` — SHA-256 hash-chained log

`AuditLog.append(event_type, payload)` writes one entry whose hash
depends on the previous entry's hash, so `verify_chain()` can prove
nothing in the log was edited after the fact. Every stage below calls
`audit.append(...)` at least once; `ECGPipeline.run()` returns the whole
log with the result so a full record of *why* a decision was made is
always available, not just the final number.

### 3. `ingest` (Stage 1) — parse raw device files into a `Recording`

- `Recording` — the dataclass every downstream stage consumes: raw
  signal + timestamps + sample rate + patient/segment IDs + any flagged
  gaps.
- `parse_vitalpatch_ecg(csv_path)` — VitalPatch's alternating
  timestamp/value CSV format; splits a file into multiple `Recording`s
  wherever a gap exceeds 200ms (so filtering never bridges a real
  dropout).
- `parse_sensio_ecg(csv_path)` — SeNSiO's metadata-header CSV format;
  detects whether the device already bandpass-filtered the signal
  (`ECG_Filtered` column present) so stage 4 knows to skip its first 4
  steps.
- `parse_wfdb_record(path)` — generic loader for public WFDB datasets
  (MITDB, Icentia11k, ...), used by the training tools, not live
  inference.
- `discover_vitalpatch_files` / `discover_sensio_files` — glob helpers
  used by both the pipeline CLI and the training tools.

### 4. `quality` (Stage 2) — Signal Quality Index (SQI) gate

`run_sqi_gate(signal, timestamps_ms, fs, clip_value)` splits the signal
into 5-second windows and calls `evaluate_window()` on each, which
checks (in this order) flatline, clipping, missing-samples, morphology
kurtosis, baseline wander, and SNR — the **first** failing check sets
`reject_code`, and a window is dropped only if it fails one of these.
Deliberately *not* checked here: RR-interval regularity — an irregular
but clean AFib strip must survive this gate (that's the point of
recommendation #6, see `Docs/README.md`). Rejected windows are logged,
not silently dropped (`WindowVerdict.reject_code`), and RR outliers get
picked up downstream by the rhythm classifier instead.

### 5. `resample` (Stage 3) — uniform 125 Hz

- `resample_linear` — for wearable data with irregular timestamps
  (VitalPatch/SeNSiO); linear, not cubic, to avoid ringing near the
  sharp QRS peak.
- `resample_decimate` — FIR anti-aliased decimation for clean
  integer-ratio sources (e.g. Icentia11k's 250 Hz → 125 Hz).
- `to_target_rate(...)` — dispatches between the two based on whether
  the input rate is an exact multiple of `TARGET_FS`.

### 6. `filters` (Stage 4) — 5-step filter chain

Applied in order by `apply_filter_chain(x, fs, already_bandpass_filtered)`:

1. `remove_baseline_median` — double median filter, removes drift without distorting the QRS.
2. `highpass_residual` — 2nd-order Butterworth high-pass, catches drift the median filter misses.
3. `powerline_notch` — IIR notch at 50Hz mains frequency.
4. `bandpass` — 0.5–40Hz Butterworth bandpass (steps 1–4 are skipped entirely if the device already pre-filtered).
5. `emg_suppress_kalman` — adaptive-gain scalar Kalman filter, suppresses muscle noise between beats while still letting a real QRS transient through.
6. `robust_zscore` (used later, per-beat, not part of the chain itself) — median/MAD normalization, resistant to motion-artefact outliers.

`_safe_filtfilt` falls back from zero-phase `filtfilt` to causal
`lfilter` when a segment is too short for `filtfilt`'s padding
requirement, rather than raising.

### 7. `beats` (Stage 5) — R-peak detection and segmentation

- `detect_r_peaks` — WFDB's XQRS adaptive-threshold detector (chosen over
  Pan-Tompkins because wearable electrodes at non-standard chest
  positions produce atypical QRS morphology XQRS handles better).
- `segment_beats(signal, fs, r_peaks)` — for every R-peak, extracts a
  **primary window** (200ms pre + 400ms post — 75 samples, used for
  features/classification) and a **wide window** (500/500ms — 125
  samples, used for the encoder), computes RR-pre/RR-post, and flags
  (never drops) RR intervals outside 300–2000ms.
- `_beat_level_sqi` — a second, beat-level quality check independent of
  the stage-2 window gate: rejects a beat if its amplitude is too low
  relative to the local noise floor, if there's excess baseline drift
  across the window, or if the detected R-peak isn't actually the local
  maximum (catches detector jitter).
- `Beat` — the dataclass carrying all of this per beat: `quality_rejected`
  beats get label `"Q"` downstream instead of being deleted from the list.

### 8. `features` (Stage 6a) — 56-dim handcrafted feature vector

- `beat_feature_vector(beat, primary_pre_samples)` — 5 morphological
  features (RR-pre, local HRV, left/right area ratio, above/below
  amplitude ratio, amplitude range) + 51 wavelet coefficients
  (`_wavelet_features`, db4 decomposition to a fixed length) = 56 dims,
  normalized with `robust_zscore` first.
- `recording_level_hrv(beats)` — SDNN, RMSSD, pNN50, LF/HF ratio
  (`_lf_hf_ratio`, via Lomb-Scargle since RR intervals are unevenly
  spaced in time), and QRS-width trend — feeds the stage-8 risk scorer,
  not the beat classifier.
- A documented, *reverted* experiment lives in the module docstring: a
  7-feature rhythm-context extension improved one validation split but
  regressed the real held-out test set — kept as a note so it isn't
  silently retried.

### 9. `encoder` (Stage 6b) — self-supervised ECG embedding

- `ECGEncoder` — a ~30K-parameter Conv1D encoder (125-sample wide window
  → 32-dim embedding), small enough for edge deployment.
- `pretrain_self_supervised(windows)` — masked-reconstruction training
  (`random_mask` zeroes random spans, `ReconstructionDecoder` tries to
  reconstruct them): needs **zero labels**, so it can run today on
  Cliniaura's own unlabeled VitalPatch/SeNSiO recordings, before any
  public labeled dataset is downloaded.
- `save_encoder` / `load_encoder` / `embed_windows` — persistence and
  batch inference. The trained weights ship at `models/ecg_encoder.pt`.
- This embedding feeds both `classify` (once a classifier head exists)
  and `similar_cases` (nearest-neighbour retrieval).

### 10. `classify` (Stage 7) — beat + rhythm classification

- `FiveClassBeatClassifier` — a trainable XGBoost 5-class (N/S/V/F/Q)
  classifier. `.fit()` trains it; until then, `predict_one()`
  transparently falls back to `RuleBasedBeatClassifier` (RR-prematurity
  + QRS-width heuristics) and **every prediction is tagged** with
  `source: "trained_model"` or `"rule_based_fallback"` so nothing
  downstream mistakes a heuristic guess for a trained model's opinion.
- `BinaryGateCNN` — a fast normal-vs-arrhythmia triage network (defined,
  wired for future use — untrained until fit on labeled data).
- `RhythmContextEngine.analyze(labels, rr_ms)` — looks at the *sequence*
  of beat labels a single-beat classifier can't reason about: VT runs
  (`_vt_runs`, ≥3 consecutive V beats), bigeminy/trigeminy (`_geminy`),
  and AFib suspicion (`_afib_suspected`, from RR coefficient-of-variation
  over a rolling window — a rhythm finding, never a quality rejection).

### 11. `risk` (Stage 8) — risk scoring, conformal prediction, temporal trend

- `score_recording(labels, findings, hrv, news2, qsofa)` — deterministic
  rules: PVC/PAC burden %, VT run count, AFib burden, HRV suppression,
  optionally escalated by NEWS2/qSOFA clinical scores, producing one of
  the 4 `RISK_LEVELS` plus human-readable reasons. Every reason string
  names the actual threshold that was crossed, not just the metric — e.g.
  `"PVC burden 24.0% > critical threshold (20.0%)"` — pulled straight from
  `RiskThresholds` (all 8 cutoffs — PVC/PAC/AFib burden, VT run length,
  HRV SDNN floor, NEWS2/qSOFA escalation points — now live there, so the
  displayed number and the comparison it's based on can never drift
  apart).
- `to_agent_ecg_risk_summary(risk_report)` — converts a `RiskReport` into
  the plain dict `MedGemma-Agent`'s `ECGRiskSummary` schema expects
  (`vitals/schemas.py` in that repo), mapping this module's `"LOW"` tier
  to that agent's `"NORMAL"`. This is the wire contract for the additive
  ECG-into-vitals-agent integration — the two projects share this JSON
  shape only, not a Python import, so `ecg_pipeline` has no dependency on
  `MedGemma-Agent` or vice versa. See that repo's README ("ECG risk
  integration") for how it's consumed: additively, escalation-only, same
  as the LLM merge rule in `report.merge_decision()` above.
- `ConformalRiskPredictor` — split-conformal prediction: `calibrate()`
  once a labeled calibration set exists, then `predict_set()` returns
  the *set* of risk levels consistent with 90% coverage instead of one
  point estimate. Returns the full 4-level set (maximally conservative)
  until calibrated — never a false guarantee.
- `TemporalRiskTracker` — per-patient rolling history (`record()`) and
  linear-trend slope (`trend()`) over the last 15 minutes, flagging slow
  deterioration that no single snapshot would trigger alone.

### 12. `similar_cases` (Stage 6 support) — nearest-neighbour retrieval

`SimilarCaseIndex` wraps scikit-learn's `NearestNeighbors` over encoder
embeddings (cosine distance) plus each case's outcome label. `query()`
returns the k nearest historical cases, given to the LLM as grounding
context in stage 9 instead of it reasoning from the current recording
alone.

### 13. `report` (Stage 9) — MedGemma clinical report

- `PROMPT_TEMPLATE` — forces step-by-step reasoning *before* the final
  JSON verdict (so the reasoning trace is available for later
  distillation, and so clinicians can see *why*).
- `call_medgemma(prompt)` — hits a local Ollama server; returns `None`
  (not an exception) if unreachable, so the system degrades gracefully.
- `merge_decision(risk_report, llm_output, audit)` — the safety layer:
  **CRITICAL alerts bypass the LLM entirely**; the LLM may only *raise*
  the deterministic risk level, never lower it; if it disagrees by more
  than one severity level its output is rejected outright. Every branch
  (bypassed / unavailable / rejected / accepted) is written to the audit
  log.
- `MergedDecision` — keeps `deterministic_decision`, `llm_decision`, and
  `final_decision` as three separate fields, so it's always visible how
  much the LLM actually changed vs. the fixed rules (recommendation #10).

### 14. `pipeline` — `ECGPipeline`, the orchestrator

`ECGPipeline.run(recording, news2_score, qsofa_score)` wires stages 2–9
together in order, writing an audit entry per stage, and returns a
`PipelineResult` with everything: kept-sample counts, per-beat labels,
rhythm findings, the risk report, the temporal trend, and the merged
decision. If too little signal survives stages 2–3 (e.g. everything got
SQI-rejected), `_insufficient_data_result()` returns a plain LOW-risk
result instead of crashing or fabricating a score from no data.

### 15. CLI (`main()`) — run it end to end

```bash
python -m ecg_pipeline.ecg_pipeline_core --source vitalpatch --limit 3
python -m ecg_pipeline.ecg_pipeline_core --source sensio --limit 3
```

Flags: `--encoder <path>` / `--classifier <path>` (defaults to
`models/ecg_encoder.pt` / `models/five_class_xgb.json`), `--no-classifier`
to force the rule-based fallback. `summarize()` formats one
`PipelineResult` as the JSON block you see printed per recording.

---

## `ecg_pipeline_tools.py` — data acquisition & training

Everything needed to go from "no labeled data" to "a trained classifier
and calibrated risk predictor" lives here, as 6 sections behind one
subcommand dispatcher. It imports the runtime building blocks it needs
(`apply_filter_chain`, `run_sqi_gate`, `FiveClassBeatClassifier`, etc.)
from `ecg_pipeline_core.py` rather than duplicating them.

### 1. `splits` — the single source of truth for every record-ID list

`MITDB_DS1` / `MITDB_DS2` (the standard AAMI train/test split),
`SVDB_RECORDS` / `INCART_RECORDS` / `LTAFDB_RECORDS` / `SDDB_RECORDS`
(train-only enrichment sets — `MITDB_DS2` always stays the only held-out
test set), and `DS1_TRAIN` / `DS1_VAL` — a **non-random** patient-level
carve-out of DS1 for honest validation (record 208 alone holds 373 of
DS1's ~415 F-class beats, so it's deliberately kept in training rather
than risking it landing in a random validation split). Asserts at import
time guarantee the three splits never silently overlap.

### 2. `download-datasets` — PhysioNet Open Access downloads

```bash
python -m ecg_pipeline.ecg_pipeline_tools download-datasets --all
python -m ecg_pipeline.ecg_pipeline_tools download-datasets --only mitdb svdb
```

`download_wfdb_database` pulls MITDB/SVDB/INCART/CUDB in full (all
small, all directly usable). `download_cinc2017` pulls the ~95MB
single-lead AFib challenge set. `download_icentia11k` pulls a **capped**
40-patient × 3-segment subset (~250MB) rather than the full 188GB
dataset — spread across the `p00`–`p09` groups so the subset isn't
biased toward one group.

### 3. `download-icentia11k-full` — the full 188GB Icentia11k via S3

```bash
python -m ecg_pipeline.ecg_pipeline_tools download-icentia11k-full --workers 64
```

Two resumable phases: `list_all_objects()` pages through S3's
`ListObjectsV2` XML API and caches the manifest to disk; `download_all()`
fetches everything with a thread pool (`_download_one` skips any file
that already exists locally at the correct size, so a re-run after an
interruption doesn't re-download anything). No AWS credentials needed —
the bucket is public over plain HTTPS, and this is meaningfully faster
than PhysioNet's own web server for this particular dataset.

### 4. `train-encoder` — self-supervised pretraining on local data

```bash
python -m ecg_pipeline.ecg_pipeline_tools train-encoder --max-files 40 --epochs 15
```

`collect_wide_windows()` runs stages 1–5 of the real pipeline (SQI gate
→ resample → filter → beat segmentation) over local VitalPatch/SeNSiO
files and collects the resulting wide beat windows — **no labels
needed**. `main_train_encoder` then calls `pretrain_self_supervised`
(from `ecg_pipeline_core`) and saves the result to
`models/ecg_encoder.pt`.

### 5. `train-classifiers` — the big one: fit + calibrate on real labels

```bash
python -m ecg_pipeline.ecg_pipeline_tools train-classifiers --dataset mitdb
```

This is the most heavily-flagged subcommand (see `--help` for the full
list) because it's also the ablation harness used to tune the
classifier:

- `_load_record_beats` / `build_dataset` — loads a WFDB record's
  *ground-truth annotation* positions (not XQRS detection) through the
  same `segment_beats`/`beat_feature_vector` path production inference
  uses, so training features are extracted identically to inference.
  `_snap_to_local_peak` corrects the small rounding offset introduced by
  rescaling annotation positions to `TARGET_FS`.
- `AAMI_SYMBOL_MAP` — maps each dataset's raw beat symbols onto the 5
  AAMI classes.
- `random_oversample` — a documented 1:3-of-majority floor (not 1:1) for
  minority classes, with a per-class override for F (the rarest real
  class). The docstring explains a real reproducibility bug this
  function had to be fixed for: iterating `set(y)` instead of
  `sorted(set(y))` silently changed results run-to-run under a fixed
  seed, because of Python's per-process string-hash randomization.
- `--include-svdb` / `--include-incart` / `--include-ltafdb`
  `--include-sddb` / `--include-icentia11k` — optional training-set
  enrichment flags (SVDB is on by default). `MITDB_DS2` is never touched
  by any of these — it stays the one report-only held-out set.
- After fitting `FiveClassBeatClassifier`, it also calibrates a
  `ConformalRiskPredictor` by mapping beat-class confidence onto a
  coarse 4-level risk-score proxy (`_beat_proba_to_risk_scores`) — real
  calibration should eventually use actual recording-level outcomes, but
  this gives the conformal predictor *some* real calibration set instead
  of staying permanently uncalibrated.
- Reports per-class sensitivity/precision/F1 on `DS1_VAL` (tuning
  signal) and then on `DS2` (the number that actually matters — printed
  as "REPORT-ONLY — never touched during tuning").

### 6. `eval-classifier` — re-check a saved model without retraining

```bash
python -m ecg_pipeline.ecg_pipeline_tools eval-classifier --model models/five_class_xgb.json --split-set ds2
```

`evaluate()` reuses `build_dataset`/`per_class_metrics` from the
`train-classifiers` section to reproduce a saved model's per-class
metrics, macro-F1, confusion matrix, and F→S misclassification rate in
isolation — this is what the timing-features ablation
(`ABLATION_REPORT.md`) was run through, since `train-classifiers` bakes
training and evaluation into one call and there was previously no way to
re-check a model's numbers alone.

### Combined CLI dispatcher

`_SUBCOMMANDS` maps each subcommand name to its `main_<name>(argv)`
function. `main()` reads `sys.argv[1:]` directly (rather than routing
through `argparse.add_subparsers`) specifically so that
`<subcommand> --help` shows *that subcommand's own* help text — an outer
argparse parser's `-h` would otherwise intercept `--help` regardless of
where it appears and print the dispatcher's help instead.
