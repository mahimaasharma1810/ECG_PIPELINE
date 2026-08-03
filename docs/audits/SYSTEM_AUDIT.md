# SYSTEM_AUDIT.md — Production-Readiness Audit

**Scope:** ECG signal-processing pipeline (`ecg_inference/`, `ecg_pipeline/`), MedGemma clinical LLM agent (`MedGemma-Agent/`), and the data/storage layer around them.
**Method:** Every number in this report was either (a) measured live in this environment by instrumenting and running the real production code against real sample data (VitalPatch CSVs, MIT-BIH WFDB records, and a live local Ollama+GPU MedGemma instance), or (b) read directly from source with file:line citations. No latency/throughput/storage figure in this report is estimated or assumed. Branch audited: `mod_1`, HEAD `e0da1f7`.
**Environment used for measurement:** 10 vCPUs, 125 GB RAM, 1× NVIDIA RTX 2080 Ti (11 GB VRAM), Python 3.10.12, xgboost 3.2.0, Ollama 0.30.9 with a locally-cached `medgemma:latest` (4.3B, Q4_K_M/gemma3) and `medgemma-student:latest` (494M, Q4_K_M/qwen2).

---

## 0. Fixes Applied Since This Audit

Four items were fixed and verified (bit-identical output on real files, live re-tests of each failure mode) in `ecg_inference/` after this audit was written:

| Fix | File(s) | Verified result |
|---|---|---|
| Batched Stage 7 classifier call (was: one `predict_proba()` per beat) | `classifier.py` (new `predict_batch()`), `pipeline.py` | Bit-identical labels vs. the old loop (0/500 mismatches, tested); real WFDB records now run in ~3.2-3.8 s instead of ~5.4-7.5 s |
| Missing/empty `--input` file crash | `run_inference.py` | Now exits with a clean message (`--input not found for --source ...` / `--input file is empty or has no parseable columns: ...`) instead of an uncaught traceback; WFDB's `.hea`/`.dat`/`.atr` triplet path handled correctly (a plain `.exists()` check would have broken every valid WFDB call — caught in testing) |
| `call_medgemma`'s 10 s timeout vs. 75.6 s measured cold start | `report.py` (new `MEDGEMMA_TIMEOUT_S = 90.0`) | Raises the ceiling above the measured cold-load time; does not eliminate the cold-start cost itself — pre-warming the model is still the real fix, noted in-code |
| Insufficient-data segments internally labeled "LOW" | `classifier.py` (`RiskReport.assessable: bool = True`), `pipeline.py` (`_insufficient_data_result`) | See correction directly below — this turned out narrower than originally reported |

**Correction to this audit's original §11/§14 finding**: further tracing showed `run_inference.py`'s actual JSON output was *already* correct — `report.py:541-542` overrides the label to `"NOT_ASSESSABLE"` whenever fewer than 5 beats were analyzed, verified live before any fix was applied. The real, narrower bug: code that reads `result.risk_report.alert_level` or `result.merged_decision.final_decision` **directly** (bypassing that JSON-assembly-layer correction) still saw a bare `"LOW"`. That direct-read pattern exists today in `ecg_pipeline/multimodal_batch.py:77` and `ecg_pipeline/agent_bridge.py` — but those two files import `RiskReport`/`ECGPipeline` from the older `ecg_pipeline_core.py` duplicate (§13's "duplicated 3 ways" finding), not from `ecg_inference`, so **the fix applied here does not reach them**. `ecg_inference.RiskReport` now carries an explicit `assessable` field (default `True`, set `False` only by `_insufficient_data_result`) so any *new* code built on the `ecg_inference` package can check it directly instead of re-deriving beat-count logic itself; mirroring the same field into `ecg_pipeline_core.py` (and updating `multimodal_batch.py`/`agent_bridge.py` to check it) is still open and is the same underlying fix, just needed a second time in the duplicate codebase.

---

## 1. Executive Summary

The ECG pipeline itself is fast and clinically well-instrumented (hash-chained audit trail, transparent rule-trace, no-crash degradation on bad signal). The two systems around it are not production-ready:

- **The single largest fixable latency bug in the whole repo** is in the beat classifier: it calls XGBoost's `predict_proba()` once per beat in a Python loop (`ecg_inference/pipeline.py:174-180`) instead of once per recording. Measured on this machine: **153× slower** than batching (3,026 ms vs 19.7 ms for 2,000 beats). This single change would cut end-to-end ECG latency roughly in half.
- **MedGemma cold-start (75.6 s measured) exceeds `ecg_inference/report.py`'s own 10 s default timeout** (`report.py:159`) by 7.5×. Any pipeline run that hits a cold/idle model will silently fall back to rule-based-only — every time, not occasionally — and nothing logs that this happened as anything other than a routine audit entry.
- **The MedGemma-Agent FastAPI service makes a synchronous, blocking Ollama call from inside an `async def` route handler** (`agent/nodes/llm_analyzer.py:80-93`, invoked from `api/routes/vitals.py:92`). Because Uvicorn runs a single event loop with no `--workers`, no thread offload, and Ollama itself is configured `OLLAMA_NUM_PARALLEL:1`, **one patient's LLM call blocks every other request in the system**, including `/health`, for up to 120 s (the configured timeout) or 75.6 s (measured real cold load).
- **There is no `logging` module usage anywhere in `ecg_inference/`, `ecg_pipeline/`, or `MedGemma-Agent/`** (repo-wide grep, zero hits in both explore passes). Diagnostics are `print()` (160 calls) or a structured-but-severity-free `AuditLog`/audit-DB-table. The MedGemma-Agent's own compliance audit table (`audit_logs`) has **0 rows against 6,760 rows in `alerts`/`vitals_snapshots`** in the live database — the HIPAA-relevant audit trail is not actually being written in whatever environment produced that data, despite the code path existing.
- Basic error handling is missing at the I/O boundary: a missing ECG file or an empty CSV **crashes with an uncaught traceback** (verified live below), not a graceful error.
- Storage: 6.0 GB of the 8.4 GB repo footprint is `MedGemma-Agent/`, of which **4.4 GB is a dead, unreferenced `venv_windows_broken_backup/` directory** (confirmed unused via grep, untracked by git). The 15 MB XGBoost model file is duplicated byte-for-byte in two locations, plus a third, different-shape 16.3 MB copy in `training/model_artifacts/`.

None of this is a rewrite. Every issue below has a scoped, concrete fix; several were verified in minutes once instrumented.

---

## 2. Overall System Architecture

```
Raw ECG (VitalPatch/SeNSiO CSV | WFDB record)
        │
        ▼
┌─────────────────────────── ecg_inference / ecg_pipeline ───────────────────────────┐
│ Stage 1 Ingest → 2 SQI gate → 3 Resample(125Hz) → 4 Filter chain → 5 R-peak+beats  │
│  → 6 Features/HRV → 7 XGBoost classify + rhythm rules → 8 Risk cascade             │
│  → 9 MedGemma call (own 10s timeout, own /api/generate call) → JSON/MD report       │
└──────────────────────────────────────────────────────────────────────────────────┘
        │ (separate integration path, multimodal_batch.py)
        ▼
┌───────────────────────────────── MedGemma-Agent (FastAPI) ─────────────────────────┐
│ /api/v1/vitals/snapshot → auth → DB write → LangGraph:                             │
│   input_validator → [CRITICAL bypass | ehr_retriever → risk_scorer → llm_analyzer  │
│   (Ollama /api/chat, 120s timeout) → output_validator] → alert_router → audit_logger│
└──────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
   SQLite (medgemma_agent.db) + ChromaDB (data/chroma) + hash-chained audit table
```

Two independent MedGemma call sites exist (`ecg_inference/report.py:159` and `MedGemma-Agent/agent/nodes/llm_analyzer.py:80`), with different timeouts (10 s vs 120 s), different HTTP endpoints (`/api/generate` vs `/api/chat`), and duplicated JSON-extraction logic. There is also a third, near-duplicate report-assembly implementation in `ecg_pipeline/agent_bridge.py`.

---

## 3. End-to-End Pipeline Timing (measured)

16 real files benchmarked: 8 VitalPatch CSV segments (`data/raw/vitalpatch/Patch_183594/*.csv`) and 8 MIT-BIH WFDB records (100, 101, 103, 106, 111, 200, 208, 219). Harness: real production functions, `time.perf_counter()` around each stage, no mocking.

| File type | n | Total latency avg | min | max |
|---|---|---|---|---|
| VitalPatch (~2.5 min segments) | 8 | 993 ms | 577 ms | 1,150 ms |
| WFDB (30 min records, more beats) | 8 | 6,700 ms | 5,377 ms | 7,545 ms |
| **All 16** | 16 | **3,846 ms** | 577 ms | 7,545 ms |

A VitalPatch segment (duration_s=151.99 s per `data/reports/vitalpatch_run_manifest.csv` row 1) processed in ~1.03 s of wall time → **~147× faster than real-time** for a single patient on ECG-only processing (MedGemma excluded). This is the good news: raw signal-processing throughput is not the bottleneck. MedGemma call latency and per-patient serialization are (see §5, §9).

Historical evidence from the repo's own batch run confirms this at scale: `data/reports/multimodal_batch/multimodal_manifest.json`, cited in `README.md:278-281`, processed **3,632 segments in 2,800.6 s = 0.77 s/segment average**, consistent with the per-file benchmark above.

---

## 4. ECG Pipeline Latency Analysis (per-stage)

Aggregated across all 16 real-file runs (`ecg_inference/pipeline.py:74-215`'s stage sequence, re-instrumented stage-by-stage):

| Stage | Function (file:line) | Avg | Min | Max | % of total |
|---|---|---|---|---|---|
| 1. Ingest/parse | `parse_vitalpatch_ecg`/`parse_wfdb_record` (`preprocess.py:238`,`:367`) | 105.9 ms | 26.4 ms | 206.4 ms | 2.75% |
| 2. SQI gate | `run_sqi_gate` (`preprocess.py:639`) | 647.8 ms | 51.2 ms | 1,251.7 ms | 16.84% |
| 3. Resample → 125 Hz | `to_target_rate` (`preprocess.py:728`) | 10.6 ms | 0.0 ms | 25.0 ms | 0.28% |
| 4. Filter chain (baseline/notch/bandpass/Kalman) | `apply_filter_chain` (`preprocess.py:851`) | 143.8 ms | 12.1 ms | 274.6 ms | 3.74% |
| 5. R-peak detect + beat segment | `detect_and_segment` (`detector.py:249`, XQRS) | 681.8 ms | 68.9 ms | 1,311.2 ms | 17.73% |
| 6. Feature extraction + HRV | `batch_feature_matrix`/`recording_level_hrv` (`features.py:295`,`:319`) | 455.7 ms | 65.0 ms | 942.5 ms | 11.85% |
| 7. **Beat classify + rhythm rules** | `predict_one` loop (`pipeline.py:174-180`) + `RhythmContextEngine.analyze` | **1,800.7 ms** | 340.9 ms | 3,797.7 ms | **46.83%** |
| 8. Risk cascade | `score_recording` (`classifier.py:292`) | 0.12 ms | 0.04 ms | 0.26 ms | 0.003% |
| 9. MedGemma call (see §5) | `call_medgemma` (`report.py:159`) | not included above — measured separately | | | |

**Bottleneck, confirmed by micro-benchmark:** Stage 7 is nearly half of total runtime because `pipeline.py:174-180` calls `self.classifier.predict_one(...)` — which internally does `self.model.predict_proba(feature_vec.reshape(1, -1))` (`classifier.py:150`) — **once per beat, in a Python `for` loop**, rather than building the full feature matrix and predicting once. Direct measurement on this machine, same model, same feature width:

```
loop  (2,000× single-row predict_proba): 3,026.2 ms
batch (1× 2,000-row predict_proba):         19.7 ms
speedup: 153.9×
```

For a 30-min WFDB record with ~2,000 beats, this loop alone plausibly accounts for 2-3 s of the 5.4-7.5 s total. **Optimization: batch-predict `feature_matrix` once via `self.classifier.model.predict_proba(feature_matrix)`, then walk the array to assign labels/probabilities to each `Beat`** — a same-file change, no model retraining, no numeric difference (XGBoost's batched and per-row predictions are numerically identical; only the Python-call overhead is amortized).

**Secondary bottlenecks:**
- **Stage 5 (R-peak/beats, 17.7%)**: XQRS (`wfdb.processing`) is a pure-Python/NumPy adaptive detector with no batching or vectorization across windows; further speedup would require replacing XQRS or pre-filtering candidate windows, a larger change than Stage 7's fix.
- **Stage 2 (SQI gate, 16.8%)**: windowed quality checks (flatline/clipping/kurtosis/SNR) run once per window per verdict; the wide min-max spread (51 ms → 1,252 ms) tracks recording length, suggesting no early-exit for already-failing windows.
- **Stage 6 (features, 11.9%)**: `batch_feature_matrix` is already vectorized per its name but still costs ~450 ms average; profiling per-feature would be the next step if Stage 7's fix doesn't bring total latency low enough for the target patient count (see §9).

No stage currently has timing instrumentation in production (`AuditLog.append()` at `preprocess.py:169-178` stamps wall-clock `time.time()` per event, not elapsed duration) — all numbers above required building a benchmark harness from scratch; this should be a permanent addition (see §16).

---

## 5. MedGemma Latency Analysis (measured live, not simulated)

A real Ollama server was started against the repo's already-cached `medgemma:latest` model and hit with the actual production prompt template (`MedGemma-Agent/agent/prompts.py`, 1,482-char system prompt + a representative 896-char user prompt = 2,378 chars / ~687 tokens).

| Call | Wall time | Ollama `load_duration` | `prompt_eval` | `eval` (generation) | Output tokens |
|---|---|---|---|---|---|
| **Cold (1st call, model not in VRAM)** | **75.57 s** | 71.72 s | 0.73 s (687 tok) | 3.09 s | 267 |
| Warm call 2 | 4.26 s | 1.12 s | 0.04 s | 3.09 s | 273 |
| Warm call 3 | 4.16 s | 1.15 s | 0.06 s | 2.93 s | 259 |
| Warm call 4 | 4.46 s | 1.12 s | 0.06 s | 3.25 s | 283 |
| Warm call 5 | 4.54 s | 1.16 s | 0.05 s | 3.32 s | 288 |
| **Warm avg (n=4)** | **4.36 s** | ~1.14 s | ~0.05 s | ~3.15 s | ~276 |

- **Median (warm)**: 4.36 s. **95th percentile / worst case observed**: the cold call, 75.57 s — a **17× gap** between warm and cold that the codebase's own timeout does not account for consistently (see below).
- **Generation throughput**: ~85-90 tokens/sec on this RTX 2080 Ti for the Q4_K_M 4.3B model — in line with GPU-quantized 4B-class model expectations.
- **GPU utilization**: 3.8 GB / 11.26 GB VRAM used while loaded (34%) — comfortable headroom, GPU sits at 0% between requests (bursty, not continuously loaded).
- **Concurrency**: `OLLAMA_NUM_PARALLEL:1` in the server's own startup config (confirmed in Ollama's log output) — the inference server itself serializes generation, on top of the FastAPI-layer blocking issue below.

**Why MedGemma is the largest bottleneck when cold, and a silent-failure risk even when warm:**

1. `ecg_inference/report.py:159` — `call_medgemma(prompt, model=MEDGEMMA_MODEL, timeout_s: float = 10.0)`. **10 seconds is less than the measured 75.57 s cold-load time by 7.5×.** Since Ollama's default `OLLAMA_KEEP_ALIVE` is 5 minutes, any pipeline run that fires after 5 minutes of MedGemma idle time will time out and silently fall back to rule-based-only via the `except (requests.RequestException, ValueError): return None` at `report.py:169-170` — this is by design (documented as graceful degradation), but it means **the LLM narrative is likely absent far more often in practice than anyone reviewing only "ACCEPTED"/"UNAVAILABLE" counts in a manifest would assume**, because a "cold" 10 s timeout will fire on essentially every request after any idle gap, not as a rare tail event.
2. `MedGemma-Agent/config/settings.py:9` uses `OLLAMA_TIMEOUT = 120` — long enough to survive a cold load (75.57 s measured) with margin, but during that entire window **the whole FastAPI process is blocked** (see §9), so the cost of a cold start is paid by every other in-flight request, not just the one that triggered it.
3. No retry logic exists at either call site (confirmed via repo-wide grep for `retry|tenacity|backoff|circuit` — zero hits) and no metric records how often the LLM path was actually reached vs. bypassed vs. timed out.

**Practical optimizations:**
- Raise `ecg_inference/report.py`'s `timeout_s` default well above 75 s (or, better, pre-warm the model at process start and periodically ping it to keep it inside `OLLAMA_KEEP_ALIVE`, eliminating cold starts from the hot path entirely).
- Set `OLLAMA_KEEP_ALIVE` higher than the interval between real requests in production, or run a lightweight keep-alive heartbeat.
- Move the Ollama call off the request-handling event loop (see §9) so a cold/slow LLM call never blocks other patients.
- Consider `medgemma-student:latest` (494M vs 4.3B, already present in the local model cache) for the hot path if its clinical-safety evaluation (`MedGemma-Agent/scripts/evaluate_clinical_safety.py`) has been passed — an 8-9× smaller model would substantially cut both cold-load and generation time, at a safety/quality tradeoff that must be validated, not assumed.
- Track and log prompt token count over time; `ehr_context` is inserted with no truncation or cap (`agent/prompts.py`, `rag/retriever.py:23` pulls top_k=5 chunks unbounded in size), so prompt-eval time (and thus latency) could grow silently as EHR ingestion volume grows.

---

## 6. Storage Analysis

| Location | Size | Contents / notes |
|---|---|---|
| `MedGemma-Agent/` | 6.0 GB | see breakdown below |
| `data/` | 2.4 GB | see breakdown below |
| `training/` | 19 MB | includes a **third** copy of a similarly-named model, `model_artifacts/five_class_xgb_with_svdb.json` (16.3 MB, different training run, not identical to production model) |
| `ecg_pipeline/` | 16 MB | includes `models/five_class_xgb.json` (15.75 MB) |
| `ecg_inference/` | 16 MB | includes `models/five_class_xgb.json` (15.75 MB) — **byte-identical** to the `ecg_pipeline/` copy (confirmed via `diff`) |
| `docs/` | 1.2 MB | markdown handoff/audit docs |
| `processed_output/` | 8 KB | a single leftover CSV under `Patch_184B2F/` — looks like a stale/orphaned scratch output, not part of any documented pipeline output path |

### `MedGemma-Agent/` breakdown (6.0 GB)
| Item | Size | Note |
|---|---|---|
| `venv_windows_broken_backup/` | **4.4 GB** | **Dead weight.** Not referenced in any `.sh`/`.md` in the repo (grep confirms zero hits); untracked by git (`git status` shows `??`); its own name says "broken." Safe to delete. |
| `venv/` | 1.6 GB | active virtualenv, correctly gitignored |
| `data/` (medgemma_agent.db + chroma) | 26 MB | see below |
| `server.log` | 272 KB | Uvicorn access log only (no app-level logging) |
| rest of source (`agent/`, `api/`, `abg/`, etc.) | ~700 KB combined | fine |

### `data/` breakdown (2.4 GB)
| Item | Size | Note |
|---|---|---|
| `raw/public/` | 1.3 GB | `incartdb` 796 MB, `challenge2017` 306 MB (includes a **99 MB `training2017.zip`** never unpacked-and-deleted), `mitdb` 90 MB, `svdb` 53 MB, `cudb` 6.8 MB — public benchmark datasets, redistributable per `.gitignore`'s own comment |
| `raw/vitalpatch/` | 639 MB | real device CSVs |
| `vitals_downloads/` | 384 MB | 6 patch folders, 26-73 MB each |
| `raw/prorhythm/` | 19 MB | includes an 11.8 MB single CSV |
| `reports/vitalpatch/` | 45 MB | **3,628 `.json` + 3,628 `.md`** — one narrative markdown per JSON report; not redundant (different consumers) but doubles file count for the same information; a stray `.json.bak_pre_rpeak_fix` backup file also sits here (`1844AC/1778227671297_..._seg0.json.bak_pre_rpeak_fix`) and should be deleted or moved to an explicit backups/ location |
| `reports/clinician_review_visual/` | 18 MB | 50 PDFs |
| `reports/multimodal_batch/` | 4.9 MB | includes 3 manifest JSONs from 3 different batch runs (full, 14-segment check, 25-segment check) — the two small-scale checks are superseded per `README.md:280-283` and could be archived/deleted once confirmed no longer needed |
| `reports/synthetic/` | 2.4 MB | synthetic test-scenario outputs |

### Duplicate / redundant / deletable, ranked by recoverable space
1. `MedGemma-Agent/venv_windows_broken_backup/` — **4.4 GB**, confirmed dead, delete.
2. `data/raw/public/challenge2017/training2017.zip` — **99 MB**, if already extracted elsewhere, delete the archive.
3. `ecg_pipeline/models/five_class_xgb.json` **or** `ecg_inference/models/five_class_xgb.json` — **15.75 MB**, byte-identical duplicate; keep one canonical copy and have the other package import/symlink it (a real duplication risk: if someone updates one without the other, the two packages silently diverge).
4. `training/model_artifacts/five_class_xgb_with_svdb.json` — **16.3 MB** — verify whether this is superseded by the production model; if so, archive outside the working tree.
5. `data/reports/multimodal_batch/sample14_manifest_preserved.json` + `sample25_manifest_smallcheck.json` — superseded per README, safe to archive.
6. The `.bak_pre_rpeak_fix` file — single stray backup, delete or relocate.
7. `processed_output/` — appears orphaned; confirm no active script writes here before deleting.

**Storage growth pattern:** report generation is 1:1 with segments processed (2 files/segment: json+md), so storage grows linearly with recordings processed — at the current ~45 MB / 3,628 segments (≈12.4 KB/segment) rate, 1M segments would be ~12.4 GB of reports alone. No compaction, rotation, or archival-to-cold-storage policy exists for `data/reports/`.

**Recommendations without affecting reproducibility:**
- Deduplicate the model file via a single canonical location + import, not two copies.
- Compress `data/raw/public/*` `.dat`/`.csv` files at rest (gzip) between batch runs — WFDB tooling can read `.dat.gz` transparently via decompression on load, or decompress-on-demand in the parser; the public datasets are static physiological recordings, ideal for compression.
- Move superseded/small-scale manifest checks and `.bak_*` files to an explicit `archive/` folder outside the active `reports/` tree.
- Set a retention policy (e.g. keep JSON, drop rendered `.md`/`.pdf` after N days, regenerate on demand) since narratives are deterministically reproducible from the JSON + rule trace.

---

## 7. Memory Analysis

Measured via `/usr/bin/time -v` on a full `run_inference.py` invocation against a 30-min WFDB record (2,204 beats):

| Metric | Value |
|---|---|
| Maximum resident set size | **468.9 MB** |
| User CPU time | 43.03 s |
| System CPU time | 2.41 s |
| Wall clock | 11.84 s |
| CPU utilization | **383%** (multi-threaded despite `n_jobs=1` pin — see §8) |
| Minor page faults | 127,763 |
| Major (I/O) page faults | 1 |

468 MB peak RSS for a single-recording CLI process is not concerning in isolation, but there is no streaming/chunked processing anywhere in the pipeline — `recording.signal_mv`, `signal_clean` (a `.copy()` at `pipeline.py:88`), `resampled`, `resampled_filled`, and `filtered` are **five full-length float arrays of the same recording held simultaneously** in `ECGPipeline.run()` (`pipeline.py:88-106`). For a single 30-min recording this is trivial; for a design meant to process many concurrent patients in one long-lived process (as opposed to one-process-per-CLI-invocation), this is the first thing to profile if scaling to dozens of simultaneous streams in one process.

**Notable non-obvious memory-lifetime issue**: `TemporalRiskTracker._history` (`classifier.py:439-440`) is a `dict[patient_id -> deque]` that lives for the lifetime of the `ECGPipeline` instance and is only ever appended to via `.record()` (`pipeline.py:196-197`), never evicted. In `run_inference.py`'s one-shot CLI usage this is harmless (process exits after one file), but if `ECGPipeline` is instantiated once and reused across many patients in a long-running service (which is exactly the multimodal batch/Agent-integration use case this repo is building toward), **this dict grows unboundedly with the number of distinct patient IDs ever seen**, with no TTL or cap — a genuine slow memory leak under sustained multi-patient production use.

No pandas DataFrames are retained beyond the parsing stage (`preprocess.py`'s parsers return numpy arrays into `Recording` dataclasses, not DataFrames) — good practice, avoids the common pandas-object-overhead leak pattern.

---

## 8. CPU / GPU Utilization

- **CPUs available**: 10 cores, 125 GB RAM, no `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` env vars set (confirmed via `env | grep`) — numpy's OpenBLAS build (`USE_OPENMP=` unset, `MAX_THREADS=2` per `numpy.show_config()`) and XGBoost's internal prediction threading are left at library defaults, not explicitly pinned outside the one `n_jobs=1` in `FiveClassBeatClassifier.fit()`/model construction (`classifier.py:144`).
- **Measured 383% CPU utilization** for a process whose model is explicitly pinned to `n_jobs=1` — confirms that XGBoost's C++ prediction path and/or BLAS spin up multiple threads per call regardless of the `n_jobs` constructor argument (which governs `.fit()`/tree-building parallelism, not necessarily every internal `predict_proba` call path in this version). This directly compounds the Stage 7 bottleneck: **2,000+ tiny `predict_proba` calls each pay thread-pool spin-up/teardown overhead**, which the batch-predict fix in §4 eliminates as a side effect (19.7 ms total instead of 3,026 ms — not just fewer Python-call overheads, but far less thread-management overhead too).
- **GPU**: RTX 2080 Ti, 11.26 GB VRAM, 0% utilization at idle. The ECG pipeline (Stages 1-8) is **pure CPU** — no GPU usage at all in `ecg_inference`/`ecg_pipeline` (confirmed: no `torch`/`cupy`/CUDA imports on the active path; the one CNN/encoder path that would have used a GPU is explicitly dead code per `pipeline.py:56-67`'s own docstring). GPU is used exclusively by MedGemma inference, at 3.8/11.26 GB (34%) VRAM when loaded.
- **Parallelization opportunities**: Filtering (Stage 4), feature extraction (Stage 6), and R-peak detection (Stage 5) all operate on one recording at a time with no cross-recording parallelism — a batch job processing N recordings (as `multimodal_batch.py`/`batch_vitalpatch_report.py` do) runs them **serially in a single process** (confirmed: no `multiprocessing`/`concurrent.futures`/`joblib` import anywhere in `ecg_pipeline/*.py` or `ecg_inference/*.py`). Since each recording's Stages 1-8 are independent of every other recording, this is an easy, high-value parallelization target: a `multiprocessing.Pool`/`concurrent.futures.ProcessPoolExecutor` across the batch would let the 10 available cores process ~10 recordings concurrently instead of 1, directly multiplying batch throughput. XGBoost inference itself (Stage 7, post-fix) and classification are inherently batchable per-recording already (§4); across-recording parallelism is the next lever specifically for batch/offline throughput.

---

## 9. Real-Time Performance

Given VitalPatch streams continuously and each ~2.5 min segment measured at ~1.0 s ECG-only processing (147× real-time headroom per patient, §3):

| Question | Answer, with evidence |
|---|---|
| Max sustainable throughput (ECG-only, single core) | ~1 segment/second sustained (993 ms avg measured) → ≈ 1 patient-segment/sec per worker process |
| Segments/sec at 10 cores, if batch-parallelized (not currently implemented, see §8) | up to ~10/sec (linear in cores, since stages are single-threaded per recording and independent across recordings) |
| Patients supported simultaneously (ECG-only) | Tens of concurrent patients, comfortably, on ECG processing alone — **not the bottleneck** |
| Patients supported simultaneously (with MedGemma in the loop) | **Effectively 1 at a time**, today. See below. |
| Expected latency per patient, ECG-only | ~1.0-1.15 s (VitalPatch segment length) measured |
| Expected latency per patient, with MedGemma warm | +4.36 s median (measured, §5) |
| Expected latency per patient, with MedGemma cold | +75.6 s measured (§5) |
| Can the system keep up with real-time ECG? | **Yes**, for the ECG-only path. **No**, for the combined path under concurrent load, because of the issue below. |
| Where does backlog begin? | **The instant a second patient's request arrives while a first patient's MedGemma call is in flight.** |

**Root cause of the backlog point**, verified by code + measurement together:
1. `MedGemma-Agent/api/routes/vitals.py:28` declares `async def submit_vitals_snapshot(...)`, but internally calls `monitoring_agent.invoke(initial_state)` (`vitals.py:92`) synchronously.
2. That graph invocation reaches `agent/nodes/llm_analyzer.py:80-93`, which opens a **blocking** `httpx.Client(...).post(...)` — not `httpx.AsyncClient`, and not wrapped in `asyncio.to_thread`/`run_in_executor`.
3. Uvicorn is started with **no `--workers` flag** (`start.sh:56-61`) — a single process, single asyncio event loop.
4. Therefore, a blocking call inside that loop **stalls the entire process** — every other endpoint (including `/health`) — for the duration of that one call: measured 4.36 s warm, 75.6 s cold.
5. Ollama's own `OLLAMA_NUM_PARALLEL:1` means even a fully-async client wouldn't get concurrent *generation* from the model itself, only concurrent *queueing* — a queue-based/async architecture would still need either a multi-slot Ollama config or a request queue with backpressure to actually serve multiple patients' LLM calls concurrently.

**Practical fix path**: move the Ollama call to `asyncio.to_thread(...)` (a one-line wrap around the existing `httpx.Client` call, no client library change needed) as a stopgap so at least non-LLM endpoints stay responsive; run Uvicorn with multiple workers behind a process-level queue; and separately, raise `OLLAMA_NUM_PARALLEL` and/or run a request queue with an explicit max-concurrency + backpressure policy so patient requests degrade to "queued, ETA N seconds" rather than silently stalling.

---

## 10. Scalability Assessment

- **Horizontal scaling of the ECG pipeline**: straightforward — each recording is independent, stateless except for `TemporalRiskTracker`'s per-patient history (§7), so multiple worker processes (one per core, or containerized replicas) can each own a shard of patients. No current code does this (batch scripts are single-process/serial, §8).
- **Horizontal scaling of MedGemma-Agent**: harder without more work, because (a) the service is single-process/single-worker today (§9), and (b) it owns local SQLite (`db/database.py`) and a local ChromaDB (`rag/vector_store.py`) — neither is a natively multi-writer, multi-instance datastore. Scaling to multiple API replicas would require migrating to a networked DB (Postgres) and a networked/sharded vector store, or accepting a single-writer bottleneck.
- **Model-serving scaling**: one Ollama instance, one GPU, `OLLAMA_NUM_PARALLEL:1`. Scaling patient throughput on the LLM side means either a bigger/multi-GPU Ollama deployment with a higher parallel slot count, a smaller/faster model (the already-cached `medgemma-student` is 8.7× smaller), or explicit queueing with defined SLA/backpressure — not "add more API replicas," since they'd all still serialize on the same underlying model server.
- **Batch throughput**: current batch scripts (`multimodal_batch.py`, `batch_vitalpatch_report.py`) process files serially in one process with no `multiprocessing` (confirmed via grep, §8) — this is the easiest, lowest-risk scaling win available (10× on this machine, near-linear in core count) and requires no architecture change, just wrapping the existing per-file loop in a process pool.

---

## 11. Failure Mode Analysis

Verified **live**, not just read from source, by feeding the real `run_inference.py` CLI adversarial inputs:

| Scenario | Verified behavior | Evidence |
|---|---|---|
| Missing ECG file | **Crashes.** Uncaught `FileNotFoundError` traceback to stderr. | Live test: `python run_inference.py --input data/raw/vitalpatch/DOES_NOT_EXIST.csv ...` → full pandas/Python traceback, no graceful message. |
| Corrupted/empty CSV | **Crashes.** Uncaught `pandas.errors.EmptyDataError: No columns to parse from file`. | Live test on a 0-byte file. |
| Garbage (non-CSV text) content | **Handled gracefully.** `"No usable segments parsed from <path>"` + clean `SystemExit`, no traceback. | Live test on a text file with random content. |
| Stray non-numeric values inside an otherwise valid CSV | **Handled.** `pd.to_numeric(..., errors="coerce")` (`preprocess.py:259`) converts to NaN, rows dropped by timestamp/value pair — a documented fix for a real historical bug that crashed 70/3,570 VitalPatch segments. | Code-verified, `preprocess.py:248-259` comment cites the original incident. |
| Missing timestamps (SeNSiO) | **Handled** — synthesized from `fs_nominal` (`preprocess.py:343`), not required from the device. | Code-verified. |
| Missing RR/temperature vitals | **Handled** — components scored as `None`/missing, never imputed (`classifier.py:334,338`); NEWS2/qSOFA degrade gracefully to partial coverage. | Code-verified, matches `docs/decisions/NEWS2_PARTIAL_COVERAGE_DECISION.md`. |
| Poor ECG signal quality | **Handled** — SQI gate + hard floor (`pipeline.py:97-101`) routes to `_insufficient_data_result` (`pipeline.py:217-239`), a plain non-crashing result. **But** it defaults `alert_level="LOW"` (`pipeline.py:226`) for a segment with *no assessable signal at all* — clinically, "we couldn't tell" and "we checked and it's fine" are different claims, and reporting the former as the latter's risk label is a mislabeling risk, not a crash risk. | Code-verified, `pipeline.py:223-227`. |
| Zero R-peaks detected | **Handled** — empty beat list flows through Stages 6-8 without crashing; `mean_rr` falls back to `0.0`, risk cascade uses `max(1, len(labels))` to avoid divide-by-zero. | Code-verified, `pipeline.py:169-170`, `classifier.py:295`. |
| Classifier fails to load / untrained weights | **Crashes intentionally** — `load_classifier` (`report.py:245-250`) raises `RuntimeError` rather than silently falling back, on the reasoning that an untrained classifier producing labels would be worse than a hard failure. Reasonable for a CLI tool; **would produce an unhandled 500 if reached inside a long-running service** that doesn't wrap pipeline construction in its own try/except. | Code-verified. |
| MedGemma / Ollama unavailable or times out | **Handled** — both call sites catch and fall back to rule-based-only (`ecg_inference/report.py:164-170`; `MedGemma-Agent/agent/nodes/llm_analyzer.py:112-118`), no crash, but see §5 for how often this actually fires given the 10 s timeout vs 75.6 s measured cold start. | Code-verified + live latency measurement. |
| JSON parsing failure from LLM response | **Handled** in both codebases — regex/`raw_decode` fallback, then `None`/`llm_error` field on total failure, downstream falls back to rule-based. | Code-verified, `report.py:172-182`, `llm_analyzer.py:20-41`. |
| Database (SQLite) unavailable | **Not handled** — no try/except around `db.commit()`/`db.query()` in `MedGemma-Agent/api/routes/vitals.py` or `compliance/audit_logger.py`; would surface as a raw, unhandled `sqlalchemy.exc.OperationalError` → default FastAPI 500. | Code-verified (explore pass; not separately re-tested live in this session since it requires simulating a DB outage). |

**Overall pattern**: failures inside the signal-processing logic (bad values, missing vitals, poor quality, zero beats) are handled thoughtfully and match documented clinical-safety reasoning. Failures at the **I/O boundary** (file doesn't exist, file is empty) and **infrastructure boundary** (DB down) are not — these are the two gaps to close first, since they're also the cheapest to fix (a `try/except FileNotFoundError`/`pd.errors.EmptyDataError` at the CLI entry point, and a try/except + graceful-degraded-response around the DB calls).

---

## 12. Logging Review

| Requirement | Present? | Evidence |
|---|---|---|
| Errors | Partial | Only via uncaught tracebacks (§11) or `AuditLog` entries like `MEDGEMMA_UNAVAILABLE`/`MEDGEMMA_REJECTED` — no `logging.error()`/`exception()` anywhere. |
| Warnings | No | No warning-level severity exists in either the `AuditLog` (`preprocess.py:153-194`, no `level` field) or MedGemma-Agent's audit table. |
| Execution time per stage | No | Confirmed absent (§4) — had to be built from scratch for this audit. |
| Pipeline stage name | Yes | `AuditLog.append("STAGE1_INGEST", ...)` etc. (`pipeline.py:77` onward) — good, structured, but JSON-audit-trail, not a log stream. |
| Model version | **No** | Confirmed by inspecting `build_risk_report_json`'s actual output keys — no `model_version`/`classifier_version` field anywhere in the report JSON. |
| Threshold version | **No** | Same check — no `threshold_version`/`config_version` field. Given `RiskThresholds` etc. are versionless module-level singletons (`preprocess.py:135-140`), there is currently no way to tell, from a stored report, which threshold values produced it if they're changed later. |
| Patient ID | Yes | On `Recording`/audit entries (`pipeline.py:77-79`) and DB rows (`Patient.id` in MedGemma-Agent). |
| Session ID / request ID | No | No request-ID correlation anywhere (confirmed via grep for `request_id`/`correlation`/`trace_id` — zero hits); impossible to correlate a MedGemma-Agent server log line with a specific patient request without cross-referencing timestamps. |
| Risk level | Yes | In both the `AuditLog` (`STAGE8_RISK`) and the DB `Alert` table. |

**Concrete anomaly found**: the live `MedGemma-Agent/data/medgemma_agent.db` has **6,760 rows in `alerts`/`vitals_snapshots` but 0 rows in `audit_logs`**, despite `audit_logger_node` (`agent/nodes/audit_logger.py:13-44`) running on every graph path. This means the one piece of infrastructure that exists specifically for HIPAA-relevant compliance audit is not actually recording anything in this environment's history — worth an immediate root-cause check (silent exception in `compliance/audit_logger.log_event`? wrong DB session? never actually wired into the invoked graph build?) before any clinical use.

**Recommendation**: adopt Python's `logging` module with structured (JSON) output, include `request_id`, `patient_id`, `stage`, `duration_ms`, `model_version`, `threshold_version`, and severity on every log line, and keep the existing `AuditLog`/audit-DB-table as the compliance record — the two serve different purposes (operational debugging vs. compliance trail) and neither substitutes for the other today.

---

## 13. Code Quality Review

- **Duplicated logic, 3 ways**: `ecg_inference/*` (2,853 lines, "extracted verbatim" per its own module docstrings) duplicates `ecg_pipeline/ecg_pipeline_core.py` (2,820 lines, the original monolith), and `ecg_pipeline/agent_bridge.py` (1,637 lines) re-implements report-assembly a third time. Any threshold or bugfix must currently be applied in up to three places to stay consistent — the `five_class_xgb.json` byte-identical duplication (§6) is a direct symptom of this split.
- **Duplicated Ollama/JSON-parsing logic**: `agent/nodes/llm_analyzer.py:79-118` and `api/routes/abg.py:76-121` duplicate the entire call/parse/error-handling block near-verbatim, including `_extract_json`/`_strip_thinking_tokens`. A shared client wrapper would fix both the duplication and let a single retry/timeout/circuit-breaker fix apply everywhere.
- **Large functions**: `ECGPipeline.run()` (`pipeline.py:74-215`, ~140 lines, all 9 stages inlined); `report.py`'s `build_risk_report_json` (~125 lines) and `render_narrative` (~60 lines); `abg/interpreter.py` (511 lines total, largest single file in MedGemma-Agent, not fully read line-by-line in this pass and worth a dedicated review if the ABG path goes to production.
- **Magic numbers inconsistent with the otherwise-good config pattern**: most thresholds are centralized in dataclasses (`preprocess.py:44-140`, e.g. `RiskThresholds`), but some are inlined at call sites instead — `pipeline.py:98` (`TARGET_FS * 2` minimum valid samples), `pipeline.py:149` (`snap_radius = 8 if ... else 15`), `detector.py:58` (`fs * 2` minimum signal length), `classifier.py:142-144` (`n_estimators=164, max_depth=11` hardcoded in `.fit()`). On the Agent side, `guardrails/clinical_rules.py:277-283` hardcodes the top-level NEWS2 alert cutoffs (`>=7` CRITICAL etc.) inline even though the per-vital sub-scores are correctly externalized to `guardrails/thresholds.yaml` — an inconsistency worth closing so all clinical thresholds live in one auditable place.
- **Global mutable state**: module-level singletons `SQI`, `FILTER`, `BEATS`, `RISK`, `CONFORMAL`, `TEMPORAL` (`preprocess.py:135-140`) are shared, mutable dataclass instances across the whole process — safe today because nothing mutates them at runtime, but a latent risk if any future code path does. `TemporalRiskTracker._history` is genuinely stateful and unbounded (§7).
- **160 `print()` calls** across `ecg_pipeline/`+`ecg_inference/` in lieu of logging (§12) — mechanical to replace with `logging` calls once a logger is introduced.
- **No dead-code deletion needed for the encoder path** — it's already correctly excised from `ecg_inference` per that package's own docstring (`pipeline.py:56-67`), a good example of the cleanup this repo does apply consistently when it decides to.

---

## 14. Clinical Readiness Assessment

| Property | Status | Evidence |
|---|---|---|
| Deterministic behaviour | **Good** | `n_jobs=1` pinned specifically to guarantee reproducible XGBoost output across runs (`classifier.py:108-109,144`, with an in-code note this was verified via back-to-back identical runs). |
| Reproducibility | **Good** | Frozen model weights, versioned thresholds-as-code, full rule-trace re-rendering (`_build_rule_trace`, `report.py:359-...`) that reconstructs exactly which threshold fired. |
| Explainability | **Good** | `rule_trace`/`deciding_rule` fields in the report JSON make the deterministic decision path fully inspectable per-recording — a real strength versus a black-box classifier-only design. |
| Threshold transparency | **Good, with a gap** | Thresholds are code-visible dataclasses, but **not versioned/stamped into output** (§12) — you can inspect today's thresholds, but a report generated last month is not self-describing about which threshold values produced it if they've since changed. |
| Model transparency | **Partial** | `CLASS_CONFIDENCE_NOTES` (`report.py:269-283`) honestly reports per-class held-out F1 (e.g. "F" class F1=0.011, explicitly called "essentially unsolved... treat as noise") directly in the report — genuinely good clinical honesty. But no `model_version` field ties a report back to a specific trained artifact (§12). |
| Failure handling | **Partial** | Strong inside the clinical logic (§11), weak at I/O/infra boundaries (§11) — a missing-file crash in a batch job is an operational problem, not a patient-safety one, but it should still not be an uncaught traceback in a clinical system. |
| Auditability | **Partial** | `AuditLog` is a genuinely good hash-chained, ordered trace per recording (mirrors MedGemma-Agent's `compliance/audit_logger.py` pattern) — but the MedGemma-Agent's own audit **DB table has 0 rows against 6,760 processed events** (§12), a live, currently-broken audit trail that must be fixed before any real clinical audit review. |
| Safety | **Good, verified in code** | CRITICAL alerts always bypass the LLM entirely (`agent/nodes/alert_router.py:21-23`, `guardrails/input_guardrails.py:72-77`; ECG-side equivalent at `report.py:221-222`/`merge_decision`'s bypass) — patient safety for the worst-case alert never depends on Ollama being reachable. LLM output can only **raise** risk level, never lower it (`report.py:210`), and disagreement >1 severity level is rejected outright (`report.py:204-208`). This is a well-designed safety invariant. |
| Clinical traceability | **Good** | Every risk level traces to either a specific rule-trace entry or an explicit "LLM raised X→Y, reason: ..." record — a clinician reviewing a report can see *why*, not just *what*. |

**Missing components for a production-grade clinical decision support system**, given the above:
1. Model/threshold version stamping on every report (currently absent).
2. A fixed audit-logging bug (0 rows recorded) before relying on the compliance trail for real oversight.
3. Graceful (not crashing) handling of missing/corrupted input files, matching the care already put into signal-quality edge cases.
4. Distinguishing "insufficient signal to assess" from "assessed as LOW risk" in the `_insufficient_data_result` path (§11) — currently both report `alert_level="LOW"`, which is a mislabeling risk in a clinical context specifically because it looks identical to a genuine clean reading.
5. Structured, severity-leveled operational logging (separate from the compliance audit trail) for on-call debugging.
6. A load test / documented SLA for concurrent patients, since none exists today and the architecture (§9) currently serializes hard on MedGemma.

---

## 15. Priority Issues

**CRITICAL**
- MedGemma-Agent's audit_logs table has 0 rows against 6,760 processed events — the compliance audit trail is not being written (§12).
- FastAPI event loop blocks on synchronous Ollama calls inside `async def` handlers — one patient's LLM call stalls the entire service, up to 75.6 s measured (§9).
- ~~`ecg_inference/report.py`'s 10 s MedGemma timeout is 7.5× shorter than the measured 75.6 s cold-start latency~~ — **FIXED** (§0): raised to `MEDGEMMA_TIMEOUT_S = 90.0`.
- ~~Missing/empty input files crash with uncaught tracebacks rather than a handled error~~ — **FIXED** (§0), verified live in `run_inference.py`.

**HIGH**
- ~~Stage 7 per-beat `predict_proba` loop is 153× slower than necessary~~ — **FIXED** (§0): batched via new `predict_batch()`, bit-identical output confirmed, real records now ~2× faster end to end.
- No structured/severity-leveled logging anywhere; no model/threshold version stamped on reports (§12, §14).
- `_insufficient_data_result` reports "LOW" risk for segments where no real assessment was possible — **partially fixed** (§0): `ecg_inference.RiskReport` now has an explicit `assessable` flag, but the same bug independently exists in the `ecg_pipeline_core.py`-based `multimodal_batch.py`/`agent_bridge.py` path, which was not touched.
- `TemporalRiskTracker._history` grows unboundedly per unique patient ID in any long-running process (§7).
- No retry/circuit-breaker on either Ollama call site; no concurrency-safety on the shared FastAPI/Ollama path beyond `OLLAMA_NUM_PARALLEL:1` serialization (§5, §9).

**MEDIUM**
- 4.4 GB dead `venv_windows_broken_backup/` directory (§6).
- Byte-identical 15 MB model file duplicated in two packages, plus a third 16.3 MB variant in `training/` (§6).
- Batch scripts process files serially with no multiprocessing despite embarrassingly parallel workload (§8, §10).
- Database (SQLite) errors unhandled in MedGemma-Agent routes (§11).
- Duplicated Ollama-call/JSON-parse code between `llm_analyzer.py` and `abg.py` (§13).

**LOW**
- Stray `.bak_pre_rpeak_fix` file and superseded small-scale batch manifests left in `data/reports/` (§6).
- Inconsistent magic-number placement for a handful of thresholds otherwise centralized in config dataclasses (§13).
- 160 `print()` calls to migrate to structured logging once introduced (§12, §13).
- `processed_output/` appears to be an orphaned scratch directory (§6).

---

## 16. Optimization Recommendations

1. **Batch the Stage 7 classifier call** — replace the per-beat `for` loop (`pipeline.py:174-180`) with one `predict_proba(feature_matrix)` call, then map results back to beats. Verified 153× speedup on the prediction step alone; numerically identical output.
2. **Instrument every stage with real timing** — add a lightweight `time.perf_counter()` wrapper (or a decorator) around each of the 9 stages and emit `duration_ms` into the existing `AuditLog` payload (it already has a slot for arbitrary payload fields) — closes the "no execution-time" logging gap and gives ongoing regression visibility for free.
3. **Fix the MedGemma timeout/cold-start mismatch** — raise `ecg_inference/report.py`'s default well past the measured 75.6 s worst case, or (better) pre-warm the model and keep it warm with a heartbeat so cold starts never occur on the patient-facing path.
4. **Move the Ollama call off the FastAPI event loop** — wrap the existing `httpx.Client(...).post(...)` in `asyncio.to_thread(...)` as an immediate low-risk fix; plan a queue-based architecture with explicit concurrency limits as the durable fix.
5. **Parallelize batch processing across recordings** — wrap `multimodal_batch.py`/`batch_vitalpatch_report.py`'s per-file loop in a `ProcessPoolExecutor`; near-linear speedup up to core count (10× on this machine) with no algorithmic change.
6. **Deduplicate the model file** and point both packages at one canonical path.
7. **Delete `venv_windows_broken_backup/`** (4.4 GB, confirmed dead) and decompress/re-archive the public WFDB datasets.
8. **Add `try/except` at the CLI/API I/O boundary** for missing/corrupted files, converting uncaught tracebacks into a clean, logged, non-zero-exit error.
9. **Cap `TemporalRiskTracker._history`** with either a TTL or an LRU eviction policy keyed on patient_id, to bound memory in any long-running deployment.
10. **Stamp `model_version`/`threshold_version` into every report JSON** — cheap (a constant string next to the existing frozen thresholds) and closes a real auditability gap.
11. **Root-cause the 0-row `audit_logs` anomaly** before any clinical audit relies on that table.

---

## 17. Estimated Improvements After Optimization

| Metric | Current (measured) | After Stage-7 batch fix (measured micro-benchmark, extrapolated to full pipeline) | After batch-parallelized offline processing (measured core count) |
|---|---|---|---|
| WFDB (30 min) total pipeline latency | 5.4-7.5 s (avg 6.7 s) | Stage 7 (1.8 s avg) drops toward ~20-50 ms → **new total ≈ 3.0-3.9 s avg**, a **~45-55% reduction** | unchanged per-file, but 10 files run in the time 1 did (near-linear to 10 cores) |
| VitalPatch (~2.5 min) total pipeline latency | 0.58-1.15 s (avg 0.99 s) | Stage 7 is a smaller fraction here (fewer beats) but still meaningfully reduced; expect ~20-35% reduction | same parallelization benefit |
| Batch of 3,632 segments (real historical run) | 2,800.6 s (~46.7 min) | with Stage 7 fix alone: plausibly ~1,500-2,000 s | with 10-way process parallelism on top: plausibly ~150-300 s (order-of-magnitude) |
| MedGemma cold-start patient-facing stall | 75.6 s, blocks entire service (§9) | N/A (config/architecture fix, not algorithmic) — with pre-warming: 0 s in steady state; with async offload: 75.6 s absorbed without blocking other patients |
| MedGemma warm latency | 4.36 s median | unchanged (model-inference-bound, not a code-efficiency issue) — reducible only via a smaller model (`medgemma-student`, pending safety validation) or more/better GPU |

These are the two figures backed by direct measurement (Stage 7 micro-benchmark, real core count); the "after" pipeline-total and batch-total numbers are arithmetic extrapolations from those measurements, not independently re-measured end-to-end post-fix, and should be re-benchmarked once the changes are made.

---

## 18. Remaining Limitations (of this audit)

- MedGemma-Agent's FastAPI service was not actually load-tested end-to-end in this session (no live multi-patient concurrent request test against the full LangGraph path) — the blocking-event-loop finding is based on direct code inspection (confirmed call chain, confirmed no `--workers`, confirmed synchronous client) plus a live single-call latency measurement, not a live concurrent-patient reproduction. A follow-up load test (e.g. `locust`/`k6` against `/api/v1/vitals/snapshot` with 5-10 concurrent virtual patients) would directly confirm the predicted backlog behavior and quantify queueing delay.
- Database-unavailable and guardrail-rejection failure modes were verified by code inspection, not live-triggered (would require intentionally corrupting/locking the SQLite file or crafting a rejection-triggering payload against a running MedGemma-Agent instance).
- The `abg/interpreter.py` path (511 lines, the largest single file in MedGemma-Agent) was not read in full during this audit and deserves a dedicated pass before production use.
- Post-fix pipeline totals in §17 are extrapolated from the verified Stage-7 micro-benchmark, not independently re-measured after actually applying the code change.
- This audit did not evaluate model clinical accuracy/validity (sensitivity/specificity, calibration) beyond what the repo's own `docs/thesis/BEAT_CLASSIFICATION_SUMMARY.md`-derived confidence notes already state in the report JSON — that is a distinct clinical-validation exercise from the systems/performance audit requested here.
