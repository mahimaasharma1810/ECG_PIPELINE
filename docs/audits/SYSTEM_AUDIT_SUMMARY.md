# SYSTEM_AUDIT_SUMMARY.md — Quick-Reference Tables

Companion to `SYSTEM_AUDIT.md` (full evidence, file:line citations, methodology). Everything below is measured or code-verified, not assumed — see the full report for proof.

---

## 0. Already Fixed (verified)

| Fix | Verified how |
|---|---|
| ✅ Beat classifier batching (153× speedup) | Ran real records before/after: identical risk labels, ~2× faster end to end |
| ✅ Missing/empty ECG file no longer crashes | Tested live with a missing file and an empty file — clean error message, no traceback |
| ✅ MedGemma timeout raised (10s → 90s) | Was shorter than the measured 75.6s real cold-start time; now has margin |
| 🟡 "Insufficient data" mislabeled as "LOW" | Turned out narrower than first reported — see note below |

**Note on the last one**: the actual saved report (what `run_inference.py` writes to disk) was already correctly saying `"NOT_ASSESSABLE"`, not `"LOW"` — that part of the original finding was wrong. The real, smaller bug is that two files in the older `ecg_pipeline/` folder (`multimodal_batch.py`, `agent_bridge.py`) read the risk label a different, more direct way that skips that correction. Fixed the current package (`ecg_inference`) to expose an explicit flag for this; the older duplicate folder still needs the same fix mirrored into it if that path is still in use.

---

## 1. One-Line Verdict

| Area | Status |
|---|---|
| ECG signal pipeline (Stages 1-8) | 🟢 Fast, safe, well-designed — one easy fix roughly halves its time |
| MedGemma (LLM) integration | 🔴 Biggest risk — blocks the whole service, times out silently |
| Storage | 🟡 Bloated but easy to clean (mostly one dead folder) |
| Logging / audit trail | 🔴 Missing in practice, and the one audit table that exists isn't recording anything |
| Failure handling | 🟡 Good for clinical edge cases, crashes on basic bad input |
| Clinical safety design | 🟢 Genuinely solid — CRITICAL never depends on the LLM, LLM can only raise risk, never lower it |

---

## 2. ECG Pipeline — Per-Stage Latency (measured, 16 real files)

| Stage | Avg time | Min | Max | % of total | Verdict |
|---|---|---|---|---|---|
| 1. Load/parse ECG | 106 ms | 26 ms | 206 ms | 2.8% | fine |
| 2. Signal quality gate | 648 ms | 51 ms | 1,252 ms | 16.8% | fine |
| 3. Resample to 125Hz | 11 ms | 0 ms | 25 ms | 0.3% | fine |
| 4. Filtering (baseline/notch/bandpass) | 144 ms | 12 ms | 275 ms | 3.7% | fine |
| 5. R-peak detection + beat cutting | 682 ms | 69 ms | 1,311 ms | 17.7% | fine |
| 6. Feature extraction | 456 ms | 65 ms | 943 ms | 11.9% | fine |
| **7. Beat classification (AI model)** | **1,801 ms** | 341 ms | 3,798 ms | **46.8%** | 🔴 **fix this** |
| 8. Risk scoring | 0.1 ms | — | — | ~0% | fine |
| **TOTAL** | **3,846 ms** | 577 ms | 7,545 ms | 100% | — |

**The one fix that matters most**: Stage 7 classifies beats one at a time in a loop. Feeding them all to the AI model at once instead measured **153× faster** (3,026 ms → 20 ms for 2,000 beats). No accuracy change, no retraining — just a coding-pattern fix.

---

## 3. MedGemma (LLM) Latency — measured on real GPU

| Scenario | Time | Notes |
|---|---|---|
| **Cold start** (model not loaded) | **75.6 sec** | Mostly just loading the model into GPU memory (71.7s of the 75.6s) |
| **Warm** (model already loaded) | **4.4 sec avg** | 4.2 – 4.5 sec range; ~3s is the AI actually "thinking," rest is overhead |
| Configured timeout (ECG pipeline side) | 10 sec | 🔴 **7.5× shorter than the real 75.6s cold start** — will time out and silently skip the AI almost every time the model isn't already warm |
| Configured timeout (Agent service side) | 120 sec | Long enough to survive a cold start, but **freezes the entire server** for that whole time |
| GPU memory used | 3.8 GB / 11.3 GB | Plenty of headroom |
| Concurrent requests supported | **Effectively 1** | Server config limits it to one at a time, and the API code blocks on it too |

---

## 4. Storage — Where the Space Goes

| Location | Size | What it is | Action |
|---|---|---|---|
| `MedGemma-Agent/venv_windows_broken_backup/` | **4.4 GB** | Dead, unused backup folder | 🔴 Delete |
| `MedGemma-Agent/venv/` | 1.6 GB | Normal Python environment | Keep |
| `data/raw/public/` (research datasets) | 1.3 GB | MIT-BIH etc. benchmark ECG data | Keep (compress if space matters) |
| `data/raw/vitalpatch/` | 639 MB | Real device recordings | Keep |
| `data/vitals_downloads/` | 384 MB | Downloaded vitals per patch | Keep |
| `data/reports/` | 71 MB | Generated reports (JSON/MD/PDF) | Grows forever, no cleanup policy |
| Duplicate AI model file | 15.75 MB × 2 | Identical copy in two folders | 🟡 Keep one, delete the other |
| Third, different model copy (`training/`) | 16.3 MB | Old/experimental version | 🟡 Verify still needed |
| Stray backup file (`.bak_pre_rpeak_fix`) | small | Leftover from a past fix | Delete |

**Total repo footprint**: ~8.4 GB, of which **~4.4 GB (over half) is one dead folder.**

---

## 5. Memory & CPU (measured)

| Metric | Value |
|---|---|
| Peak memory per ECG file processed | 469 MB |
| CPU usage during processing | 383% (uses ~4 cores at once, even though it's "pinned" to 1) |
| GPU usage during ECG processing | 0% — GPU is only used by the LLM, never by the ECG math |
| GPU usage during MedGemma | 34% of VRAM, 0% between requests |
| Available hardware | 10 CPU cores, 125 GB RAM, 1 GPU (11 GB) — plenty of headroom, mostly unused |

**Slow memory leak risk**: one internal tracker (`TemporalRiskTracker`) keeps a growing record per patient forever with no cleanup — fine for a quick script, a real problem if the service runs for weeks with many patients.

---

## 6. Real-Time / Throughput

| Question | Answer |
|---|---|
| Can it keep up with one patient's live ECG stream? | **Yes, easily** — processes 147× faster than the data arrives |
| Can it keep up with many patients at once (ECG only)? | **Yes** — tens of patients, no problem |
| Can it keep up with many patients at once (with the AI/LLM step)? | **No** — only one patient's AI call can run at a time; everyone else waits |
| Where does the traffic jam start? | The moment a 2nd patient's request arrives while the 1st is waiting on the AI |

---

## 7. Failure Handling — What Happens When Things Go Wrong (tested live)

| Problem | What actually happens | OK? |
|---|---|---|
| ECG file doesn't exist | Program crashes with an error dump | 🔴 No |
| ECG file is empty/corrupted | Program crashes with an error dump | 🔴 No |
| ECG file has garbage (non-CSV) content | Clean error message, no crash | 🟢 Yes |
| Some bad values mixed into otherwise-good data | Automatically cleaned and handled | 🟢 Yes |
| Missing heart-rate/temperature vitals | Skipped safely, not guessed/faked | 🟢 Yes |
| Very poor signal quality | Reported without crashing — but currently labeled "LOW risk" instead of "couldn't assess," which is misleading | 🟡 Needs wording fix |
| Zero heartbeats detected | Handled without crashing | 🟢 Yes |
| AI (MedGemma) unreachable or too slow | Falls back to rule-based scoring, no crash | 🟢 Yes |
| AI returns garbled/invalid response | Falls back to rule-based scoring, no crash | 🟢 Yes |
| Database unavailable | Would crash with a raw server error | 🔴 No (not directly tested, confirmed by reading the code) |

---

## 8. Logging & Audit Trail

| Does it log... | Present? |
|---|---|
| Errors | 🟡 Only sort of — no real error log, just crash dumps or audit notes |
| Warnings | 🔴 No |
| Time taken per step | 🔴 No (had to be measured from scratch for this audit) |
| Which pipeline stage | 🟢 Yes |
| AI model version | 🔴 No |
| Threshold/rules version | 🔴 No |
| Patient ID | 🟢 Yes |
| Request/session ID | 🔴 No |
| Risk level | 🟢 Yes |

**Found a live bug**: the compliance audit table meant to record every patient interaction has **0 entries**, even though the database shows 6,760 patient events happened. The audit trail is not actually working right now.

---

## 9. Code Quality Snapshot

| Issue | Extent |
|---|---|
| Same logic duplicated across files | 3 separate copies of the core pipeline logic exist |
| Duplicated AI-call/error-handling code | Copy-pasted in 2 places in the Agent service |
| Largest single functions | ~140 lines (pipeline orchestrator), ~510 lines (one AI-interpreter file) |
| Hardcoded numbers that should be config | A handful, mostly minor |
| `print()` statements instead of real logging | 160 |

---

## 10. Clinical Safety Design — What's Actually Good

| Safety property | Verified |
|---|---|
| Same input always gives same output (reproducible) | 🟢 Yes |
| Critical alerts never depend on the AI being available | 🟢 Yes — hardcoded bypass |
| AI can only raise a risk level, never lower one | 🟢 Yes |
| Every risk decision traces back to a specific rule | 🟢 Yes |
| Report honestly says which AI predictions are unreliable | 🟢 Yes (e.g. one beat type is flagged "essentially unsolved, treat as noise") |
| Reports show which model/threshold version made the call | 🔴 No — this should be added |

---

## 11. Priority Fix List

| Priority | Issue | Effort to fix |
|---|---|---|
| 🔴 Critical | Audit log recording 0 rows — compliance trail broken | Investigate + small fix |
| 🔴 Critical | AI call freezes the entire server for other patients | Medium (async wrapper) |
| ✅ Fixed | ~~10-second AI timeout vs. real 75-second cold start~~ | Done — see §0 |
| ✅ Fixed | ~~Missing/empty files crash instead of erroring cleanly~~ | Done — see §0 |
| ✅ Fixed | ~~Beat classifier loop is 153× slower than it needs to be~~ | Done — see §0 |
| 🟠 High | No real logging, no model/threshold version on reports | Medium |
| 🟡 Partially fixed | "Insufficient data" mislabeled as "LOW risk" | Fixed in `ecg_inference`; same bug still open in the older `ecg_pipeline/` batch path — see §0 |
| 🟡 Medium | 4.4 GB dead folder + duplicate model files | Trivial (delete/dedupe) |
| 🟡 Medium | Batch processing runs one file at a time instead of parallel | Small, high payoff for batch speed |
| 🟢 Low | Stray backup files, minor magic numbers, print() cleanup | Low priority |

---

## 12. Expected Impact If the Top Fixes Are Applied

| Metric | Before | After (estimated from measurements) |
|---|---|---|
| ECG processing time (30-min recording) | 6.7 sec avg | ~3-4 sec avg (~50% faster) |
| Batch of 3,632 files | 46.7 minutes | Potentially 3-5 minutes (with parallel processing) |
| AI cold-start service freeze | Freezes everyone for 75.6 sec | Down to 0 sec with model pre-warming |
| Dead storage | 4.4 GB wasted | 0 GB (one deletion) |

For full details, evidence, and file-level citations behind every row above, see **`SYSTEM_AUDIT.md`**.
