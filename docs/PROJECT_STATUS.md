# ECG Pipeline — Project Status

Audit date: 2026-07-27. Documentation-only pass — no pipeline, model, threshold,
or AAMI-scheme code was modified to produce this report. Every claim below
cites the on-disk artifact it came from. Anything I could not verify on disk
is marked **[UNVERIFIED]** rather than assumed. Default stance: **NOT READY**
unless a specific claim is backed by evidence stated below.

---

## PART 1 — WHAT WORKS (with evidence)

### 1.1 Beat classifier — real, reproducible, held-out numbers

Source: `docs/archive/ABLATION_REPORT.md`, "Production baseline" section
(lines 17–47), command shown is `eval-classifier --model
models/five_class_xgb.json --split-set ds2`. DS2 is MITDB, held out at the
patient level, never touched during training/tuning.

| Class | Sensitivity | Precision | F1 | Support (beats) |
|---|---|---|---|---|
| N (normal) | 0.966 | 0.965 | **0.966** | 40,711 |
| S (supraventricular) | 0.140 | 0.165 | **0.152** | 1,795 |
| V (ventricular) | 0.907 | 0.765 | **0.830** | 3,005 |
| F (fusion) | 0.003 | 0.083 | **0.005** | 363 |
| Q (unknown) | 0.000 | 0.000 | **0.000** | 7 |

Macro-F1: **0.3906**. Accuracy: 0.9224 (explicitly *not* the success metric
per the report's own protocol — accuracy is dominated by the 89%-majority N
class).

**Honest read:** N and V are reliable (F1 0.97 / 0.83). S and F are not — S
recall is 13%, F recall is 0.5%, and 62.8% of true F beats get called S
(`ABLATION_REPORT.md` line 47). The stated reason, cross-checked against the
report's feature-family ablation (lines 458–594): S's dominant failure is
*prematurity detection*, not S/V shape confusion (67.6% of true S beats are
missed as N entirely — a single-lead device without P-wave/PR-interval
information structurally struggles to flag "this beat is premature" at all).
F is data-starved (403 total F beats in DS1, 363 concentrated in one MITDB
record) — not a modeling failure, a data-scarcity ceiling. Q has only 7 DS2
examples — a real number, but not enough support to claim anything about
Q reliability at all.

**A two-stage classifier (timing-gate + morphology-discriminator) was built
and extensively tested to fix S specifically** (`ABLATION_REPORT.md` lines
619–983, five full versions v1–v5, ~10 model artifacts in
`ecg_pipeline/models/`). Result: **CLOSED, genuinely negative** — every
version that improved S regressed V beyond the pre-registered safety margin,
and the result held after the validation split was independently enlarged
2.5x to rule out a sampling artifact (line 937–976). **Production remains
`five_class_xgb.json`, unmodified**, per the report's own summary table
(lines 986–997).

**Training/eval methodology note (verified by grep, `ecg_pipeline_tools.py`
lines 521, 570–586):** DS1/DS2 training and evaluation use ground-truth
`.atr` expert annotation sample positions (`wfdb.rdann`) for R-peak location,
**not** the runtime XQRS detector. This means the F1 numbers above are
**not** affected by the R-peak/SQI bugs described in Part 2 — those bugs only
affect live inference on new, unlabeled recordings, not this evaluation.

### 1.2 End-to-end pipeline — real batch manifests

Sources: `data/reports/vitalpatch_run_manifest.csv` (3,570 rows, computed
directly this audit) and `data/reports/prorhythm_run_manifest.csv` (18 rows).
Neither file is tracked in git — both are local run artifacts, generated
2026-07-25/26 per file mtime.

**VitalPatch (the actual deployment source), 3,570 segments across 6 patients:**

| Outcome | Count | % |
|---|---|---|
| Assessable | 2,795 | 78.3% |
| NOT_ASSESSABLE (insufficient signal survived quality gating) | 705 | 19.7% |
| Hard parse failure (see Part 2.3) | 70 | 2.0% |

| Risk level | Count | % |
|---|---|---|
| LOW | 1,832 | 51.3% |
| MEDIUM | 159 | 4.5% |
| HIGH | 527 | 14.8% |
| CRITICAL | 277 | 7.8% |
| NOT_ASSESSABLE | 705 | 19.7% |
| (blank, parse error) | 70 | 2.0% |

**SeNSiO/prorhythm, 18 segments, single device (`FD:DC:A1:9C:48:40`):**
assessable 13/18 (72.2%), NOT_ASSESSABLE 5/18 (27.8%). Risk: LOW 8, HIGH 3,
CRITICAL 2, NOT_ASSESSABLE 5. **Sample size is thin (n=18, one device) — not
enough to generalize a SeNSiO-specific quality rate with confidence.**

### 1.3 MedGemma integration — verified live, not just fallback

Two independent paths exist in `agent_bridge.py` (docstring, lines 1–28):

- **`report` path** (used by `report_ui.py`/`demo_stream.py`, production
  path): talks directly to Ollama via `call_medgemma`, no dependency on the
  separate MedGemma-Agent FastAPI app. `medgemma_status` in the manifest
  proves the LLM actually ran and was **accepted** (not just attempted) for
  **2,413 of 3,570 VitalPatch rows (67.6%)** and **11 of 18 SeNSiO rows
  (61%)**. `UNAVAILABLE_FALLBACK` fired for 105 rows (2.9%) — proof the
  degrade-gracefully path is real and was actually exercised, not just
  theoretical. `SKIPPED_CRITICAL` fired for 277 rows (7.8%, matches the
  CRITICAL risk-level count exactly) — the CRITICAL-bypass safety rule is
  confirmed active in real runs.
- **`push` path**: pairs real ECG risk output with the MedGemma-Agent
  app's own vitals schema, but the vitals values are explicitly
  **simulated**, borrowed from `MedGemma-Agent/vitals/test_cases.yaml`
  (`agent_bridge.py` lines 23–27, self-documented as an open question, not
  production-ready).

**Right now, this session (2026-07-27):** I started `report_ui.py` live,
confirmed the landing page (HTTP 200), POSTed a synthetic AFib replay, and
confirmed `/api/state/<run_id>` returned real `beat_summary`/`risk_level`
JSON mid-replay. I did not wait for that specific run to reach the MedGemma
stage before shutting the server down, so this session's own check does not
by itself reprove live MedGemma — that proof is the 2026-07-26 manifest
above. Separately: the `ollama` CLI on this machine (`~/bin/ollama`) is
currently broken (`Not Found` error on invocation) even though the model
blobs exist locally (`~/.ollama/models/{medgemma,medgemma-student}`, 3.5GB
combined) — so live MedGemma availability is **intermittent on this
machine**, consistent with the batch manifest's own 2.9% fallback rate.

### 1.4 Report/demo/UI — demonstrably functional

- `ecg_pipeline/test_pipeline_synthetic.py`: **ran live in this audit,
  5/5 scenarios PASS** (NORMAL, PVC_BURDEN, VT_RUN, AFIB_LIKE, NOISY) against
  known injected ground truth.
- `ecg_pipeline/report_ui.py` live replay server: **confirmed live in this
  audit** — landing page 200, `/start` → valid `run_id` redirect,
  `/api/state/<run_id>` streaming real risk/beat data.
- 8 saved, real `ECGPipeline` output reports on disk:
  `ecg_pipeline/example_reports/` (7 WFDB + 1 VitalPatch, dated 2026-07-24).
- Prior-session screenshots verifying the replay UI (deciding-rule wording,
  AFib/LOW consistency, legend, highlight labeling) exist in this session's
  and the prior session's scratch directories — these are ephemeral
  verification artifacts, not committed to the repo, so cite them as
  session evidence, not permanent proof.

### 1.5 Bug fixes landed (commits, with before/after numbers from the commit messages themselves)

| Commit | Fix | Before → After (measured) |
|---|---|---|
| `3add727` | WFDB SQI over-rejection (baseline/DC scored on raw signal, not detrended) | WFDB rejection 48–100% → 7–23% across 7 test records. VitalPatch rejection 26.7% → 13.3%, beats analyzed 99 → 122 (not a regression). Synthetic degenerate signals still 100% rejected (gate is not a pass-through). |
| `2420ecc` | R-peak beat-level over-culling (snap XQRS peaks to true local max) | WFDB retention (analyzed/ground-truth beats) 8–70% → 84–131% across 7 records. VitalPatch unaffected/improved: 189 detected, 185 analyzed (up from 122). |
| `414c9fc` | SeNSiO 100Hz SQI collapse; MedGemma false "exceeds" claims | SeNSiO went from 0 beats/100% rejection to functioning (see 1.2 manifest). MedGemma narrative now pre-computes every rule verdict in Python and strips any surviving false "exceeds" claim from the model's free text. |

All three commits explicitly state the classifier was **not** retrained
through the fixed detection path — this is fine, not a gap, because (per
1.1) classifier training/eval never used XQRS detection in the first place.

---

## PART 2 — WHAT'S STILL BROKEN OR UNFINISHED

| # | Issue | Measured or suspected | Severity / blast radius |
|---|---|---|---|
| 1 | **WFDB R-peak over-detection, unfixed** | Measured (`2420ecc` commit message): records 103/111 show 1.5–1.9x true beat count. Root cause investigated and ruled out (not a units/scale bug); leading hypothesis is XQRS adaptive-rate "hunting" on noisy records, unconfirmed. | Low for actual deployment — VitalPatch (the real device path) is explicitly confirmed unaffected/improved by the same fix, and classifier eval never touches XQRS (1.1). High for benchmark credibility if this pipeline is ever compared against public WFDB records for anything beyond training/eval. |
| 2 | **NOT_ASSESSABLE rate ~20% VitalPatch / ~28% SeNSiO** | Measured (1.2). **Attribution is not fully verified**: some fraction is certainly genuine (insufficient signal survived quality gating, by design — the `MIN_BEATS_FOR_ASSESSMENT` guard exists specifically to make this an honest state, not a bug). I could not establish on disk what fraction, if any, is a residual symptom of pre-fix beat loss vs. genuinely low-quality real-world segments (short segments, disconnection, motion artifact). **[UNVERIFIED]** as a precise split — do not read either "it's all genuine" or "it's a bug" into this number without a dedicated investigation. |
| 3 | **`'-'` sentinel parse failures** | Measured: 70/3,570 VitalPatch segments (2.0%), across all 6 patients, hard-fail with `ValueError: could not convert string to float: '-'` (`batch_vitalpatch_report.py` line 99). Root cause identified by reading `parse_vitalpatch_ecg` (`ecg_pipeline_core.py` lines 281–301): raw source CSVs contain a literal `'-'` string cell that `pd.isna()` does not catch (it's not NaN), so `.astype(np.float64)` crashes on the whole file. **Not fixed.** | These rows silently drop out of the manifest as `error` rows rather than being scored NOT_ASSESSABLE or LOW — a real data-loss bug, though it fails loudly (exception, not a wrong number) rather than producing a wrong clinical output. |
| 4 | **NEWS2/qSOFA vitals never wired** | Measured by grep: `score_recording()`/`RiskReport` accept `news2_score`/`qsofa_score` (`ecg_pipeline_core.py` lines 1934–1992), and `agent_bridge.py` reads them into the rule trace — but **no caller anywhere in the codebase** (`report_ui.py`, `demo_stream.py`, both batch scripts) ever passes a real value in. The rules are permanently `evaluated: False`. This is honestly self-disclosed in the report's own `known_limitations` text (`agent_bridge.py` line 415: "NEWS2/qSOFA vitals pairing is not wired into this ECG-only bridge — safety_overrides above are reported as not-evaluated rather than fabricated"). | Part of the documented risk cascade is inert in every real run to date. Not a silent-wrong-output risk (it's disclosed, not fabricated) but it means any claim of "NEWS2/qSOFA-aware risk scoring" would be false as shipped. |
| 5 | **Encoder / CNN-transformer models orphaned** | Measured: `models/ecg_encoder.pt` (36KB, self-supervised pretraining, `main_train_encoder` in `ecg_pipeline_tools.py`, 15 epochs / 6,498 unlabeled windows per its own `meta.json`) is never loaded by `ECGPipeline`, `agent_bridge.py`, or any inference path — only referenced as a CLI default *output* path for training it. `models/five_class_cnntx_v1.pt` and `_v2_nosddb.pt` (504KB each) have **zero references anywhere in the current `.py` codebase** — no training script, no loader found. **[UNVERIFIED]** what these were for; treat as dead/orphaned artifacts, not part of the production system. | None for current production (nothing depends on them), but they represent unfinished/abandoned work that should either be documented or removed. |
| 6 | **Display/consistency bugs (this session)** | Fixed this session, verified by re-render + curl: deciding-rule text no longer prints `None`/`null`/misleading `FIRED`; AFib-suspected vs LOW risk wording now states the real burden-vs-threshold mechanism instead of being self-contradictory; rhythm-finding highlight bands are now labeled on-plot; legend clipping fixed (`bbox_inches="tight"`, `ncol` capped). **Still open, found but not fixed (out of scope for that task):** `RhythmContextEngine._afib_suspected` (`ecg_pipeline_core.py` ~lines 1866–1894) indexes AFib findings into `clean_rr` (a `None`-filtered RR list) rather than the original `beats` list — a latent index-mapping bug that would only manifest when RR-flagged beats exist upstream of a flagged window. Measured `n_rr_flagged=0` on the one recording tested, so no observed misalignment yet — **not proven absent in general.** | Cosmetic/wording issues: none now (fixed). Latent indexing bug: unknown severity, not reproduced, needs a targeted test case with RR-flagged beats before it can be called safe. |
| 7 | **Non-code blocker: `main` branch on GitHub has none of this work** | Measured directly (`git ls-remote --heads origin`): `origin/main` = `840d8e7`, the very first "Initial commit" only. All subsequent work — both SQI/R-peak fixes, the full ablation report, the batching/report pipeline, README/PROJECT_OVERVIEW — exists only on `mod_1`, which **is** pushed and in sync with `origin/mod_1` (verified: `git rev-list --left-right --count mod_1...origin/mod_1` → `0 0`). Two stale local branches (`fix-rpeak-overdetection`, `fix-sqi-baseline`) are unpushed but their commits are already contained in `mod_1`'s history (`git branch --contains 2420ecc` lists `mod_1`) — no work is actually at risk of loss. | The team's shared "main" is effectively a stub. Nothing is lost, but nothing real is merged either — this needs a PR/merge decision, not a data-recovery effort. |
| 8 | **`MedGemma-Agent` submodule dirty, diverged from its own `dev` branch** | Measured: `git status --short --branch` inside the submodule shows tracking `origin/dev` with 65 files / ~7,900 line diff, uncommitted. Not investigated further — this is a separate repo/component, out of scope for this ECG-pipeline audit, but flagged since it shows in top-level `git status` as `m MedGemma-Agent`. | Unknown — **[UNVERIFIED]**, needs its own audit if it matters for deployment. |

---

## PART 3 — EDGE-DEPLOYMENT READINESS

**Does it run on Jetson-class hardware?** **[UNVERIFIED — no evidence found.]**
Grepped the full `ecg_pipeline/*.py` tree and `docs/` for `Jetson`,
`TensorRT`, `ONNX` — zero hits. No cross-compilation, quantization, or
on-device benchmark artifacts exist in this repo. This is consistent with
[[project_vitalpatch_scope]] (Jetson/quantization/distillation is explicitly
a teammate's separate scope) — but it does mean nothing in *this* repo
demonstrates the pipeline has ever run on target hardware.

**Latency/throughput measured?** **[UNVERIFIED — no evidence found.]** No
profiling script or timing manifest exists. `demo_stream.py`'s `--speed`
flag is a simulated-real-time replay multiplier for the demo UI, not a
hardware throughput benchmark — do not conflate the two.

**Memory/model footprint (measured, on disk):**
- Production classifier `five_class_xgb.json`: **15MB** on disk (the only
  model file actually loaded by `ECGPipeline`'s default inference path).
  In-memory footprint for an XGBoost tree ensemble is the same order of
  magnitude.
- `ecg_pipeline/models/` also contains ~10 two-stage-experiment artifacts
  (10–16MB each, ~150MB+ combined) and the orphaned encoder/CNN-transformer
  files (Part 2.5) — none of these are loaded by production code, but if the
  whole `models/` directory were shipped as-is to an edge device, that's
  unnecessary weight that should be pruned first.
- MedGemma: 3.5GB combined local blob storage for `medgemma` +
  `medgemma-student` (`~/.ollama/models`). This is almost certainly too
  large for a Jetson Nano's typical RAM/storage budget without the
  quantization/distillation work that's explicitly out of this repo's scope.

**Does the WFDB/SeNSiO beat-loss bug affect the deployment data path?**
Mostly **no**. VitalPatch is the actual deployment source, and both fix
commits explicitly measured VitalPatch as unaffected-or-improved. The
classifier's own accuracy numbers are unaffected because training/eval never
used XQRS detection (1.1). The one confirmed-unfixed defect (WFDB record
103/111 over-detection) only matters if raw public WFDB signals are ever
ingested live at the edge, which is not the deployment scenario — it mainly
affects benchmark comparability, not the real device path.

**Is MedGemma required at the edge, or optional?** **Optional, and this is
a confirmed safety property, not just a design intent.** `render_narrative()`
always produces a complete `_deterministic_narrative` when Ollama is
unreachable or its output is rejected (`agent_bridge.py` lines 644–655), and
this path was actually exercised 105 times in the real VitalPatch batch run
(1.3). The risk-level decision itself (`score_recording`/`merge_decision`)
never depends on MedGemma — MedGemma only narrates an already-decided value
and can escalate, never downgrade or invent (per the code's own
"escalate-only" design, confirmed in `agent_bridge.py`'s module docstring).
**The classifier + cascade path can run edge-only; MedGemma is a
cloud/Ollama-dependent narrative layer that degrades gracefully without it.**

**Can a recording silently under-process and report a false LOW?**
This is the central safety question, and I checked it directly rather than
assuming. `agent_bridge.py` lines 383–385:
```python
n_beats_analyzed = result.n_beats_accepted
assessable = n_beats_analyzed >= MIN_BEATS_FOR_ASSESSMENT   # = 5
risk_level = result.risk_report.alert_level if assessable else "NOT_ASSESSABLE"
```
**Confirmed: if fewer than 5 beats survive quality gating, the risk level is
force-overridden to `NOT_ASSESSABLE` regardless of what the cascade
computed** — this specific "empty/near-empty signal silently reads as clean
LOW" failure mode is guarded against, and the guard is exercised at scale in
the real manifest (705 VitalPatch rows, 19.7%). This is a genuine, verified
safety property — say so plainly rather than defaulting to alarmism.

What I did **not** verify, and cannot rule out from what's on disk: a
recording that clears the 5-beat floor but where imperfect SQI gating on a
noisy segment still lets through beats that get spuriously but confidently
labeled — the floor guards against too-little-data, not against
plausible-looking-but-wrong data. This is a real residual gap, not a
disproven one.

### Per-component readiness verdict

| Component | Verdict | Blocker |
|---|---|---|
| Beat classifier | **NEEDS-WORK** | Real, reproducible, disclosed numbers exist (N/V reliable, S/F genuinely weak, Q untested with n=7). Not a blocker to ship as a *disclosed, clinician-reviewed screening flag* — but any claim of uniform reliability across all 5 classes would be false. Two-stage fix attempt closed negative; no further improvement path currently in progress. |
| Pipeline (parse → SQI → detect → classify → cascade) | **NEEDS-WORK** | VitalPatch path is measured and mostly functioning (78.3% assessable) with two real bug fixes landed and verified. Three unresolved, measured defects directly touch the deployment path: the 2.0% hard parse-crash on `'-'` sentinel values, unresolved NOT_ASSESSABLE attribution, and permanently-inert NEWS2/qSOFA rules. None currently produce a silently-wrong LOW as far as verified (the 5-beat floor is real and exercised) — but the SQI-noise residual gap above is unproven-safe, not proven-safe. |
| Report/narrative layer | **NEEDS-WORK, closer to READY** | Deterministic fallback is proven correct and exercised at scale; live MedGemma is proven to work but is intermittent (Ollama down on this dev machine right now); false-claim filtering is real and tested. Main gap is that the richer MedGemma-Agent vitals-pairing path (`push`) still uses simulated, not real, vitals. |
| Demo/replay UI | **READY for its stated purpose (demo/visualization only)** | Confirmed live and functional this session; explicitly documented as read-only/non-diagnostic tooling, not a clinical interface — do not read this verdict as "clinical UI ready." |
| Edge deployment (Jetson-class) | **NOT READY** | No evidence anywhere in this repo that the pipeline has run on target hardware, no latency/throughput numbers, no quantization/export path, MedGemma's 3.5GB footprint is un-sized for the target. This matches the known scope split (Jetson work belongs to a teammate) but means, factually, edge deployment readiness cannot be claimed from this repo's contents today. |

**No blanket "ready" verdict is given, per the task's instruction — the
system is genuinely partway there: the classifier and VitalPatch pipeline
path are measured and mostly working with disclosed weaknesses, the report
layer degrades safely, but edge-hardware readiness specifically is
unverified/not-ready, and three concrete pipeline defects remain unfixed on
the deployment data path.**

---

## PART 4 — PRIORITIZED FIX LIST TO REACH EDGE DEPLOYMENT

Ordered highest-impact/safety-first. Each item: action + why.

1. **Fix the `'-'` sentinel parse crash in `parse_vitalpatch_ecg`.**
   Currently 2.0% of every real VitalPatch batch silently drops out as a
   hard error instead of being scored. This is the only issue in this list
   that causes outright *data loss* on the real deployment source — highest
   priority because it's small, root-caused, and safety-adjacent (a
   recording that should have been assessed instead vanishes from the
   manifest with no risk level at all).

2. **Resolve the non-code blocker: get `mod_1` merged/pushed into `main` on
   GitHub, or make `mod_1` the shared branch of record.** Right now the
   team's actual shared branch is a one-commit stub; every fix and every
   piece of evidence in this report lives only on a branch that nobody has
   merged. This doesn't block the code from working, but it blocks anyone
   else from building on this work safely. (No commits are at risk — this
   is a merge decision, not a recovery task.)

3. **Investigate the true composition of the ~20% VitalPatch NOT_ASSESSABLE
   rate** (genuine low-quality signal vs. any residual beat-loss symptom).
   This number directly determines how often the system will legitimately
   say "I can't tell you anything" in the field — worth knowing precisely
   before committing to it as an expected operating rate.

4. **Either wire real NEWS2/qSOFA vitals in, or explicitly scope them out of
   the near-term deployment plan.** They're currently dead code that's
   honestly disclosed as dead — the risk is someone assuming they're active
   because the fields exist in the schema. A short doc note or a real
   integration both resolve this; leaving it silently unresolved does not.

5. **Confirm/rule out the WFDB R-peak over-detection root cause**, or at
   minimum document clearly that it's scoped to non-VitalPatch, raw-signal
   ingestion paths only, so nobody mistakes it for a VitalPatch-path risk
   later.

6. **Write and run a targeted test for the latent `_afib_suspected`
   index-mapping bug** (`clean_rr` vs `beats` indexing) with a recording
   that actually has RR-flagged beats upstream of a flagged window — this
   session's fix only confirmed it's *not currently triggered*, not that
   it's safe in general.

7. **Get real latency/throughput/memory numbers on the actual target
   hardware**, or explicitly confirm edge deployment is out of scope for
   this repo (per the current teammate-owned Jetson scope) so the project's
   own documentation doesn't imply readiness that hasn't been measured.

8. **Prune or document the orphaned model artifacts** (`ecg_encoder.pt`,
   `five_class_cnntx_v1/v2.pt`, the ~10 two-stage experiment files) —
   optional polish, but they add ~150MB+ of unexplained weight to
   `models/` and would confuse anyone packaging this for an edge image.

9. **Resolve the `MedGemma-Agent` submodule's dirty/diverged state**
   (65 files, ~7,900-line diff vs. its own `dev` branch) — flagged but not
   investigated in this audit; needs its own look before it's relied on.

10. **If real-vitals pairing is wanted for the `push`/MedGemma-Agent
    integration path**, replace the simulated `vitals/test_cases.yaml`
    values with a real source — currently explicitly a demo-only path by
    the code's own admission.

---

*Every specific claim above cites a file path, command, or a live check I
ran during this audit (2026-07-27). Items marked [UNVERIFIED] are things I
looked for and did not find evidence of — treat them as open questions, not
as either "working" or "broken."*
