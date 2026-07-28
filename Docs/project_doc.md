# ECG 5-Class Beat Classifier

AAMI 5-class (N/S/V/F/Q) ECG beat classifier. XGBoost on handcrafted
morphology + wavelet features. This README is the handover doc — what's
done, what's next, and where the work is currently stuck, waiting on a
decision.

## Where things are

*(CORRECTION, 2026-07-28: the layout below reflects this file's original
1844AC session. Since then, an inference-ready packaging pass moved
`ecg_pipeline_tools.py`, `_original_stages/`, and every non-production
model artifact — including `ecg_encoder.pt` — out of `ecg_pipeline/` and
into a new top-level `training/` directory. See the root `README.md` for
the current layout and `Docs/inference_ready.md` for the full move
rationale. Paths below are historical.)*

```
ecg_pipeline/
  ecg_pipeline_core.py    # runtime pipeline: ingest, filter, segment, features, classify, risk
  models/
    five_class_xgb.json          # PRODUCTION model — never overwrite this
    five_class_xgb.classes.json
training/
  ecg_pipeline_tools.py   # training/eval CLI: download-datasets, train-classifiers, eval-classifier
  model_artifacts/
    ecg_encoder.pt                # optional learned encoder, not used by the classifier below
  _original_stages/       # frozen historical reference implementation, kept for diffing
data/raw/public/          # datasets — NOT in git (too large), see "Getting the data" below
```

Run everything from `/home2/mahimakopalley/projects` as:
```
python -m training.ecg_pipeline_tools <subcommand> [args]
```

## Getting the data

```
python -m training.ecg_pipeline_tools download-datasets --all
```
MITDB, SVDB, INCART, LTAFDB, SDDB, challenge2017, CUDB. `icentia11k` is
~257GB — download separately (`download-icentia11k-full`) only if you
actually need it; it's a large-volume, lower-priority train-only source
and was found not worth using as-is (see below).

If disk quota is tight, route large downloads to a scratch/local disk and
symlink into `data/raw/public/<name>` — that's what this project did on
its original machine (`/ssd_scratch/...`), but that path is
machine-specific and won't exist if you're setting up fresh elsewhere.

## What's been done

**Production model** (`five_class_xgb.json`): trained on MITDB DS1_TRAIN
(12 records) + SVDB, Q-class dropped, ROS (1:3 floor) + balanced class
weights. Retrained 2026-07-20 on the enlarged DS1_VAL split (see below) —
same recipe as before, DS2 (held-out, never trained/tuned on) numbers are
statistically unchanged:

| Class | Sensitivity | Precision | F1 |
|---|---|---|---|
| N | 0.969 | 0.964 | 0.967 |
| S | 0.143 | 0.167 | 0.154 |
| V | 0.903 | 0.797 | **0.847** |
| F | 0.000 | 0.000 | 0.000 |

Macro-F1: 0.3935. **S and F are the known weak points** — this whole body
of work exists to try to fix that without regressing N/V.

**Reproducibility fixed:** XGBoost's `random_state` alone didn't guarantee
identical results across sessions (thread-count-dependent floating point
in split-finding). Fixed by pinning `n_jobs=1` in
`FiveClassBeatClassifier.fit()` — verified via two back-to-back identical
runs. The numbers above are the pinned, reproducible reference.

**Experiments tried and discarded** (each is a real run with a full
DS2 confusion matrix — not guesses):
- **+LTAFDB/SDDB train-only enrichment** — discarded. LTAFDB alone added
  8.5M beats (40-80x DS1's native size), diluted rather than helped the
  rare classes, made S and F *worse*.
- **+7 local-rhythm-context ("timing") features** (rolling RR-ratio,
  prematurity score, compensatory-pause flag, etc.) — discarded. Improved
  S (F1 0.139→0.182) but regressed V (F1 0.826→0.775): timing features
  cause the model to confuse premature S beats with premature V beats.
  Removing just the single dominant timing feature made it *worse*, not
  better — the entanglement isn't one bad feature, timing as a family
  blurs the S/V boundary.
- **R-peak amplitude (`r_amp`) feature** — found as dead code (computed,
  never returned by `_morphological_features`), fixed as a bug regardless,
  but tested and found to be noise/mildly negative as a feature. Not used.

**The decisive finding — go/no-go test for a two-stage classifier:** a
feature-family ablation (morphology-only vs timing-only vs combined) shows
morphology alone separates S from V *well* (S→V confusion rate 0.169, V F1
0.866 — better than the flat production recipe). Timing alone is much
worse at this (S→V 0.469). So the flat model's S/V confusion isn't a
morphology problem, it's timing actively degrading otherwise-good
morphology signal. Meanwhile, morphology-only's dominant S error is
**S→N** (67.6% of true S beats missed entirely, only 16.9% confused with
V) — a **detection** failure, not a **discrimination** failure.

**Conclusion: GREEN LIGHT for a two-stage classifier.**
- Stage 1 (gate): binary Normal-vs-Abnormal, *may* use timing features
  (timing is good at flagging "this beat looks premature/ectopic" — that's
  exactly the S→N gap).
  Stage 2 (discriminator): S vs V vs F among whatever Stage 1 flags,
  **morphology-only** (that's what already works well).

Full detail, every command run, every confusion matrix: this repo's
detailed ablation log was kept locally on the machine this work was done
on (not included in this git repo — see "What's not in this repo" below).
If you have access to that machine, it's at `Docs/archive/ABLATION_REPORT.md`.

## Two-stage classifier — closed, genuinely negative (2026-07-20)

**The two-stage classifier was built and tested across five versions
(v1–v5) — none is promotable, and this is now a settled result, not an
open question.** Every version was evaluated chained (Stage 1 → Stage 2,
on DS2, never tuned against it) per the honest-result rule below. The
pattern across all five: Stage 1 reliably catches S beats the flat model
misses, but no Stage-2 design has been able to do that *without* also
regressing V:

- Stage 2 restricted to S/V/F only (no Normal output) recovers S but lets
  Normal beats that leak past Stage 1 get force-mislabeled as V, regressing
  V's precision.
- Giving Stage 2 a Normal output fixes that regression, but Stage 2 then
  re-derives "is this abnormal?" from morphology alone — the same signal
  that couldn't catch these S beats in the first place — and most of the
  S gain disappears.
- Retuning Stage 1's operating threshold, and feeding Stage 2 Stage 1's own
  confidence score, both help partially but don't resolve it: a meaningful
  share of Stage 1's false-positive Normal beats are ones it is
  *confidently* wrong about, and no downstream mechanism can undo a
  confident upstream error.

**The open question — whether that last finding was a fundamental limit
or an artifact of the original 4-patient validation carve-out — is now
resolved.** `DS1_VAL` was doubled to 10 patient records (`splits.py`'s
`_DS1_VAL_RECORDS` in `ecg_pipeline_tools.py`) and a fresh model (v5,
same recipe as v1) retrained from scratch on the correspondingly-shrunk
`DS1_TRAIN`. The larger, more patient-diverse split reproduced the
original split's steep S-recall-vs-threshold trade-off almost exactly
(not DS2's gentler one) — confirming the trade-off is a real property of
the DS1 patient population, not small-sample noise. At v5's own selected
threshold, V still regresses beyond the zero-tolerance margin (V F1
0.866→0.789, V precision 0.828→0.690) even though S improves (S F1
0.160→0.383). **Verdict: not promotable, and no further iteration on
this Stage-1/Stage-2 architecture is expected to change that.**

Full methodology, every command, every confusion matrix, and the specific
numbers behind each version (including the closing v5 analysis): 
`Docs/archive/ABLATION_REPORT.md`, kept locally per "What's not in this
repo" below.

## Where this goes next

Fixing S/F detection needs a different approach than two-stage gating on
this feature set — not attempted here.

## MedGemma-Agent integration and student distillation (2026-07-20/21)

The production ECG classifier above is now wired into the broader
VitalPatch system end-to-end, and a distilled sub-1B student model has
been trained, quantized, and deployed:

1. **`ecg_pipeline/agent_bridge.py`** — runs a real VitalPatch ECG segment
   through `ECGPipeline`, converts the resulting `RiskReport` to an
   `ECGRiskSummary` (aggregated PVC/PAC/AFib burden, VT runs, HRV
   suppression — not raw beats), and POSTs it alongside a vitals snapshot
   to MedGemma-Agent's live `/api/v1/vitals/snapshot`. Verified end-to-end
   against a real recording: the LLM's reasoning explicitly cited the ECG
   findings, confirming `escalate_with_ecg` and prompt rule 6 fire
   correctly on a live request (previously nothing populated `ecg_risk` in
   the request path at all). One caveat surfaced: that real segment showed
   a 100% AFib burden — plausibly a genuine finding, or domain shift
   between MITDB/SVDB (training data) and real VitalPatch signals; not yet
   investigated further.
2. **`MedGemma-Agent/scripts/generate_teacher_dataset.py`** — generated
   2,500 examples from the live agent (Ollama + `medgemma`) across 20
   stratified vitals × ECG severity combinations. 56% hit real LLM
   reasoning, 44% hit the rule-based fallback (1000 as the deliberate
   immediate-CRITICAL safety bypass, 98 from real Ollama timeouts/errors
   under sustained load — both are legitimate system behavior and kept).
3. **`MedGemma-Agent/scripts/prepare_student_jsonl.py`** →
   **`train_student_lora.py`** — QLoRA fine-tuned `Qwen2.5-0.5B-Instruct`
   (2250 train / 250 held-out val examples, 3 epochs, bf16 on an RTX
   2080 Ti — fp16 was tried first but crashed on a GradScaler/BFloat16
   incompatibility). Eval loss 0.208, mean token accuracy 93.5%.
4. **`merge_lora_weights.py`** → **`quantize_student_gguf.py`**-equivalent
   (via a fresh local `llama.cpp` build) → Q4_K_M GGUF, 374 MB, deployed
   via `ollama create` as `medgemma-student`. Note: trained on raw
   prompt/completion text, not Qwen's ChatML format, so the Modelfile uses
   `TEMPLATE "{{ .Prompt }}"` (raw passthrough) rather than the default
   chat template.
5. **Validation** (`validate_student_outputs.py` +
   `evaluate_clinical_safety.py`, 250 held-out examples): 99.6% schema
   validity; risk-level exact match 92.8%, 100% within one severity tier;
   **CRITICAL false-negative rate 0.8% (1/118)** — one case downgraded
   CRITICAL→HIGH, the exact zero-tolerance failure mode, flagged not
   hidden. `recommended_action` field agreement is weaker (57.8%) — a
   real, unresolved gap: the student tracks overall severity well but is
   less reliable at reproducing the specific recommended action.
6. **`benchmark_student_on_jetson.py`** — run here as a same-machine GPU
   proxy only (**no physical Jetson Nano is reachable from this
   environment**): student vs. full-size `medgemma`, both via Ollama —
   1.34s vs 6.92s mean total latency, 434MB vs 2.6GB resident memory,
   234 vs 89 tokens/s. Directionally exactly what distillation should
   achieve, but these are dev-box numbers, not Jetson Nano numbers — rerun
   this script unmodified on the actual device for the real figures.

**Scope note (2026-07-21):** items 2-6 above (teacher dataset generation,
QLoRA distillation, quantization, Ollama student deployment, edge
benchmarking) are now a teammate's responsibility, not this workstream's.
They're left in place as a working first pass, not maintained further here.
This repo's ongoing scope is (1) the classifier above and (2) the
integration bridge in item 1 plus its evaluation, below.

## ECG → MedGemma-Agent integration evaluation (2026-07-21)

Data-flow/HTTP-200 checks aren't enough to know an integration works —
`MedGemma-Agent/scripts/evaluate_ecg_integration.py` checks four levels
against the live agent (Ollama + the full-size `medgemma` teacher model,
not the student):

1. **Data flow** — does a submitted `ecg_risk` round-trip through
   `/api/v1/vitals/snapshot` with every field intact? **Pass.**
2. **Rule engine** — does `escalate_with_ecg()` correctly raise the alert
   level per ECG severity tier (NORMAL/MEDIUM/HIGH/CRITICAL), and never let
   a normal ECG downgrade a vitals-derived alert? **5/5 pass**, including
   the explicit never-downgrade check.
3. **LLM reasoning content** — for the cases where the LLM actually runs
   (i.e. not the immediate-CRITICAL safety bypass), does its free-text
   reasoning cite the specific ECG finding rather than ignoring the ECG
   section? A PVC-burden case correctly cited PVC; a quiet-ECG case
   correctly did *not* hallucinate arrhythmia findings. **Pass** — caught
   and fixed a false-positive in the check itself first (naive substring
   match flagged "no PVC burden" as a PVC mention; fixed with a
   negation-aware check).
4. **Prompt utilization** — (a) gross: response changes substantially
   between a quiet and a severe ECG (**pass**, though this is via rule
   escalation, not proof the LLM itself is attentive); (b) meaningful: two
   ECG profiles mapped to the *same* rule alert level (HIGH), so the LLM
   runs both times, compared for whether reasoning differs to reflect the
   different underlying finding (PVC vs. AFib) rather than being generic
   boilerplate.

**Honest result:** 13/14 checks pass on a typical run. The one recurring
"failure" is level 4b intermittently — the AFib-burden case occasionally
lands in the rule-based fallback (`rule_based_only=True`) instead of
running the LLM, which makes that specific comparison inconclusive on that
run. Investigated directly: re-running the identical payload 3x got
2 real-LLM responses and 1 fallback — this is genuine Ollama flakiness
under load (consistent with the ~4% instability rate seen during teacher
dataset generation), not a deterministic bug tied to the AFib input, and
the fallback is the system's safety design working as intended (degrade to
rule-based rather than fail). Re-run the script to get a clean pass; don't
treat one flaky run as a real regression.

## Ground rules for anyone continuing this work

- **DS2 is report-only.** Never train, tune, sample, or threshold using
  it. It's the only honest signal for "does this actually work."
- **All splits are patient-level**, no patient's beats in more than one
  split. Check a validation split's per-class beat counts before trusting
  it — a technically-valid split can still be too small to see a real
  regression.
- **Never overwrite `models/five_class_xgb.json`.** New experiments get
  new filenames. Promotion to production is a separate, deliberate,
  human-approved step.
- **Accuracy is not a success metric** (N is ~90% of everything). Always
  report per-class F1, macro-F1, the confusion matrix, and whichever
  off-diagonal rate is relevant to what you're testing.
- **Fix the seed, pin `n_jobs=1`.** Two runs of the same config should
  give byte-identical results — if they don't, that's a bug to fix before
  trusting any comparison built on top of it.
- **No "done" claim without a checkable artifact.** A comment, a status
  note, or a task-list entry is not evidence that something happened. A
  file that exists, a metric from a fresh run, or a diff is. (This project
  hit three cases of confidently-worded but unverifiable "already done"
  claims in one session — don't add a fourth.)
- **Honest-result rule.** Evaluate the full chained system, not just an
  isolated stage's flattering number, and if it doesn't beat the current
  baseline on the class you're targeting without regressing another
  (V and N are zero-tolerance — see `Docs/archive/AGENT_RULES.md` rule 8),
  say so plainly. A clean negative is a valid, useful outcome; don't paper
  over it.

## What's not in this repo

To keep this handover lean:
- **Raw datasets** (`data/`) — re-download via the commands above.
- **~300MB of experimental model files** from the ablation work above —
  every one is reproducible from a documented command; only the
  production model is kept here.
- **Detailed docs** (full ablation report with every command/confusion
  matrix, a research audit, a bug log, agent working-rules) — kept locally
  on the original machine, not pushed, to keep this repo to the essentials.
  This README is the distilled version of all of it.
