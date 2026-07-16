# ECG 5-Class Beat Classifier

AAMI 5-class (N/S/V/F/Q) ECG beat classifier. XGBoost on handcrafted
morphology + wavelet features. This README is the handover doc — what's
done, what's next, and where the work is currently stuck, waiting on a
decision.

## Where things are

```
ecg_pipeline/
  ecg_pipeline_core.py    # runtime pipeline: ingest, filter, segment, features, classify, risk
  ecg_pipeline_tools.py   # training/eval CLI: download-datasets, train-classifiers, eval-classifier
  models/
    five_class_xgb.json          # PRODUCTION model — never overwrite this
    five_class_xgb.classes.json
    ecg_encoder.pt                # optional learned encoder, not used by the classifier below
data/raw/public/          # datasets — NOT in git (too large), see "Getting the data" below
_original_stages/         # frozen historical reference implementation, kept for diffing
```

Run everything from `/home2/mahimakopalley/projects` as:
```
python -m ecg_pipeline.ecg_pipeline_tools <subcommand> [args]
```

## Getting the data

```
python -m ecg_pipeline.ecg_pipeline_tools download-datasets --all
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

**Production model** (`five_class_xgb.json`): trained on MITDB DS1 + SVDB,
Q-class dropped, ROS (1:3 floor) + balanced class weights. DS2 (held-out,
never trained/tuned on):

| Class | Sensitivity | Precision | F1 |
|---|---|---|---|
| N | 0.968 | 0.964 | 0.966 |
| S | 0.155 | 0.166 | 0.160 |
| V | 0.908 | 0.828 | **0.866** |
| F | 0.011 | 0.143 | 0.021 |

Macro-F1: 0.4026. **S and F are the known weak points** — this whole body
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

## Where this is stuck — the next step

**The two-stage classifier has not been built yet.** This is the next
piece of work, and it's a real architecture decision, not a small tweak —
whoever picks this up should design it deliberately, not just start
coding. Requirements already worked out:

1. **Stage 1 threshold tuning**: tune on a validation split (patient-level,
   carved from DS1 — check per-class beat counts are large enough to be
   meaningful before trusting it, small validation splits have hidden real
   regressions before in this project). Optimize for **abnormal recall**,
   not accuracy — a real S/V beat wrongly gated out as Normal by Stage 1
   can never be recovered by Stage 2. The number to beat: today's flat
   model effectively gives S only ~33% abnormal-detection recall (67.6%
   of S beats end up misclassified as N). Stage 1 needs to clear that bar
   by a wide margin to be worth building.
2. **Stage 2**: morphology-only (56-dim, no timing). Train on TRUE abnormal
   beats (clean labels) for stage-2-in-isolation numbers, but **evaluate
   the full system chained** (Stage 1 → Stage 2) on DS2 — the chained
   number is what counts, because it reflects real error propagation
   (beats Stage 1 misses are gone for good; report both numbers, don't
   only report the flattering isolated one).
3. **Report**: full 5-class DS2 confusion matrix for the chained system,
   per-class sensitivity/precision/F1, macro-F1, S→V / V→S / S→N /
   F→(N,S,V) rates, compared against the pinned baseline above.
4. **Honest-result rule**: if the chained system doesn't beat the flat
   baseline on S without regressing V/N, say so plainly. A clean negative
   here is a valid, useful outcome — don't paper over it by only reporting
   the nicer-looking isolated stage-2 number.

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
