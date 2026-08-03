# Handoff — Cliniaura ECG Pipeline

**Written:** 2026-07-29
**Branch:** `mod_1`
**Machine:** `ada` cluster, project root `/home2/mahimakopalley/projects`

This document is for a teammate picking up work on this pipeline. It assumes you
know Python and Git, but nothing about this project. It explains what exists,
what is broken, and exactly what to do about it — in order.

Every number here was measured from a file on this machine. Where a number is an
estimate, it says so. Nothing is guessed.

---

## 1. What this project does, in one paragraph

Patients recovering from surgery at home wear a small chest patch (VitalPatch)
that continuously records their heart's electrical signal (an ECG) and some basic
vital signs. Nobody can watch days of raw signal by eye. This pipeline reads the
raw recordings, finds every individual heartbeat, labels each one as normal or
one of four abnormal types, looks across many beats for dangerous rhythm
patterns, and assigns a risk level (LOW / MEDIUM / HIGH / CRITICAL) using fixed
human-written rules. A separate service, `MedGemma-Agent`, scores the patient's
vital signs using two standard clinical scores (NEWS2 and qSOFA) and can push the
risk level *up* — never down. A small local language model writes the result up
in plain English, but it never decides the risk level.

---

## 2. Where everything is

All paths are relative to `/home2/mahimakopalley/projects`.

### Code you will actually touch

| Path | What it is |
|---|---|
| `ecg_pipeline/ecg_pipeline_core.py` | The whole 9-stage pipeline. ~2,600 lines. The risk rules live at lines 2062–2091. |
| `ecg_pipeline/agent_bridge.py` | Connects the ECG pipeline to `MedGemma-Agent`. **The vitals-pairing bug is here** (lines 188–199). |
| `ecg_inference/` | A copy of the pipeline with training code stripped out, for deployment. Kept byte-identical to the core — **if you change a rule in one, change it in both.** |
| `MedGemma-Agent/` | A Git submodule. Separate repo (`github.com/saranambiar/MedGemma-Agent`). The vitals-scoring service. |
| `training/ecg_pipeline_tools.py` | Training and evaluation CLI. Not used at inference time. |

### Data (all of `data/` is gitignored — it is NOT on GitHub)

| Path | What it is |
|---|---|
| `data/raw/vitalpatch/Patch_<ID>/*_ecg.csv` | **2,375 raw ECG files**, 6 patients. The real input data. |
| `data/vitals_downloads/Patch_<ID>/*_vitals.csv` | **2,512 vitals files** (heart rate, breathing rate, temperature, posture). |
| `data/raw/prorhythm/` | 18 recordings from a second device (SeNSiO). |
| `data/raw/public/` | Labelled hospital datasets used to train and score the model (MITDB, SVDB, LTAFDB, etc.). See `docs/thesis/DATASETS.md`. |

### Results already produced

| Path | What it contains |
|---|---|
| `data/reports/vitalpatch_run_manifest.csv` | **The big ECG-only result.** 3,570 rows, one per segment. |
| `data/reports/prorhythm_run_manifest.csv` | Same for the second device. 18 rows. |
| `data/reports/multimodal_batch/sample14_manifest_preserved.json` | **The only multimodal result that exists.** 14 segments. |
| `data/reports/vitalpatch/`, `data/reports/prorhythm/` | One JSON + one Markdown report per segment. |
| `ecg_pipeline/example_reports/` | 8 sample reports committed to Git, so you can see the output format without the data. |

### Documentation

| Path | Read it when |
|---|---|
| `README.md` | First. The map of the repo and the honest numbers. |
| `docs/thesis/PIPELINE_METHODS_AND_RESULTS.md` | **The most important one.** Stage-by-stage methods, every measured result, and where each number came from. |
| `docs/decisions/AGENT_RULES.md` | **Before touching the classifier or any model file.** Every rule exists because it was broken once. |
| `docs/thesis/DATASETS.md` | You need to re-download a dataset. |
| `docs/archive/ABLATION_REPORT.md` | You want the full history of failed experiments. **Gitignored — local only, not on GitHub.** |

---

## 3. The situation right now, honestly

**What works and is proven:**

- The ECG-only pipeline runs end to end on real data. On 3,570 segments
  (81.9 hours of signal, 357,909 heartbeats), **78.3% were assessable** — good
  enough quality to say something about. The rest were correctly refused rather
  than guessed at.
- The heartbeat classifier is reliable for the two classes that matter most:
  Normal (F1 0.966) and Ventricular (F1 0.830, the dangerous class).
- The atrial-fibrillation rule was validated against real ground truth (LTAFDB,
  449,749 windows): sensitivity 0.971, F1 0.893.
- The connection to `MedGemma-Agent` works. All 14 test segments pushed
  successfully and got NEWS2/qSOFA scores back. All 25 of its guardrail tests
  pass.

**What is not done:**

- **The big multimodal run has never finished.** One attempt was started; the
  cluster killed it because the job ran out of time. So we have 14 segments of
  multimodal data, not 3,570.
- **The vitals-pairing logic is buggy** and throws away about 40% of the vitals
  data it could be using. Details in Task 1 below.
- **The ECG-only results are slightly stale** — they were produced before a
  parser bug was fixed, so 70 segments are missing from them.

**The most important thing to understand:** do not read "vitals never changed the
risk level in 14 out of 14 segments" as meaning vitals don't matter. VitalPatch
can only supply 3 of the 6 things NEWS2 needs, and with only 3 the score can
never reach 7 — which is the threshold that would escalate the risk. The
escalation is arithmetically impossible right now, so of course it never fired.
That tells us nothing about whether vitals are clinically useful.

---

## 4. Tasks, in the order they should be done

### TASK 1 — Fix the vitals-pairing bug ⚠️ DO THIS FIRST

**What's wrong**

Each ECG file must be matched to a vitals file so we know the patient's heart
rate, breathing rate and temperature at that moment. The current code matches
them by comparing the **timestamps in their filenames** and accepting the match
only if they are within **30 seconds** of each other.

That 30-second rule was tuned by looking at one patient. It does not work for the
others.

**The evidence** (measured across all 2,375 ECG files on 2026-07-29)

| Patient | ECG files | Matched | Coverage |
|---|---|---|---|
| Patch_183594 | 289 | 286 | 99.0% |
| Patch_1844AC | 501 | 76 | **15.2%** |
| Patch_184635 | 479 | 427 | 89.1% |
| Patch_1849DF | 502 | 91 | **18.1%** |
| Patch_184B27 | 180 | 135 | 75.0% |
| Patch_184B2F | 424 | 417 | 98.3% |
| **Total** | **2,375** | **1,432** | **60.3%** |

**Why it happens**

A vitals file is not a single instant — it covers about **20 minutes** of
readings, each row with its own timestamp (measured: median internal span 20.0
minutes, median gap between files 1,200,258 ms). On some patients the ECG
recording starts at the same moment a new vitals file starts, so the filenames
match closely. On other patients the ECG recording is free-running, so it starts
in the *middle* of a vitals file's 20-minute window. The data is right there in
the file — but the filename timestamp is minutes away, so the 30-second rule
rejects it.

**The fix**

Match on **interval containment** instead: pair the ECG with the vitals file
whose internal timestamp range (first row to last row) *contains* the ECG's
timestamp. Keep the existing nearest-within-30-seconds rule as a fallback for
ECGs that land in a gap between two vitals files.

**What you'll gain** (already measured, so you can check your work)

| Patient | Now | After the fix |
|---|---|---|
| Patch_183594 | 99.0% | 95.8% |
| Patch_1844AC | 15.2% | **100.0%** |
| Patch_184635 | 89.1% | 92.3% |
| Patch_1849DF | 18.1% | **100.0%** |
| Patch_184B27 | 75.0% | **100.0%** |
| Patch_184B2F | 98.3% | 85.1% |
| **Total** | **60.3%** | **95.3%** |

That is **831 more segments** with vitals. Note two patients get slightly *worse*
under pure containment (their ECGs fall in gaps between vitals files) — this is
exactly why you keep the old rule as a fallback and take whichever matches.

**Where the code is**

`ecg_pipeline/agent_bridge.py`, function `load_real_vitals()`, starting line 131.
The matching loop is lines **188–199**:

```python
best_path, best_offset = None, None
if ecg_ts is not None and patient_dir.is_dir():
    for candidate in patient_dir.glob("*_vitals.csv"):
        ...
        offset = abs(cand_ts - ecg_ts)          # <-- filename timestamps only
        if best_offset is None or offset < best_offset:
            best_offset, best_path = offset, candidate

vitals_path = best_path if (best_path is not None
                            and best_offset <= max_offset_ms) else None
```

**Performance warning:** reading the first and last row of every vitals file, for
every ECG file, would be very slow (2,375 × ~420 files). Build an index of
`(first_timestamp, last_timestamp, path)` per patient **once**, then look up each
ECG against it.

**How to check you're done**

Write a small script that runs the new matching over all 2,375 files and prints
coverage per patient. It should reproduce the "After the fix" column above,
give or take the fallback cases. Do not move on until it does.

**Rules for this task**

- Never invent a vitals value. If a value is missing, it stays `None`.
- Never mark a value `real: True` unless it came from a sensor reading.
- SpO2 and blood pressure are **always** unavailable — VitalPatch has no sensor
  for either. Do not fill them in, not even with "normal" defaults.

---

### TASK 2 — Re-run the ECG-only batch to refresh the results

**What's wrong**

Some real device files contain a literal `-` where a sample should be. This used
to crash the parser and lose the whole file. **That bug is already fixed**
(`ecg_pipeline/ecg_pipeline_core.py`, lines 292–307 — bad values become NaN and
are dropped in timestamp/value *pairs*, so nothing gets misaligned).

But `data/reports/vitalpatch_run_manifest.csv` was produced *before* the fix, so
70 of its 3,570 rows are still `PARSE_ERROR`. Every headline number we quote —
including the 78.3% assessability figure — is computed from that stale file.

**Proof the fix works** (run 2026-07-29): all 2,375 raw files re-parsed with the
current code — **0 failures, 27 seconds**.

**What to do**

Re-run the batch and regenerate the manifest:

```bash
cd /home2/mahimakopalley/projects
python -m ecg_pipeline.batch_vitalpatch_report    # resumable
python -m ecg_pipeline.manifest_summary
```

**How to check you're done**

`data/reports/vitalpatch_run_manifest.csv` has **zero** rows containing
`PARSE_ERROR`, and slightly more than 3,570 rows. Update the numbers in
`README.md` and `docs/thesis/PIPELINE_METHODS_AND_RESULTS.md` §6.1 to match.

---

### TASK 3 — Run the full multimodal batch

**Do Tasks 1 and 2 first.** Running this before Task 1 produces a vitals-coverage
number that measures the bug, not the data.

**What's wrong**

This has never completed. The one attempt (cluster job 2660129) was killed
because the job hit its wall-clock time limit before finishing even the first
patient. It was not a code failure.

**What you need before starting**

1. **A cluster allocation with enough time.** Estimate: the 14-segment test took
   15.5 seconds including 14 live round-trips to the Agent (~1.1 s per segment).
   Scaled to ~3,570 segments that is roughly **65 minutes** — but this is an
   extrapolation, and the Agent's speed under sustained load has never been
   measured. **Ask for at least 3 hours.**

2. **The `MedGemma-Agent` service running on your allocated node:**

```bash
cd /home2/mahimakopalley/projects/MedGemma-Agent
./start.sh
sleep 20
curl -s http://localhost:8000/api/v1/health | python3 -m json.tool
```

Confirm the health check reports the database as OK before running anything.
Default endpoint the pipeline pushes to:
`http://localhost:8000/api/v1/vitals/snapshot`, API key `change-me-clinician-key`
(both overridable with `--agent-url` / `--api-key`).

If `start.sh` fails with `bash\r: not found`, the file has Windows line endings:
`sed -i 's/\r$//' start.sh`.

3. **The batch runner script.** `ecg_pipeline/agent_bridge.py`'s `push` command
   loads the 15 MB model on every invocation, so calling it once per file would
   waste hours. Use `ecg_pipeline/multimodal_batch.py`, which loads the model
   once and loops in-process. It writes
   `data/reports/multimodal_batch/multimodal_manifest.json`.

**Small-scale check first**

```bash
python ecg_pipeline/multimodal_batch.py 3     # 3 files per patient
```

Confirm `agent_push_status` is `SUCCESS` and `agent_news2_coverage` shows
`3/6 NEWS2 components` for segments that have breathing rate and temperature.
Only then run the full thing with no argument.

**What the results should answer**

- How often are vitals available to pair with an ECG? (After Task 1, expect
  ~95%.)
- How often do the vitals actually change the final risk level compared to ECG
  alone?
- Is there any relationship between the NEWS2 score and what the ECG found?

**Do not report any of these as fact until the run completes.** Right now the
answer to all three is "we don't know."

---

### TASK 4 — Push the `MedGemma-Agent` submodule commits

**What's wrong**

`MedGemma-Agent/` is a submodule pointing at a **different person's repository**
(`github.com/saranambiar/MedGemma-Agent`). Three commits that make the partial
vitals work — `37f4432`, `1f36dda`, `087f261` — exist **only on this machine**,
on a local branch called `dev`.

The main repo records a pointer to commit `087f261`. Because that commit isn't on
GitHub, **anyone who clones this project gets a submodule they cannot check out.**

**What to do**

Talk to the submodule's owner. This is their repository, so it is their call
whether these commits are pushed to it, go on a branch, or come in as a pull
request. Do not force-push to someone else's repo.

**Also be aware:** the submodule's working directory has uncommitted changes —
56 files differ, but 50 of those are line-ending (CRLF vs LF) noise only. The
remaining 6 contain real, pre-existing ECG-integration edits that were not
written as part of this pipeline's work and have not been reviewed. Do not
commit the whole lot blindly. Check with `git diff --ignore-all-space` to see
what is actually different.

---

### TASK 5 — Get a clinician to review some CRITICAL segments

**Why this matters more than any code change**

Of the assessable segments, **9.9% came back CRITICAL** (277 out of 2,795). That
is a very high rate for patients who are recovering normally. It could mean the
system is catching real events. It could equally mean it is producing false
alarms.

**We cannot tell from inside the pipeline.** There are no ground-truth labels on
this patient data. And we know the ventricular-beat classifier has precision
0.754 — roughly **1 in 4 of its "ventricular beat" calls is wrong** — and
ventricular-beat burden is what drives both CRITICAL rules.

**What to do**

Take a stratified sample of about 50 CRITICAL segments and have a clinician
review the waveforms. The reports are already written and human-readable:

```
data/reports/vitalpatch/<segment_id>.md      # readable report
data/reports/vitalpatch/<segment_id>.json    # structured data
```

Find CRITICAL segments with:

```bash
cd /home2/mahimakopalley/projects
python3 -c "
import csv
rows = [r for r in csv.DictReader(open('data/reports/vitalpatch_run_manifest.csv'))
        if r['final_risk_level'] == 'CRITICAL']
print(len(rows))
for r in rows[:20]:
    print(r['patient_id'], r['segment_id'], r['n_beats_analyzed'])
"
```

**Why this blocks things:** how many false alarms a nurse gets per day, whether
the thresholds need changing, and whether this is anywhere near deployable all
depend on the answer. No amount of further engineering can produce it.

---

## 5. Smaller improvements, once the above is done

| # | What | Why |
|---|---|---|
| 6 | Pick the vitals **rows** nearest the ECG time, instead of averaging the whole file | A vitals file covers 20 minutes. Averaging all of it is coarser than the data allows and coarser than NEWS2 assumes. Natural follow-on from Task 1. |
| 7 | Sweep the AFib rule's `window=20` | Only the *threshold* was ever tested (against 449,749 real windows). Window size is untested. The same test harness can do it. |
| 8 | Decide what the NEWS2 escalation should do with only 3 of 6 components | With 3 components the score cannot reach 7, so the CRITICAL escalation can never fire on this hardware. Either set a documented device-specific threshold, or state clearly that it is inactive. Shipping a safety rule that silently cannot fire is the worst option. **The qSOFA rule is fine** — it reached 2 in one of the 14 test segments and works. |
| 9 | Calibrate the confidence tiers | They are currently a rule of thumb, not real probabilities. `ConformalConfig` already exists in `ecg_pipeline_core.py` but nothing uses it. |
| 10 | Add a consciousness (AVPU) input | It is the only remaining NEWS2 component obtainable without new hardware, and would take coverage from 3/6 to 4/6. |
| 11 | Clean up the Git object store | `git gc` currently refuses to run: "too many unreachable loose objects". Fix the cause, then delete `.git/gc.log`. |

---

## 6. Do NOT do these

**Do not try to improve the S (supraventricular) or F (fusion) beat classes on
the current data.**

Four genuinely different approaches were tried. All four failed the same way —
every one that improved S made V (the dangerous class) worse, past the project's
zero-tolerance safety limit:

| What was tried | S F1 | V F1 | Result |
|---|---|---|---|
| Added 7 timing features | 0.182 (better) | 0.775 (worse) | Rejected |
| Added 8.5M more training beats | 0.098 (worse) | 0.808 (worse) | Rejected |
| Two-stage classifier, 5 versions | 0.383 (much better) | 0.789 (worse) | Rejected |
| CNN + Transformer deep model, 2 versions | 0.121 (worse) | 0.730 (much worse) | Rejected |

The likely real reason: a single ECG lead often cannot see the P-wave, the
electrical signature that distinguishes an early atrial beat from a normal one.
**67.6% of true S beats are missed entirely** — not confused with something else,
just not visible. This is a hardware and data limitation, not a tuning problem.
Full history: `docs/archive/ABLATION_REPORT.md`.

**Do not change any threshold or model file without reading
`docs/decisions/AGENT_RULES.md` first.** Every rule in it exists because it was broken once
at real cost.

**Do not report a number you have not measured yourself from a file on disk.** If
you are unsure, write `[UNVERIFIED]` or leave it out.

---

## 7. Getting set up

```bash
cd /home2/mahimakopalley/projects
git checkout mod_1

# ECG pipeline only (no PyTorch needed)
pip install -r requirements_inference.txt

# Try it on one file
python run_inference.py \
    --input  data/raw/vitalpatch/Patch_1844AC/1778423422284_VC2B008BF_1844AC_ecg.csv \
    --output /tmp/test.json \
    --source vitalpatch

# MedGemma-Agent has its own environment
cd MedGemma-Agent
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill it in
python -m pytest tests/test_guardrails.py -q     # should print "25 passed"
```

**Known environment problems and their fixes:**

- The `MedGemma-Agent` venv was originally built on Windows (`Include/`, `Lib/`,
  `Scripts/` instead of `bin/`). If you see that, delete it and rebuild natively.
- `pip install` may fail with a disk-quota error while downloading NVIDIA CUDA
  packages. Clear the pip cache (`pip cache purge`) and install the CPU-only
  build of PyTorch.
- Scripts may have Windows line endings, causing `bash\r: No such file or
  directory`. Fix with `sed -i 's/\r$//' <file>`.

---

## 8. Open questions to raise, not solve alone

1. **Is 9.9% CRITICAL believable for this patient group?** Needs a clinician
   (Task 5).
2. **What should NEWS2 escalation do with only 3 of 6 components?** This is a
   clinical safety decision, not an engineering one (item 8).
3. **Can the submodule commits be pushed upstream, and by whom?** Needs the
   submodule owner (Task 4).
4. **Is a 30-second freshness limit on vitals clinically acceptable at all?**
   After Task 1, some pairings will use vitals recorded up to ~20 minutes from
   the ECG. Whether that is close enough to combine into one risk score is a
   clinical question that has never been asked.

---

## 9. Quick reference — commands that actually work

```bash
# Single ECG file -> JSON report
python run_inference.py --input <file.csv> --output out.json --source vitalpatch --narrative

# Single file -> report with a live MedGemma narrative (needs `ollama serve`)
python -m ecg_pipeline.agent_bridge report --file <file.csv> --source vitalpatch

# Push ECG + real vitals to MedGemma-Agent (needs the service running)
python -m ecg_pipeline.agent_bridge push \
    --vitalpatch-root data/raw/vitalpatch \
    --vitals-root     data/vitals_downloads \
    --limit 5

# Full ECG-only batch (resumable)
python -m ecg_pipeline.batch_vitalpatch_report
python -m ecg_pipeline.manifest_summary

# Full multimodal batch (loads the model once)
python ecg_pipeline/multimodal_batch.py        # add a number to limit files per patient

# Agent tests
cd MedGemma-Agent && ./venv/bin/python -m pytest tests/test_guardrails.py -q
```

**Flags that do NOT exist**, despite looking like they should:
`agent_bridge push` has no `--input` and no `--output`. Files are found by
scanning `--vitalpatch-root` (which must be the *parent* of the `Patch_*`
folders, not one of them) and processed in sorted order up to `--limit`.
