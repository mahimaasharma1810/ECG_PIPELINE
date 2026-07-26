# ECG Pipeline — `mod_1` branch

This branch takes the ECG 5-class beat classifier (see [`project_doc.md`](project_doc.md)
for the classifier's own training/eval history) and wires it into a full,
end-to-end **clinical report pipeline**: raw device file → parse → filter →
detect beats → classify → detect rhythm abnormalities → risk cascade →
live MedGemma clinical narrative → saved report. It also fixes three
correctness bugs found while building that pipeline out. Nothing here
retrains, reweights, or reconfigures the classifier, the risk cascade
thresholds, or the AAMI 5-class scheme — every fix below is either in the
signal path *before* the classifier ever sees a beat, or in the narrative
*after* the risk decision has already been made.

> **New here? Start with [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md)
> instead.** It's a plain-English writeup of this whole project — what an
> ECG and a "beat" are, why the problem is hard, the real classifier
> results with every number cited to the file it came from, what was
> tried and rejected to fix the weak classes, and honest limitations. No
> ML or ECG background assumed. This README below is the engineering
> reference (what changed, where, and why) for people already familiar
> with the codebase.

## Pipeline stages

```
raw file (VitalPatch CSV | SeNSiO/prorhythm CSV | WFDB record)
  -> Stage 2: SQI gate (native sample rate)          evaluate_window()
  -> Stage 3: resample to 125Hz                      resample_linear()
  -> Stage 4: filter chain (baseline/highpass/notch/bandpass/Kalman EMG)
  -> Stage 5: R-peak detection + beat segmentation   detect_and_segment()
  -> Stage 6: feature extraction
  -> Stage 7: classification (five_class_xgb.json, AAMI N/S/V/F/Q)
  -> Stage 8: risk cascade (rule_trace, deciding_rule, risk_level)
  -> Stage 9: MedGemma clinical narrative             agent_bridge.py
  -> saved report: <segment_id>.json + <segment_id>.md
```

All of Stages 2–8 live in `ecg_pipeline/ecg_pipeline_core.py`. Stage 9 and
the report assembly live in `ecg_pipeline/agent_bridge.py`.

## What changed on this branch

### 1. R-peak beat-level over-culling (commit `2420ecc`)

`detect_and_segment()` fed XQRS's raw peak indices straight into
`segment_beats()`, but on resampled data those indices land ~3 samples off
the true `|amplitude|` local max. The beat-level SQI check
(`R_PEAK_NOT_LOCAL_MAX`) requires the peak to *be* that local max within a
±3-sample window, so nearly every real beat failed it — 1400–2400 beats
lost per WFDB test record, the dominant cause of beat loss even after the
SQI-gate fix below. Fixed by snapping each XQRS peak to its true local max
before segmentation (mirroring a fix that already existed in the
training/eval path but was never ported to the runtime path).

- Retention (beats analyzed / ground-truth beats) went from 8–70% to
  84–131% across 7 WFDB test records.
- VitalPatch unaffected (no regression: 189 detected, 185 analyzed, up
  from 122).
- **Known, explicitly NOT fixed here** (see commit message for full
  reasoning): true XQRS over-detection on two noisier WFDB records
  (103/111), where XQRS's adaptive rate-tracking appears to "hunt" at its
  hr_max=200bpm floor. No safe low-risk fix identified; flagged as a
  separate follow-up rather than reconfiguring XQRS without validation.

### 2. WFDB SQI over-rejection (commit `3add727`)

`evaluate_window()` scored `baseline_wander_ratio` and `snr_db` directly
on the raw window. That's correct for VitalPatch (already filtered by the
device) but wrong for raw WFDB input, where Stage 4's baseline removal
hasn't run yet — a clean record's own real DC/baseline offset got scored
as "noise," driving some records to 100% window rejection. Fixed by
scoring those two metrics on a locally detrended copy (the same 0.5Hz
highpass Stage 4 already uses) instead of the raw window; flatline/
clipping/missing/kurtosis still score the raw window unchanged.

- WFDB rejection dropped from 48–100% to 7–23% across 7 test records.
- VitalPatch improved too (26.7% → 13.3% rejection) — not a regression.
- Synthetic flatline/white-noise/clipped signals still 100% rejected (the
  gate still gates).

### 3. SeNSiO/prorhythm zero-beat bug (uncommitted on this branch, `ecg_pipeline_core.py`)

A prorhythm/SeNSiO recording (100Hz native rate) returned 0 beats
detected and 100% SQI rejection end-to-end — `NOT_ASSESSABLE`, MedGemma
correctly skipped. Root cause: `snr_db` defines "signal" as 0.5–40Hz
content and "noise" as everything else. At SeNSiO's 100Hz native rate,
Nyquist is exactly 50Hz — the same frequency as the mains notch filter,
and `powerline_notch()` correctly no-ops whenever `notch_hz >= Nyquist`
(you cannot design a notch at/above Nyquist). That leaves a narrow,
un-notch-able sliver right at Nyquist that any near-Nyquist artifact can
dominate, reading as overwhelming "noise" even though the actual QRS
structure is physiologically plausible (verified via an SQI-bypass test:
89.2% RR-plausibility on the target recording).

Fixed by scoring `snr_db` on a copy resampled to `TARGET_FS` (125Hz, the
same resample Stage 3 already performs downstream) *before* the
notch/highpass are applied for scoring purposes — this only activates
when the native rate itself can't support the notch
(`fs / 2.0 <= FILTER.notch_hz`), so WFDB and VitalPatch (both already
≥ 125Hz) are untouched.

- Verified across 3 SeNSiO files with a before/after comparison; no
  VitalPatch regression; synthetic flatline still fully rejected.

### 4. MedGemma narrative quality (uncommitted on this branch, `agent_bridge.py`)

Two defects found in the LLM-authored clinical narrative:

- It ignored `beat_summary` (N/S/V/F/Q counts) entirely.
- It sometimes misread its own numbers — e.g. asserting *"PAC burden
  8.642 exceeds threshold 15.0"* (false; 8.642 < 15.0). Both numbers were
  real, just the comparison was wrong. Observed in roughly 30% of
  generations even after the prompt fix below — a genuine small-model
  (4B) limitation, not something prompting alone can fully close.

Fix, in two parts:

1. **Prompt fix**: every rule's exceeds/does-not-exceed verdict is now
   pre-computed in Python (`_rule_verdict_line`, using the cascade's own
   comparison operators) and handed to MedGemma as an explicit fact,
   alongside explicit `beat_summary` and `rhythm_findings` blocks. The
   prompt forbids MedGemma from inferring its own comparisons.
2. **Deterministic composition + post-hoc guard**: the saved narrative is
   assembled in code with a guaranteed structure — (a) beat summary
   (b) rhythm findings (c) the deciding rule with its real value vs.
   threshold (d) final risk level (e) a clinician-review disclaimer —
   with MedGemma's own prose embedded as a "Clinical interpretation"
   sub-section. A validator (`_narrative_asserts_false_exceed`) scans that
   prose for a false exceed/cross/trigger claim against a rule that did
   **not** fire, and drops just that sub-section (never the structured
   parts) if it finds one. This guarantees the *saved* report is never
   wrong, even though the model itself remains occasionally unreliable.

Verified: 9/9 runs (3 recordings × 3 repeats, including a SeNSiO file and
a VitalPatch file) show complete structure, zero false-exceed claims,
zero invented numbers, stable risk levels.

### 5. Batch pipeline + saved reports (new: `batch_vitalpatch_report.py`,
   `batch_prorhythm_report.py`, `manifest_summary.py`)

- `run_full_report()` (in `agent_bridge.py`) is now run over every raw
  recording in `data/raw/vitalpatch/Patch_*/` and every SeNSiO/prorhythm
  recording in `data/raw/prorhythm/`.
- Every segment gets a saved `<segment_id>.json` (full structured
  `RiskReport`: beats, rhythm findings, rule trace, risk level, narrative)
  and a companion `<segment_id>.md` (human-readable clinical report),
  written under `data/reports/<source>/<patient>/`.
- Each file is wrapped in its own try/except so one bad file never aborts
  the batch — failures are logged to a run manifest (`error` column)
  instead of crashing. This is needed in practice: ~70 VitalPatch files
  contain a literal `'-'` sentinel value in place of a numeric field and
  fail to parse; they're now skipped and logged instead of killing the
  run.
- The `NOT_ASSESSABLE` guard is preserved end-to-end: recordings with too
  few surviving beats stay `NOT_ASSESSABLE`, MedGemma is skipped
  (`medgemma_status = SKIPPED_NOT_ASSESSABLE`), and no narrative is
  fabricated for them.
- `batch_vitalpatch_report.py` is resumable: since parsing + running a
  given raw file is a pure function of that file's bytes and the current
  code, the batch skips any file whose output JSON already exists on disk
  and reconstructs its manifest row from that JSON instead of
  recomputing — important because the batch is long-running and
  serialized MedGemma calls make re-processing expensive.
- `manifest_summary.py` combines both run manifests into one CSV, prints
  summary counts (assessable vs. `NOT_ASSESSABLE`, risk-level
  distribution, MedGemma status distribution), and prints a few full
  "showcase" reports end-to-end (raw file → beat counts → rhythm findings
  → deciding rule → final clinical report).

## Running it

```bash
# one segment, live MedGemma report:
python -m ecg_pipeline.agent_bridge --file <path-to-raw-csv>

# full VitalPatch batch (all patients, resumable):
python -m ecg_pipeline.batch_vitalpatch_report

# full SeNSiO/prorhythm batch:
python -m ecg_pipeline.batch_prorhythm_report

# combined manifest + summary + showcase, once both batches have run:
python -m ecg_pipeline.manifest_summary
```

MedGemma must be served locally via Ollama (`ollama serve`, model
`medgemma:latest`) for Stage 9 to produce a live narrative; without it,
`medgemma_status` falls back to `UNAVAILABLE_FALLBACK` and the structured
report is still saved, just without an LLM-authored interpretation.

## Known limitations / honest caveats

- MedGemma (4B, local) occasionally still misstates a comparison in its
  free-text prose (~30% of generations pre-filter); the post-hoc validator
  catches and strips this before saving, but the model itself was not
  fixed and cannot be fully fixed by prompting alone.
- XGBoost XQRS over-detection on two specific noisy WFDB records
  (103/111) is diagnosed but not fixed — see commit `2420ecc`.
- The VitalPatch batch (2375 files) is serialized on a single local
  Ollama instance (`OLLAMA_NUM_PARALLEL=1`), so a full run takes several
  hours; it is resumable and safe to kill/restart.
- Nothing in this branch changes `five_class_xgb.json`, the risk cascade
  thresholds, or the AAMI N/S/V/F/Q scheme — every fix is upstream (signal
  quality / R-peak detection) or downstream (narrative composition) of
  the classifier itself.
