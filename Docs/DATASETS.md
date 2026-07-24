# Datasets — what we have and their exact format

Everything below is what's physically present on disk right now (confirmed
by directly inspecting the files, not from memory/notes). Two categories:
**VitalPatch** (the real target device, unlabeled) and **public labeled
datasets** (used to train/evaluate the beat classifier — none of these are
VitalPatch data, they're all standard cardiology research databases).

None of `data/raw/` is in git (too large) — see each dataset's download
notes if it needs to be re-fetched.

---

## 1. VitalPatch (target device — real, unlabeled wearable ECG)

**Location:** `data/raw/vitalpatch/Patch_<device_id>/`

6 patients present:

| Patient folder | # ECG files |
|---|---|
| Patch_183594 | 289 |
| Patch_1844AC | 501 |
| Patch_184635 | 479 |
| Patch_1849DF | 502 |
| Patch_184B27 | 180 |
| Patch_184B2F | 424 |

Total size: 639MB.

**Filename format:** `{unix_epoch_ms}_{device_mac}_{patient_id}_ecg.csv`
e.g. `1778744267337_VC2B008BF_183594_ecg.csv`.

**File content format:** no header row. Each row is a flat, comma-separated
sequence of **alternating (timestamp_ms, adc_value) pairs** — not one
column per field:
```
1778744262706,-400,1778744262714,-171,1778744262722,-78,1778744262730,-59,...
```
So row 1 = `[t0, v0, t1, v1, t2, v2, ...]`, not `[t0, v0]` per row. Parsing
(`parse_vitalpatch_ecg` in `ecg_pipeline_core.py:281`) flattens the whole
file, splits into even/odd indices to recover `timestamps_ms` and `values`
arrays, sorts by timestamp, de-duplicates, then splits into sub-segments
wherever there's an internal timing gap (>1.5x the nominal step) — so **one
CSV file can produce more than one `Recording`** if it has an internal
dropout.

- **Sampling rate:** nominal 125Hz (8ms step between timestamps) — this is
  assumed, not read from a header (VitalPatch CSVs carry no metadata).
  Each file is a chunk of roughly 5 minutes (per collector-side
  segmentation), ~178 lines/file, ~200 samples/line.
- **Units:** raw ADC counts, not mV/calibrated — the pipeline's own
  filtering/normalization (robust z-score, per-beat) doesn't require
  calibrated units, so this is fine as-is.
- **Labels: none.** VitalPatch has no ground-truth beat annotations —
  it's the real deployment target, evaluated only by running the trained
  classifier on it (see `Docs/BEAT_CLASSIFICATION_SUMMARY.md` for how this
  is used) and by pipeline-internal signal-quality checks, not by
  comparing to a true label.
- **No vitals/BP/SpO2 sensor on this hardware** — a companion 11-column
  vitals CSV exists in the format (`parse_vitalpatch_vitals`) but any
  SpO2/BP columns in it are sentinel placeholders, not real readings.

---

## 2. Public labeled datasets (WFDB format — used to train/evaluate the classifier)

All of these are standard **PhysioNet WFDB-format** databases — same
three-file layout per record:
- `<id>.hea` — plain-text header: signal name(s), sample rate, sample
  count, gain/ADC info, one line per channel.
- `<id>.dat` — the actual signal, binary, encoded per the `.hea` header
  (usually 212 or 16-bit format).
- `<id>.atr` — expert cardiologist beat-by-beat annotations (sample index +
  a 1-character beat-type symbol), read via `wfdb.rdann(path, "atr")`.

Loaded via `wfdb.rdrecord()` / `wfdb.rdann()` (the `wfdb` Python package),
not hand-parsed. Only **channel 0** of any record is used
(`record.p_signal[:, 0]`), even for multi-lead records.

**Beat-symbol → AAMI 5-class mapping** (same for every dataset below,
`AAMI_SYMBOL_MAP` in `ecg_pipeline_tools.py:531`):

| Raw WFDB symbol(s) | AAMI class |
|---|---|
| `N`, `L`, `R`, `e`, `j` | **N** (normal) |
| `A`, `a`, `J`, `S` | **S** (supraventricular ectopic) |
| `V`, `E` | **V** (ventricular ectopic) |
| `F` | **F** (fusion) |
| `/`, `f`, `Q` | **Q** (unknown/paced/unclassifiable) |

Any other raw symbol (rhythm-change markers, artifact flags, etc.) is
simply not in this map and is dropped, not miscounted.

### 2a. MITDB — MIT-BIH Arrhythmia Database
- **Location:** `data/raw/public/mitdb/` — 48 records, 90MB.
- **Format:** 2 channels (commonly `MLII` + `V1`), **fs = 360Hz**, ~650,000
  samples/record (~30 min each).
- **Role in the pipeline:** the primary dataset. Split into three
  non-overlapping, patient-level groups (record IDs, defined in
  `ecg_pipeline_tools.py:111-159`):
  - `MITDB_DS1` (22 records) → further split into:
    - `DS1_TRAIN` (12 records): 101,106,108,109,112,115,119,122,203,208,209,230
    - `DS1_VAL` (10 records, enlarged from an original 4 — see
      `ABLATION_REPORT.md`): 114,116,118,124,201,205,207,215,220,223
  - `MITDB_DS2` (22 records, **held-out, report-only, never used to pick
    anything**): 100,103,105,111,113,117,121,123,200,202,210,212,213,214,
    219,221,222,228,231,232,233,234
  - This is the standard "DS1/DS2" inter-patient split from the original
    de Chazal et al. AAMI-classification literature, not something
    invented here.

### 2b. SVDB — MIT-BIH Supraventricular Arrhythmia Database
- **Location:** `data/raw/public/svdb/` — 78 records, 53MB.
- **Format:** 2 channels (`ECG1`/`ECG2`), **fs = 128Hz**, ~230,400
  samples/record (~30 min each).
- **Role:** train-only S-class enrichment (`SVDB_RECORDS`, all 78 records,
  IDs 800-894) — added to `DS1_TRAIN` specifically because MITDB alone has
  very few true S-class beats. Never touches DS2.

### 2c. INCARTDB — St Petersburg INCART 12-lead Arrhythmia Database
- **Location:** `data/raw/public/incartdb/` — 75 records, 796MB.
- **Format:** **12 channels** (`I`, `II`, `III`, `AVR`, ...), **fs = 257Hz**,
  ~462,600 samples/record (~30 min each). Filenames are `I01`..`I75`
  (strings, not bare integers).
- **Role:** train-only enrichment (`INCART_RECORDS`), same purpose as SVDB.

### 2d. LTAFDB — Long-Term AF Database
- **Location:** `data/raw/public/ltafdb/` — 84 records, ~3.4GB of signal
  data (`du` misreports this directory as 0 on this filesystem; verified
  directly via `.dat` file sizes).
- **Format:** 2 channels, **fs = 128Hz**, long Holter recordings —
  durations vary by record; record `00` alone is 9,661,440 samples
  (~21 hours).
- **Annotations:** unlike the other datasets, LTAFDB's clinically useful
  labels are **rhythm-interval annotations** in `aux_note` (e.g. `(AFIB`,
  `(N`, `(VT`), not just per-beat symbols — this is what the AFib-burden
  validation work used (`Docs/archive/ABLATION_REPORT.md`,
  `afib_batch_ltafdb.py`), reading via `wfdb.rdann(path, "atr")` and
  filtering `aux_note` for non-empty rhythm-change markers.
- **Role:** train-only enrichment for the beat classifier (`LTAFDB_RECORDS`,
  84 records); separately, also the **ground-truth AF/Normal reference**
  for validating the AFib-burden pipeline logic (not beat classification).

### 2e. SDDB — Sudden Cardiac Death Holter Database
- **Location:** `data/raw/public/sddb/` — 23 records, ~1.2GB.
- **Format:** 2 channels, **fs = 250Hz**, long recordings (record `30` is
  22,099,250 samples ≈ 24.5 hours). Also ships a `.ari` (automated, less
  reliable) annotation alongside the expert `.atr` — only `.atr` is used.
- **Role:** train-only. Notable as one of the very few real sources of
  **F-class (fusion) beats** — one sample record alone had 75 F beats,
  meaningful next to MITDB+SVDB's combined ~410 F examples total.

### 2f. CUDB — Creighton University Ventricular Tachyarrhythmia Database
- **Location:** `data/raw/public/cudb/` — 35 records, 6.8MB.
- **Format:** **1 channel**, **fs = 250Hz**, short recordings (record
  `cu01` is 127,232 samples ≈ 8.5 min — these are brief VT/VF episodes,
  not full Holters).
- **Role:** downloaded, but **not currently wired into any
  `build_dataset`/training call** in `ecg_pipeline_tools.py` — present on
  disk as an available-but-unused enrichment option.

### 2g. CHALLENGE2017 — PhysioNet/CinC Challenge 2017 (AF Classification)
- **Location:** `data/raw/public/challenge2017/training2017/` — 8,528
  records, 306MB.
- **Format: different from the rest.** Single-lead, **fs = 300Hz**,
  variable length (record `A00761` is 9,000 samples = 30s; lengths vary
  per recording). Signal stored as `.mat` (MATLAB format, not WFDB `.dat`),
  paired with a `.hea` header. No per-beat `.atr` annotations — this
  dataset's original task is **record-level** rhythm classification
  (Normal / AF / Other / Noisy), not per-beat labeling.
- **Role: downloaded, not currently used** by the beat classifier's
  `build_dataset` pipeline (which expects per-beat `.atr` annotations) —
  present on disk as a candidate for a future *rhythm-level* (not
  beat-level) task, not the current one.

### 2h. ICENTIA11K
- **Location:** `data/raw/public/icentia11k/` — **empty, 0 bytes.**
  Referenced in `ecg_pipeline_tools.py` (a `main_download_icentia11k_full`
  CLI command exists) but not actually downloaded onto this machine yet.

---

## 3. Other formats present but not part of the classifier's data pipeline

### SeNSiO device data
- **Location:** `data/raw/sense_io/` — 19MB, informal CSV exports (e.g.
  `ECG_2026-07-07T12-21-42.647304.csv`, and a `ECG_Filtered_...` variant).
- **Format:** 11 metadata rows (key,value pairs) + blank rows + a real
  header at row 16 (`Index,...`). Two column variants: raw has `ECG`;
  pre-filtered exports have `ECG_Raw` + `ECG_Filtered` (the pipeline skips
  its own filtering stage if `ECG_Filtered` is present, to avoid
  double-filtering). Sample rate is parsed out of a `Command Sent`
  metadata field (`STARTECG_F:<rate>`), not fixed.
- **Role:** a different piece of hardware from VitalPatch, supported by
  its own parser (`parse_sensio_ecg`) but not part of the current beat-
  classifier training/eval data — informal test recordings, not a
  training/eval dataset.

---

## Summary table

| Dataset | Records | Channels | fs (Hz) | Format | Labels | Role |
|---|---|---|---|---|---|---|
| VitalPatch | 6 patients, 180-502 files each | 1 | 125 (assumed) | flat CSV, ts/value pairs | none | real deployment target |
| MITDB | 48 | 2 | 360 | WFDB | per-beat `.atr` | primary train+val+held-out test |
| SVDB | 78 | 2 | 128 | WFDB | per-beat `.atr` | train-only, S-class boost |
| INCARTDB | 75 | 12 | 257 | WFDB | per-beat `.atr` | train-only enrichment |
| LTAFDB | 84 | 2 | 128 | WFDB | per-beat `.atr` + rhythm `aux_note` | train enrichment + AFib ground truth |
| SDDB | 23 | 2 | 250 | WFDB | per-beat `.atr` (+ `.ari`) | train-only, F-class source |
| CUDB | 35 | 1 | 250 | WFDB | per-beat `.atr` | downloaded, unused |
| Challenge2017 | 8,528 | 1 | 300 | `.mat` + `.hea` (not WFDB `.dat`) | record-level rhythm label | downloaded, unused (different task) |
| Icentia11k | 0 | — | — | — | — | not downloaded |
| SeNSiO | informal exports | 1 | varies | custom CSV | none | different hardware, not in training pipeline |
