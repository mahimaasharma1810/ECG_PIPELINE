"""ECG pipeline — training / data-acquisition tooling, consolidated.

This single file merges what used to be 6 separate modules
(splits.py, download_datasets.py, download_icentia11k_full.py,
train_encoder.py, train_classifiers.py, eval_classifier.py) into one
runnable script, per request. NOTHING was removed — every function,
class, constant, and docstring below is verbatim from its original
file; only the intra-package `from .xxx import yyy` lines were dropped
in favour of one `from .ecg_pipeline_core import ...` (the merged runtime
pipeline lives there), a few `module.function(...)` call sites were
flattened to `function(...)` accordingly, and each script's own
`main()` was renamed `main_<script>()` (parse_args now takes an
optional `argv`) so all five can coexist in one namespace behind a
single subcommand dispatcher at the bottom.

The original per-script files are preserved unchanged under
`_original_stages/` for reference/diffing.

Usage (one subcommand per original script):
    python -m ecg_pipeline.ecg_pipeline_tools download-datasets --all
    python -m ecg_pipeline.ecg_pipeline_tools download-datasets --only mitdb svdb
    python -m ecg_pipeline.ecg_pipeline_tools download-icentia11k-full --workers 64
    python -m ecg_pipeline.ecg_pipeline_tools train-encoder --max-files 40 --epochs 15
    python -m ecg_pipeline.ecg_pipeline_tools train-classifiers --dataset mitdb
    python -m ecg_pipeline.ecg_pipeline_tools eval-classifier --model models/five_class_xgb.json --split-set ds2
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
import wfdb
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight

from .ecg_pipeline_core import (
    AAMI_CLASSES, BEATS, DATA_RAW, MODELS_DIR, TARGET_FS,
    ConformalRiskPredictor, FiveClassBeatClassifier, INPUT_LEN, PretrainResult,
    N_FEATURES, N_FEATURES_WITH_TIMING, TIMING_FEATURE_NAMES, _feature_width,
    apply_filter_chain, beat_feature_vector, detect_and_segment,
    discover_sensio_files, discover_vitalpatch_files, parse_sensio_ecg,
    parse_vitalpatch_ecg, pretrain_self_supervised, robust_zscore, run_sqi_gate,
    save_encoder, segment_beats, to_target_rate,
)


# ============================================================================
# splits.py — Single source of truth for every record-ID split
# ============================================================================
"""Single source of truth for every record-ID split used in training/eval.

MITDB_DS1 / MITDB_DS2 / SVDB_RECORDS were previously hardcoded inside
train_classifiers.py; moved here so eval_classifier.py (and any future
script) can reference the same lists without a circular import, and so
there is exactly one place that defines "what is DS1" (patient-level
splits are meaningless if two files can silently drift out of sync).

DS1_TRAIN / DS1_VAL: patient-level validation carve-out of DS1, added so
tuning decisions (feature engineering, class-imbalance handling) have an
honest held-out signal instead of being tuned directly against DS2 (DS2
must stay report-only — see RESEARCH_AUDIT.md).

Record selection for DS1_VAL was NOT random: MITDB record 208 alone
contains 373 of DS1's ~415 real F-class beats (F is already critically
scarce). Randomly assigning 208 to validation would gut training's only
real F signal. Records 118/201/207/223 were chosen instead because,
together, they carry meaningful S (404 of DS1's 944 S beats) and V (897)
support for validation while leaving 208 and the bulk of F-bearing
records (108/109/114/124/203/205/215) in training. Verified by direct
annotation count (AAMI-mapped) before finalizing:

  DS1_VAL beats  : S=404  V=897  F=16    (118:S96, 201:S128/V198/F2,
                    207:S107/V210, 223:V473/S73/F14)
  DS1_TRAIN beats: S=540  V~=rest F=399  (keeps 208's 373 F beats)

This is a one-time, documented, non-random choice — re-run the same
annotation-count check before changing it.

2026-07-20 update — enlarged DS1_VAL (4 -> 10 records): the 4-record
carve-out above turned out to be too small a *patient* sample for its
governing use (Stage-1 threshold tuning for the two-stage classifier,
see ABLATION_REPORT.md Experiment C) — DS1_VAL and DS2 disagreed
substantially on how fast S-recall degrades with threshold, and a
4-patient split can't distinguish "real effect" from "which 4 patients
happened to be in it." Doubled to 10 records by adding 114/116/124/
205/215/220 to the original 4, using the same protective logic as
208/F above: records 209 (S=351, the single largest S carrier outside
the original 4) and 106/119 (V=516/444, the two largest V carriers)
were deliberately KEPT IN TRAINING rather than added to VAL, so
training doesn't lose its dominant S/V signal the same way it would if
208 were pulled for F. Verified by direct annotation count:

  DS1_VAL  (10 records): N=20384 S=510  V=1285 F=36  Q=0     total=22215
  DS1_TRAIN(12 records): N=23130 S=363  V=2338 F=367 Q=7     total=26205

Same one-time, documented, non-random methodology as the original
choice — re-run the same annotation-count check before changing it
again.
"""

MITDB_DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
             201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
MITDB_DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
             213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

SVDB_RECORDS = [800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812,
                820, 821, 822, 823, 824, 825, 826, 827, 828, 829,
                840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850,
                851, 852, 853, 854, 855, 856, 857, 858, 859, 860,
                861, 862, 863, 864, 865, 866, 867, 868, 869, 870,
                871, 872, 873, 874, 875, 876, 877, 878, 879, 880,
                881, 882, 883, 884, 885, 886, 887, 888, 889, 890,
                891, 892, 893, 894]

# INCART: all 75 records, already fully present at data/raw/public/incartdb
# (downloaded in an earlier session, per download_datasets.py's budget
# plan). Train-only enrichment, same as SVDB -- MITDB DS2 stays the only
# held-out test set. Filenames are "I01".."I75", not bare integers, so
# these are strings (build_dataset/_load_record_beats take str(rid) and
# glob for f"{rid}.hea", which works for either type).
INCART_RECORDS = [f"I{i:02d}" for i in range(1, 76)]

# LTAFDB (Long Term AF Database): 84 records, 24h+ Holter recordings, real
# per-beat N/V/A(->S) annotations (confirmed via a sample record before
# downloading the rest -- see ABLATION_REPORT.md / SESSION_LOG). Train-only.
# Fetched verbatim via wfdb.get_record_list("ltafdb") -- do not hand-edit.
LTAFDB_RECORDS = ["00", "01", "03", "05", "06", "07", "08", "10", "100", "101",
                   "102", "103", "104", "105", "11", "110", "111", "112", "113", "114",
                   "115", "116", "117", "118", "119", "12", "120", "121", "122", "13",
                   "15", "16", "17", "18", "19", "20", "200", "201", "202", "203",
                   "204", "205", "206", "207", "208", "21", "22", "23", "24", "25",
                   "26", "28", "30", "32", "33", "34", "35", "37", "38", "39",
                   "42", "43", "44", "45", "47", "48", "49", "51", "53", "54",
                   "55", "56", "58", "60", "62", "64", "65", "68", "69", "70",
                   "71", "72", "74", "75"]

# SDDB (Sudden Cardiac Death Holter Database): 23 records. Smaller than
# LTAFDB but notable for actually containing real F-class beats (a sample
# record had 75 F beats -- F has essentially no other real-data source in
# this pipeline besides MITDB/SVDB's ~410 examples). Train-only.
# Fetched verbatim via wfdb.get_record_list("sddb") -- do not hand-edit.
SDDB_RECORDS = ["30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
                 "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
                 "50", "51", "52"]

_DS1_VAL_RECORDS = [114, 116, 118, 124, 201, 205, 207, 215, 220, 223]

DS1_VAL = sorted(_DS1_VAL_RECORDS)
DS1_TRAIN = sorted(set(MITDB_DS1) - set(_DS1_VAL_RECORDS))

assert set(DS1_TRAIN) | set(DS1_VAL) == set(MITDB_DS1)
assert set(DS1_TRAIN) & set(DS1_VAL) == set()
assert set(DS1_VAL) & set(MITDB_DS2) == set()
assert set(DS1_TRAIN) & set(MITDB_DS2) == set()


# ============================================================================
# download_datasets.py — CLI: download a disk-conscious subset of public
# labeled ECG datasets
# ============================================================================
_DOWNLOAD_DATASETS_DOC = """CLI: download a disk-conscious subset of public labeled ECG datasets.

Fills in the labeled-data gap the pipeline was built without (see the
"Known limitations" section of README.md). Every dataset here is
PhysioNet Open Access — no credentialing needed.

Budget (confirmed with the user given ~7TB free but a shared, already
78%-full research filesystem):
  - MITDB, SVDB, INCART, CUDB: downloaded in FULL — each is small
    (104MB / 52MB / 795MB / 6MB) and all four are directly usable by
    FiveClassBeatClassifier (recommendation #7).
  - CinC2017 (challenge-2017): only `training2017.zip` (~95MB) — the
    single-lead AFib/Normal/Other/Noisy labels this pipeline's
    AFIB_SUSPECTED heuristic needs a real classifier for (recommendation
    #6's rhythm classifier).
  - Icentia11k: capped at a SMALL subset (40 patients x 3 segments each,
    ~250MB) rather than the full 188.3GB ZIP — the closest domain match
    to VitalPatch/SeNSiO (continuous single-lead wearable data), but far
    too large to pull in full here.

Usage:
    python -m ecg_pipeline.ecg_pipeline_tools download-datasets --all
    python -m ecg_pipeline.ecg_pipeline_tools download-datasets --only mitdb svdb
"""

PUBLIC_DIR = DATA_RAW / "public"
PHYSIONET_FILES = "https://physionet.org/files"

WFDB_DATABASES = {
    "mitdb": "mitdb",
    "svdb": "svdb",
    "incartdb": "incartdb",
    "cudb": "cudb",
}

ICENTIA11K_DB = "icentia11k-continuous-ecg/1.0"
ICENTIA11K_N_PATIENTS = 40
ICENTIA11K_SEGMENTS_PER_PATIENT = 3
# Spread across the p00-p09 groups (1000 patients/group) instead of one
# contiguous block, so the subset isn't biased toward a single group.
ICENTIA11K_PATIENT_IDS = [
    f"p{group:02d}{offset:03d}"
    for group in range(10)
    for offset in (1, 251, 501, 751)
]  # 10 groups x 4 patients = 40


def download_wfdb_database(name: str, db_dir: str) -> None:
    import wfdb

    dest = PUBLIC_DIR / name
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] downloading full database ({db_dir}) -> {dest}")
    wfdb.dl_database(db_dir, str(dest))
    print(f"[{name}] done.")


def download_cinc2017(chunk_size: int = 1 << 20) -> None:
    dest = PUBLIC_DIR / "challenge2017"
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "training2017.zip"

    url = f"{PHYSIONET_FILES}/challenge-2017/1.0.0/training2017.zip"
    if zip_path.exists():
        print(f"[challenge2017] {zip_path.name} already present, skipping download.")
    else:
        print(f"[challenge2017] downloading {url} -> {zip_path}")
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    f.write(chunk)

    print("[challenge2017] extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    print(f"[challenge2017] done -> {dest}")


def download_icentia11k(patient_ids: list[str] = ICENTIA11K_PATIENT_IDS,
                         n_segments: int = ICENTIA11K_SEGMENTS_PER_PATIENT) -> None:
    dest_root = PUBLIC_DIR / "icentia11k"
    dest_root.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    for pid in patient_ids:
        group = pid[:3]  # e.g. "p00"
        patient_dest = dest_root / group / pid
        patient_dest.mkdir(parents=True, exist_ok=True)

        for seg in range(n_segments):
            seg_name = f"{pid}_s{seg:02d}"
            for ext in ("hea", "dat", "atr"):
                fname = f"{seg_name}.{ext}"
                out_path = patient_dest / fname
                if out_path.exists():
                    total_bytes += out_path.stat().st_size
                    continue
                url = f"{PHYSIONET_FILES}/{ICENTIA11K_DB}/{group}/{pid}/{fname}"
                resp = requests.get(url, timeout=60)
                if resp.status_code == 404:
                    continue  # some patients have fewer than n_segments recorded
                resp.raise_for_status()
                out_path.write_bytes(resp.content)
                total_bytes += len(resp.content)
        print(f"[icentia11k] {pid}: {n_segments} segment(s) fetched "
              f"({total_bytes / 1e6:.1f} MB total so far)")

    print(f"[icentia11k] done. {len(patient_ids)} patients, "
          f"{total_bytes / 1e6:.1f} MB -> {dest_root}")


def main_download_datasets(argv=None):
    parser = argparse.ArgumentParser(description=_DOWNLOAD_DATASETS_DOC)
    all_names = list(WFDB_DATABASES) + ["challenge2017", "icentia11k"]
    parser.add_argument("--all", action="store_true", help="download everything in the budget plan")
    parser.add_argument("--only", nargs="+", choices=all_names, help="download only these datasets")
    args = parser.parse_args(argv)

    if not args.all and not args.only:
        parser.error("pass --all or --only <names>")

    selected = all_names if args.all else args.only

    for name, db_dir in WFDB_DATABASES.items():
        if name in selected:
            download_wfdb_database(name, db_dir)
    if "challenge2017" in selected:
        download_cinc2017()
    if "icentia11k" in selected:
        download_icentia11k()


# ============================================================================
# download_icentia11k_full.py — Download the FULL Icentia11k dataset via S3
# ============================================================================
_DOWNLOAD_ICENTIA11K_FULL_DOC = """Download the FULL Icentia11k dataset (~188GB, 11,000 patients) via its
public AWS S3 mirror (s3://physionet-open/icentia11k-continuous-ecg/1.0/),
which is far faster than PhysioNet's own web server for this dataset
(~240KB/s single-connection there vs. ~10MB/s achievable here with
concurrent requests against S3).

No AWS credentials needed — bucket is public, accessed over plain HTTPS.

Two phases, both resumable:
  1. List every object under the prefix (paginated ListObjectsV2 XML API)
     and cache the manifest to disk.
  2. Download every object with a thread pool, skipping any file that
     already exists locally with the correct size (so this is safe to
     re-run after an interruption, and won't re-fetch the existing
     40-patient x 3-segment subset already on disk).

Usage:
    python -m ecg_pipeline.ecg_pipeline_tools download-icentia11k-full
    python -m ecg_pipeline.ecg_pipeline_tools download-icentia11k-full --workers 64
"""

BUCKET_URL = "https://physionet-open.s3.amazonaws.com"
PREFIX = "icentia11k-continuous-ecg/1.0/"
DEST_ROOT = DATA_RAW / "public" / "icentia11k"
MANIFEST_PATH = DEST_ROOT / "_s3_manifest.tsv"

S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def list_all_objects() -> list[tuple[str, int]]:
    """Returns [(key, size_bytes), ...] for every object under PREFIX,
    via paginated ListObjectsV2. Cached to MANIFEST_PATH once complete."""
    if MANIFEST_PATH.exists():
        print(f"Using cached manifest at {MANIFEST_PATH}")
        objects = []
        for line in MANIFEST_PATH.read_text().splitlines():
            key, size = line.rsplit("\t", 1)
            objects.append((key, int(size)))
        return objects

    print("Listing all objects (paginated) — this takes a few minutes...")
    objects: list[tuple[str, int]] = []
    token = None
    page = 0
    while True:
        params = {"list-type": "2", "prefix": PREFIX, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        resp = requests.get(BUCKET_URL, params=params, timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for content in root.findall(f"{S3_NS}Contents"):
            key = content.find(f"{S3_NS}Key").text
            size = int(content.find(f"{S3_NS}Size").text)
            objects.append((key, size))
        page += 1
        if page % 20 == 0:
            print(f"  ...{page} pages, {len(objects)} objects so far")

        is_truncated = root.find(f"{S3_NS}IsTruncated").text == "true"
        if not is_truncated:
            break
        token = root.find(f"{S3_NS}NextContinuationToken").text

    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        for key, size in objects:
            f.write(f"{key}\t{size}\n")
    print(f"Manifest complete: {len(objects)} objects -> {MANIFEST_PATH}")
    return objects


def _download_one(key: str, size: int) -> int:
    """Returns bytes actually downloaded (0 if skipped as already present)."""
    rel_path = key[len(PREFIX):]  # e.g. "p00/p00001/p00001_s00.dat"
    if not rel_path:
        return 0
    out_path = DEST_ROOT / rel_path
    if out_path.exists() and out_path.stat().st_size == size:
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BUCKET_URL}/{key}"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    return len(resp.content)


def download_all(workers: int = 48) -> None:
    objects = list_all_objects()
    total_bytes = sum(size for _, size in objects)
    print(f"Total dataset size: {total_bytes / 1e9:.2f} GB across {len(objects)} objects")

    done_count = 0
    done_bytes = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one, key, size): (key, size) for key, size in objects}
        for fut in as_completed(futures):
            key, size = futures[fut]
            try:
                downloaded = fut.result()
            except Exception as e:
                print(f"  FAILED {key}: {e}")
                continue
            done_count += 1
            done_bytes += downloaded if downloaded else size  # count skipped bytes as "already done"
            if done_count % 2000 == 0:
                elapsed = time.time() - start
                rate_mb_s = (done_bytes / 1e6) / elapsed if elapsed > 0 else 0
                pct = 100 * done_bytes / total_bytes
                print(f"  {done_count}/{len(objects)} objects, "
                      f"{done_bytes / 1e9:.2f}/{total_bytes / 1e9:.2f} GB ({pct:.1f}%), "
                      f"{rate_mb_s:.2f} MB/s avg")

    elapsed = time.time() - start
    print(f"\nDone. {done_count} objects processed in {elapsed / 3600:.2f} hours.")


def main_download_icentia11k_full(argv=None):
    parser = argparse.ArgumentParser(description=_DOWNLOAD_ICENTIA11K_FULL_DOC)
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args(argv)
    download_all(workers=args.workers)


# ============================================================================
# train_encoder.py — CLI: self-supervised pretrain the ECG foundation encoder
# ============================================================================
_TRAIN_ENCODER_DOC = """CLI: self-supervised pretrain the ECG foundation encoder on our own
unlabeled VitalPatch/SeNSiO recordings (recommendation #2).

Usage:
    python -m ecg_pipeline.ecg_pipeline_tools train-encoder --max-files 20 --epochs 20
"""


def collect_wide_windows(max_files: int = 20, clip_value: float | None = None) -> np.ndarray:
    """Runs stages 1-5 on real local recordings and collects the resulting
    wide beat windows (unlabeled) as pretraining input."""
    windows = []

    vp_files = discover_vitalpatch_files(DATA_RAW / "vitalpatch")[:max_files]
    for f in vp_files:
        try:
            for rec in parse_vitalpatch_ecg(f):
                windows.extend(_windows_from_recording(rec.signal_mv, rec.timestamps_ms, rec.fs_nominal,
                                                         rec.already_bandpass_filtered, clip_value))
        except Exception as e:  # noqa: BLE001 - best-effort corpus building over many real files
            print(f"  skip {f.name}: {e}", file=sys.stderr)

    se_files = discover_sensio_files(DATA_RAW / "sense_io")[:max_files]
    for f in se_files:
        try:
            rec = parse_sensio_ecg(f)
            windows.extend(_windows_from_recording(rec.signal_mv, rec.timestamps_ms, rec.fs_nominal,
                                                     rec.already_bandpass_filtered, clip_value))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {f.name}: {e}", file=sys.stderr)

    return np.array(windows) if windows else np.zeros((0, INPUT_LEN))


def _windows_from_recording(signal, timestamps_ms, fs_nominal, already_filtered, clip_value) -> list[np.ndarray]:
    keep_mask, _ = run_sqi_gate(signal, timestamps_ms, fs_nominal, clip_value=clip_value)
    clean = signal.copy().astype(float)
    clean[~keep_mask] = np.nan

    resampled, t_resampled = to_target_rate(clean, timestamps_ms, fs_nominal, TARGET_FS)
    valid = ~np.isnan(resampled)
    if not valid.any():
        return []
    filled = np.interp(t_resampled, t_resampled[valid], resampled[valid])
    filtered = apply_filter_chain(filled, TARGET_FS, already_bandpass_filtered=already_filtered)

    beats = detect_and_segment(filtered, TARGET_FS, BEATS)
    return [robust_zscore(b.wide_window) for b in beats
            if b.wide_window is not None and not b.quality_rejected]


def main_train_encoder(argv=None):
    parser = argparse.ArgumentParser(description=_TRAIN_ENCODER_DOC)
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out", type=Path, default=MODELS_DIR / "ecg_encoder.pt")
    args = parser.parse_args(argv)

    print("Collecting unlabeled beat windows from real local recordings...")
    windows = collect_wide_windows(max_files=args.max_files)
    print(f"Collected {len(windows)} beat windows.")
    if len(windows) < 50:
        print("Too few windows to pretrain meaningfully; point --max-files at more recordings.")
        return

    encoder, result = pretrain_self_supervised(windows, epochs=args.epochs)
    print(f"Pretraining done: {result.epochs} epochs, final loss {result.final_loss:.4f}, "
          f"{result.n_windows} windows.")

    save_encoder(encoder, args.out, meta={"epochs": result.epochs, "final_loss": result.final_loss,
                                           "n_windows": result.n_windows, "source": "vitalpatch+sensio (unlabeled)"})
    print(f"Saved encoder to {args.out}")


# ============================================================================
# train_classifiers.py — CLI: fit FiveClassBeatClassifier + calibrate
# ConformalRiskPredictor
# ============================================================================
_TRAIN_CLASSIFIERS_DOC = """CLI: fit FiveClassBeatClassifier on real labeled public data and
calibrate ConformalRiskPredictor — the "flip .fit() on" step that
`classify.py`'s docstring says is all that's needed once labeled data
exists (recommendation #7).

Beats are segmented using the SAME `beats.segment_beats` / `features.py`
code path production inference uses, but seeded with each dataset's
ground-truth annotation sample positions instead of XQRS detection, so
training features are extracted identically to how they'll be extracted
at inference time.

Usage:
    python -m ecg_pipeline.ecg_pipeline_tools train-classifiers --dataset mitdb
"""

# Standard AAMI EC57 beat-symbol mapping (the same grouping Zhu et al. 2021
# and the rest of the review's reference papers use).
AAMI_SYMBOL_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}


def _snap_to_local_peak(signal: np.ndarray, r_peaks: np.ndarray, search_radius: int = 15) -> np.ndarray:
    """Annotation sample positions, rescaled from the original fs (e.g.
    MITDB's 360Hz) to TARGET_FS and run through resampling + the filter
    chain, land close to but not exactly on the true local extremum
    (empirically ~3 samples off on MITDB record 100, consistent enough to
    be a rounding/interpolation artefact rather than noise). Beat
    segmentation's own beat-level SQI check rejects any beat whose R-peak
    isn't the true local max — appropriate for XQRS-detected peaks, which
    are true maxima by construction, but not for rescaled annotations. So
    snap each annotation to the true nearby extremum before segmenting,
    same as any annotation-to-resampled-timeline training pipeline needs.
    """
    snapped = r_peaks.copy()
    for i, r in enumerate(r_peaks):
        lo, hi = max(0, r - search_radius), min(len(signal), r + search_radius)
        if hi <= lo:
            continue
        snapped[i] = lo + int(np.argmax(np.abs(signal[lo:hi])))
    return snapped


def _load_record_beats(record_path: Path, ann_ext: str = "atr",
                        include_timing: bool = False,
                        drop_compensatory_pause: bool = False,
                        timing_only: bool = False,
                        include_r_amp: bool = False,
                        include_qrs_shape: bool = False) -> tuple[np.ndarray, list[str]]:
    """Returns (feature_matrix, labels) for every valid, non-rejected beat
    in one WFDB record, using ground-truth annotation positions."""
    record = wfdb.rdrecord(str(record_path))
    ann = wfdb.rdann(str(record_path), ann_ext)
    fs = float(record.fs)
    signal = record.p_signal[:, 0].astype(np.float64)

    keep = [i for i, sym in enumerate(ann.symbol) if sym in AAMI_SYMBOL_MAP]
    if not keep:
        return np.zeros((0, _feature_width(include_timing, drop_compensatory_pause,
                                            timing_only, include_r_amp, include_qrs_shape))), []
    ann_samples = ann.sample[keep]
    ann_labels = [AAMI_SYMBOL_MAP[ann.symbol[i]] for i in keep]

    timestamps_ms = np.arange(len(signal)) * (1000.0 / fs)
    resampled, t_resampled = to_target_rate(signal, timestamps_ms, fs, TARGET_FS)
    filtered = apply_filter_chain(resampled, TARGET_FS, already_bandpass_filtered=False)

    scale = TARGET_FS / fs
    r_peaks_resampled = np.round(ann_samples * scale).astype(int)
    r_peaks_resampled = np.clip(r_peaks_resampled, 0, len(filtered) - 1)
    r_peaks_resampled = _snap_to_local_peak(filtered, r_peaks_resampled)

    # NOTE: r_peaks_resampled here is built from the ANNOTATION file's beat
    # positions (ground truth), not from run_time R-peak detection -- so
    # rr_ms/the K=8/16/32 windows segment_beats computes from it reflect the
    # true rhythm, matching what a correctly-functioning runtime R-peak
    # detector would see. This is the same beats/segment_beats call runtime
    # inference uses (via detect_and_segment), so train and inference share
    # one computation path end to end, not just for beat_feature_vector.
    beats = segment_beats(filtered, TARGET_FS, r_peaks_resampled, BEATS)
    primary_pre_samples = int(round(BEATS.primary_pre_ms / 1000.0 * TARGET_FS))

    rows, labels = [], []
    for beat, label in zip(beats, ann_labels):
        vec = beat_feature_vector(beat, primary_pre_samples, include_timing=include_timing,
                                   drop_compensatory_pause=drop_compensatory_pause,
                                   timing_only=timing_only, include_r_amp=include_r_amp,
                                   include_qrs_shape=include_qrs_shape)
        if vec is not None:
            rows.append(vec)
            labels.append(label)

    return (np.vstack(rows) if rows else
            np.zeros((0, _feature_width(include_timing, drop_compensatory_pause,
                                         timing_only, include_r_amp, include_qrs_shape)))), labels


def build_dataset(db_dir: Path, record_ids: list[int],
                   include_timing: bool = False,
                   drop_compensatory_pause: bool = False,
                   timing_only: bool = False,
                   include_r_amp: bool = False,
                   include_qrs_shape: bool = False) -> tuple[np.ndarray, list[str]]:
    all_X, all_y = [], []
    for rid in record_ids:
        record_path = db_dir / str(rid)
        if not record_path.with_suffix(".hea").exists():
            print(f"  skip {rid}: not downloaded")
            continue
        if not record_path.with_suffix(".atr").exists():
            # Some records ship only unaudited (.ari) annotations on PhysioNet,
            # no expert-reviewed .atr at all -- e.g. 11 of SDDB's 23 records.
            # Skip rather than mixing in noisier auto-generated labels.
            print(f"  skip {rid}: no .atr (expert) annotations available")
            continue
        X, y = _load_record_beats(record_path, include_timing=include_timing,
                                   drop_compensatory_pause=drop_compensatory_pause,
                                   timing_only=timing_only, include_r_amp=include_r_amp,
                                   include_qrs_shape=include_qrs_shape)
        print(f"  {rid}: {len(y)} labeled beats")
        all_X.append(X)
        all_y.extend(y)
    X = np.vstack(all_X) if all_X else np.zeros(
        (0, _feature_width(include_timing, drop_compensatory_pause, timing_only, include_r_amp,
                            include_qrs_shape)))
    return X, all_y


def per_class_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=AAMI_CLASSES, zero_division=0)
    return {c: {"sensitivity": round(recall[i], 4), "precision": round(precision[i], 4),
                "f1": round(f1[i], 4), "support": int(support[i])}
            for i, c in enumerate(AAMI_CLASSES)}


def random_oversample(X: np.ndarray, y: list[str], minority_ratio: float = 1.0 / 3.0,
                       seed: int = 0,
                       per_class_ratio_overrides: dict[str, float] | None = None
                       ) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Mild Random Oversampling: any class below `minority_ratio` of the
    majority class's count is oversampled (with replacement) up to that
    floor. Classes already at or above it are left untouched — this is a
    1:3 floor, not a 1:1 rebalance, since 1:1 over-represents rare classes
    like F given how few real examples exist (Talukder et al. found ROS
    beats SMOTE/ADASYN/GAN for this exact imbalance regime).

    Also returns the row-index array used to build the output (original
    indices followed by duplicated ones), so a caller can apply the same
    resampling to a parallel array (e.g. a per-row data-source tag).

    `per_class_ratio_overrides` (e.g. {"F": 0.6}) replaces `minority_ratio`
    for just that class, so F can be given its own floor independent of
    S/V — added because lumping F under the same 1:3 floor as S/V means F
    (already the rarest real class, ~410 examples) gets the exact same
    proportional boost as S despite being an order of magnitude scarcer,
    which was one hypothesis for why F gets absorbed into S/V at
    classification time (see train_classifiers.py Task 3 ablation).
    """
    rng = np.random.default_rng(seed)
    y_arr = np.array(y)
    # sorted(), not set() iteration order: Python randomizes str hash order
    # per-process (PYTHONHASHSEED), so `for cls in set(y)` visits classes in
    # a different order each run -- since every class's oversampled rows are
    # drawn from the same shared `rng` in sequence, a different visit order
    # consumes the seeded random stream differently and silently produces a
    # different ROS result on every run despite an identical `seed`. This is
    # what task_4's reproducibility rerun caught (two seed=42 runs of the
    # exact same config gave different DS2 Macro-F1: 0.3983 vs 0.3945).
    counts = {c: int((y_arr == c).sum()) for c in sorted(set(y))}
    majority_count = max(counts.values())
    overrides = per_class_ratio_overrides or {}

    idx_parts = [np.arange(len(y))]
    for cls, count in counts.items():
        ratio = overrides.get(cls, minority_ratio)
        floor = int(majority_count * ratio)
        if count < floor and count > 0:
            idxs = np.where(y_arr == cls)[0]
            extra = rng.choice(idxs, size=floor - count, replace=True)
            idx_parts.append(extra)

    all_idx = np.concatenate(idx_parts)
    X_ros = X[all_idx]
    y_ros = y_arr[all_idx].tolist()
    return X_ros, y_ros, all_idx


def list_icentia11k_patients(icentia_dir: Path) -> list[Path]:
    """All patient directories currently on disk under the bucketed
    `pXX/pXXXXX` layout — works with however much of the dataset has been
    downloaded so far, not the full 11,000-patient set."""
    return sorted(icentia_dir.glob("p*/p*"))


def select_icentia11k_patients(patient_dirs: list[Path], n_patients: int, seed: int) -> list[Path]:
    rng = np.random.default_rng(seed)
    if n_patients >= len(patient_dirs):
        return patient_dirs
    idx = rng.choice(len(patient_dirs), size=n_patients, replace=False)
    return [patient_dirs[i] for i in sorted(idx)]


def build_icentia11k_dataset(patient_dirs: list[Path], segments_per_patient: int,
                              seed: int) -> tuple[np.ndarray, list[str]]:
    """Loads a random `segments_per_patient` ~1-hour segment(s) from each
    given Icentia11k patient. Reuses `_load_record_beats` unchanged —
    Icentia11k's raw N/S/V/Q beat symbols already map onto
    `AAMI_SYMBOL_MAP` as-is (no F symbol exists in this dataset at all),
    its `.atr` annotation extension matches the default, and its 250Hz
    native rate hits the exact-integer-ratio decimation path in
    `resample.to_target_rate` (already written with this dataset in mind).
    A patient-subsample is used, not the full download, because at
    ~5,000 beats/segment even a few hundred patients dwarfs the current
    ~220K-beat MITDB+SVDB training set.
    """
    rng = np.random.default_rng(seed)
    all_X, all_y = [], []
    for pdir in patient_dirs:
        hea_files = sorted(pdir.glob("*.hea"))
        if not hea_files:
            continue
        n_pick = min(segments_per_patient, len(hea_files))
        chosen = rng.choice(len(hea_files), size=n_pick, replace=False)
        for i in chosen:
            record_path = hea_files[i].with_suffix("")
            try:
                X, y = _load_record_beats(record_path)
            except Exception as e:
                print(f"  skip {record_path.name}: {e}")
                continue
            if len(y):
                all_X.append(X)
                all_y.extend(y)
    X = np.vstack(all_X) if all_X else np.zeros((0, N_FEATURES))
    return X, all_y


def main_train_classifiers(argv=None):
    parser = argparse.ArgumentParser(description=_TRAIN_CLASSIFIERS_DOC)
    parser.add_argument("--dataset", choices=["mitdb"], default="mitdb")
    parser.add_argument("--data-root", type=Path, default=DATA_RAW / "public")
    parser.add_argument("--out", type=Path, default=MODELS_DIR / "five_class_xgb.json")
    parser.add_argument("--include-svdb", action=argparse.BooleanOptionalAction, default=True,
                         help="add all 78 SVDB records into training (S-class boost); DS2 stays pure MITDB")
    parser.add_argument("--include-incart", action=argparse.BooleanOptionalAction, default=False,
                         help="add all 75 INCART records into training (12-lead source, channel 0 only, "
                              "same as every other WFDB record here); DS2 stays pure MITDB")
    parser.add_argument("--include-ltafdb", action=argparse.BooleanOptionalAction, default=False,
                         help="add all 84 LTAFDB records into training (24h+ Holter, real per-beat "
                              "N/V/A(->S) annotations, plus AFib rhythm context); DS2 stays pure MITDB")
    parser.add_argument("--include-sddb", action=argparse.BooleanOptionalAction, default=False,
                         help="add all 23 SDDB records into training -- notable for containing real "
                              "F-class beats, unlike Icentia11k which has none; DS2 stays pure MITDB")
    parser.add_argument("--drop-q", action=argparse.BooleanOptionalAction, default=True,
                         help="exclude Q-class beats from the training objective (too few real examples "
                              "to be learnable — 7 in MITDB DS2 alone; handle paced beats with a rule instead)")
    parser.add_argument("--include-icentia11k", action=argparse.BooleanOptionalAction, default=False,
                         help="add a random patient subsample from the downloaded Icentia11k data "
                              "(S/V boost only — it has no F-class labels)")
    parser.add_argument("--icentia11k-patients", type=int, default=750,
                         help="number of Icentia11k patients to randomly sample from whatever has "
                              "been downloaded so far")
    parser.add_argument("--icentia11k-segments-per-patient", type=int, default=1,
                         help="how many ~1-hour segments to load per sampled patient")
    parser.add_argument("--icentia11k-weight", type=float, default=0.4,
                         help="relative sample_weight multiplier applied to Icentia11k-origin training "
                              "rows, down-weighting them vs MITDB/SVDB's more rigorously verified labels")
    parser.add_argument("--icentia11k-seed", type=int, default=None,
                         help="defaults to --seed if not given")
    parser.add_argument("--train-split", choices=["ds1_train", "ds1_full"], default="ds1_train",
                         help="ds1_train (default): train on splits.DS1_TRAIN, hold out splits.DS1_VAL "
                              "for honest validation reporting. ds1_full: train on all of DS1 (no "
                              "validation report) — only for building a final model AFTER tuning "
                              "decisions have already been made on the validation set.")
    parser.add_argument("--seed", type=int, default=42,
                         help="fixed random_state for ROS, XGBoost, and Icentia11k sampling — "
                              "two runs with the same config and same seed must match exactly")
    parser.add_argument("--ros", action=argparse.BooleanOptionalAction, default=True,
                         help="apply Random Oversampling (1:3-of-majority floor) to the minority classes")
    parser.add_argument("--balanced-weights", action=argparse.BooleanOptionalAction, default=True,
                         help="apply sklearn compute_sample_weight('balanced') on top of "
                              "(whatever the --ros result is). Default True+True stacks both, which "
                              "STATUS_QA.md flagged as a possible double-correction — use --no-ros / "
                              "--no-balanced-weights to ablate each in isolation.")
    parser.add_argument("--f-ros-ratio", type=float, default=None,
                         help="dedicated ROS floor for F only (fraction of majority-class count), "
                              "overriding the shared 1:3 floor used for S/V. E.g. 0.6 gives F its own, "
                              "much higher floor without changing S/V's oversampling.")
    parser.add_argument("--f-weight-multiplier", type=float, default=1.0,
                         help="extra sample_weight multiplier applied ONLY to F-labeled rows, on top of "
                              "whatever ROS/balanced-weight combination is used — a misclassification-cost "
                              "knob distinct from the general class-imbalance handling above.")
    parser.add_argument("--timing-features", action=argparse.BooleanOptionalAction, default=False,
                         help="append the 7 local-rhythm-context features (rolling RR-ratio at "
                              "K=8/16/32, rr_pre/rr_post ratio, RR CV, prematurity score, "
                              "compensatory-pause flag) to the 56-dim base vector, giving a 63-dim "
                              "vector (see beat_feature_vector in ecg_pipeline_core.py). Opt-in only "
                              "-- production's 56-dim vector is the default everywhere. Not "
                              "compatible with --include-icentia11k in this build (build_icentia11k_dataset "
                              "does not thread this flag through; combining the two will crash on the "
                              "vstack dimension mismatch rather than silently misbuild).")
    parser.add_argument("--drop-compensatory-pause", action=argparse.BooleanOptionalAction, default=False,
                         help="only meaningful with --timing-features: drop just the "
                              "compensatory_pause_flag column, keeping the other 6 timing features "
                              "(62-dim vector). Micro-ablation from ABLATION_REPORT.md's timing-v1 "
                              "run, where that one feature was rank 1/63 by gain and is the "
                              "suspected driver of the S->V confusion that regressed V's DS2 F1.")
    parser.add_argument("--timing-only", action=argparse.BooleanOptionalAction, default=False,
                         help="use ONLY the 7 (or 6 with --drop-compensatory-pause) timing "
                              "features -- no morphology/wavelet at all. For the feature-family "
                              "ablation (morphology-only vs timing-only vs combined) in "
                              "ABLATION_REPORT.md, testing whether morphology alone can separate "
                              "S/V before committing to a two-stage classifier design.")
    parser.add_argument("--include-r-amp", action=argparse.BooleanOptionalAction, default=False,
                         help="append R-peak amplitude as a 6th morphological feature. "
                              "_morphological_features computed this (r_amp) but never returned "
                              "it -- dead code, found during two-stage-classifier prerequisite "
                              "work (ABLATION_REPORT.md 'Prerequisite 2'). Appended after the "
                              "existing 5, so their indices never shift. Opt-in only -- production's "
                              "56-dim vector is unaffected by default.")
    args = parser.parse_args(argv)
    if (args.timing_features or args.timing_only or args.include_r_amp) and args.include_icentia11k:
        parser.error("--timing-features/--timing-only/--include-r-amp are not compatible with "
                     "--include-icentia11k in this build: build_icentia11k_dataset() does not "
                     "thread these flags through, so the two sources would vstack at mismatched widths.")
    if args.icentia11k_seed is None:
        args.icentia11k_seed = args.seed
    if not args.out.is_absolute():
        # `-m ecg_pipeline.ecg_pipeline_tools train-classifiers` resolves relative paths
        # against the caller's cwd, not this package's directory, so a
        # relative --out can silently point outside models/ (and crash at
        # save time once training has already finished).
        args.out = (MODELS_DIR.parent / args.out).resolve()

    db_dir = args.data_root / args.dataset
    ds1_record_ids = DS1_TRAIN if args.train_split == "ds1_train" else MITDB_DS1
    print(f"Building DS1 train ({args.train_split}) from {db_dir} ...")
    X_train, y_train = build_dataset(db_dir, ds1_record_ids, include_timing=args.timing_features,
        drop_compensatory_pause=args.drop_compensatory_pause,
            timing_only=args.timing_only,
                include_r_amp=args.include_r_amp)
    print(f"Building DS2 (held-out test, report-only) from {db_dir} ...")
    X_test, y_test = build_dataset(db_dir, MITDB_DS2, include_timing=args.timing_features,
        drop_compensatory_pause=args.drop_compensatory_pause,
            timing_only=args.timing_only,
                include_r_amp=args.include_r_amp)
    X_val, y_val = (None, None)
    if args.train_split == "ds1_train":
        print(f"Building DS1_VAL (patient-level validation carve-out) from {db_dir} ...")
        X_val, y_val = build_dataset(db_dir, DS1_VAL, include_timing=args.timing_features,
            drop_compensatory_pause=args.drop_compensatory_pause,
                timing_only=args.timing_only,
                    include_r_amp=args.include_r_amp)

    if args.include_svdb:
        svdb_dir = args.data_root / "svdb"
        print(f"Building SVDB training data (S-class boost) from {svdb_dir} ...")
        X_svdb, y_svdb = build_dataset(svdb_dir, SVDB_RECORDS, include_timing=args.timing_features,
            drop_compensatory_pause=args.drop_compensatory_pause,
                timing_only=args.timing_only,
                    include_r_amp=args.include_r_amp)
        print(f"  SVDB contributes {len(y_svdb)} beats to training")
        X_train = np.vstack([X_train, X_svdb])
        y_train = y_train + y_svdb

    if args.include_incart:
        incart_dir = args.data_root / "incartdb"
        print(f"Building INCART training data from {incart_dir} ...")
        X_incart, y_incart = build_dataset(incart_dir, INCART_RECORDS, include_timing=args.timing_features,
            drop_compensatory_pause=args.drop_compensatory_pause,
                timing_only=args.timing_only,
                    include_r_amp=args.include_r_amp)
        print(f"  INCART contributes {len(y_incart)} beats to training")
        X_train = np.vstack([X_train, X_incart])
        y_train = y_train + y_incart

    if args.include_ltafdb:
        ltafdb_dir = args.data_root / "ltafdb"
        print(f"Building LTAFDB training data from {ltafdb_dir} ...")
        X_ltafdb, y_ltafdb = build_dataset(ltafdb_dir, LTAFDB_RECORDS, include_timing=args.timing_features,
            drop_compensatory_pause=args.drop_compensatory_pause,
                timing_only=args.timing_only,
                    include_r_amp=args.include_r_amp)
        print(f"  LTAFDB contributes {len(y_ltafdb)} beats to training")
        X_train = np.vstack([X_train, X_ltafdb])
        y_train = y_train + y_ltafdb

    if args.include_sddb:
        sddb_dir = args.data_root / "sddb"
        print(f"Building SDDB training data (F-class boost -- real F beats) from {sddb_dir} ...")
        X_sddb, y_sddb = build_dataset(sddb_dir, SDDB_RECORDS, include_timing=args.timing_features,
            drop_compensatory_pause=args.drop_compensatory_pause,
                timing_only=args.timing_only,
                    include_r_amp=args.include_r_amp)
        print(f"  SDDB contributes {len(y_sddb)} beats to training")
        X_train = np.vstack([X_train, X_sddb])
        y_train = y_train + y_sddb

    source_train = None
    if args.include_icentia11k:
        icentia_dir = args.data_root / "icentia11k"
        print(f"\nSelecting Icentia11k patient subsample from {icentia_dir} ...")
        all_patients = list_icentia11k_patients(icentia_dir)
        print(f"  {len(all_patients)} patient directories available on disk")
        chosen_patients = select_icentia11k_patients(all_patients, args.icentia11k_patients,
                                                       args.icentia11k_seed)
        print(f"  sampling {len(chosen_patients)} patients, "
              f"{args.icentia11k_segments_per_patient} segment(s) each")
        X_icentia, y_icentia = build_icentia11k_dataset(
            chosen_patients, args.icentia11k_segments_per_patient, args.icentia11k_seed)
        print(f"  Icentia11k contributes {len(y_icentia)} beats to training "
              f"(no F-class beats exist in this dataset)")
        source_train = ["primary"] * len(y_train) + ["icentia11k"] * len(y_icentia)
        X_train = np.vstack([X_train, X_icentia])
        y_train = y_train + y_icentia

    print(f"\nDS1(+SVDB{'+Icentia11k' if args.include_icentia11k else ''}) train: "
          f"{len(y_train)} beats, DS2 test: {len(y_test)} beats")
    if len(y_train) < 100 or len(y_test) < 100:
        print("Not enough labeled beats to train meaningfully — check --data-root.")
        return

    if args.drop_q:
        keep = [i for i, lab in enumerate(y_train) if lab != "Q"]
        n_dropped = len(y_train) - len(keep)
        X_train = X_train[keep]
        y_train = [y_train[i] for i in keep]
        if source_train is not None:
            source_train = [source_train[i] for i in keep]
        print(f"Dropped {n_dropped} Q-class beats from training (not learnable at this sample size)")

    print(f"Train class counts before ROS: {dict(Counter(y_train))}")
    if args.ros:
        f_overrides = {"F": args.f_ros_ratio} if args.f_ros_ratio is not None else None
        X_train_ros, y_train_ros, ros_idx = random_oversample(
            X_train, y_train, minority_ratio=1.0 / 3.0, seed=args.seed,
            per_class_ratio_overrides=f_overrides)
        print(f"Train class counts after ROS (1:3 floor{', F override=' + str(args.f_ros_ratio) if f_overrides else ''}): "
              f"{dict(Counter(y_train_ros))}")
    else:
        X_train_ros, y_train_ros, ros_idx = X_train, y_train, np.arange(len(y_train))
        print("ROS disabled (--no-ros): training on raw class counts")

    if args.balanced_weights:
        sample_weight = compute_sample_weight("balanced", y_train_ros)
    else:
        sample_weight = np.ones(len(y_train_ros), dtype=float)
        print("Balanced class weights disabled (--no-balanced-weights): uniform sample_weight")

    if args.f_weight_multiplier != 1.0:
        y_ros_arr = np.array(y_train_ros)
        sample_weight = sample_weight.copy()
        sample_weight[y_ros_arr == "F"] *= args.f_weight_multiplier
        print(f"Applied extra {args.f_weight_multiplier}x sample_weight multiplier to "
              f"{int((y_ros_arr == 'F').sum())} F-labeled training rows")

    if source_train is not None:
        source_train_ros = [source_train[i] for i in ros_idx]
        icentia_mask = np.array([s == "icentia11k" for s in source_train_ros])
        sample_weight = sample_weight.copy()
        sample_weight[icentia_mask] *= args.icentia11k_weight
        print(f"Down-weighted {int(icentia_mask.sum())} Icentia11k-origin training rows "
              f"by {args.icentia11k_weight}x")

    classifier = FiveClassBeatClassifier()
    classifier.fit(X_train_ros, y_train_ros, sample_weight=sample_weight, random_state=args.seed)
    classifier.save(args.out)
    print(f"Saved trained classifier -> {args.out}  (seed={args.seed})")

    if args.timing_features or args.timing_only:
        importances = classifier.model.feature_importances_
        timing_names = ([n for n in TIMING_FEATURE_NAMES if n != "compensatory_pause_flag"]
                         if args.drop_compensatory_pause else TIMING_FEATURE_NAMES)
        n_timing = len(timing_names)
        timing_importances = importances[-n_timing:]
        print(f"\nXGBoost feature importances (gain-based) for the {n_timing} timing features "
              f"(rank out of {len(importances)} total features, 1=most important):")
        ranks = (-importances).argsort().argsort()  # 0-based rank per feature, ties broken by index
        for name, imp, rank in zip(timing_names, timing_importances, ranks[-n_timing:]):
            print(f"  {name}: importance={imp:.4f}  rank={rank + 1}/{len(importances)}")
        if args.timing_only:
            print(f"  (timing-only model: all {n_timing} features ARE the model, "
                  f"importances sum to {timing_importances.sum():.4f} by construction)")
        else:
            print(f"  (sum of all 56 base-feature importances: {importances[:56].sum():.4f}, "
                  f"sum of {n_timing} timing importances: {timing_importances.sum():.4f})")

    def _predict(X):
        y_pred, proba_all = [], []
        for x in X:
            proba = classifier.model.predict_proba(x.reshape(1, -1))[0]
            proba_all.append(proba)
            classes = classifier._label_encoder.inverse_transform(np.arange(len(proba)))
            y_pred.append(classes[np.argmax(proba)])
        return y_pred, proba_all

    if X_val is not None and len(y_val) > 0:
        # Validation report — this is where all tuning decisions should be
        # made. Never used to fit/oversample/weight anything above.
        y_val_pred, _ = _predict(X_val)
        val_metrics = per_class_metrics(y_val, y_val_pred)
        val_macro_f1 = f1_score(y_val, y_val_pred, labels=AAMI_CLASSES, average="macro", zero_division=0)
        print(f"\nPer-class metrics on DS1_VAL (patient-level validation, tuning signal — NOT DS2):")
        for c, m in val_metrics.items():
            print(f"  {c}: sensitivity={m['sensitivity']:.3f} precision={m['precision']:.3f} "
                  f"f1={m['f1']:.3f} support={m['support']}")
        print(f"Macro-F1 on DS1_VAL: {val_macro_f1:.4f}")

    y_pred, proba_all = _predict(X_test)

    metrics = per_class_metrics(y_test, y_pred)
    print("\nPer-class metrics on DS2 (held-out test set, REPORT-ONLY — never touched during tuning):")
    for c, m in metrics.items():
        print(f"  {c}: sensitivity={m['sensitivity']:.3f} precision={m['precision']:.3f} "
              f"f1={m['f1']:.3f} support={m['support']}")

    macro_f1 = f1_score(y_test, y_pred, labels=AAMI_CLASSES, average="macro", zero_division=0)
    overall_acc = float(np.mean(np.array(y_pred) == np.array(y_test)))
    print(f"\nMacro-F1 on DS2: {macro_f1:.4f}  (this is the number that matters here, not accuracy)")
    print(f"Overall DS2 accuracy: {overall_acc:.4f}")

    # Confusion matrix + S<->V confusion rates: unconditional (AGENT_RULES.md
    # rule 4 -- always report the confusion matrix and the relevant
    # off-diagonal rates, not just for the timing-feature experiments).
    cm = confusion_matrix(y_test, y_pred, labels=AAMI_CLASSES)
    print(f"\nDS2 confusion matrix (rows=true, cols=pred), order {AAMI_CLASSES}:")
    print(cm)
    n_idx, s_idx, v_idx = AAMI_CLASSES.index("N"), AAMI_CLASSES.index("S"), AAMI_CLASSES.index("V")
    s_support, v_support = cm[s_idx].sum(), cm[v_idx].sum()
    s_to_n_rate = cm[s_idx, n_idx] / s_support if s_support else 0.0
    s_to_v_rate = cm[s_idx, v_idx] / s_support if s_support else 0.0
    v_to_s_rate = cm[v_idx, s_idx] / v_support if v_support else 0.0
    print(f"S->N misclassification rate: {s_to_n_rate:.3f} "
          f"({cm[s_idx, n_idx]}/{s_support} true S beats predicted N)")
    print(f"S->V misclassification rate: {s_to_v_rate:.3f} "
          f"({cm[s_idx, v_idx]}/{s_support} true S beats predicted V)")
    print(f"V->S misclassification rate: {v_to_s_rate:.3f} "
          f"({cm[v_idx, s_idx]}/{v_support} true V beats predicted S)")

    if args.timing_features and not args.timing_only:
        # Production baseline (five_class_xgb.json) and full timing-v1 (all 7
        # timing features), both reproduced fresh via eval-classifier on
        # 2026-07-16 -- see ABLATION_REPORT.md. Hardcoded here (not re-run
        # live) so this comparison is cheap to print every time; if either
        # model is ever retrained, update these dicts and cite the new
        # eval-classifier run in ABLATION_REPORT.md.
        baseline_ds2 = {"N": 0.9720, "S": 0.1390, "V": 0.8260, "F": 0.0110, "Q": 0.0000}
        timing_v1_ds2 = {"N": 0.9750, "S": 0.1820, "V": 0.7750, "F": 0.0100, "Q": 0.0000}
        timing_v1_s_to_v_rate = 0.558  # 1001/1795, from timing-v1's eval-classifier run

        print("\nDS2 per-class F1: production baseline -> timing-v1 (all 7) -> this run:")
        v_regressed_vs_baseline = False
        for c in AAMI_CLASSES:
            new_f1 = metrics[c]["f1"]
            base_f1 = baseline_ds2[c]
            v1_f1 = timing_v1_ds2[c]
            flag = ""
            if c == "V" and (new_f1 - base_f1) < -0.02:
                flag = "  <-- still a V REGRESSION vs baseline (>0.02 drop)"
                v_regressed_vs_baseline = True
            print(f"  {c}: {base_f1:.4f} -> {v1_f1:.4f} -> {new_f1:.4f}"
                  f"  (vs baseline: {new_f1 - base_f1:+.4f}; vs timing-v1: {new_f1 - v1_f1:+.4f}){flag}")
        print(f"\nS->V rate: baseline 0.354 -> timing-v1 {timing_v1_s_to_v_rate:.3f} -> this run {s_to_v_rate:.3f}")

        if args.drop_compensatory_pause:
            # Explicit verdict rule from ABLATION_REPORT.md's micro-ablation spec:
            # only a win if S keeps most of its gain AND V recovers to within a
            # trivial margin of baseline. Otherwise S/V are entangled on this
            # signal and the next move is a two-stage classifier, not more
            # feature tweaking.
            s_gain_kept = (metrics["S"]["f1"] - baseline_ds2["S"]) / (timing_v1_ds2["S"] - baseline_ds2["S"]) \
                if (timing_v1_ds2["S"] - baseline_ds2["S"]) > 1e-9 else 0.0
            v_recovered = (baseline_ds2["V"] - metrics["V"]["f1"]) <= 0.02
            print(f"\nVERDICT CHECK: S kept {s_gain_kept:.0%} of timing-v1's S-F1 gain over baseline; "
                  f"V is {'within' if v_recovered else 'NOT within'} 0.02 of baseline "
                  f"({metrics['V']['f1']:.4f} vs {baseline_ds2['V']:.4f}).")
            if s_gain_kept >= 0.5 and v_recovered:
                print("--> WIN: S kept most of its gain and V recovered. Promotable candidate "
                      "(pending human review) -- compensatory_pause_flag was the entangling feature.")
            else:
                print("--> NOT a win by the pre-registered rule. If S's gain collapsed along with "
                      "removing compensatory_pause_flag, S and V are entangled on the shared "
                      "timing signal generally, not just on that one feature -- the next move is "
                      "a two-stage classifier, not further feature-level tweaking.")
        elif v_regressed_vs_baseline:
            print("\n*** V-class regressed on DS2. This is the exact failure mode the (unverified) "
                  "docstring note warned about -- timing features helping S/validation while quietly "
                  "breaking V on the real held-out test. Do not promote this model. ***")

    # Calibrate the conformal risk predictor on the same held-out probabilities,
    # mapping AAMI beat-class confidence to a coarse risk-level proxy so
    # ConformalRiskPredictor.calibrate() has a real (if approximate) calibration
    # set instead of remaining permanently uncalibrated (recommendation #4).
    print("\nCalibrating ConformalRiskPredictor (beat-class confidence as a risk-level proxy)...")
    conformal = ConformalRiskPredictor()
    try:
        risk_level_idx = np.array([_label_to_risk_idx(l) for l in y_test])
        risk_scores = _beat_proba_to_risk_scores(np.array(proba_all), classifier)
        conformal.calibrate(risk_scores, risk_level_idx)
        print(f"Calibrated. qhat={conformal._qhat:.4f}")
    except ValueError as e:
        print(f"Skipped calibration: {e}")


def _label_to_risk_idx(aami_label: str) -> int:
    # N -> LOW, S/F -> MEDIUM, V -> HIGH (coarse proxy; real calibration should
    # use actual recording-level risk outcomes once available)
    return {"N": 0, "S": 1, "F": 1, "V": 2, "Q": 0}[aami_label]


def _beat_proba_to_risk_scores(beat_proba: np.ndarray, classifier: FiveClassBeatClassifier) -> np.ndarray:
    classes = list(classifier._label_encoder.inverse_transform(np.arange(beat_proba.shape[1])))
    risk_groups = {0: ["N", "Q"], 1: ["S", "F"], 2: ["V"], 3: []}
    out = np.zeros((len(beat_proba), 4))
    for level, members in risk_groups.items():
        idxs = [classes.index(m) for m in members if m in classes]
        out[:, level] = beat_proba[:, idxs].sum(axis=1) if idxs else 1e-6
    out[:, 3] = 1e-6  # no beat-level class maps to CRITICAL; kept as a near-zero floor
    return out / out.sum(axis=1, keepdims=True)


# ============================================================================
# eval_classifier.py — Standalone evaluation CLI
# ============================================================================
_EVAL_CLASSIFIER_DOC = """Standalone evaluation CLI — reproduces per-class DS2 metrics for an
ALREADY-TRAINED classifier without retraining anything.

This exists because train_classifiers.py bakes training and evaluation
into one main() call; there was no way to re-check a saved model's
numbers in isolation. Used as the baseline/ablation harness -- e.g. for
the timing-features experiment, see ABLATION_REPORT.md for the actual
run's numbers (created 2026-07-16; do not cite this docstring itself as
evidence anything happened -- an earlier version of this file made that
mistake, see the CORRECTION note in ecg_pipeline_core.py's features
section).

Usage:
    python -m ecg_pipeline.ecg_pipeline_tools eval-classifier \
        --model models/five_class_xgb.json --split-set ds2

    python -m ecg_pipeline.ecg_pipeline_tools eval-classifier \
        --model models/five_class_xgb_timing_v1.json --split-set ds2 --timing-features
"""


def evaluate(model_path: Path, record_ids: list[int], db_dir: Path,
             n_features: int | None = None, label: str = "",
             include_timing: bool = False, drop_compensatory_pause: bool = False,
             timing_only: bool = False, include_r_amp: bool = False) -> dict:
    X, y = build_dataset(db_dir, record_ids, include_timing=include_timing,
                          drop_compensatory_pause=drop_compensatory_pause, timing_only=timing_only,
                          include_r_amp=include_r_amp)
    if n_features is not None and X.shape[1] > n_features:
        X = X[:, :n_features]

    classifier = FiveClassBeatClassifier()
    classifier.load(model_path)

    y_pred = []
    for x in X:
        proba = classifier.model.predict_proba(x.reshape(1, -1))[0]
        classes = classifier._label_encoder.inverse_transform(np.arange(len(proba)))
        y_pred.append(classes[np.argmax(proba)])

    metrics = per_class_metrics(y, y_pred)

    macro_f1 = f1_score(y, y_pred, labels=AAMI_CLASSES, average="macro", zero_division=0)
    overall_acc = float(np.mean(np.array(y_pred) == np.array(y)))
    cm = confusion_matrix(y, y_pred, labels=AAMI_CLASSES)

    print(f"\n=== {label or model_path.name} on {len(y)} beats ===")
    print(f"{'class':<6}{'sensitivity':<13}{'precision':<12}{'f1':<8}{'support':<8}")
    for c in AAMI_CLASSES:
        m = metrics[c]
        print(f"{c:<6}{m['sensitivity']:<13.3f}{m['precision']:<12.3f}{m['f1']:<8.3f}{m['support']:<8}")
    print(f"Macro-F1: {macro_f1:.4f}   Overall accuracy: {overall_acc:.4f}  (accuracy is NOT the success metric)")
    print(f"\nConfusion matrix (rows=true, cols=pred), order {AAMI_CLASSES}:")
    print(cm)

    n_idx, f_idx, s_idx, v_idx = (AAMI_CLASSES.index("N"), AAMI_CLASSES.index("F"),
                                   AAMI_CLASSES.index("S"), AAMI_CLASSES.index("V"))
    f_support, s_support, v_support = cm[f_idx].sum(), cm[s_idx].sum(), cm[v_idx].sum()
    f_to_s_rate = cm[f_idx, s_idx] / f_support if f_support else 0.0
    s_to_n_rate = cm[s_idx, n_idx] / s_support if s_support else 0.0
    s_to_v_rate = cm[s_idx, v_idx] / s_support if s_support else 0.0
    v_to_s_rate = cm[v_idx, s_idx] / v_support if v_support else 0.0
    print(f"F->S misclassification rate: {f_to_s_rate:.3f} ({cm[f_idx, s_idx]}/{f_support} true F beats predicted S)")
    print(f"S->N misclassification rate: {s_to_n_rate:.3f} ({cm[s_idx, n_idx]}/{s_support} true S beats predicted N)")
    print(f"S->V misclassification rate: {s_to_v_rate:.3f} ({cm[s_idx, v_idx]}/{s_support} true S beats predicted V)")
    print(f"V->S misclassification rate: {v_to_s_rate:.3f} ({cm[v_idx, s_idx]}/{v_support} true V beats predicted S)")

    return {"metrics": metrics, "macro_f1": macro_f1, "overall_acc": overall_acc,
            "confusion_matrix": cm.tolist(), "f_to_s_rate": f_to_s_rate, "s_to_n_rate": s_to_n_rate,
            "s_to_v_rate": s_to_v_rate, "v_to_s_rate": v_to_s_rate, "n": len(y)}


def main_eval_classifier(argv=None):
    parser = argparse.ArgumentParser(description=_EVAL_CLASSIFIER_DOC)
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "five_class_xgb.json")
    parser.add_argument("--data-root", type=Path, default=DATA_RAW / "public")
    parser.add_argument("--split-set", choices=["ds1_train", "ds1_val", "ds2"], default="ds2",
                         help="ds1_train/ds1_val are the new patient-level carve-out of DS1 "
                              "(splits.py); ds2 is the literature held-out test set")
    parser.add_argument("--n-features", type=int, default=None,
                         help="slice feature vectors to first N columns (use 56 to evaluate the "
                              "legacy production model against the extended feature extractor)")
    parser.add_argument("--label", default="")
    parser.add_argument("--timing-features", action=argparse.BooleanOptionalAction, default=False,
                         help="build 63-dim (56 base + 7 timing) vectors to match a model trained "
                              "with train-classifiers --timing-features; must match how the model "
                              "being loaded was actually trained or predict_proba will raise a "
                              "feature-count mismatch")
    parser.add_argument("--drop-compensatory-pause", action=argparse.BooleanOptionalAction, default=False,
                         help="only meaningful with --timing-features: build 62-dim vectors "
                              "(compensatory_pause_flag dropped) to match a model trained with "
                              "train-classifiers --timing-features --drop-compensatory-pause")
    parser.add_argument("--timing-only", action=argparse.BooleanOptionalAction, default=False,
                         help="build timing-only vectors (7, or 6 with --drop-compensatory-pause) "
                              "to match a model trained with train-classifiers --timing-only")
    parser.add_argument("--include-r-amp", action=argparse.BooleanOptionalAction, default=False,
                         help="append R-peak amplitude to match a model trained with "
                              "train-classifiers --include-r-amp")
    args = parser.parse_args(argv)

    db_dir = args.data_root / "mitdb"
    record_ids = {"ds1_train": DS1_TRAIN, "ds1_val": DS1_VAL, "ds2": MITDB_DS2}[args.split_set]
    evaluate(args.model, record_ids, db_dir, n_features=args.n_features, label=args.label,
             include_timing=args.timing_features, drop_compensatory_pause=args.drop_compensatory_pause,
             timing_only=args.timing_only, include_r_amp=args.include_r_amp)


# ============================================================================
# train_twostage.py — CLI: two-stage classifier (Stage 1 gate + Stage 2 S/V/F)
# ============================================================================
_TRAIN_TWOSTAGE_DOC = """CLI: train and evaluate the two-stage classifier proposed in
ABLATION_REPORT.md's feature-family ablation (GREEN LIGHT result).

Stage 1 (gate): binary Normal-vs-Abnormal, morphology + timing (63-dim) --
timing is a genuinely useful "is this beat premature/ectopic" signal for
this binary decision (the discarded flat-model timing experiments showed
timing hurts S/V *discrimination*, but that failure mode is now entirely
Stage 2's problem -- Stage 1 never has to tell S from V, only
normal from not-normal).

Stage 2 (discriminator): S vs V vs F among whatever Stage 1 flags as
Abnormal, morphology-only (56-dim) -- this is Model A from the
feature-family ablation, already shown to separate S/V well
(S->V rate 0.169, V F1 0.866 on DS2).

Both stages reuse FiveClassBeatClassifier unmodified -- it fits on
`sorted(set(y))`, so it already works as a 2-class or 3-class model, not
just 5-class. No new classifier class needed (AGENT_RULES.md rule 7 spirit:
don't build new abstractions the existing ones already cover).

Stage 1's decision threshold is tuned on DS1_VAL (patient-level,
tuning-only split -- never DS2, per AGENT_RULES.md rule 1) via a sweep,
defaulting to the Youden's-J-maximizing point (sensitivity + specificity -
1), a standard non-arbitrary screening-gate criterion.

Usage:
    python -m ecg_pipeline.ecg_pipeline_tools train-twostage \
        --out-prefix models/five_class_xgb_twostage_v1
"""

# Reference floors, derived from the pinned baseline's DS2 confusion matrix
# (ABLATION_REPORT.md "Prerequisite 1") -- what an implicit "not predicted N"
# gate would already achieve with the flat model, without any Stage 1 at all:
#   true abnormal (S+V+F) = 1795+3005+363 = 5163; predicted N among these =
#   1212(S->N)+148(V->N)+97(F->N) = 1457 -> non-N rate = 3706/5163 = 0.7178
#   true S = 1795; S->N = 1212 -> S non-N rate = 583/1795 = 0.3248
_FLAT_MODEL_ABNORMAL_GATE_FLOOR = 0.7178
_FLAT_MODEL_S_GATE_FLOOR = 0.3248

# Trivial-margin V-precision floor for Experiment A's threshold sweep (v1/v2
# result: v1's threshold=0.05 leaked 8.2% of N past the gate, collapsing V
# precision 0.828->0.655; v2 fixed V by giving Stage 2 an N-escape-hatch but
# that undid Stage 1's S gain instead -- see ABLATION_REPORT.md). Same 0.02
# trivial-margin convention as the rest of this file's Rule-8 checks.
_V_PRECISION_FLOOR = 0.828 - 0.02

# Best two-stage result so far (v1: Stage 2 = S/V/F-only, threshold=0.05) --
# used by the auto-verdict below so a new run is judged against the best
# already-demonstrated two-stage result, not just the flat baseline. v2's
# auto-verdict compared only to the flat baseline and mislabeled a wash (S
# gain gone, V fixed) as "promotable CANDIDATE" -- see ABLATION_REPORT.md's
# two-stage v2 section for the correction. Update this dict by hand whenever
# a new run actually beats it.
_BEST_PRIOR_TWOSTAGE = {"label": "v1 (Stage 2 = S/V/F-only, threshold=0.05)",
                         "s_f1": 0.364, "macro_f1": 0.4218}


def _stage1_labels(y: list[str]) -> list[str]:
    return ["N" if lab == "N" else "Abnormal" for lab in y]


def _fit_balanced(X: np.ndarray, y: list[str], seed: int, ros: bool = True,
                   balanced_weights: bool = True,
                   class_weight_multiplier: dict[str, float] | None = None) -> FiveClassBeatClassifier:
    print(f"  class counts before ROS: {dict(Counter(y))}")
    if ros:
        X_ros, y_ros, _ = random_oversample(X, y, minority_ratio=1.0 / 3.0, seed=seed)
        print(f"  class counts after ROS (1:3 floor): {dict(Counter(y_ros))}")
    else:
        X_ros, y_ros = X, y

    sample_weight = (compute_sample_weight("balanced", y_ros) if balanced_weights
                      else np.ones(len(y_ros), dtype=float))

    clf = FiveClassBeatClassifier()
    clf.fit(X_ros, y_ros, sample_weight=sample_weight, random_state=seed,
            class_weight_multiplier=class_weight_multiplier)
    return clf


def _stage1_abnormal_proba(clf: FiveClassBeatClassifier, X: np.ndarray) -> np.ndarray:
    proba = clf.model.predict_proba(X)
    classes = list(clf._label_encoder.inverse_transform(np.arange(proba.shape[1])))
    return proba[:, classes.index("Abnormal")]


def tune_stage1_threshold(clf: FiveClassBeatClassifier, X_val: np.ndarray, y_val: list[str]) -> dict:
    """Sweeps threshold 0.05-0.95, reports abnormal/S/V/F recall and N
    specificity at each point, picks the Youden's-J-maximizing threshold as
    the default reported operating point. DS1_VAL only -- never DS2
    (AGENT_RULES.md rule 1)."""
    y_val_arr = np.array(y_val)
    abnormal_proba = _stage1_abnormal_proba(clf, X_val)

    n_mask, s_mask, v_mask, f_mask = (y_val_arr == "N", y_val_arr == "S",
                                       y_val_arr == "V", y_val_arr == "F")
    abn_mask = y_val_arr != "N"

    sweep = []
    best_j, best_threshold = -1.0, 0.5
    print(f"\n{'thresh':<8}{'abn_recall':<12}{'S_recall':<10}{'V_recall':<10}{'F_recall':<10}{'N_specificity':<14}")
    for threshold in np.arange(0.05, 1.0, 0.05):
        threshold = float(threshold)
        predicted_abnormal = abnormal_proba >= threshold
        abn_recall = float((predicted_abnormal & abn_mask).sum() / abn_mask.sum()) if abn_mask.any() else 0.0
        s_recall = float((predicted_abnormal & s_mask).sum() / s_mask.sum()) if s_mask.any() else 0.0
        v_recall = float((predicted_abnormal & v_mask).sum() / v_mask.sum()) if v_mask.any() else 0.0
        f_recall = float((predicted_abnormal & f_mask).sum() / f_mask.sum()) if f_mask.any() else 0.0
        n_specificity = float((~predicted_abnormal & n_mask).sum() / n_mask.sum()) if n_mask.any() else 0.0
        j = abn_recall + n_specificity - 1.0
        sweep.append({"threshold": round(threshold, 2), "abnormal_recall": abn_recall, "s_recall": s_recall,
                       "v_recall": v_recall, "f_recall": f_recall, "n_specificity": n_specificity,
                       "youden_j": j})
        print(f"{threshold:<8.2f}{abn_recall:<12.3f}{s_recall:<10.3f}{v_recall:<10.3f}{f_recall:<10.3f}{n_specificity:<14.3f}")
        if j > best_j:
            best_j, best_threshold = j, threshold

    print(f"\nSelected threshold (Youden's J maximizing, DS1_VAL): {best_threshold:.2f}  (J={best_j:.3f})")
    return {"threshold": best_threshold, "sweep": sweep}


def chained_eval(stage1: FiveClassBeatClassifier, stage2: FiveClassBeatClassifier,
                  X_63: np.ndarray, y_true: list[str], threshold: float,
                  n_features: int = N_FEATURES,
                  append_stage1_proba_to_stage2: bool = False) -> dict:
    """One Stage1->Stage2 chained pass. Generic over whatever classes Stage 2
    knows (S/V/F-only, per v1/Experiment A, or N/S/V/F, per v2/Fix 1) --
    beats gated Normal by Stage 1 are predicted N directly; beats gated
    Abnormal are handed to Stage 2, whose own predicted classes (whatever
    they are) become the final label.

    `append_stage1_proba_to_stage2` (EXPERIMENT B): appends Stage 1's own
    P(abnormal) for each beat as an extra Stage-2 input column, alongside
    the `n_features` morphology columns -- reuses `abnormal_proba` (already
    computed here for the gate decision) rather than recomputing it. At eval
    time this is always the real, fully-trained Stage 1's output (DS1_VAL
    and DS2 are both genuinely held-out from Stage 1's training, so there is
    no leakage concern here -- only Stage 2's TRAINING data needs the
    out-of-fold treatment, done separately by `cross_fit_stage1_proba`)."""
    y_true_arr = np.array(y_true)
    abnormal_proba = _stage1_abnormal_proba(stage1, X_63)
    predicted_abnormal = abnormal_proba >= threshold
    y_pred = np.full(len(y_true), "N", dtype=object)
    if predicted_abnormal.any():
        X_stage2_input = X_63[predicted_abnormal][:, :n_features]
        if append_stage1_proba_to_stage2:
            X_stage2_input = np.hstack([X_stage2_input, abnormal_proba[predicted_abnormal].reshape(-1, 1)])
        proba2 = stage2.model.predict_proba(X_stage2_input)
        classes2 = stage2._label_encoder.inverse_transform(np.arange(proba2.shape[1]))
        y_pred[predicted_abnormal] = classes2[np.argmax(proba2, axis=1)]
    y_pred = y_pred.tolist()

    metrics = per_class_metrics(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=AAMI_CLASSES, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=AAMI_CLASSES)
    n_idx, s_idx = AAMI_CLASSES.index("N"), AAMI_CLASSES.index("S")
    s_support = cm[s_idx].sum()
    s_to_n_rate = float(cm[s_idx, n_idx] / s_support) if s_support else 0.0
    return {"metrics": metrics, "macro_f1": macro_f1, "confusion_matrix": cm,
            "s_to_n_rate": s_to_n_rate, "y_pred": y_pred}


def sweep_chained_threshold(stage1: FiveClassBeatClassifier, stage2: FiveClassBeatClassifier,
                             X_eval_63: np.ndarray, y_eval: list[str],
                             v_precision_floor: float = _V_PRECISION_FLOOR,
                             append_stage1_proba_to_stage2: bool = False) -> dict:
    """EXPERIMENT A: sweeps Stage 1's threshold and scores each point by the
    CHAINED (Stage1->Stage2) macro-F1, not Stage-1-only Youden's J -- Youden's
    J is blind to what a leaked-N false positive does once it reaches Stage 2
    (that blindness is exactly what let v1's threshold=0.05 through with an
    undetected V-precision collapse). Must be run on a tuning split
    (DS1_VAL), never DS2 -- AGENT_RULES.md rule 1 -- so this function takes
    whatever eval set the caller passes and does not itself enforce which
    one; main_train_twostage always calls it with DS1_VAL.
    Picks the macro-F1-maximizing threshold AMONG those meeting the
    V-precision floor; if none meet the floor, returns chosen_threshold=None
    so the caller can report "no sweet spot found" honestly instead of
    silently picking a floor-violating point."""
    rows = []
    best_macro_f1, best_threshold = -1.0, None
    print(f"\n{'thresh':<8}{'S_f1':<8}{'S_recall':<10}{'S->N':<8}{'V_prec':<9}{'V_f1':<8}{'N_f1':<8}{'macro_f1':<10}{'meets_V_floor':<14}")
    for threshold in np.arange(0.05, 1.0, 0.05):
        threshold = float(threshold)
        result = chained_eval(stage1, stage2, X_eval_63, y_eval, threshold,
                               append_stage1_proba_to_stage2=append_stage1_proba_to_stage2)
        m = result["metrics"]
        meets_floor = bool(m["V"]["precision"] >= v_precision_floor)
        row = {"threshold": round(threshold, 2), "s_f1": m["S"]["f1"], "s_recall": m["S"]["sensitivity"],
               "s_to_n_rate": result["s_to_n_rate"], "v_precision": m["V"]["precision"], "v_f1": m["V"]["f1"],
               "n_f1": m["N"]["f1"], "macro_f1": result["macro_f1"], "meets_v_floor": meets_floor}
        rows.append(row)
        print(f"{threshold:<8.2f}{m['S']['f1']:<8.3f}{m['S']['sensitivity']:<10.3f}{result['s_to_n_rate']:<8.3f}"
              f"{m['V']['precision']:<9.3f}{m['V']['f1']:<8.3f}{m['N']['f1']:<8.3f}{result['macro_f1']:<10.4f}"
              f"{'yes' if meets_floor else 'NO':<14}")
        if meets_floor and result["macro_f1"] > best_macro_f1:
            best_macro_f1, best_threshold = result["macro_f1"], threshold

    if best_threshold is None:
        print(f"\nNo threshold in the sweep meets the V-precision floor ({v_precision_floor:.3f}). "
              "No sweet spot found -- Experiment B territory.")
    else:
        print(f"\nSelected threshold (max chained macro-F1 subject to V-precision >= "
              f"{v_precision_floor:.3f}, DS1_VAL): {best_threshold:.2f}  (macro-F1={best_macro_f1:.4f})")
    return {"threshold": best_threshold, "sweep": rows}


def _build_dataset_with_fold_ids(record_specs: list[tuple[Path, object]], n_folds: int,
                                  seed: int) -> tuple[np.ndarray, list[str], np.ndarray]:
    """EXPERIMENT B: same per-record beat extraction as build_dataset (63-dim,
    timing included), but additionally tags each output row with a fold id
    (0..n_folds-1). Fold assignment is per RECORD, not per beat, so a single
    recording's beats never split across the train/held-out boundary of a
    fold -- a recording's beats are highly correlated (same patient, same
    device, same noise characteristics), so per-beat fold assignment would
    leak across folds in a way per-record assignment does not, mirroring why
    DS1_TRAIN/DS1_VAL/DS2 are patient-level splits in the first place."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(record_specs))
    fold_of_record = np.empty(len(record_specs), dtype=int)
    for rank, record_idx in enumerate(order):
        fold_of_record[record_idx] = rank % n_folds

    X_parts, y_parts, fold_parts = [], [], []
    for i, (db_dir, rid) in enumerate(record_specs):
        record_path = db_dir / str(rid)
        if not record_path.with_suffix(".hea").exists():
            print(f"  skip {rid}: not downloaded")
            continue
        if not record_path.with_suffix(".atr").exists():
            print(f"  skip {rid}: no .atr (expert) annotations available")
            continue
        X_r, y_r = _load_record_beats(record_path, include_timing=True)
        print(f"  {rid}: {len(y_r)} labeled beats (fold {fold_of_record[i]})")
        if len(y_r) == 0:
            continue
        X_parts.append(X_r)
        y_parts.extend(y_r)
        fold_parts.extend([int(fold_of_record[i])] * len(y_r))

    X = np.vstack(X_parts) if X_parts else np.zeros((0, _feature_width(True)))
    return X, y_parts, np.array(fold_parts)


def cross_fit_stage1_proba(X_train: np.ndarray, y_train: list[str], fold_id: np.ndarray,
                            seed: int) -> np.ndarray:
    """EXPERIMENT B: out-of-fold Stage-1 P(abnormal) for every training beat,
    used as an extra Stage-2 input feature. Each fold's beats are scored by a
    Stage 1 trained on every OTHER fold only -- Stage 1 never predicts on its
    own training beats here, so this feature can't leak Stage 1's
    memorization of a beat's own label into Stage 2's training signal (the
    real, fully-trained Stage 1 is still what gets deployed and evaluated;
    this cross-fitting only ever touches Stage 2's TRAINING inputs)."""
    y_stage1_arr = np.array(_stage1_labels(y_train))
    oof_proba = np.zeros(len(y_train), dtype=float)
    n_folds = int(fold_id.max()) + 1
    for k in range(n_folds):
        held_mask = fold_id == k
        train_mask = ~held_mask
        print(f"  cross-fit fold {k + 1}/{n_folds}: training Stage 1 on {int(train_mask.sum())} beats, "
              f"scoring {int(held_mask.sum())} held-out beats")
        fold_stage1 = _fit_balanced(X_train[train_mask], y_stage1_arr[train_mask].tolist(), seed=seed)
        oof_proba[held_mask] = _stage1_abnormal_proba(fold_stage1, X_train[held_mask])
    return oof_proba


def main_train_twostage(argv=None):
    parser = argparse.ArgumentParser(description=_TRAIN_TWOSTAGE_DOC)
    parser.add_argument("--data-root", type=Path, default=DATA_RAW / "public")
    parser.add_argument("--out-prefix", type=Path, default=MODELS_DIR / "five_class_xgb_twostage_v1")
    parser.add_argument("--include-svdb", action=argparse.BooleanOptionalAction, default=True,
                         help="add all 78 SVDB records into training (same as production's recipe)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--f-weight-multiplier", type=float, default=1.0,
                         help="extra sample_weight multiplier on F-labeled rows in Stage 2 training")
    parser.add_argument("--threshold", type=float, default=None,
                         help="skip DS1_VAL tuning and use this Stage-1 threshold directly")
    parser.add_argument("--stage2-include-normal", action=argparse.BooleanOptionalAction, default=False,
                         help="FIX 1 (see ABLATION_REPORT.md two-stage v1 result): train Stage 2 on "
                              "N/S/V/F instead of S/V/F-only, so a Normal beat that leaks past Stage 1's "
                              "gate can be recovered as N instead of being FORCED into S/V/F -- v1's "
                              "root cause for V's precision collapse (0.828->0.655). Default False "
                              "reproduces v1's S/V/F-only Stage 2 exactly.")
    parser.add_argument("--threshold-objective", choices=["youden_j", "chained_macro_f1"], default="youden_j",
                         help="youden_j (default, v1/v2 behavior): tune Stage 1's threshold on its own "
                              "sensitivity+specificity, blind to what Stage 2 does with a leaked beat. "
                              "chained_macro_f1 (EXPERIMENT A): tune on the full chained Stage1->Stage2 "
                              "macro-F1, subject to a hard V-precision floor -- both computed on DS1_VAL, "
                              "never DS2 (AGENT_RULES.md rule 1).")
    parser.add_argument("--stage2-use-stage1-proba", action=argparse.BooleanOptionalAction, default=False,
                         help="EXPERIMENT B: append Stage 1's P(abnormal) as an extra (57th) Stage 2 "
                              "input feature, alongside the 56 morphology features -- Stage 2 stays "
                              "S/V/F-only (v1's architecture), no raw timing features added. Training-time "
                              "values are generated via K-fold cross-fitting (--cv-folds) so Stage 1 "
                              "never scores its own training beats; eval-time values use the real, "
                              "fully-trained Stage 1 (DS1_VAL/DS2 are already held out from Stage 1's "
                              "training, so no cross-fitting is needed there).")
    parser.add_argument("--cv-folds", type=int, default=5,
                         help="number of record-level folds for Experiment B's Stage-1 cross-fitting")
    args = parser.parse_args(argv)
    if not args.out_prefix.is_absolute():
        args.out_prefix = (MODELS_DIR.parent / args.out_prefix).resolve()

    db_dir = args.data_root / "mitdb"
    svdb_dir = args.data_root / "svdb"

    fold_id = None
    if args.stage2_use_stage1_proba:
        print("Building DS1_TRAIN(+SVDB) with per-record fold ids (63-dim, EXPERIMENT B cross-fitting) ...")
        record_specs = [(db_dir, rid) for rid in DS1_TRAIN]
        if args.include_svdb:
            record_specs += [(svdb_dir, rid) for rid in SVDB_RECORDS]
        X_train, y_train, fold_id = _build_dataset_with_fold_ids(record_specs, args.cv_folds, args.seed)
    else:
        print("Building DS1_TRAIN (63-dim: 56 morphology/wavelet + 7 timing) ...")
        X_train, y_train = build_dataset(db_dir, DS1_TRAIN, include_timing=True)
    print("Building DS1_VAL (patient-level tuning split) ...")
    X_val, y_val = build_dataset(db_dir, DS1_VAL, include_timing=True)
    print("Building DS2 (held-out, report-only) ...")
    X_test, y_test = build_dataset(db_dir, MITDB_DS2, include_timing=True)

    if args.include_svdb and not args.stage2_use_stage1_proba:
        print("Building SVDB training data (S-class boost) ...")
        X_svdb, y_svdb = build_dataset(svdb_dir, SVDB_RECORDS, include_timing=True)
        print(f"  SVDB contributes {len(y_svdb)} beats to training")
        X_train = np.vstack([X_train, X_svdb])
        y_train = y_train + y_svdb

    keep = [i for i, lab in enumerate(y_train) if lab != "Q"]
    print(f"Dropped {len(y_train) - len(keep)} Q-class beats from training (not learnable at this sample size)")
    X_train = X_train[keep]
    y_train = [y_train[i] for i in keep]
    if fold_id is not None:
        fold_id = fold_id[keep]

    # ---- Stage 1: Normal vs Abnormal, morphology + timing (63-dim) ----
    print("\n=== Stage 1: Normal-vs-Abnormal gate (63-dim, timing included) ===")
    y_train_stage1 = _stage1_labels(y_train)
    stage1 = _fit_balanced(X_train, y_train_stage1, seed=args.seed)
    stage1_path = Path(f"{args.out_prefix}_stage1.json")
    stage1.save(stage1_path)
    print(f"Saved Stage 1 -> {stage1_path}  (seed={args.seed})")

    # ---- EXPERIMENT B: out-of-fold Stage-1 P(abnormal) for Stage 2's TRAINING data only ----
    oof_abnormal_proba = None
    if args.stage2_use_stage1_proba:
        print(f"\n=== EXPERIMENT B: {args.cv_folds}-fold cross-fitting Stage 1 for out-of-fold "
              "P(abnormal) (Stage 2 training feature only -- the deployed Stage 1 above is untouched) ===")
        oof_abnormal_proba = cross_fit_stage1_proba(X_train, y_train, fold_id, seed=args.seed)

    # ---- Stage 2: discriminator, morphology-only (56-dim, +1 if Experiment B) ----
    # Trained BEFORE threshold selection: chained_macro_f1 objective needs a
    # trained Stage 2 to score each candidate threshold against (that's the
    # whole point -- Youden's J on Stage 1 alone can't see what Stage 2 does
    # with a leaked beat).
    if args.stage2_include_normal:
        print("\n=== Stage 2: N/S/V/F discriminator (56-dim, morphology-only, FIX 1: N included "
              "so a leaked Normal beat can be recovered instead of forced into S/V/F) ===")
        stage2_idx = list(range(len(y_train)))
    else:
        print("\n=== Stage 2: S/V/F discriminator (56-dim, morphology-only, true-abnormal beats only"
              + (", + out-of-fold Stage-1 P(abnormal) as feature 57 -- EXPERIMENT B" if oof_abnormal_proba is not None else "")
              + ") ===")
        stage2_idx = [i for i, lab in enumerate(y_train) if lab != "N"]
    X_train_stage2 = X_train[stage2_idx][:, :N_FEATURES]
    if oof_abnormal_proba is not None:
        X_train_stage2 = np.hstack([X_train_stage2, oof_abnormal_proba[stage2_idx].reshape(-1, 1)])
        print(f"  Stage 2 input is now {X_train_stage2.shape[1]}-dim (56 morphology + 1 out-of-fold "
              "Stage-1 P(abnormal))")
    y_train_stage2 = [y_train[i] for i in stage2_idx]
    f_mult = {"F": args.f_weight_multiplier} if args.f_weight_multiplier != 1.0 else None
    stage2 = _fit_balanced(X_train_stage2, y_train_stage2, seed=args.seed, class_weight_multiplier=f_mult)
    stage2_path = Path(f"{args.out_prefix}_stage2.json")
    stage2.save(stage2_path)
    print(f"Saved Stage 2 -> {stage2_path}  (seed={args.seed})")

    # ---- Stage 1 threshold selection (DS1_VAL only -- never DS2, AGENT_RULES.md rule 1) ----
    if args.threshold_objective == "chained_macro_f1":
        print("\n=== EXPERIMENT A: threshold sweep scored by CHAINED macro-F1 (DS1_VAL) ===")
        tuning = sweep_chained_threshold(stage1, stage2, X_val, y_val,
                                          append_stage1_proba_to_stage2=args.stage2_use_stage1_proba)
        if tuning["threshold"] is None and args.threshold is None:
            print("Falling back to threshold=0.05 (v1's operating point) since no point in the "
                  "sweep met the V-precision floor -- report this run as 'no sweet spot found', "
                  "not as a tuned result.")
            tuning_threshold = 0.05
        else:
            tuning_threshold = tuning["threshold"]
    else:
        tuning = tune_stage1_threshold(stage1, X_val, y_val)
        tuning_threshold = tuning["threshold"]
    threshold = args.threshold if args.threshold is not None else tuning_threshold

    threshold_path = Path(f"{args.out_prefix}_threshold.json")
    threshold_path.write_text(json.dumps({"threshold": threshold, "objective": args.threshold_objective,
                                           "sweep": tuning["sweep"]}, indent=2))
    print(f"Saved Stage-1 threshold + sweep -> {threshold_path}")

    # ---- Stage-1-in-isolation report on DS2 (diagnostic, not the promotion number) ----
    print(f"\n=== Stage 1 in isolation on DS2 (threshold={threshold:.2f}) -- diagnostic only ===")
    y_test_arr = np.array(y_test)
    abnormal_proba_test = _stage1_abnormal_proba(stage1, X_test)
    predicted_abnormal_test = abnormal_proba_test >= threshold
    s_mask_test, v_mask_test, f_mask_test = y_test_arr == "S", y_test_arr == "V", y_test_arr == "F"
    abn_mask_test = y_test_arr != "N"
    s_recall_ds2 = float((predicted_abnormal_test & s_mask_test).sum() / s_mask_test.sum()) if s_mask_test.any() else 0.0
    v_recall_ds2 = float((predicted_abnormal_test & v_mask_test).sum() / v_mask_test.sum()) if v_mask_test.any() else 0.0
    f_recall_ds2 = float((predicted_abnormal_test & f_mask_test).sum() / f_mask_test.sum()) if f_mask_test.any() else 0.0
    abn_recall_ds2 = float((predicted_abnormal_test & abn_mask_test).sum() / abn_mask_test.sum()) if abn_mask_test.any() else 0.0
    print(f"Stage-1-ISOLATED S-recall on DS2:        {s_recall_ds2:.3f}  (flat-model implicit floor: {_FLAT_MODEL_S_GATE_FLOOR:.3f})")
    print(f"Stage-1-ISOLATED V-recall on DS2:        {v_recall_ds2:.3f}")
    print(f"Stage-1-ISOLATED F-recall on DS2:        {f_recall_ds2:.3f}")
    print(f"Stage-1-ISOLATED abnormal-recall on DS2: {abn_recall_ds2:.3f}  (flat-model implicit floor: {_FLAT_MODEL_ABNORMAL_GATE_FLOOR:.3f})")
    print("(these measure whether Stage 1 GATES the beat as Abnormal -- not the final chained label; "
          "see chained S-recall below for what the full system actually outputs)")

    if args.stage2_use_stage1_proba:
        print("\n=== EXPERIMENT B diagnostic: does P(abnormal) separate leaked-N from true-abnormal "
              "beats reaching Stage 2? (DS2) ===")
        gated_proba = abnormal_proba_test[predicted_abnormal_test]
        gated_true = y_test_arr[predicted_abnormal_test]
        true_abnormal_proba = gated_proba[gated_true != "N"]
        leaked_n_proba = gated_proba[gated_true == "N"]

        def _qs(x):
            if len(x) == 0:
                return "n=0"
            q = np.percentile(x, [0, 10, 25, 50, 75, 90, 100])
            return (f"n={len(x)}  min={q[0]:.3f} p10={q[1]:.3f} p25={q[2]:.3f} median={q[3]:.3f} "
                    f"p75={q[4]:.3f} p90={q[5]:.3f} max={q[6]:.3f}")

        print(f"  True abnormal beats (S/V/F) reaching Stage 2:  {_qs(true_abnormal_proba)}")
        print(f"  Leaked false-positive N beats reaching Stage 2: {_qs(leaked_n_proba)}")
        if len(true_abnormal_proba) and len(leaked_n_proba):
            gap = float(np.median(true_abnormal_proba) - np.median(leaked_n_proba))
            print(f"  Median gap (true-abnormal minus leaked-N): {gap:+.3f}  "
                  f"({'separated -- feature carries real signal Stage 2 can exploit' if gap > 0.1 else 'heavily overlapping -- feature likely a wash'})")

    # ---- Chained end-to-end (Stage 1 -> Stage 2) prediction on DS2 ----
    print("\n=== Chained end-to-end (Stage 1 -> Stage 2) DS2 report -- the number that matters ===")
    result = chained_eval(stage1, stage2, X_test, y_test, threshold,
                           append_stage1_proba_to_stage2=args.stage2_use_stage1_proba)
    metrics, macro_f1, cm = result["metrics"], result["macro_f1"], result["confusion_matrix"]
    overall_acc = float(np.mean(np.array(result["y_pred"]) == y_test_arr))

    print(f"{'class':<6}{'sensitivity':<13}{'precision':<12}{'f1':<8}{'support':<8}")
    for c in AAMI_CLASSES:
        m = metrics[c]
        print(f"{c:<6}{m['sensitivity']:<13.3f}{m['precision']:<12.3f}{m['f1']:<8.3f}{m['support']:<8}")
    print(f"Macro-F1: {macro_f1:.4f}   Overall accuracy: {overall_acc:.4f}  (not the success metric)")
    print(f"\nConfusion matrix (rows=true, cols=pred), order {AAMI_CLASSES}:")
    print(cm)

    n_idx, s_idx, v_idx = AAMI_CLASSES.index("N"), AAMI_CLASSES.index("S"), AAMI_CLASSES.index("V")
    s_support, v_support = cm[s_idx].sum(), cm[v_idx].sum()
    s_to_n_rate = result["s_to_n_rate"]
    s_to_v_rate = cm[s_idx, v_idx] / s_support if s_support else 0.0
    v_to_s_rate = cm[v_idx, s_idx] / v_support if v_support else 0.0
    print(f"S->N misclassification rate: {s_to_n_rate:.3f} ({cm[s_idx, n_idx]}/{s_support} true S beats predicted N)")
    print(f"S->V misclassification rate: {s_to_v_rate:.3f} ({cm[s_idx, v_idx]}/{s_support} true S beats predicted V)")
    print(f"V->S misclassification rate: {v_to_s_rate:.3f} ({cm[v_idx, s_idx]}/{v_support} true V beats predicted S)")

    # ---- Comparison vs the pinned baseline (ABLATION_REPORT.md "Prerequisite 1") ----
    baseline_ds2 = {"N": 0.966, "S": 0.160, "V": 0.866, "F": 0.021, "Q": 0.000}
    baseline_macro_f1 = 0.4026
    baseline_s_to_n, baseline_s_to_v, baseline_v_to_s = 0.676, 0.169, 0.041
    baseline_v_precision = 0.828  # the number two-stage v1 broke (0.828 -> 0.655)
    print("\n=== vs pinned baseline (ABLATION_REPORT.md Prerequisite 1) ===")
    v_regressed, n_regressed = False, False
    for c in AAMI_CLASSES:
        delta = metrics[c]["f1"] - baseline_ds2[c]
        flag = ""
        if c in ("V", "N") and delta < -0.02:
            flag = "  <-- REGRESSION (AGENT_RULES.md Rule 8: V/N are zero-tolerance)"
            if c == "V":
                v_regressed = True
            else:
                n_regressed = True
        print(f"  {c}: {baseline_ds2[c]:.4f} -> {metrics[c]['f1']:.4f}  ({delta:+.4f}){flag}")
    print(f"  Macro-F1: {baseline_macro_f1:.4f} -> {macro_f1:.4f}  ({macro_f1 - baseline_macro_f1:+.4f})")
    print(f"  V precision: {baseline_v_precision:.3f} -> {metrics['V']['precision']:.3f}  "
          f"({metrics['V']['precision'] - baseline_v_precision:+.3f})  <- this is the number v1 broke")
    print(f"  S->N rate: {baseline_s_to_n:.3f} -> {s_to_n_rate:.3f}  ({s_to_n_rate - baseline_s_to_n:+.3f})")
    print(f"  S->V rate: {baseline_s_to_v:.3f} -> {s_to_v_rate:.3f}  ({s_to_v_rate - baseline_s_to_v:+.3f})")
    print(f"  V->S rate: {baseline_v_to_s:.3f} -> {v_to_s_rate:.3f}  ({v_to_s_rate - baseline_v_to_s:+.3f})")

    # ---- Comparison vs the BEST PRIOR two-stage result, not just the flat baseline ----
    # v2's auto-verdict compared only to baseline_ds2 and printed "promotable CANDIDATE"
    # for a run that beat the flat baseline by +0.0066 S F1 while losing v1's entire
    # +0.204 S F1 gain -- a wash mislabeled as progress. See ABLATION_REPORT.md's
    # two-stage v2 section for the correction this check exists to prevent.
    print(f"\n=== vs best prior two-stage result ({_BEST_PRIOR_TWOSTAGE['label']}) ===")
    s_vs_best_prior = metrics["S"]["f1"] - _BEST_PRIOR_TWOSTAGE["s_f1"]
    print(f"  S F1: {_BEST_PRIOR_TWOSTAGE['s_f1']:.4f} -> {metrics['S']['f1']:.4f}  ({s_vs_best_prior:+.4f})")
    is_wash_vs_prior = s_vs_best_prior < -0.02

    if v_regressed or n_regressed:
        print("\n*** Per AGENT_RULES.md Rule 8: V and/or N regressed beyond the trivial margin. "
              "This candidate is NOT promotable as-is, even if S improved. Report honestly, do not "
              "promote. ***")
    elif metrics["S"]["f1"] <= baseline_ds2["S"]:
        print("\n*** S did not improve over the flat baseline. Two-stage is a clean negative result "
              "here -- report honestly per README.md's Honest-result rule, do not promote. ***")
    elif is_wash_vs_prior:
        print(f"\n*** S beats the flat baseline but is WORSE than the best prior two-stage result "
              f"({_BEST_PRIOR_TWOSTAGE['label']}, S F1 {_BEST_PRIOR_TWOSTAGE['s_f1']:.4f}) by more than "
              f"the trivial margin. This is a WASH relative to what two-stage has already been shown "
              f"to achieve, not a genuine improvement -- do NOT label this 'promotable' just because "
              f"it beats the flat baseline. Report honestly. ***")
    else:
        print("\nNo V/N regression, S improved over the flat baseline, AND S is not worse than the "
              "best prior two-stage result -- a promotable CANDIDATE pending explicit human review "
              "(AGENT_RULES.md Rule 8(c)). STOP here; do not promote automatically.")

    return {"stage1_threshold": threshold, "metrics": metrics, "macro_f1": macro_f1,
            "confusion_matrix": cm.tolist(), "s_recall_ds2_stage1": s_recall_ds2,
            "abnormal_recall_ds2_stage1": abn_recall_ds2}


# ============================================================================
# analyze_twostage_stage1.py — CLI: EXPERIMENT C -- is Stage 1's precision/
# S-recall trade-off fixable by threshold alone?
# ============================================================================
_ANALYZE_TWOSTAGE_STAGE1_DOC = """CLI: EXPERIMENT C -- is there ANY Stage-1 operating threshold where
abnormal-PRECISION is high (few confident false-positive N leaks) AND
S-recall stays acceptable? This is the decisive test of whether two-stage's
proven bottleneck (Stage 1 emits confidently-wrong N beats -- see the
"Two-stage: consolidated conclusion" section of ABLATION_REPORT.md) is
fixable by threshold alone, or whether S-recall and N-precision are
fundamentally in tension at the gate.

Loads the EXISTING v1 Stage 1 / Stage 2 artifacts (no retraining -- "one
change at a time" means only the operating threshold varies here; v1's
recipe, features, and Stage 2 are unchanged by construction, not just by
claim). Reports, per threshold, on DS1_VAL (tuning signal) and DS2
(report-only -- these numbers are read, never used to pick anything, so
this stays within AGENT_RULES.md rule 1):
  - Stage-1-ISOLATED abnormal-precision, abnormal-recall, S/V/F-recall,
    N-specificity (does Stage 1's OWN call agree with the truth, before
    Stage 2 touches anything)
  - Chained (Stage1->Stage2) S F1, V F1, V precision, macro-F1 (what the
    full system actually delivers at that threshold, reusing v1's
    unmodified Stage 2 so the effect is attributable to Stage 1 alone)
Plus a fixed, threshold-independent diagnostic: how many true-N beats does
this Stage 1 assign P(abnormal) > 0.85 -- beats it is CONFIDENT about, not
just beats that happen to leak at today's threshold.

Usage:
    python -m ecg_pipeline.ecg_pipeline_tools analyze-twostage-stage1
"""


def main_analyze_twostage_stage1(argv=None):
    parser = argparse.ArgumentParser(description=_ANALYZE_TWOSTAGE_STAGE1_DOC)
    parser.add_argument("--data-root", type=Path, default=DATA_RAW / "public")
    parser.add_argument("--stage1-model", type=Path, default=MODELS_DIR / "five_class_xgb_twostage_v1_stage1.json")
    parser.add_argument("--stage2-model", type=Path, default=MODELS_DIR / "five_class_xgb_twostage_v1_stage2.json")
    parser.add_argument("--confident-threshold", type=float, default=0.85,
                         help="P(abnormal) above which a leaked N beat counts as 'confidently wrong', "
                              "not just leaked")
    args = parser.parse_args(argv)

    db_dir = args.data_root / "mitdb"
    print("Building DS1_VAL (patient-level tuning split, 63-dim) ...")
    X_val, y_val = build_dataset(db_dir, DS1_VAL, include_timing=True)
    print("Building DS2 (held-out, report-only, 63-dim) ...")
    X_test, y_test = build_dataset(db_dir, MITDB_DS2, include_timing=True)

    stage1 = FiveClassBeatClassifier()
    stage1.load(args.stage1_model)
    stage2 = FiveClassBeatClassifier()
    stage2.load(args.stage2_model)
    print(f"Loaded Stage 1 <- {args.stage1_model}")
    print(f"Loaded Stage 2 <- {args.stage2_model}  (classes: {list(stage2._label_encoder.classes_)})")

    for split_name, X_eval, y_eval in [("DS1_VAL (tuning signal)", X_val, y_val),
                                        ("DS2 (report-only)", X_test, y_test)]:
        print(f"\n=== EXPERIMENT C: Stage-1-isolated precision/S-recall trade-off "
              f"+ chained effect on {split_name} ===")
        y_eval_arr = np.array(y_eval)
        abnormal_proba = _stage1_abnormal_proba(stage1, X_eval)
        n_mask, s_mask, v_mask, f_mask = (y_eval_arr == "N", y_eval_arr == "S",
                                           y_eval_arr == "V", y_eval_arr == "F")
        abn_mask = y_eval_arr != "N"

        print(f"{'thresh':<8}{'abn_prec':<10}{'abn_rec':<9}{'S_rec':<8}{'V_rec':<8}{'F_rec':<8}{'N_spec':<8}"
              f"{'chS_f1':<8}{'chV_f1':<8}{'chV_prec':<9}{'ch_macroF1':<11}")
        candidates = []
        for threshold in np.arange(0.05, 1.0, 0.05):
            threshold = float(threshold)
            predicted_abnormal = abnormal_proba >= threshold
            abn_precision = float((predicted_abnormal & abn_mask).sum() / predicted_abnormal.sum()) \
                if predicted_abnormal.any() else 0.0
            abn_recall = float((predicted_abnormal & abn_mask).sum() / abn_mask.sum()) if abn_mask.any() else 0.0
            s_recall = float((predicted_abnormal & s_mask).sum() / s_mask.sum()) if s_mask.any() else 0.0
            v_recall = float((predicted_abnormal & v_mask).sum() / v_mask.sum()) if v_mask.any() else 0.0
            f_recall = float((predicted_abnormal & f_mask).sum() / f_mask.sum()) if f_mask.any() else 0.0
            n_specificity = float((~predicted_abnormal & n_mask).sum() / n_mask.sum()) if n_mask.any() else 0.0

            chained = chained_eval(stage1, stage2, X_eval, y_eval, threshold)
            cm = chained["metrics"]
            ch_s_f1, ch_v_f1, ch_v_prec = cm["S"]["f1"], cm["V"]["f1"], cm["V"]["precision"]

            print(f"{threshold:<8.2f}{abn_precision:<10.3f}{abn_recall:<9.3f}{s_recall:<8.3f}{v_recall:<8.3f}"
                  f"{f_recall:<8.3f}{n_specificity:<8.3f}{ch_s_f1:<8.3f}{ch_v_f1:<8.3f}{ch_v_prec:<9.3f}"
                  f"{chained['macro_f1']:<11.4f}")

            if abn_precision >= 0.90 and s_recall >= 0.70:
                candidates.append(round(threshold, 2))

        if candidates:
            print(f"\nCandidate thresholds with isolated abnormal-precision >= 0.90 AND S-recall >= 0.70: "
                  f"{candidates}")
        else:
            print("\nNo threshold achieves BOTH isolated abnormal-precision >= 0.90 AND S-recall >= 0.70 "
                  f"on {split_name}.")

    # ---- Fixed diagnostic: confident false-positive N beats on DS2 (threshold-independent) ----
    abnormal_proba_test = _stage1_abnormal_proba(stage1, X_test)
    y_test_arr = np.array(y_test)
    n_mask_test = y_test_arr == "N"
    confident_fp = int(((abnormal_proba_test > args.confident_threshold) & n_mask_test).sum())
    print(f"\n=== Fixed diagnostic: confident false-positive N beats on DS2 "
          f"(P(abnormal) > {args.confident_threshold}) ===")
    print(f"{confident_fp} of {int(n_mask_test.sum())} true N beats have P(abnormal) > "
          f"{args.confident_threshold} -- threshold-INDEPENDENT: no gate threshold at or below "
          f"{args.confident_threshold} on THIS Stage 1 model can exclude them without also excluding "
          "every true abnormal beat Stage 1 is similarly confident about.")


# ============================================================================
# Combined CLI dispatcher
# ============================================================================

_SUBCOMMANDS = {
    "download-datasets": main_download_datasets,
    "download-icentia11k-full": main_download_icentia11k_full,
    "train-encoder": main_train_encoder,
    "train-classifiers": main_train_classifiers,
    "eval-classifier": main_eval_classifier,
    "train-twostage": main_train_twostage,
    "analyze-twostage-stage1": main_analyze_twostage_stage1,
}


def main():
    # Deliberately not routed through argparse: an outer parser's own
    # `-h`/`--help` action fires regardless of where it appears in argv,
    # which would swallow e.g. `eval-classifier --help` and print this
    # dispatcher's help instead of eval-classifier's own. Handling the
    # subcommand name directly lets each `main_<name>`'s own
    # ArgumentParser see and handle its own `--help`/args untouched.
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print(f"Subcommands: {', '.join(_SUBCOMMANDS)}")
        return
    command, remaining = argv[0], argv[1:]
    if command not in _SUBCOMMANDS:
        print(f"Unknown subcommand: {command!r}. Choose from: {', '.join(_SUBCOMMANDS)}", file=sys.stderr)
        sys.exit(2)
    _SUBCOMMANDS[command](remaining)


if __name__ == "__main__":
    main()
