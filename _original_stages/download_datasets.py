"""CLI: download a disk-conscious subset of public labeled ECG datasets.

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
    python -m cliniaura_pipeline.download_datasets --all
    python -m cliniaura_pipeline.download_datasets --only mitdb svdb
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import requests

from .config import DATA_RAW

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    all_names = list(WFDB_DATABASES) + ["challenge2017", "icentia11k"]
    parser.add_argument("--all", action="store_true", help="download everything in the budget plan")
    parser.add_argument("--only", nargs="+", choices=all_names, help="download only these datasets")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
