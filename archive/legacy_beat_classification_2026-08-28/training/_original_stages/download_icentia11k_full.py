"""Download the FULL Icentia11k dataset (~188GB, 11,000 patients) via its
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
    python -m cliniaura_pipeline.download_icentia11k_full
    python -m cliniaura_pipeline.download_icentia11k_full --workers 64
"""
from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from .config import DATA_RAW

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()
    download_all(workers=args.workers)


if __name__ == "__main__":
    main()
