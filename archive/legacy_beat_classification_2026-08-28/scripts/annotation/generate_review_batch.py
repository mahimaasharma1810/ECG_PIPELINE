#!/usr/bin/env python3
"""Generate clinician review PDFs for every patient/device ID, one folder each.

    python scripts/annotation/generate_review_batch.py

Layout produced:
    validation/clinician_review/pending/vitalpatch/<Patch_ID>/<segment>.pdf
    validation/clinician_review/pending/prorhythm/<device_id>/<segment>.pdf
    validation/clinician_review/pending/generation_manifest.csv

SAFETY GUARD -- why some segments are skipped rather than rendered:
to_target_rate() early-returns unchanged when the device's native rate is
already TARGET_FS (VitalPatch, 125 Hz), so SQI-blanked samples survive as
NaN and are shaded "NO DATA". At any other rate (ProRhythm, 100 Hz) the
call reaches resample_linear(), which DROPS NaN samples and rebuilds its
grid over only the survivors -- so scattered surviving windows get spliced
into a continuous-looking trace whose adjacent samples come from different
times, and the output's time axis no longer matches the recording's. RR
intervals measured across such a splice are meaningless, and beat
timestamps are wrong. Measured on this corpus: VitalPatch retains 100.0%
of its time span on every file; ProRhythm collapses to 5.1-80.9% on 13 of
18 files. Any segment retaining less than MIN_TIME_RETENTION of its span is
therefore skipped with a recorded reason instead of being put in front of a
clinician.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# Marker-anchored, not hop-counted -- see the matching comment in
# scripts/verify_sqi_gate.py.
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ecg_pipeline").is_dir())
sys.path.insert(0, str(REPO_ROOT))

from ecg_inference.preprocess import (  # noqa: E402
    parse_vitalpatch_ecg, parse_prorhythm_ecg, run_sqi_gate, to_target_rate, TARGET_FS,
)
from scripts.annotation.export_beats_for_review import (  # noqa: E402
    _run_to_beats, _select, build_entries,
)
from scripts.annotation.generate_annotation_pdf import build_pdf  # noqa: E402

ROOT = REPO_ROOT
OUT = ROOT / "validation" / "clinician_review" / "pending"
MIN_TIME_RETENTION = 0.95
PIPELINE_VERSION = "models/production/five_class_xgb.json"


def time_retention(rec) -> float:
    """Fraction of the recording's true span that survives into the
    resampled grid. <1.0 means the time axis collapsed (see module docstring)."""
    keep, _ = run_sqi_gate(rec.signal_mv, rec.timestamps_ms, rec.fs_nominal, clip_value=None)
    sig = rec.signal_mv.copy()
    sig[~keep] = np.nan
    _, t = to_target_rate(sig, rec.timestamps_ms, rec.fs_nominal, TARGET_FS)
    if len(t) < 2:
        return 0.0
    raw = rec.timestamps_ms[-1] - rec.timestamps_ms[0]
    return float((t[-1] - t[0]) / raw) if raw else 0.0


def process(rec, out_dir: Path, n_beats: int, strategy: str) -> dict:
    row = {"patient_id": rec.patient_id, "segment_id": rec.segment_id,
           "source": rec.source, "fs_nominal": rec.fs_nominal}
    ret = time_retention(rec)
    row["time_retention_pct"] = round(100 * ret, 1)
    if ret < MIN_TIME_RETENTION:
        row.update(status="SKIPPED_TIME_AXIS_COLLAPSED", n_beats=0, pdf="",
                   reason=f"only {100*ret:.1f}% of the recording span survives resampling; "
                          f"strips would splice non-contiguous time")
        return row

    filtered, beats, labels, confs, valid = _run_to_beats(rec)
    if not beats:
        row.update(status="SKIPPED_NO_BEATS", n_beats=0, pdf="",
                   reason="no beats detected after quality gating")
        return row

    sel = _select(labels, confs, n_beats, strategy)
    entries = build_entries(rec, filtered, beats, labels, confs, sel, valid_mask=valid)
    if not entries:
        row.update(status="SKIPPED_NO_ENTRIES", n_beats=len(beats), pdf="",
                   reason="no reviewable beats selected")
        return row

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{rec.segment_id}.pdf"
    build_pdf(entries, pdf, PIPELINE_VERSION)
    row.update(status="OK", n_beats=len(beats), pdf=str(pdf.relative_to(ROOT)),
               reason="", n_pages=len(entries),
               label_dist=json.dumps({c: labels.count(c) for c in sorted(set(labels))}))
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segments-per-id", type=int, default=3)
    p.add_argument("--beats-per-segment", type=int, default=5)
    p.add_argument("--strategy", choices=["balanced", "low-confidence"], default="balanced")
    args = p.parse_args(argv)

    rows = []

    # ---------------- VitalPatch: one folder per Patch_ID ----------------
    vp_root = ROOT / "data" / "raw" / "vitalpatch"
    for pdir in sorted(d for d in vp_root.iterdir() if d.is_dir()):
        files = sorted(pdir.glob("*.csv"))
        if not files:
            continue
        sel = [files[i] for i in np.linspace(0, len(files) - 1,
                                             min(args.segments_per_id, len(files))).astype(int)]
        for f in sel:
            try:
                segs = parse_vitalpatch_ecg(f)
            except Exception as e:
                rows.append({"patient_id": pdir.name, "segment_id": f.stem, "source": "vitalpatch",
                             "status": "FAILED", "reason": f"{type(e).__name__}: {e}"})
                continue
            if not segs:
                continue
            r = process(segs[0], OUT / "vitalpatch" / pdir.name,
                        args.beats_per_segment, args.strategy)
            rows.append(r)
            print(f"  vitalpatch/{pdir.name}/{segs[0].segment_id}: {r['status']}")

    # ---------------- ProRhythm: one folder per device ID ----------------
    pr_root = ROOT / "data" / "raw" / "prorhythm"
    pr_files = sorted(pr_root.glob("*.csv"))
    for f in pr_files:
        try:
            rec = parse_prorhythm_ecg(f)
        except Exception as e:
            rows.append({"patient_id": "unknown", "segment_id": f.stem, "source": "prorhythm",
                         "status": "FAILED", "reason": f"{type(e).__name__}: {e}"})
            continue
        folder = rec.patient_id.replace(":", "_") or "unknown"
        r = process(rec, OUT / "prorhythm" / folder, args.beats_per_segment, args.strategy)
        rows.append(r)
        print(f"  prorhythm/{folder}/{rec.segment_id}: {r['status']}")

    OUT.mkdir(parents=True, exist_ok=True)
    man = OUT / "generation_manifest.csv"
    cols = ["source", "patient_id", "segment_id", "fs_nominal", "time_retention_pct",
            "status", "n_beats", "n_pages", "label_dist", "pdf", "reason"]
    with man.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r.get("status") == "OK"]
    print(f"\nGenerated {len(ok)} PDF(s); {len(rows) - len(ok)} segment(s) skipped/failed.")
    print(f"Manifest -> {man.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
