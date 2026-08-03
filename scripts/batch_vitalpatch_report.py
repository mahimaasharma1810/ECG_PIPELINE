"""batch_vitalpatch_report.py -- run the transparent ECG->RiskReport bridge
(agent_bridge.run_full_report) over every raw VitalPatch CSV, one report per
segment, with a per-segment try/except so a single bad file never aborts the
batch, plus a run manifest so empty/failed runs are visible rather than
silently absent.

Reuses agent_bridge.run_full_report() and ecg_pipeline_core.py exactly as-is
-- this script only discovers files, loops, writes JSON, and tallies a
manifest. It does not touch the classifier, AAMI scheme, risk cascade, or any
model/training code.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from collections import Counter
import sys
from pathlib import Path

# Marker-anchored, not hop-counted -- see scripts/verify_sqi_gate.py. Needed
# because this script moved out of the ecg_pipeline package and is now run
# as `python scripts/<name>.py`, which puts scripts/ on sys.path, not the root.
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ecg_pipeline").is_dir())
sys.path.insert(0, str(REPO_ROOT))

from ecg_pipeline.agent_bridge import load_classifier, run_full_report, save_report  # noqa: E402
from ecg_pipeline.ecg_pipeline_core import DATA_RAW, MODELS_DIR, discover_vitalpatch_files, parse_vitalpatch_ecg  # noqa: E402

DATA_REPORTS = DATA_RAW.parent / "reports" / "vitalpatch"

MANIFEST_FIELDS = [
    "patient_id", "segment_id", "source", "duration_s", "n_beats_detected",
    "n_beats_analyzed", "quality_score", "sqi_window_rejection_rate",
    "assessable", "final_risk_level", "n_rhythm_findings",
    "has_afib_vt_pvc_finding", "medgemma_status", "error",
]


def _notable_finding(report_json: dict) -> bool:
    kinds = {f["kind"] for f in report_json.get("rhythm_findings", [])}
    if kinds & {"VT_RUN", "BIGEMINY", "TRIGEMINY", "AFIB_SUSPECTED"}:
        return True
    return any(r.get("fired") and "PVC burden" in r.get("condition", "")
               for r in report_json.get("rule_trace", []))


def run_batch(vitalpatch_root: Path, classifier_path: Path, out_dir: Path,
              manifest_path: Path, progress_every: int = 100,
              vitals_root: Path | None = None) -> None:
    classifier = load_classifier(classifier_path)
    files = discover_vitalpatch_files(vitalpatch_root)
    print(f"Discovered {len(files)} VitalPatch CSV files under {vitalpatch_root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    n_files_ok = 0
    n_files_errored = 0
    n_segments = 0
    start = time.time()

    for i, f in enumerate(files, 1):
        patient_id = f.parent.name.replace("Patch_", "")
        patient_out_dir = out_dir / patient_id

        # Resume support: this loop can run for hours over 2375 files and has
        # already been killed mid-run once by session teardown outside our
        # control. parse+run is a deterministic function of the raw file, so
        # if any segment JSON for this file's stem already exists, reload its
        # manifest row instead of re-parsing/re-running (which would also
        # burn a real, serialized MedGemma call for no new information).
        existing = sorted(patient_out_dir.glob(f"{f.stem}_seg*.json")) if patient_out_dir.exists() else []
        if existing:
            n_files_ok += 1
            for ep in existing:
                n_segments += 1
                rj = json.loads(ep.read_text())
                rows.append({
                    "patient_id": patient_id, "segment_id": rj["recording"]["segment_id"], "source": "vitalpatch",
                    "duration_s": rj["recording"]["duration_s"], "n_beats_detected": rj["recording"]["n_beats_detected"],
                    "n_beats_analyzed": rj["recording"]["n_beats_analyzed"], "quality_score": rj["recording"]["quality_score"],
                    "sqi_window_rejection_rate": rj["recording"]["sqi_window_rejection_rate"],
                    "assessable": rj["assessable"], "final_risk_level": rj["final_risk_level"],
                    "n_rhythm_findings": len(rj["rhythm_findings"]), "has_afib_vt_pvc_finding": _notable_finding(rj),
                    "medgemma_status": rj["medgemma"]["status"], "error": None,
                })
            if i % progress_every == 0 or i == len(files):
                elapsed = time.time() - start
                print(f"[{i}/{len(files)} files, {n_segments} segments so far -- resumed] "
                      f"elapsed={elapsed:.1f}s ({i/elapsed:.1f} files/s)")
            continue

        try:
            recordings = parse_vitalpatch_ecg(f)
        except Exception as e:
            n_files_errored += 1
            rows.append({
                "patient_id": patient_id, "segment_id": None, "source": "vitalpatch",
                "duration_s": None, "n_beats_detected": None, "n_beats_analyzed": None,
                "quality_score": None, "sqi_window_rejection_rate": None,
                "assessable": False, "final_risk_level": None, "n_rhythm_findings": None,
                "has_afib_vt_pvc_finding": None, "medgemma_status": None,
                "error": f"PARSE_ERROR: {type(e).__name__}: {e}",
            })
            continue
        n_files_ok += 1

        for recording in recordings:
            n_segments += 1
            try:
                report_json, result = run_full_report(recording, classifier,
                                                       vitals_root=vitals_root)
            except Exception as e:
                rows.append({
                    "patient_id": patient_id, "segment_id": recording.segment_id, "source": "vitalpatch",
                    "duration_s": None, "n_beats_detected": None, "n_beats_analyzed": None,
                    "quality_score": None, "sqi_window_rejection_rate": None,
                    "assessable": False, "final_risk_level": None, "n_rhythm_findings": None,
                    "has_afib_vt_pvc_finding": None, "medgemma_status": None,
                    "error": f"RUN_ERROR: {type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
                })
                continue

            save_report(report_json, patient_out_dir, recording.segment_id)

            rows.append({
                "patient_id": patient_id,
                "segment_id": recording.segment_id,
                "source": "vitalpatch",
                "duration_s": report_json["recording"]["duration_s"],
                "n_beats_detected": report_json["recording"]["n_beats_detected"],
                "n_beats_analyzed": report_json["recording"]["n_beats_analyzed"],
                "quality_score": report_json["recording"]["quality_score"],
                "sqi_window_rejection_rate": report_json["recording"]["sqi_window_rejection_rate"],
                "assessable": report_json["assessable"],
                "final_risk_level": report_json["final_risk_level"],
                "n_rhythm_findings": len(report_json["rhythm_findings"]),
                "has_afib_vt_pvc_finding": _notable_finding(report_json),
                "medgemma_status": report_json["medgemma"]["status"],
                "error": None,
            })

        if i % progress_every == 0 or i == len(files):
            elapsed = time.time() - start
            print(f"[{i}/{len(files)} files, {n_segments} segments so far] "
                  f"elapsed={elapsed:.1f}s ({i/elapsed:.1f} files/s)")

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nManifest written -> {manifest_path} ({len(rows)} rows)")

    # SUMMARY COUNTS
    total = len(rows)
    n_errored = sum(1 for r in rows if r["error"] is not None)
    n_assessable = sum(1 for r in rows if r["assessable"] is True)
    n_not_assessable = sum(1 for r in rows if r["assessable"] is False and r["error"] is None)
    level_dist = Counter(r["final_risk_level"] for r in rows if r["error"] is None)
    n_notable = sum(1 for r in rows if r["has_afib_vt_pvc_finding"] is True)

    print("\n" + "=" * 70)
    print("SUMMARY COUNTS")
    print("=" * 70)
    print(f"Files discovered:        {len(files)}  (parsed ok: {n_files_ok}, parse errors: {n_files_errored})")
    print(f"Total segments:          {total}")
    print(f"  Errored:               {n_errored}")
    print(f"  Assessable:            {n_assessable}")
    print(f"  NOT_ASSESSABLE:        {n_not_assessable}  "
          f"({100.0 * n_not_assessable / total:.1f}% of all segments)" if total else "")
    print(f"final_risk_level distribution (non-errored): {dict(level_dist)}")
    print(f"Segments with any AFIB_SUSPECTED/VT/PVC finding: {n_notable}")
    print("=" * 70)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vitalpatch-root", type=Path, default=DATA_RAW / "vitalpatch")
    parser.add_argument("--classifier", type=Path, default=MODELS_DIR / "five_class_xgb.json")
    parser.add_argument("--out-dir", type=Path, default=DATA_REPORTS)
    parser.add_argument("--manifest", type=Path, default=DATA_REPORTS.parent / "vitalpatch_run_manifest.csv")
    parser.add_argument("--progress-every", type=int, default=100)
    # Vitals pairing is OPT-IN. Passing this matches real vitals to each segment and
    # feeds the resulting partial NEWS2 / qSOFA proxy into the risk cascade, so those
    # safety overrides are actually evaluated. Omitting it reproduces the previous
    # ECG-only behaviour byte-for-byte.
    parser.add_argument("--vitals-root", type=Path, default=None,
                        help="Root of the per-patient vitals CSV tree (e.g. data/vitals_downloads). "
                             "When given, NEWS2/qSOFA overrides are evaluated from real vitals.")
    args = parser.parse_args(argv)
    run_batch(args.vitalpatch_root, args.classifier, args.out_dir, args.manifest,
              args.progress_every, vitals_root=args.vitals_root)


if __name__ == "__main__":
    main()
