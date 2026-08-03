"""batch_prorhythm_report.py -- run the transparent ECG->RiskReport bridge
(agent_bridge.run_full_report) over a selected set of prorhythm ProRhythm CSVs,
one report per file, with a per-file try/except so a single bad file never
aborts the batch, plus a run manifest so failures are visible.

prorhythm/ is not yet in docs/thesis/DATASETS.md. It was inspected directly for
this run: same "ProRhythm" CSV layout already documented for data/raw/sense_io/
(11 metadata rows, blank rows, header at row 16, sample rate parsed from the
"Command Sent" field e.g. STARTECG_F:100 -> 100Hz here, not the 125Hz
VitalPatch nominal rate) -- so the existing parse_prorhythm_ecg() parser is
reused as-is, no new parser needed.

Reuses agent_bridge.run_full_report(), parse_prorhythm_ecg(), and
ecg_pipeline_core.py exactly as-is -- this script only discovers files,
loops, writes JSON, and tallies a manifest. It does not touch the
classifier, AAMI scheme, risk cascade, or any model/training code.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import traceback
import sys
from pathlib import Path

# Marker-anchored, not hop-counted -- see scripts/verify_sqi_gate.py. Needed
# because this script moved out of the ecg_pipeline package and is now run
# as `python scripts/<name>.py`, which puts scripts/ on sys.path, not the root.
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ecg_pipeline").is_dir())
sys.path.insert(0, str(REPO_ROOT))

from ecg_pipeline.agent_bridge import load_classifier, run_full_report, save_report  # noqa: E402
from ecg_pipeline.ecg_pipeline_core import DATA_RAW, MODELS_DIR, parse_prorhythm_ecg  # noqa: E402

DATA_REPORTS = DATA_RAW.parent / "reports" / "prorhythm"

MANIFEST_FIELDS = [
    "patient_id", "segment_id", "source", "duration_s", "n_beats_detected",
    "n_beats_analyzed", "quality_score", "sqi_window_rejection_rate",
    "assessable", "final_risk_level", "n_rhythm_findings",
    "medgemma_status", "error",
]


def run_batch(files: list[Path], classifier_path: Path, out_dir: Path, manifest_path: Path) -> None:
    classifier = load_classifier(classifier_path)
    print(f"Processing {len(files)} prorhythm CSV files")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for f in files:
        print(f"\n--- {f.name} ---")
        try:
            recording = parse_prorhythm_ecg(f)
        except Exception as e:
            rows.append({
                "patient_id": None, "segment_id": f.stem, "source": "prorhythm", "duration_s": None,
                "n_beats_detected": None, "n_beats_analyzed": None, "quality_score": None,
                "sqi_window_rejection_rate": None, "assessable": False, "final_risk_level": None,
                "n_rhythm_findings": None, "medgemma_status": None,
                "error": f"PARSE_ERROR: {type(e).__name__}: {e}",
            })
            print(f"  PARSE_ERROR: {e}")
            continue

        try:
            report_json, result = run_full_report(recording, classifier)
        except Exception as e:
            rows.append({
                "patient_id": recording.patient_id, "segment_id": recording.segment_id, "source": "prorhythm",
                "duration_s": None,
                "n_beats_detected": None, "n_beats_analyzed": None, "quality_score": None,
                "sqi_window_rejection_rate": None, "assessable": False, "final_risk_level": None,
                "n_rhythm_findings": None, "medgemma_status": None,
                "error": f"RUN_ERROR: {type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
            })
            print(f"  RUN_ERROR: {e}")
            continue

        patient_dir_name = re.sub(r'[^A-Za-z0-9_.-]', '_', recording.patient_id) or "unknown"
        out_path, _ = save_report(report_json, out_dir / patient_dir_name, recording.segment_id)

        rows.append({
            "patient_id": recording.patient_id,
            "segment_id": recording.segment_id,
            "source": "prorhythm",
            "duration_s": report_json["recording"]["duration_s"],
            "n_beats_detected": report_json["recording"]["n_beats_detected"],
            "n_beats_analyzed": report_json["recording"]["n_beats_analyzed"],
            "quality_score": report_json["recording"]["quality_score"],
            "sqi_window_rejection_rate": report_json["recording"]["sqi_window_rejection_rate"],
            "assessable": report_json["assessable"],
            "final_risk_level": report_json["final_risk_level"],
            "n_rhythm_findings": len(report_json["rhythm_findings"]),
            "medgemma_status": report_json["medgemma"]["status"],
            "error": None,
        })
        print(f"  beats_analyzed={report_json['recording']['n_beats_analyzed']} "
              f"risk={report_json['final_risk_level']} "
              f"medgemma={report_json['medgemma']['status']}")
        print(f"  saved -> {out_path}")

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nManifest written -> {manifest_path} ({len(rows)} rows)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prorhythm-root", type=Path, default=DATA_RAW / "prorhythm")
    parser.add_argument("--files", nargs="*", default=None,
                         help="specific filenames (within --prorhythm-root) to process; default: all ECG_*.csv "
                              "(both the raw-channel and on-device-prefiltered 'ECG_Filtered_*' variants -- "
                              "these are separate, non-overlapping recording sessions here, not duplicates, "
                              "see the timestamps; parse_prorhythm_ecg already detects the ECG_Filtered column "
                              "and marks already_bandpass_filtered accordingly)")
    parser.add_argument("--classifier", type=Path, default=MODELS_DIR / "five_class_xgb.json")
    parser.add_argument("--out-dir", type=Path, default=DATA_REPORTS)
    parser.add_argument("--manifest", type=Path, default=DATA_REPORTS.parent / "prorhythm_run_manifest.csv")
    args = parser.parse_args(argv)

    if args.files:
        files = [args.prorhythm_root / name for name in args.files]
    else:
        files = sorted(args.prorhythm_root.glob("ECG_*.csv"))

    run_batch(files, args.classifier, args.out_dir, args.manifest)


if __name__ == "__main__":
    main()
