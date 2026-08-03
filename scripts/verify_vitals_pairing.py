"""verify_vitals_pairing.py — measures real vitals-pairing coverage of
load_real_vitals() (agent_bridge.py) against every real ECG file on disk.

Run: python -m ecg_pipeline.verify_vitals_pairing

Prints per-patient and overall coverage, plus a breakdown of how many matches
came from interval containment vs. the nearest-filename-within-30s fallback.
Written to check Task 1 (fix vitals-pairing bug) against the "after the fix"
numbers measured in docs/HANDOFF.md -- not to produce those numbers, to verify
them independently.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Marker-anchored, not hop-counted -- see scripts/verify_sqi_gate.py. Needed
# because this script moved out of the ecg_pipeline package and is now run
# as `python scripts/<name>.py`, which puts scripts/ on sys.path, not the root.
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ecg_pipeline").is_dir())
sys.path.insert(0, str(REPO_ROOT))

from ecg_pipeline.agent_bridge import load_real_vitals  # noqa: E402

ECG_ROOT = REPO_ROOT / "data" / "raw" / "vitalpatch"
VITALS_ROOT = REPO_ROOT / "data" / "vitals_downloads"


def main() -> None:
    patient_dirs = sorted(p for p in ECG_ROOT.iterdir() if p.is_dir() and p.name.startswith("Patch_"))

    grand_total = 0
    grand_matched = 0
    grand_containment = 0
    grand_fallback = 0

    print(f"{'Patient':<16}{'ECG files':>10}{'Matched':>10}{'Coverage':>10}{'Contain':>10}{'Fallback':>10}")
    for patient_dir in patient_dirs:
        ecg_files = sorted(patient_dir.glob("*_ecg.csv"))
        total = len(ecg_files)
        matched = 0
        containment = 0
        fallback = 0
        for ecg_file in ecg_files:
            vitals = load_real_vitals(str(ecg_file), str(VITALS_ROOT))
            if vitals["vitals_file_found"]:
                matched += 1
                if vitals["vitals_match_method"] == "interval_containment":
                    containment += 1
                elif vitals["vitals_match_method"] == "nearest_filename_fallback":
                    fallback += 1

        coverage = (matched / total * 100) if total else 0.0
        print(f"{patient_dir.name:<16}{total:>10}{matched:>10}{coverage:>9.1f}%{containment:>10}{fallback:>10}")

        grand_total += total
        grand_matched += matched
        grand_containment += containment
        grand_fallback += fallback

    overall_coverage = (grand_matched / grand_total * 100) if grand_total else 0.0
    print("-" * 76)
    print(f"{'Total':<16}{grand_total:>10}{grand_matched:>10}{overall_coverage:>9.1f}%"
          f"{grand_containment:>10}{grand_fallback:>10}")


if __name__ == "__main__":
    main()
