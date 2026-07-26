"""manifest_summary.py -- combine the VitalPatch and prorhythm run manifests
into one CSV, print summary counts, and print full showcase examples.

Reuses the manifests already written by batch_vitalpatch_report.py and
batch_prorhythm_report.py -- this script only reads/aggregates/prints, it
does not re-run the pipeline or touch the classifier/cascade/model.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from ecg_pipeline.ecg_pipeline_core import DATA_RAW

DATA_REPORTS = DATA_RAW.parent / "reports"
COMBINED_FIELDS = ["source", "patient_id", "segment_id", "duration_s", "n_beats_analyzed",
                    "assessable", "final_risk_level", "n_rhythm_findings", "medgemma_status"]


def _read_manifest(path: Path, source: str) -> list[dict]:
    if not path.exists():
        print(f"WARNING: manifest not found: {path}")
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append({
            "source": source,
            "patient_id": r.get("patient_id"),
            "segment_id": r.get("segment_id"),
            "duration_s": r.get("duration_s"),
            "n_beats_analyzed": r.get("n_beats_analyzed"),
            "assessable": r.get("assessable"),
            "final_risk_level": r.get("final_risk_level"),
            "n_rhythm_findings": r.get("n_rhythm_findings"),
            "medgemma_status": r.get("medgemma_status"),
            "error": r.get("error"),
        })
    return out


def build_combined_manifest(vitalpatch_manifest: Path, prorhythm_manifest: Path,
                             out_path: Path) -> list[dict]:
    rows = _read_manifest(vitalpatch_manifest, "vitalpatch") + _read_manifest(prorhythm_manifest, "sensio")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMBINED_FIELDS + ["error"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in COMBINED_FIELDS + ["error"]})
    print(f"Combined manifest written -> {out_path} ({len(rows)} rows)")
    return rows


def print_summary(rows: list[dict]) -> None:
    total = len(rows)
    errored = [r for r in rows if r.get("error")]
    ok = [r for r in rows if not r.get("error")]
    assessable = [r for r in ok if r.get("assessable") == "True"]
    not_assessable = [r for r in ok if r.get("assessable") == "False"]
    level_dist = Counter(r["final_risk_level"] for r in ok)
    medgemma_dist = Counter(r["medgemma_status"] for r in ok)
    live_narrative = sum(1 for r in ok if r.get("medgemma_status") == "ACCEPTED")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total segments/files: {total}")
    print(f"  Errored (parse/run failures, skipped gracefully): {len(errored)}")
    print(f"  Assessable: {len(assessable)}")
    print(f"  NOT_ASSESSABLE: {len(not_assessable)}")
    print(f"Risk level distribution: {dict(level_dist)}")
    print(f"MedGemma status distribution: {dict(medgemma_dist)}")
    print(f"Segments with a LIVE MedGemma narrative (ACCEPTED): {live_narrative}")
    print("=" * 70)


def print_showcase(rows: list[dict], reports_root: Path, n: int = 3) -> None:
    def load(r):
        seg = r["segment_id"]
        pat = r["patient_id"]
        src_dir = "vitalpatch" if r["source"] == "vitalpatch" else "prorhythm"
        if src_dir == "vitalpatch":
            path = reports_root / src_dir / pat / f"{seg}.json"
        else:
            import re
            patient_dir = re.sub(r'[^A-Za-z0-9_.-]', '_', pat) or "unknown"
            path = reports_root / src_dir / patient_dir / f"{seg}.json"
        return json.loads(path.read_text()) if path.exists() else None

    low = [r for r in rows if r.get("final_risk_level") == "LOW" and r.get("medgemma_status") == "ACCEPTED"]
    high = [r for r in rows if r.get("final_risk_level") == "HIGH" and r.get("medgemma_status") == "ACCEPTED"]
    rhythm = [r for r in rows if int(r.get("n_rhythm_findings") or 0) > 0 and r.get("medgemma_status") == "ACCEPTED"]

    picks = []
    if low:
        picks.append(("LOW risk example", low[0]))
    if high:
        picks.append(("HIGH risk example", high[0]))
    for r in rhythm:
        if r not in [p[1] for p in picks]:
            picks.append(("Rhythm-finding example", r))
            break

    print("\n" + "=" * 70)
    print("SHOWCASE EXAMPLES")
    print("=" * 70)
    for label, r in picks[:n]:
        report_json = load(r)
        if report_json is None:
            print(f"\n[{label}] {r['source']}/{r['patient_id']}/{r['segment_id']} -- JSON not found, skipping")
            continue
        print(f"\n{'#'*70}\n[{label}] {r['source']} / {r['patient_id']} / {r['segment_id']}\n{'#'*70}")
        rec = report_json["recording"]
        print(f"Raw file -> parsed: duration={rec['duration_s']}s, "
              f"beats detected/analyzed={rec['n_beats_detected']}/{rec['n_beats_analyzed']}, "
              f"sqi_rejection={rec['sqi_window_rejection_rate']}")
        print("Beat summary:")
        for cls, info in report_json["beat_summary"].items():
            print(f"  {cls}: {info['count']} ({info['pct_of_analyzed_beats']}%)")
        print("Rhythm findings:")
        if report_json["rhythm_findings"]:
            for f_ in report_json["rhythm_findings"]:
                print(f"  {f_['kind']}: {f_['evidence_text']}")
        else:
            print("  (none)")
        print(f"Deciding rule: {report_json['deciding_rule']['condition']} "
              f"(fired={report_json['deciding_rule']['fired']})")
        print(f"Final risk level: {report_json['final_risk_level']}")
        print(f"MedGemma status: {report_json['medgemma']['status']}")
        print("\n--- Clinical report ---")
        print(report_json["narrative"])
    print("=" * 70)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vitalpatch-manifest", type=Path, default=DATA_REPORTS / "vitalpatch_run_manifest.csv")
    parser.add_argument("--prorhythm-manifest", type=Path, default=DATA_REPORTS / "prorhythm_run_manifest.csv")
    parser.add_argument("--out", type=Path, default=DATA_REPORTS / "combined_run_manifest.csv")
    parser.add_argument("--reports-root", type=Path, default=DATA_REPORTS)
    args = parser.parse_args(argv)

    rows = build_combined_manifest(args.vitalpatch_manifest, args.prorhythm_manifest, args.out)
    print_summary(rows)
    print_showcase(rows, args.reports_root)


if __name__ == "__main__":
    main()
