"""TASK C - pin current outputs across all 37 captures.

Does NOT prove correctness. Proves CHANGES ARE VISIBLE: any future edit that
alters a verdict or a feature on any capture will show up as a diff.

    build:  python3 scripts_rhythm/s18_golden_manifest.py --build
    check:  python3 scripts_rhythm/s18_golden_manifest.py --check
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm.pipeline import process_capture
from rhythm.verdict import THRESHOLD_CV

GOLDEN = Path("reports_rhythm/golden_manifest.json")
KEYS = ("rr_cv", "rmssd_ms", "sdnn_ms", "hr_bpm", "mean_rr_ms",
        "n_flagged", "bsqi", "sqi_passed")


def summarise(path):
    rows = process_capture(path)
    out = []
    for r in rows:
        if "error" in r:
            out.append({"error": r["error"]}); continue
        out.append({k: (round(r[k], 6) if isinstance(r.get(k), float) else r.get(k))
                    for k in KEYS})
    blob = json.dumps(out, sort_keys=True, default=str)
    return dict(file=f"{Path(path).parent.name}/{Path(path).name}",
                n_windows=len(out),
                sha256=hashlib.sha256(blob.encode()).hexdigest(),
                windows=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    files = sorted(Path("prorithm_ecg").glob("*/*.csv"))
    with Pool(6) as pool:
        cur = pool.map(summarise, [str(f) for f in files])
    manifest = {"threshold": THRESHOLD_CV,
                "n_captures": len(cur),
                "captures": {c["file"]: {"sha256": c["sha256"],
                                         "n_windows": c["n_windows"]} for c in cur}}

    if a.build:
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps({**manifest,
                                      "detail": {c["file"]: c["windows"] for c in cur}},
                                     indent=1, default=str))
        print(f"wrote {GOLDEN}  ({len(cur)} captures, "
              f"{sum(c['n_windows'] for c in cur):,} windows, threshold {THRESHOLD_CV})")
        return

    if not GOLDEN.exists():
        raise SystemExit("no golden manifest - run with --build first")
    old = json.loads(GOLDEN.read_text())
    diffs = []
    if old.get("threshold") != THRESHOLD_CV:
        diffs.append(f"THRESHOLD changed {old.get('threshold')} -> {THRESHOLD_CV}")
    for c in cur:
        o = old["captures"].get(c["file"])
        if o is None:
            diffs.append(f"NEW capture {c['file']}")
        elif o["sha256"] != c["sha256"]:
            diffs.append(f"CHANGED {c['file']}: {o['n_windows']} -> {c['n_windows']} windows")
    for f in old["captures"]:
        if f not in {c["file"] for c in cur}:
            diffs.append(f"MISSING capture {f}")
    if diffs:
        print(f"GOLDEN DIFF - {len(diffs)} change(s):")
        for d in diffs:
            print(f"  {d}")
        raise SystemExit(1)
    print(f"GOLDEN MATCH - {len(cur)} captures unchanged")


if __name__ == "__main__":
    main()
