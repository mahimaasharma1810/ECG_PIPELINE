#!/usr/bin/env python3
"""Independently reproduce and audit the Stage-2 SQI gate's decisions.

    python scripts/verify_sqi_gate.py --input <ecg file> --source vitalpatch

Prints, for EVERY window the gate evaluates: its time span, pass/fail, the
reject code, and every metric it was judged on side by side with the
threshold it was compared against. Nothing here is summarised or rounded
away -- the point is that the reader can check the gate's arithmetic
rather than trust a summary of it.

It also directly tests the stage-ordering question. Stage 2 scores the
RAW signal (pipeline.py: "SQI gate (before any filtering)"), while
remove_baseline_median() does not run until Stage 4's apply_filter_chain.
So for every window rejected as BASELINE_WANDER this script recomputes
the same ratio on the same samples AFTER running that baseline filter,
and reports whether the window would still have failed. Windows that pass
post-filter are data the pipeline discarded for a defect the very next
stage removes.

Read-only: imports the production functions and thresholds, mutates
nothing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Anchored on a marker directory, not a fixed number of `.parent` hops: this
# script has already moved once (tools/ -> scripts/) and a hop count silently
# resolves to the wrong directory the next time it moves.
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ecg_pipeline").is_dir())
sys.path.insert(0, str(REPO_ROOT))

from ecg_inference.preprocess import (  # noqa: E402
    SQI, FILTER, TARGET_FS, parse_vitalpatch_ecg, parse_prorhythm_ecg, parse_wfdb_record,
    run_sqi_gate, _baseline_wander_ratio, remove_baseline_median,
)

# metric name -> (threshold value, comparison direction, reject code)
CHECKS = [
    ("flatline_frac", SQI.flatline_frac_max, ">", "FLATLINE"),
    ("clipping_frac", SQI.clipping_frac_max, ">", "CLIPPING"),
    ("missing_frac", SQI.missing_frac_max, ">", "MISSING_SAMPLES"),
    ("morphology_kurtosis", SQI.kurtosis_min, "<", "NO_QRS_IMPULSE_CHARACTER"),
    ("baseline_wander_ratio", SQI.baseline_wander_ratio_max, ">", "BASELINE_WANDER"),
    ("snr_db", SQI.snr_db_min, "<", "LOW_SNR"),
]


def load(source: str, path: Path):
    if source == "vitalpatch":
        segs = parse_vitalpatch_ecg(path)
        if not segs:
            raise SystemExit(f"No segments parsed from {path}")
        return segs[0]
    # "sensio" is the pre-rename spelling; accepted for back-compat.
    if source in ("prorhythm", "sensio"):
        return parse_prorhythm_ecg(path)
    if source == "wfdb":
        return parse_wfdb_record(path)
    raise ValueError(source)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--source", choices=["vitalpatch", "prorhythm", "wfdb", "sensio"], default="vitalpatch")
    p.add_argument("--segment-index", type=int, default=0)
    args = p.parse_args(argv)

    rec = load(args.source, args.input)
    fs = rec.fs_nominal
    sig, ts = rec.signal_mv, rec.timestamps_ms

    keep, verdicts = run_sqi_gate(sig, ts, fs, clip_value=None)

    print("=" * 100)
    print(f"SQI GATE AUDIT — {args.input.name}")
    print(f"segment={rec.segment_id}  source={rec.source}  fs_nominal={fs} Hz  "
          f"n_samples={len(sig)}  duration={(ts[-1] - ts[0]) / 1000:.1f}s")
    print("=" * 100)
    print("\nTHRESHOLDS IN FORCE (ecg_inference/preprocess.py, SQIThresholds):")
    for name, thr, op, code in CHECKS:
        print(f"   reject as {code:<26} if {name:<22} {op} {thr}")
    print("\nEvaluation order is a first-match cascade (preprocess.py:616-626): the FIRST")
    print("condition that trips decides reject_code, so later metrics may also be out of")
    print("range without being named as the reason.\n")

    hdr = f"{'win':>4} {'t_start':>9} {'t_end':>9} {'verdict':>8}  {'reject_code':<26}"
    for name, _, _, _ in CHECKS:
        hdr += f"{name[:13]:>15}"
    print(hdr)
    print("-" * len(hdr))

    t0 = ts[0]
    n_rejected = 0
    reject_counts: dict[str, int] = {}
    for i, v in enumerate(verdicts):
        if not v.passed:
            n_rejected += 1
            reject_counts[v.reject_code] = reject_counts.get(v.reject_code, 0) + 1
        row = (f"{i:>4} {(v.start_ms - t0) / 1000:>9.2f} {(v.end_ms - t0) / 1000:>9.2f} "
               f"{'PASS' if v.passed else 'REJECT':>8}  {str(v.reject_code or '-'):<26}")
        for name, thr, op, _ in CHECKS:
            val = v.metrics.get(name, float('nan'))
            bad = (val > thr) if op == ">" else (val < thr)
            row += f"{val:>14.3f}{'*' if bad else ' '}"
        print(row)

    print("\n('*' marks a metric out of range, whether or not it was the deciding code.)")
    print(f"\nWindows: {len(verdicts)}   rejected: {n_rejected}   "
          f"({100.0 * n_rejected / max(1, len(verdicts)):.1f}%)")
    print(f"Reject code counts: {reject_counts}")
    blanked = int((~keep).sum())
    print(f"Samples blanked: {blanked} / {len(keep)} ({100.0 * blanked / len(keep):.1f}% of signal)")

    # contiguous blanked stretches
    bad = ~keep
    if bad.any():
        edges = np.diff(bad.astype(int))
        st = list(np.where(edges == 1)[0] + 1)
        en = list(np.where(edges == -1)[0] + 1)
        if bad[0]:
            st = [0] + st
        if bad[-1]:
            en = en + [len(bad) - 1]
        runs = sorted(((ts[e] - ts[s]) / 1000.0, (ts[s] - t0) / 1000.0) for s, e in zip(st, en))
        print("\nContiguous blanked stretches (duration_s @ start_s):")
        for dur, start in sorted(runs, reverse=True):
            print(f"   {dur:6.2f}s @ t={start:.1f}s")

    # ---- the ordering test -------------------------------------------------
    print("\n" + "=" * 100)
    print("STAGE-ORDERING TEST — would BASELINE_WANDER rejects survive the Stage-4 filter?")
    print("=" * 100)
    bw_rejects = [(i, v) for i, v in enumerate(verdicts) if v.reject_code == "BASELINE_WANDER"]
    if not bw_rejects:
        print("No windows were rejected for BASELINE_WANDER in this recording.")
    else:
        print(f"{'win':>4} {'ratio_raw':>11} {'ratio_after_baseline_filter':>29} {'threshold':>11}  verdict")
        recovered = 0
        for i, v in bw_rejects:
            # same samples the gate scored, re-scored after Stage 4's first step
            lo = int(np.searchsorted(ts, v.start_ms))
            hi = int(np.searchsorted(ts, v.end_ms, side="right"))
            seg = sig[lo:hi].astype(float)
            seg = seg[~np.isnan(seg)]
            if len(seg) < 20:
                continue
            filt = remove_baseline_median(seg, fs, FILTER.median_baseline_window_ms)
            after = _baseline_wander_ratio(filt, fs)
            before = v.metrics.get("baseline_wander_ratio", float("nan"))
            ok = after <= SQI.baseline_wander_ratio_max
            recovered += ok
            print(f"{i:>4} {before:>11.3f} {after:>29.3f} {SQI.baseline_wander_ratio_max:>11.2f}  "
                  f"{'WOULD PASS after filter' if ok else 'still fails'}")
        print(f"\n{recovered} of {len(bw_rejects)} BASELINE_WANDER-rejected window(s) would pass "
              f"if the baseline filter ran before the gate.")
        if recovered:
            secs = sum((v.end_ms - v.start_ms) / 1000.0 for _, v in bw_rejects)
            print(f"Up to ~{secs:.1f}s of signal in this file is discarded for a defect "
                  f"Stage 4 removes.")


if __name__ == "__main__":
    main()
