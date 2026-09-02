"""Step 1c - are duplicate-timestamp samples DISTINCT samples or RETRANSMITS?

This is the hinge of the whole ingest design:
  - distinct samples that collided on a coarse clock  -> average them (brief Stage 2)
  - retransmitted/repeated packets                    -> drop them; n/span overstates fs

Test: within each duplicate-timestamp group, are the amplitudes identical?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm.ingest import load_capture

FILES = [
    "prorithm_ecg/1789/2026-08-20_10.csv",  # the brief's reference file, dup 46.6%
    "prorithm_ecg/1794/2026-08-09_12.csv",  # highest dup 74.8%, fs_active 178 Hz
    "prorithm_ecg/1793/2026-08-08_06.csv",  # lowest dup 15.6%, fs_active 90 Hz
    "prorithm_ecg/1789/2026-08-17_10.csv",  # dup 22.4%, fs_active 89.5 Hz
]

for f in FILES:
    cap = load_capture(f)
    t, a = cap.t_ms, cap.amplitude
    uniq, inv, counts = np.unique(t, return_inverse=True, return_counts=True)

    # group-wise amplitude spread
    order = np.lexsort((a, t))
    ts, as_ = t[order], a[order]
    grp_start = np.flatnonzero(np.r_[True, np.diff(ts) != 0])
    grp_end = np.r_[grp_start[1:], ts.size] - 1
    gmin, gmax = as_[grp_start], as_[grp_end]
    gsize = grp_end - grp_start + 1
    multi = gsize > 1
    identical = (gmax - gmin) == 0

    active = float(np.sum(np.diff(t)[np.diff(t) <= 1000]) / 1000.0)
    print(f"\n=== {f}")
    print(f"  n={cap.n_samples:,}  unique_ts={uniq.size:,}  dup_frac={cap.duplicate_fraction:.1%}")
    print(f"  group sizes: {dict(zip(*np.unique(counts, return_counts=True)))}")
    print(f"  multi-sample groups        : {multi.sum():,}")
    print(f"    all amplitudes IDENTICAL : {int((multi & identical).sum()):,} "
          f"({(multi & identical).sum()/max(multi.sum(),1):.1%})")
    print(f"    amplitudes DIFFER        : {int((multi & ~identical).sum()):,} "
          f"({(multi & ~identical).sum()/max(multi.sum(),1):.1%})")
    if (multi & ~identical).any():
        spread = (gmax - gmin)[multi & ~identical]
        print(f"    within-group spread      : median {np.median(spread):.2f}, "
              f"p95 {np.percentile(spread,95):.2f} (full range {a.max()-a.min():.1f})")
    print(f"  rate if duplicates AVERAGED (unique/active): {uniq.size/active:.2f} Hz")
    print(f"  rate as recorded            (n/active)     : {cap.n_samples/active:.2f} Hz")
