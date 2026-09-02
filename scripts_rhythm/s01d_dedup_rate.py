"""Step 1d - hypothesis: the device samples at ~90 Hz, and files reporting
120-178 Hz contain REPLAYED stream content (exact (timestamp, amplitude) rows
delivered more than once).

Test: drop exact duplicate (t, amp) pairs, recompute the active-time rate,
and see whether the 37 captures collapse onto a single rate.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm.ingest import load_capture

GAP_S = 1.0
rows = []
for f in sorted(Path("prorithm_ecg").glob("*/*.csv")):
    cap = load_capture(f)
    t, a = cap.t_ms, cap.amplitude

    pairs = np.stack([t, a], axis=1)
    _, keep_idx = np.unique(pairs, axis=0, return_index=True)
    keep_idx.sort()
    td, ad = t[keep_idx], a[keep_idx]

    def active_time(x):
        d = np.diff(x) / 1000.0
        return float(d[d <= GAP_S].sum())

    act_raw, act_ded = active_time(t), active_time(td)
    rows.append(dict(
        subject=cap.subject, file=Path(cap.path).name,
        n_raw=cap.n_samples, n_dedup=int(td.size),
        exact_dup_frac=round(1 - td.size / t.size, 3),
        fs_raw=round(cap.n_samples / act_raw, 2) if act_raw else np.nan,
        fs_dedup=round(td.size / act_ded, 2) if act_ded else np.nan,
    ))

df = pd.DataFrame(rows)
df.to_csv("reports_rhythm/s01d_dedup_rate.csv", index=False)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
print("\n--- fs BEFORE exact-duplicate removal ---")
print(f"  median {df.fs_raw.median():.2f} Hz  range [{df.fs_raw.min():.2f}, {df.fs_raw.max():.2f}]  "
      f"CV {df.fs_raw.std()/df.fs_raw.mean():.1%}")
print("--- fs AFTER exact-duplicate removal ---")
print(f"  median {df.fs_dedup.median():.2f} Hz  range [{df.fs_dedup.min():.2f}, {df.fs_dedup.max():.2f}]  "
      f"CV {df.fs_dedup.std()/df.fs_dedup.mean():.1%}")
print(f"\nfiles with >5% exact duplicate rows: {(df.exact_dup_frac > .05).sum()} / {len(df)}")
