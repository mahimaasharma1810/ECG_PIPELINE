"""TASK 4 - derive the hysteresis N from data.

N=3 was an admitted arbitrary constant. Overlapping windows share most of
their beats, so a single borderline interval can flip consecutive verdicts.
This measures the verdict flip-rate as a function of N on LTAFDB windows and
picks the point where the curve flattens.

The cost of a larger N is LAG: the reported state changes N windows after the
underlying rhythm does. Both are reported so the trade is explicit.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm import SCOPE_STATEMENT
from rhythm.degrade import DEVICE_FS
from rhythm.features import window_features, BEATS_PER_WINDOW
from rhythm.verdict import decide
from rhythm.smoothing import smooth

DB = Path("/ssd_scratch/mahimakopalley/data/raw/public/ltafdb")
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")
GRID_MS = 1000.0 / DEVICE_FS
SLIDE_BEATS = 8          # 87.5% overlap - the regime the brief warns about


def clean(a):
    return (a or "").replace("\x00", "").strip()


recs = sorted(b for b in (p.stem for p in DB.glob("*.dat"))
              if (DB / f"{b}.hea").exists() and (DB / f"{b}.atr").exists())
print(f"LTAFDB records: {len(recs)}   slide {SLIDE_BEATS} beats "
      f"({(1-SLIDE_BEATS/BEATS_PER_WINDOW):.0%} overlap)\n")

seqs = []
for r in recs:
    h = wfdb.rdheader(str(DB / r)); ann = wfdb.rdann(str(DB / r), "atr")
    keep = np.array([s in BEAT_SYMBOLS for s in ann.symbol])
    bs = ann.sample[keep]
    t = np.round((bs / float(h.fs) * 1000.0) / GRID_MS) * GRID_MS
    vs, ts = [], []
    for s in range(0, bs.size - BEATS_PER_WINDOW + 1, SLIDE_BEATS):
        f = window_features(np.diff(t[s:s + BEATS_PER_WINDOW]))
        vs.append(decide(f, True, "ok")); ts.append(t[s])
    if len(vs) > 50:
        seqs.append((r, vs, ts))
    print(f"  {r}: {len(vs):,} sliding windows")

print(f"\n{'N':>3} {'state changes':>14} {'changes/hour':>13} {'lag (windows)':>14} {'lag (s)':>9}")
print("-" * 60)
rows = []
for N in range(1, 11):
    tot_ch, tot_h, lag_s = 0, 0.0, []
    for r, vs, ts in seqs:
        states, trans = smooth(vs, ts, n_consecutive=N)
        tot_ch += len(trans)
        tot_h += (ts[-1] - ts[0]) / 1000.0 / 3600.0
        step = np.median(np.diff(ts)) / 1000.0
        lag_s.append(N * step)
    rate = tot_ch / tot_h if tot_h else np.nan
    rows.append((N, tot_ch, rate, N, float(np.median(lag_s))))
    print(f"{N:>3} {tot_ch:>14,} {rate:>13.2f} {N:>14} {np.median(lag_s):>9.1f}")

df = pd.DataFrame(rows, columns=["N", "changes", "per_hour", "lag_windows", "lag_s"])
df.to_csv("reports_rhythm/s15_hysteresis_n.csv", index=False)

r1 = df.per_hour.iloc[0]
print(f"\nflip-rate reduction relative to N=1 (no hysteresis, {r1:.1f}/h):")
for _, x in df.iterrows():
    bar = "#" * int(40 * x.per_hour / r1)
    print(f"  N={int(x.N):<2} {x.per_hour:>7.2f}/h  ({x.per_hour/r1:>5.1%})  "
          f"lag {x.lag_s:>5.1f}s  {bar}")

# knee: first N where the marginal gain drops below 5% of the N=1 rate
d = -df.per_hour.diff()
knee = None
for i in range(1, len(df)):
    if d.iloc[i] / r1 < 0.05:
        knee = int(df.N.iloc[i]); break
print(f"\ncurve flattens at N = {knee}  "
      f"(first N whose marginal reduction is < 5% of the N=1 rate)")
sel = df[df.N == knee].iloc[0]
print(f"  at N={knee}: {sel.per_hour:.2f} state changes/hour "
      f"({sel.per_hour/r1:.1%} of unsmoothed), lag {sel.lag_s:.1f} s")
print(f"\n{SCOPE_STATEMENT}")
