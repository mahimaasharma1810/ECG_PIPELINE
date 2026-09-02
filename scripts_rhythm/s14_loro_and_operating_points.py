"""TASK 2 - leave-one-record-out robustness of the LTAFDB threshold.
TASK 3 - operating-point sweep for rule-out vs rule-in framing.

Task 2 exists because 2 of 6 LTAFDB records supply 76% of IRREGULAR windows.
If dropping a single record moves the threshold outside 0.11-0.125, the value
is a fit to individual patients rather than a threshold.

Task 3 exists because Youden weights Se and Sp equally, which at 8.7%
prevalence produced 1,096 false positives against 723 true positives.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm import SCOPE_STATEMENT

df = pd.read_csv("reports_rhythm/s10_ltafdb_windows.csv")
mit = pd.read_csv("reports_rhythm/s13_mitdb_windows.csv")
GRID = np.arange(0.02, 0.40, 0.0025)


def curve(reg, irr):
    out = []
    for t in GRID:
        TP = (irr >= t).sum(); FN = (irr < t).sum()
        TN = (reg < t).sum();  FP = (reg >= t).sum()
        se = TP/(TP+FN) if TP+FN else np.nan
        sp = TN/(TN+FP) if TN+FP else np.nan
        ppv = TP/(TP+FP) if TP+FP else np.nan
        npv = TN/(TN+FN) if TN+FN else np.nan
        out.append((t, se, sp, ppv, npv, se+sp-1, int(TP), int(FP), int(TN), int(FN)))
    return pd.DataFrame(out, columns="thr Se Sp PPV NPV J TP FP TN FN".split())


print("=" * 74)
print("TASK 2 - leave-one-record-out (LTAFDB)")
print("=" * 74)
recs = sorted(df.record.unique())
rows = []
for drop in [None] + recs:
    d = df if drop is None else df[df.record != drop]
    reg = d[d.label == "REGULAR"].rr_cv.dropna().to_numpy()
    irr = d[d.label == "IRREGULAR"].rr_cv.dropna().to_numpy()
    if irr.size == 0:
        rows.append(dict(dropped=drop, n_irr=0, thr=np.nan, note="no IRREGULAR left"))
        continue
    c = curve(reg, irr)
    b = c.loc[c.J.idxmax()]
    rows.append(dict(dropped="none (full)" if drop is None else f"record {drop}",
                     n_reg=int(reg.size), n_irr=int(irr.size),
                     thr=round(float(b.thr), 4), Se=round(float(b.Se), 4),
                     Sp=round(float(b.Sp), 4), J=round(float(b.J), 4)))
loro = pd.DataFrame(rows)
print(loro.to_string(index=False))
sub = loro[loro.dropped != "none (full)"].thr.dropna()
print(f"\nthresholds when each record is dropped: {sorted(sub.round(4).tolist())}")
print(f"  range {sub.min():.4f} - {sub.max():.4f}   spread {sub.max()-sub.min():.4f}")
inside = ((sub >= 0.11) & (sub <= 0.125)).all()
print(f"  all within 0.110-0.125? {'YES - STABLE' if inside else 'NO - fit to individual patients'}")
if not inside:
    out = sub[(sub < 0.11) | (sub > 0.125)]
    print(f"  outside: {out.round(4).tolist()}")

print()
print("=" * 74)
print("TASK 3 - operating points")
print("=" * 74)
for name, d in (("LTAFDB (derivation)", df), ("MITDB (held-out database)", mit)):
    reg = d[d.label == "REGULAR"].rr_cv.dropna().to_numpy()
    irr = d[d.label == "IRREGULAR"].rr_cv.dropna().to_numpy()
    c = curve(reg, irr)
    prev = irr.size / (irr.size + reg.size)
    print(f"\n{name}   n={irr.size+reg.size:,}  prevalence {prev:.1%}")
    print(f"  {'operating point':<22} {'thr':>7} {'Se':>7} {'Sp':>7} {'PPV':>7} "
          f"{'NPV':>7} {'TP':>5} {'FP':>6} {'FN':>4}")
    picks = [("Youden-optimal", c.loc[c.J.idxmax()])]
    for target in (0.95, 0.99):
        ok = c[c.Sp >= target]
        picks.append((f"Sp >= {target:.2f}", ok.loc[ok.Se.idxmax()] if len(ok) else None))
    base_se = float(picks[0][1].Se)
    for label, r in picks:
        if r is None:
            print(f"  {label:<22} unattainable"); continue
        cost = base_se - float(r.Se)
        print(f"  {label:<22} {r.thr:>7.4f} {r.Se:>7.4f} {r.Sp:>7.4f} {r.PPV:>7.4f} "
              f"{r.NPV:>7.4f} {int(r.TP):>5} {int(r.FP):>6} {int(r.FN):>4}"
              + (f"   sensitivity cost {cost:+.4f}" if label != "Youden-optimal" else ""))
print(f"\n{SCOPE_STATEMENT}")
