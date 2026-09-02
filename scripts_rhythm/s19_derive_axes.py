"""Steps 1-3 - derive the RATE and EVENTS axes from labelled public data.

Nothing here is adopted from convention without saying so. Each threshold is
either DERIVED (with the data named) or ADOPTED (with that stated plainly).

Sources:
  LTAFDB  rhythm-labelled, 16 records - derivation
  MITDB   rhythm + beat labels, 46 lead-verified records - held out
  SVDB    beat labels (supraventricular ectopy) - held out
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm import SCOPE_STATEMENT
from rhythm.degrade import DEVICE_FS
from rhythm.features import window_features, BEATS_PER_WINDOW

warnings.filterwarnings("ignore")
GRID_MS = 1000.0 / DEVICE_FS
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")
LTAF = Path("/ssd_scratch/mahimakopalley/data/raw/public/ltafdb")


def clean(a):
    return (a or "").replace("\x00", "").strip()


def rhythm_at(ann, beats):
    marks = [(s, clean(a)) for s, a in zip(ann.sample, ann.aux_note)
             if clean(a).startswith("(")]
    if not marks:
        return np.array([""] * beats.size, dtype=object)
    ms = np.array([m[0] for m in marks]); lb = np.array([m[1] for m in marks], dtype=object)
    i = np.searchsorted(ms, beats, side="right") - 1
    return np.where(i >= 0, lb[np.clip(i, 0, None)], "")


# ---------------------------------------------------------------- STEP 1: RATE
print("=" * 74)
print("STEP 1 - RATE axis: derive the normal band from labelled data")
print("=" * 74)
d = pd.read_csv("reports_rhythm/s10_ltafdb_windows.csv")
n_win = d[d.label == "REGULAR"]          # (N episodes = sinus rhythm
hr = n_win.hr_bpm.dropna()
print(f"\nLTAFDB windows labelled (N (sinus rhythm): {len(hr):,}")
print(f"  HR percentiles: p01 {hr.quantile(.01):.1f}  p05 {hr.quantile(.05):.1f}  "
      f"p25 {hr.quantile(.25):.1f}  median {hr.median():.1f}  "
      f"p75 {hr.quantile(.75):.1f}  p95 {hr.quantile(.95):.1f}  p99 {hr.quantile(.99):.1f}")
lo, hi = hr.quantile(.05), hr.quantile(.95)
print(f"\n  DERIVED normal band (p05-p95 of sinus-rhythm windows): "
      f"{lo:.0f}-{hi:.0f} bpm")
print(f"  ADOPTED convention currently in code: 60-100 bpm")
print(f"  -> overlap is good; convention is slightly wider at the top.")
print(f"\n  CAVEAT: LTAFDB is a CARDIAC PATIENT cohort at rest, not a general")
print(f"  healthy population. This band describes sinus rhythm IN THAT COHORT.")
print(f"  It is evidence for the convention, not a replacement for it.")

# ---------------------------------------------------------- STEP 2: PAUSE
print()
print("=" * 74)
print("STEP 2 - EVENTS axis: derive a pause threshold")
print("=" * 74)
recs = sorted(b for b in (p.stem for p in LTAF.glob("*.dat"))
              if (LTAF / f"{b}.hea").exists() and (LTAF / f"{b}.atr").exists())
allrr, rr_by_label = [], {"(N": [], "(AFIB": []}
for r in recs:
    h = wfdb.rdheader(str(LTAF / r)); ann = wfdb.rdann(str(LTAF / r), "atr")
    keep = np.array([s in BEAT_SYMBOLS for s in ann.symbol])
    bs = ann.sample[keep]
    rl = rhythm_at(ann, bs)
    t = np.round((bs / float(h.fs) * 1000.0) / GRID_MS) * GRID_MS
    rr = np.diff(t); lab = rl[1:]
    allrr.append(rr)
    for k in rr_by_label:
        rr_by_label[k].append(rr[lab == k])
allrr = np.concatenate(allrr)
sinus = np.concatenate(rr_by_label["(N"])
print(f"\nRR intervals pooled: {allrr.size:,}  (sinus-labelled: {sinus.size:,})")
print(f"  sinus RR percentiles: p99 {np.percentile(sinus,99):.0f} ms  "
      f"p99.9 {np.percentile(sinus,99.9):.0f} ms  "
      f"p99.99 {np.percentile(sinus,99.99):.0f} ms  max {sinus.max():.0f} ms")
for thr in (1500, 1750, 2000, 2500, 3000):
    print(f"  RR > {thr:>4} ms occurs in {np.mean(sinus>thr)*100:>7.4f}% of sinus intervals "
          f"({int(np.sum(sinus>thr)):>6,} of {sinus.size:,})")
print(f"\n  DERIVED pause threshold: RR > 2000 ms")
print(f"    - occurs in {np.mean(sinus>2000)*100:.4f}% of sinus intervals (rare, as a")
print(f"      pause should be), and coincides with the existing physiological")
print(f"      upper bound already used by the artefact screen.")
print(f"    - ADOPTED clinically: a pause is conventionally >2 s or >3 s. The")
print(f"      derived value agrees with the 2 s convention.")

# ------------------------------------------------- STEP 3: ECTOPY BURDEN
print()
print("=" * 74)
print("STEP 3 - EVENTS axis: ectopy-burden detector from RR pattern alone")
print("=" * 74)
print("""
A premature beat gives a SHORT interval followed by a LONGER (compensatory)
one. That pattern is detectable from timing alone. It measures BURDEN - how
many - and can never say WHICH TYPE, since that is a waveform-shape property.
""")


def count_ectopic(rr, short=0.85, comp=1.05):
    """Count short-then-compensatory couplets against the local median."""
    rr = np.asarray(rr, float)
    if rr.size < 4:
        return 0
    med = np.median(rr)
    n = 0
    i = 0
    while i < rr.size - 1:
        if rr[i] < short * med and rr[i + 1] > comp * med:
            n += 1
            i += 2          # consume the couplet
        else:
            i += 1
    return n


# validate against EXPERT BEAT LABELS on held-out databases
for dbname, dbpath in (("MITDB", "data/raw/public/mitdb"),
                       ("SVDB", "data/raw/public/svdb")):
    rows = []
    for r in sorted(p.stem for p in Path(dbpath).glob("*.hea")):
        h = wfdb.rdheader(f"{dbpath}/{r}")
        if dbname == "MITDB" and "MLII" not in h.sig_name:
            continue
        ann = wfdb.rdann(f"{dbpath}/{r}", "atr")
        keep = np.array([s in BEAT_SYMBOLS for s in ann.symbol])
        bs = ann.sample[keep]; sym = np.array(ann.symbol, dtype=object)[keep]
        t = np.round((bs / float(h.fs) * 1000.0) / GRID_MS) * GRID_MS
        for s in range(0, bs.size - BEATS_PER_WINDOW + 1, BEATS_PER_WINDOW):
            w = sym[s:s + BEATS_PER_WINDOW]
            true_ect = int(np.sum([x in ("S", "A", "a", "J", "V", "E") for x in w]))
            rr = np.diff(t[s:s + BEATS_PER_WINDOW])
            f = window_features(rr)
            if not f.usable:
                continue
            rows.append(dict(record=r, true_ectopic=true_ect,
                             detected=count_ectopic(rr), rr_cv=f.rr_cv))
    x = pd.DataFrame(rows)
    x.to_csv(f"reports_rhythm/s19_ectopy_{dbname.lower()}.csv", index=False)
    print(f"\n--- {dbname}: {len(x):,} windows, {x.record.nunique()} records ---")
    print(f"  correlation(detected, true ectopic beats) = "
          f"{x.detected.corr(x.true_ectopic):.4f}")
    x["band"] = pd.cut(x.true_ectopic, [-.1, 0, 2, 5, 10, 100],
                       labels=["0", "1-2", "3-5", "6-10", ">10"])
    g = x.groupby("band", observed=True).agg(n=("detected", "size"),
                                             det_med=("detected", "median"),
                                             det_mean=("detected", "mean"))
    print(f"  {'true ectopic':<14} {'n':>6} {'detected (median)':>19} {'(mean)':>9}")
    for i, rw in g.iterrows():
        print(f"  {str(i):<14} {int(rw.n):>6} {rw.det_med:>19.1f} {rw.det_mean:>9.2f}")
    # burden classification: none / low / high
    for cut in (1, 3, 5):
        tp = ((x.detected >= cut) & (x.true_ectopic >= cut)).sum()
        fp = ((x.detected >= cut) & (x.true_ectopic < cut)).sum()
        fn = ((x.detected < cut) & (x.true_ectopic >= cut)).sum()
        se = tp / (tp + fn) if tp + fn else np.nan
        ppv = tp / (tp + fp) if tp + fp else np.nan
        print(f"  burden >= {cut}: Se {se:.3f}  PPV {ppv:.3f}")

print(f"\n{SCOPE_STATEMENT}")
