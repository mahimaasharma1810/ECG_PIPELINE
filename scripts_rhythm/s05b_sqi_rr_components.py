"""Does an RR-plausibility component catch what bSQI misses?

bSQI only sees UNCORRELATED detector disagreement. When both detectors make
the same T-wave error (record 231), bSQI stays high and the window passes.

Over-detection has a distinct RR signature that needs no annotations:
inserting a spurious peak splits one interval into a SHORT one followed by a
shorter-than-normal remainder, producing (a) intervals well below the window
median and (b) strong short/long ALTERNATION, i.e. negative lag-1
autocorrelation of the RR series.

Both are computable from detector output alone, so both are usable on device
data. Tested here against the known-bad MITDB windows.
"""
from __future__ import annotations

import sys, warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm.degrade import DEVICE_FS, resample_to
from rhythm.features import window_features, BEATS_PER_WINDOW
from rhythm.sqi import bsqi

warnings.filterwarnings("ignore")
DB = Path("data/raw/public/mitdb")
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")


def rr_plausibility(rr: np.ndarray) -> dict:
    rr = np.asarray(rr, float)
    rr = rr[np.isfinite(rr) & (rr > 0)]
    if rr.size < 4:
        return dict(frac_short=1.0, frac_long=1.0, lag1=0.0)
    med = np.median(rr)
    d = rr - rr.mean()
    denom = np.sum(d * d)
    lag1 = float(np.sum(d[:-1] * d[1:]) / denom) if denom > 0 else 0.0
    return dict(frac_short=float(np.mean(rr < 0.5 * med)),
                frac_long=float(np.mean(rr > 1.75 * med)),
                lag1=lag1)


def run(rec: str):
    import neurokit2 as nk
    hdr = wfdb.rdheader(str(DB / rec))
    if "MLII" not in hdr.sig_name:
        return []
    ch = hdr.sig_name.index("MLII")
    sig = wfdb.rdrecord(str(DB / rec), channels=[ch]).p_signal[:, 0]
    fs = float(hdr.fs)
    ann = wfdb.rdann(str(DB / rec), "atr")
    keep = np.array([s in BEAT_SYMBOLS for s in ann.symbol])
    ref_ms = ann.sample[keep] / fs * 1000.0
    x, fsa = resample_to(sig, fs, DEVICE_FS)
    _, ia = nk.ecg_peaks(x, sampling_rate=fsa)
    _, ib = nk.ecg_peaks(x, sampling_rate=fsa, method="pantompkins1985")
    pa = np.asarray(ia["ECG_R_Peaks"], float) / fsa * 1000.0
    pb = np.asarray(ib["ECG_R_Peaks"], float) / fsa * 1000.0

    rows = []
    for s in range(0, ref_ms.size - BEATS_PER_WINDOW + 1, BEATS_PER_WINDOW):
        w = ref_ms[s:s + BEATS_PER_WINDOW]
        t0, t1 = w[0], w[-1]
        da = pa[(pa >= t0) & (pa <= t1)]
        db = pb[(pb >= t0) & (pb <= t1)]
        if da.size < 5:
            continue
        f_ref = window_features(np.diff(w)); f_det = window_features(np.diff(da))
        rows.append(dict(record=rec, t0_ms=t0, cv_ref=f_ref.rr_cv, cv_det=f_det.rr_cv,
                         sdnn_det=f_det.sdnn_ms, mean_rr_det=f_det.mean_rr_ms,
                         bsqi=bsqi(da, db), **rr_plausibility(np.diff(da))))
    return rows


if __name__ == "__main__":
    recs = sorted(p.stem for p in DB.glob("*.hea"))
    with Pool(8) as pool:
        res = pool.map(run, recs)
    df = pd.DataFrame([r for rr in res for r in rr]).dropna(subset=["cv_ref", "cv_det"])
    df["abs_d"] = (df.cv_det - df.cv_ref).abs()
    df.to_csv("reports_rhythm/s05b_sqi_rr.csv", index=False)
    bad = df.abs_d > 0.05

    print(f"windows {len(df):,}; bad (|CV err|>0.05): {bad.sum()} ({bad.mean():.1%})")
    print(f"\ncomponent separation (median value, good vs bad windows):")
    for c in ("bsqi", "frac_short", "frac_long", "lag1"):
        print(f"  {c:<11} good {df.loc[~bad, c].median():+.4f}   "
              f"bad {df.loc[bad, c].median():+.4f}")

    print(f"\n--- windows that PASS bsqi>=0.95 but are still bad ---")
    slip = df[(df.bsqi >= 0.95) & bad]
    print(f"  {len(slip)} windows, records {sorted(slip.record.unique())}")
    print(f"  frac_short: median {slip.frac_short.median():.4f} "
          f"(clean windows {df.loc[(df.bsqi>=.95) & ~bad, 'frac_short'].median():.4f})")
    print(f"  lag1      : median {slip.lag1.median():+.4f} "
          f"(clean windows {df.loc[(df.bsqi>=.95) & ~bad, 'lag1'].median():+.4f})")

    print(f"\n--- SDNN of bad vs good windows (does the guard keep discrimination?) ---")
    print(f"  good: median SDNN {df.loc[~bad,'sdnn_det'].median():.1f} ms")
    print(f"  bad : median SDNN {df.loc[bad,'sdnn_det'].median():.1f} ms")
    for g in (0.0, 10.0, 15.0, 20.0):
        sub = df[df.sdnn_det > g]
        print(f"  SDNN>{g:>4.0f} ms: {len(sub):>5,} windows ({len(sub)/len(df):>5.1%}), "
              f"of which bad {sub.abs_d.gt(0.05).sum():>4} "
              f"({sub.abs_d.gt(0.05).mean():>5.1%})")

    print(f"\n--- GUARDED gate: bsqi>=0.95 AND (SDNN<=guard OR lag1>=l) ---")
    print(f"{'guard':>6} {'lag1_min':>9} {'kept':>7} {'kept%':>7} {'p95':>8} {'p99':>8} {'max':>8} {'>0.05':>7}")
    base = df[df.bsqi >= 0.95]
    for guard in (10.0, 15.0, 20.0):
        for lg in (-0.5, -0.3, -0.2):
            k = base[(base.sdnn_det <= guard) | (base.lag1 >= lg)]
            print(f"{guard:>6.0f} {lg:>9.2f} {len(k):>7,} {len(k)/len(df):>6.1%} "
                  f"{k.abs_d.quantile(.95):>8.4f} {k.abs_d.quantile(.99):>8.4f} "
                  f"{k.abs_d.max():>8.4f} {(k.abs_d>0.05).mean():>6.1%}")

    print(f"\n--- UNGUARDED (for comparison) ---")
    print(f"{'fs_max':>7} {'lag1_min':>9} {'kept':>7} {'kept%':>7} {'p95':>8} {'p99':>8} {'max':>8} {'>0.05':>7}")
    base = df[df.bsqi >= 0.95]
    for fsm in (1.0, 0.10, 0.05, 0.02):
        for lg in (-1.0, -0.5, -0.3):
            k = base[(base.frac_short <= fsm) & (base.lag1 >= lg)]
            if k.empty: continue
            print(f"{fsm:>7.2f} {lg:>9.2f} {len(k):>7,} {len(k)/len(df):>6.1%} "
                  f"{k.abs_d.quantile(.95):>8.4f} {k.abs_d.quantile(.99):>8.4f} "
                  f"{k.abs_d.max():>8.4f} {(k.abs_d>0.05).mean():>6.1%}")
