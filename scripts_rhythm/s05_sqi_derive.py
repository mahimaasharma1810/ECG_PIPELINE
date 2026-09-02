"""Stage 3 - derive and validate the SQI gate on MITDB.

Windows are 64 EXPERT beats (63 intervals). The detector's CV is computed
over its own peaks in the IDENTICAL time span, so ref and det windows pair
exactly - this removes the selection bias in the earlier s04 diagnostic,
where the two window streams desynchronised on badly-detected records.

Success criterion: gating must suppress the tail of detector-induced CV
error, because that tail is what would flip a REGULAR window to IRREGULAR.
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
from rhythm.features import window_features, INTERVALS_PER_WINDOW, BEATS_PER_WINDOW
from rhythm.sqi import signal_quality, bsqi

warnings.filterwarnings("ignore")
DB = Path("data/raw/public/mitdb")
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")


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

    x, fsa = resample_to(sig, fs, DEVICE_FS)          # arm C: no extra filtering
    try:
        _, ia = nk.ecg_peaks(x, sampling_rate=fsa)
        _, ib = nk.ecg_peaks(x, sampling_rate=fsa, method="pantompkins1985")
    except Exception as e:
        return [dict(record=rec, error=str(e)[:80])]
    pa = np.asarray(ia["ECG_R_Peaks"], float) / fsa * 1000.0
    pb = np.asarray(ib["ECG_R_Peaks"], float) / fsa * 1000.0

    rows = []
    for s in range(0, ref_ms.size - BEATS_PER_WINDOW + 1, BEATS_PER_WINDOW):
        w = ref_ms[s:s + BEATS_PER_WINDOW]
        t0, t1 = w[0], w[-1]
        f_ref = window_features(np.diff(w))
        da = pa[(pa >= t0) & (pa <= t1)]
        db = pb[(pb >= t0) & (pb <= t1)]
        if da.size < 5:
            continue
        f_det = window_features(np.diff(da))
        i0, i1 = int(t0 / 1000 * fsa), int(t1 / 1000 * fsa)
        comp = signal_quality(x[i0:i1], fsa)
        rows.append(dict(
            record=rec, t0_ms=t0,
            n_ref_beats=int(w.size), n_det_beats=int(da.size),
            cv_ref=f_ref.rr_cv, cv_det=f_det.rr_cv,
            rmssd_ref=f_ref.rmssd_ms, rmssd_det=f_det.rmssd_ms,
            bsqi=bsqi(da, db), **comp,
        ))
    return rows


if __name__ == "__main__":
    recs = sorted(p.stem for p in DB.glob("*.hea"))
    with Pool(8) as pool:
        res = pool.map(run, recs)
    df = pd.DataFrame([r for rr in res for r in rr])
    df = df.dropna(subset=["cv_ref", "cv_det"])
    df["d_cv"] = df.cv_det - df.cv_ref
    df["abs_d"] = df.d_cv.abs()
    df.to_csv("reports_rhythm/s05_sqi_windows.csv", index=False)

    print(f"windows: {len(df):,} across {df.record.nunique()} records "
          f"(64 expert beats each, paired by time span)")
    print(f"\nUNGATED detector-induced CV error:")
    for q in (.5, .9, .95, .99):
        print(f"  p{q*100:>4.1f}: {df.abs_d.quantile(q):.4f}")
    print(f"  max   : {df.abs_d.max():.4f}   |err|>0.05: {(df.abs_d>0.05).mean():.1%}")

    print(f"\n--- bSQI vs induced error ---")
    df["bin"] = pd.cut(df.bsqi, [0, .5, .7, .8, .9, .95, 1.01],
                       labels=["<.5", ".5-.7", ".7-.8", ".8-.9", ".9-.95", ">=.95"])
    g = df.groupby("bin", observed=True).agg(
        n=("abs_d", "size"), med=("abs_d", "median"),
        p95=("abs_d", lambda s: s.quantile(.95)), mx=("abs_d", "max")).round(4)
    print(g.to_string())

    print(f"\n--- sweep bsqi_min: what does the gate buy? ---")
    print(f"{'bsqi_min':>9} {'kept':>7} {'kept%':>7} {'p95':>8} {'p99':>8} {'max':>8} "
          f"{'>0.05':>7}")
    for thr in (0.0, 0.5, 0.7, 0.8, 0.85, 0.90, 0.95):
        k = df[df.bsqi >= thr]
        if k.empty:
            continue
        print(f"{thr:>9.2f} {len(k):>7,} {len(k)/len(df):>6.1%} "
              f"{k.abs_d.quantile(.95):>8.4f} {k.abs_d.quantile(.99):>8.4f} "
              f"{k.abs_d.max():>8.4f} {(k.abs_d>0.05).mean():>6.1%}")

    print(f"\n--- worst records, and whether the gate catches them ---")
    w = df.groupby("record").agg(n=("abs_d","size"), bsqi_med=("bsqi","median"),
                                 d_med=("abs_d","median"), d_max=("abs_d","max"))
    print(w.sort_values("d_med", ascending=False).head(8).round(4).to_string())
