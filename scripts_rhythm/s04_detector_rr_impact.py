"""Step 2b - how much irregularity does DETECTOR ERROR manufacture?

A false R-peak splits one RR into two; a missed peak fuses two into one.
Either way the rhythm features move. This measures, per 64-beat window,
CV and RMSSD computed from expert annotations vs from the detector output,
on MITDB degraded to device resolution via the device path (arm C).

Any window whose verdict is driven by detection error rather than by the
heart is a lie the SQI gate (Stage 3) has to catch.
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
from rhythm.features import window_features, iter_windows, BEATS_PER_WINDOW

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

    x, fs_a = resample_to(sig, fs, DEVICE_FS)
    _, info = nk.ecg_peaks(x, sampling_rate=fs_a)          # arm C: no extra filtering
    det_ms = np.asarray(info["ECG_R_Peaks"], float) / fs_a * 1000.0

    rows = []
    # index windows by TIME so ref and det windows are comparable
    for src, peaks in (("ref", ref_ms), ("det", det_ms)):
        for s, rr in iter_windows(peaks, step_beats=BEATS_PER_WINDOW):
            f = window_features(rr)
            rows.append(dict(record=rec, src=src, start_ms=float(peaks[s]),
                             **f.as_dict()))
    return rows


if __name__ == "__main__":
    recs = sorted(p.stem for p in DB.glob("*.hea"))
    with Pool(8) as pool:
        res = pool.map(run, recs)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv("reports_rhythm/s04_rr_impact.csv", index=False)

    # pair ref/det windows by nearest start time within 5 s
    pairs = []
    for rec, g in df.groupby("record"):
        a = g[g.src == "ref"].sort_values("start_ms")
        b = g[g.src == "det"].sort_values("start_ms")
        if a.empty or b.empty:
            continue
        idx = np.searchsorted(b.start_ms.values, a.start_ms.values)
        idx = np.clip(idx, 0, len(b) - 1)
        for (_, ra), k in zip(a.iterrows(), idx):
            rb = b.iloc[k]
            if abs(rb.start_ms - ra.start_ms) > 5000:
                continue
            pairs.append(dict(record=rec,
                              cv_ref=ra.rr_cv, cv_det=rb.rr_cv,
                              rmssd_ref=ra.rmssd_ms, rmssd_det=rb.rmssd_ms,
                              usable_ref=ra.usable, usable_det=rb.usable))
    p = pd.DataFrame(pairs).dropna(subset=["cv_ref", "cv_det"])
    p["d_cv"] = p.cv_det - p.cv_ref
    p.to_csv("reports_rhythm/s04_rr_impact_paired.csv", index=False)

    print(f"paired 64-beat windows: {len(p):,} across {p.record.nunique()} records")
    print(f"\nCV from expert annotations : median {p.cv_ref.median():.4f}, "
          f"p95 {p.cv_ref.quantile(.95):.4f}")
    print(f"CV from detector           : median {p.cv_det.median():.4f}, "
          f"p95 {p.cv_det.quantile(.95):.4f}")
    print(f"\ndetector-induced CV error (det - ref):")
    for q in (.5, .9, .95, .99):
        print(f"  p{q*100:>4.1f}: {p.d_cv.quantile(q):+.4f}")
    print(f"  max  : {p.d_cv.max():+.4f}")
    print(f"  |error| > 0.02 in {(p.d_cv.abs()>0.02).mean():.1%} of windows")
    print(f"  |error| > 0.05 in {(p.d_cv.abs()>0.05).mean():.1%} of windows")

    print(f"\n--- worst records by median induced CV error ---")
    w = (p.groupby("record")
           .agg(n=("d_cv","size"), cv_ref=("cv_ref","median"),
                cv_det=("cv_det","median"), d_med=("d_cv","median"),
                d_max=("d_cv","max"))
           .sort_values("d_med", ascending=False).head(8).round(4))
    print(w.to_string())
