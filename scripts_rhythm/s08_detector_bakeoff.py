"""Which R-peak detector is right on THIS device?

MITDB validation passed (F1 0.9890) but MITDB is not this signal - and the
device cross-check says the chosen detector finds ~4.4% too many beats,
which alone reproduces the observed CV of ~0.15.

The device's reported median RR is the only device-side reference available.
It cannot validate rhythm, but it can rank detectors by beat COUNT, which is
the failure in play. Each detector is scored by the ratio of its median RR to
the device's median RR; 1.0 is agreement.
"""
import sys, warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm.ingest import load_capture
from rhythm.resample import to_uniform_segments

warnings.filterwarnings("ignore")
VIT = Path("data/raw/cliniaura_live/vitals")
METHODS = ["neurokit", "pantompkins1985", "hamilton2002", "elgendi2010",
           "christov2004", "engzeemod2012", "kalidas2017", "rodrigues2021"]
FILES = ["1789/2026-08-17_08.csv", "1789/2026-08-20_10.csv", "1789/2026-08-09_14.csv",
         "1793/2026-08-08_07.csv", "1793/2026-08-08_10.csv", "1794/2026-08-07_10.csv",
         "1794/2026-08-09_13.csv", "1049/2026-08-08_06.csv"]


def run(rel):
    import neurokit2 as nk
    cap = load_capture(Path("prorithm_ecg") / rel)
    segs, _ = to_uniform_segments(cap.t_ms, cap.amplitude)
    if not segs:
        return []
    seg = max(segs, key=lambda s: s.duration_s)
    v = pd.read_csv(VIT / rel)
    dev = v.hrv.dropna(); dev = dev[(dev > 300) & (dev < 2000)]
    if dev.empty:
        return []
    out = []
    for m in METHODS:
        try:
            _, info = nk.ecg_peaks(seg.x, sampling_rate=seg.fs, method=m)
            pk = np.asarray(info["ECG_R_Peaks"], np.int64)
            if pk.size < 20:
                raise ValueError("too few peaks")
            rr = np.diff(pk) * 1000.0 / seg.fs
            rr = rr[(rr > 300) & (rr < 2000)]
            if rr.size < 20:
                raise ValueError("too few plausible RR")
            out.append(dict(file=rel, method=m, n_peaks=int(pk.size),
                            pipe_rr=float(np.median(rr)), dev_rr=float(dev.median()),
                            ratio=float(np.median(rr) / dev.median()),
                            cv=float(rr.std(ddof=1) / rr.mean()),
                            rmssd=float(np.sqrt(np.mean(np.diff(rr) ** 2)))))
        except Exception as e:
            out.append(dict(file=rel, method=m, error=str(e)[:40]))
    return out


if __name__ == "__main__":
    with Pool(4) as pool:
        res = pool.map(run, FILES)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv("reports_rhythm/s08_detector_bakeoff.csv", index=False)
    ok = df[df.get("ratio").notna()] if "ratio" in df else df.iloc[:0]
    print(f"{'method':<18} {'n ok':>5} {'RR ratio (1.0=agree)':>22} {'CV':>16} {'RMSSD ms':>16}")
    print(f"{'':<18} {'':>5} {'median   IQR':>22} {'median':>16} {'median':>16}")
    print("-" * 82)
    for m, g in ok.groupby("method"):
        print(f"{m:<18} {len(g):>5} {g.ratio.median():>8.4f} "
              f"[{g.ratio.quantile(.25):.3f},{g.ratio.quantile(.75):.3f}] "
              f"{g.cv.median():>15.4f} {g.rmssd.median():>15.1f}")
    if "error" in df and df.error.notna().any():
        print("\nfailures:")
        for m, g in df[df.error.notna()].groupby("method"):
            print(f"  {m}: {len(g)} files, e.g. {g.error.iloc[0]}")
