"""Choose the SECOND detector for bSQI, on device data.

The MITDB-inherited pair (neurokit + pantompkins1985) refused 81.7% of device
windows because pantompkins finds ~half the beats on this signal - bSQI was
measuring detector B's failure, not the signal's.

B is chosen here by agreement with A ON DEVICE DATA. A good B agrees with A on
clean stretches (so bSQI is high where the signal is good) while still being
an INDEPENDENT algorithm, so genuine disagreement remains informative.
"""
import sys, warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm.ingest import load_capture
from rhythm.resample import to_uniform_segments
from rhythm.sqi import bsqi

warnings.filterwarnings("ignore")
CANDIDATES = ["pantompkins1985", "hamilton2002", "elgendi2010",
              "christov2004", "engzeemod2012", "kalidas2017", "rodrigues2021"]
FILES = ["1789/2026-08-17_08.csv", "1789/2026-08-20_10.csv", "1789/2026-08-20_09.csv",
         "1793/2026-08-08_10.csv", "1793/2026-08-08_07.csv", "1794/2026-08-07_10.csv",
         "1794/2026-08-09_13.csv", "1049/2026-08-08_06.csv"]


def run(rel):
    import neurokit2 as nk
    cap = load_capture(Path("prorithm_ecg") / rel)
    segs, _ = to_uniform_segments(cap.t_ms, cap.amplitude)
    if not segs:
        return []
    seg = max(segs, key=lambda s: s.duration_s)
    pa = np.asarray(nk.ecg_peaks(seg.x, sampling_rate=seg.fs)[1]["ECG_R_Peaks"], np.int64)
    out = []
    for m in CANDIDATES:
        try:
            pb = np.asarray(nk.ecg_peaks(seg.x, sampling_rate=seg.fs, method=m)[1]["ECG_R_Peaks"], np.int64)
            # bSQI over 64-beat windows of A
            vals = []
            for s in range(0, pa.size - 64 + 1, 64):
                w = pa[s:s+64]; i0, i1 = int(w[0]), int(w[-1])
                sub = pb[(pb >= i0) & (pb <= i1)]
                vals.append(bsqi(w * 1000.0 / seg.fs, sub * 1000.0 / seg.fs))
            out.append(dict(file=rel, method=m, n_a=int(pa.size), n_b=int(pb.size),
                            count_ratio=float(pb.size / pa.size),
                            bsqi_med=float(np.median(vals)) if vals else np.nan,
                            frac_ge_095=float(np.mean(np.array(vals) >= 0.95)) if vals else np.nan))
        except Exception as e:
            out.append(dict(file=rel, method=m, error=str(e)[:40]))
    return out


if __name__ == "__main__":
    with Pool(4) as pool:
        res = pool.map(run, FILES)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv("reports_rhythm/s11_detector_b.csv", index=False)
    ok = df[df.get("bsqi_med").notna()]
    print(f"{'candidate B':<18} {'files':>5} {'n_B/n_A':>9} {'median bSQI':>12} {'% windows >=0.95':>18}")
    print("-" * 68)
    for m, g in ok.groupby("method"):
        print(f"{m:<18} {len(g):>5} {g.count_ratio.median():>9.3f} "
              f"{g.bsqi_med.median():>12.4f} {g.frac_ge_095.median():>17.1%}")
    best = ok.groupby("method").bsqi_med.median().idxmax()
    print(f"\nbest agreement with neurokit on device data: {best}")
