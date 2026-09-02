"""Step 2 - validate R-peak detection against MITDB expert annotations.

GATE: every rhythm number downstream depends on peak timing, so this runs
before any rhythm work.

Five arms, so detector quality is separated from degradation cost, and the
delivered rate is separated from the device's native rate:
  A  360 Hz, cleaned       - detector ceiling on native data
  B  89.7 Hz, cleaned      - cost of resolution matching (brief 4.5)
  C  89.7 Hz, NOT cleaned  - emulates the HISTORICAL device path (the 37
                             captures we hold), whose firmware already filters
                             (`ecg_clean`); brief 3.2 forbids a second filter
                             chain on device data
  D  133.83 Hz, cleaned    - the device's native rate (session 2026-09-02)
  E  133.83 Hz, NOT cleaned - emulates the CORRECTED device path

Arms B/C measure what the 2:3-decimated delivery path could achieve. Arms D/E
measure what the device itself can achieve once the path delivers every
sample. A 100 ms QRS spans ~9.0 samples at 89.70 Hz and ~13.4 at 133.83 Hz, so
D/E are expected to beat B/C - that expectation is the point of the comparison
and is not assumed anywhere downstream.

Lead handling per brief 4.4: the MLII channel is selected per record from the
header. Records with no MLII channel are EXCLUDED, not silently substituted.

Metrics are sensitivity and PPV with a +/-150 ms matching tolerance
(ANSI/AAMI EC57). Accuracy is not reported - it is meaningless here.
"""
from __future__ import annotations

import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm import SCOPE_STATEMENT
from rhythm.degrade import (DEVICE_FS, DEVICE_FS_NATIVE, resample_to,
                            map_sample_indices)

warnings.filterwarnings("ignore")

DB = Path("data/raw/public/mitdb")
TOL_MS = 150.0
# AAMI beat-annotation symbols; everything else is rhythm/quality markup
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")
PACED = {"102", "104", "107", "217"}


def match(det_ms: np.ndarray, ref_ms: np.ndarray, tol: float = TOL_MS):
    """Greedy nearest matching within tol. Returns (TP, FN, FP)."""
    if det_ms.size == 0:
        return 0, ref_ms.size, 0
    if ref_ms.size == 0:
        return 0, 0, det_ms.size
    used = np.zeros(det_ms.size, bool)
    tp = 0
    j = 0
    for r in ref_ms:
        while j < det_ms.size and det_ms[j] < r - tol:
            j += 1
        k, best, bd = j, -1, tol
        while k < det_ms.size and det_ms[k] <= r + tol:
            if not used[k] and abs(det_ms[k] - r) <= bd:
                bd, best = abs(det_ms[k] - r), k
            k += 1
        if best >= 0:
            used[best] = True
            tp += 1
    return tp, ref_ms.size - tp, det_ms.size - tp


def run_record(rec: str) -> list[dict]:
    import neurokit2 as nk

    hdr = wfdb.rdheader(str(DB / rec))
    if "MLII" not in hdr.sig_name:
        return [dict(record=rec, arm="-", lead="|".join(hdr.sig_name),
                     excluded="no MLII channel")]
    ch = hdr.sig_name.index("MLII")
    sig = wfdb.rdrecord(str(DB / rec), channels=[ch]).p_signal[:, 0]
    fs = float(hdr.fs)

    ann = wfdb.rdann(str(DB / rec), "atr")
    keep = np.array([s in BEAT_SYMBOLS for s in ann.symbol])
    ref_ms = ann.sample[keep] / fs * 1000.0

    out = []
    for arm, target_fs, clean in (("A_360_clean", fs, True),
                                  ("B_89.7_clean", DEVICE_FS, True),
                                  ("C_89.7_raw", DEVICE_FS, False),
                                  ("D_133.8_clean", DEVICE_FS_NATIVE, True),
                                  ("E_133.8_raw", DEVICE_FS_NATIVE, False)):
        if target_fs == fs:
            x, fs_a = sig, fs
        else:
            x, fs_a = resample_to(sig, fs, target_fs)
        try:
            y = nk.ecg_clean(x, sampling_rate=fs_a) if clean else x
            _, info = nk.ecg_peaks(y, sampling_rate=fs_a)
            det_ms = np.asarray(info["ECG_R_Peaks"], float) / fs_a * 1000.0
        except Exception as exc:
            out.append(dict(record=rec, arm=arm, error=str(exc)[:60]))
            continue
        tp, fn, fp = match(det_ms, ref_ms)
        se = tp / (tp + fn) if tp + fn else np.nan
        ppv = tp / (tp + fp) if tp + fp else np.nan
        out.append(dict(
            record=rec, arm=arm, lead="MLII", fs=round(fs_a, 2), paced=rec in PACED,
            n_ref=int(ref_ms.size), n_det=int(det_ms.size),
            TP=tp, FN=fn, FP=fp,
            sensitivity=round(se, 4), ppv=round(ppv, 4),
            f1=round(2 * se * ppv / (se + ppv), 4) if se and ppv else np.nan,
        ))
    return out


def main() -> None:
    recs = sorted(p.stem for p in DB.glob("*.hea"))
    with Pool(8) as pool:
        results = pool.map(run_record, recs)
    df = pd.DataFrame([r for rr in results for r in rr])
    Path("reports_rhythm").mkdir(exist_ok=True)
    df.to_csv("reports_rhythm/s03_detector_mitdb.csv", index=False)

    exc = df[df.get("excluded").notna()] if "excluded" in df else df.iloc[:0]
    if len(exc):
        print("EXCLUDED (brief 4.4 - lead must be verified, not assumed):")
        for r in exc.itertuples():
            print(f"  {r.record}: leads={r.lead} -> {r.excluded}")

    ok = df[df.arm != "-"].dropna(subset=["sensitivity"])
    print(f"\n{'arm':<14} {'recs':>5} {'ref beats':>10} {'Se':>8} {'PPV':>8} "
          f"{'F1':>8}   (pooled, beat-weighted)")
    print("-" * 68)
    for arm, g in ok.groupby("arm"):
        tp, fn, fp = g.TP.sum(), g.FN.sum(), g.FP.sum()
        se, ppv = tp / (tp + fn), tp / (tp + fp)
        print(f"{arm:<14} {len(g):>5} {int(g.n_ref.sum()):>10,} "
              f"{se:>8.4f} {ppv:>8.4f} {2*se*ppv/(se+ppv):>8.4f}")

    print(f"\n--- excluding the 4 paced records {sorted(PACED)} ---")
    for arm, g in ok[~ok.paced.astype(bool)].groupby("arm"):
        tp, fn, fp = g.TP.sum(), g.FN.sum(), g.FP.sum()
        se, ppv = tp / (tp + fn), tp / (tp + fp)
        print(f"{arm:<14} {len(g):>5} {int(g.n_ref.sum()):>10,} "
              f"{se:>8.4f} {ppv:>8.4f} {2*se*ppv/(se+ppv):>8.4f}")

    print("\n--- worst 8 records at device resolution (arm B) ---")
    b = ok[ok.arm == "B_89.7_clean"].nsmallest(8, "f1")
    print(b[["record", "paced", "n_ref", "TP", "FN", "FP", "sensitivity", "ppv", "f1"]]
          .to_string(index=False))
    print(f"\nwrote reports_rhythm/s03_detector_mitdb.csv\n\n{SCOPE_STATEMENT}")


if __name__ == "__main__":
    main()
