"""Steps 5-6 - IRREGULAR/REGULAR distributions and threshold derivation.

Source: the 6 recovered LTAFDB records (146 h, 128 Hz), degraded to the
device's measured 89.70 Hz per brief 4.5.

Labels follow the brief: (AFIB -> IRREGULAR, (N -> REGULAR. All other rhythm
annotations (SBR, B, AB, SVTA, VT, T) are EXCLUDED rather than mapped, since
neither label is defensible for them.

RR comes from EXPERT BEAT ANNOTATIONS, not from a detector. This is
deliberate: it isolates the physiological boundary from detection error,
which was shown to manufacture CV ~0.15 on device data. Peak times are
snapped to the 89.70 Hz grid so the RR quantisation matches the device.

LEAD CAVEAT (brief 4.4): LTAFDB headers name both channels simply 'ECG' and
do NOT identify the actual lead. Unlike MITDB - where MLII was verified per
record - the lead here is UNVERIFIED. Any boundary derived from this data is
not a like-for-like Lead-II result and must be reported as such.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm import SCOPE_STATEMENT
from rhythm.degrade import DEVICE_FS
from rhythm.features import window_features, BEATS_PER_WINDOW, INTERVALS_PER_WINDOW

DB = Path("/ssd_scratch/mahimakopalley/data/raw/public/ltafdb")
LABEL = {"(AFIB": "IRREGULAR", "(N": "REGULAR"}
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")
GRID_MS = 1000.0 / DEVICE_FS


def rhythm_at(ann, beat_samples):
    """Assign the rhythm episode in force at each beat."""
    marks = [(s, a) for s, a in zip(ann.sample, ann.aux_note)
             if a and a.startswith("(")]
    if not marks:
        return np.array([""] * beat_samples.size, dtype=object)
    ms = np.array([m[0] for m in marks])
    lb = np.array([m[1].strip() for m in marks], dtype=object)
    idx = np.searchsorted(ms, beat_samples, side="right") - 1
    out = np.where(idx >= 0, lb[np.clip(idx, 0, None)], "")
    return out


def main() -> None:
    recs = sorted(b for b in (p.stem for p in DB.glob("*.dat"))
                  if (DB / f"{b}.hea").exists() and (DB / f"{b}.atr").exists())
    print(f"LTAFDB records available: {len(recs)} -> {recs}")
    print(f"degrading RR timing to device grid: {DEVICE_FS} Hz ({GRID_MS:.3f} ms)\n")

    rows = []
    for r in recs:
        hdr = wfdb.rdheader(str(DB / r))
        ann = wfdb.rdann(str(DB / r), "atr")
        fs = float(hdr.fs)
        keep = np.array([s in BEAT_SYMBOLS for s in ann.symbol])
        bs = ann.sample[keep]
        rl = rhythm_at(ann, bs)
        t_ms = bs / fs * 1000.0
        # resolution matching: snap peak times to the device grid
        t_dev = np.round(t_ms / GRID_MS) * GRID_MS

        n_win = 0
        for s in range(0, bs.size - BEATS_PER_WINDOW + 1, BEATS_PER_WINDOW):
            lab_slice = rl[s:s + BEATS_PER_WINDOW]
            uniq = set(lab_slice)
            if len(uniq) != 1:            # window spans an episode boundary
                continue
            lab = LABEL.get(uniq.pop())
            if lab is None:
                continue
            f = window_features(np.diff(t_dev[s:s + BEATS_PER_WINDOW]))
            if not f.usable:
                continue
            rows.append(dict(record=r, lead="UNVERIFIED", fs_native=fs,
                             fs_grid=DEVICE_FS, label=lab, **f.as_dict()))
            n_win += 1
        print(f"  {r}: {bs.size:,} beats, {hdr.sig_len/fs/3600:.1f} h -> {n_win:,} labelled windows")

    df = pd.DataFrame(rows)
    df.to_csv("reports_rhythm/s10_ltafdb_windows.csv", index=False)
    print(f"\ntotal labelled windows: {len(df):,}")
    print(df.label.value_counts().to_string())

    print(f"\n--- distributions at {DEVICE_FS} Hz (64 beats = "
          f"{INTERVALS_PER_WINDOW} intervals) ---")
    for feat in ("rr_cv", "rmssd_ms", "sdnn_ms", "hr_bpm"):
        print(f"\n  {feat}")
        for lab, g in df.groupby("label"):
            s = g[feat].dropna()
            print(f"    {lab:<10} n={len(s):>6}  p05 {s.quantile(.05):>8.4f}  "
                  f"p25 {s.quantile(.25):>8.4f}  median {s.median():>8.4f}  "
                  f"p75 {s.quantile(.75):>8.4f}  p95 {s.quantile(.95):>8.4f}")

    print(f"\n--- separability and boundary (RR CV) ---")
    reg = df[df.label == "REGULAR"].rr_cv.dropna()
    irr = df[df.label == "IRREGULAR"].rr_cv.dropna()
    print(f"  REGULAR   n={len(reg):,}  IRREGULAR n={len(irr):,}")
    print(f"  overlap: REGULAR p95 = {reg.quantile(.95):.4f}, "
          f"IRREGULAR p05 = {irr.quantile(.05):.4f}"
          f"  -> {'OVERLAP' if reg.quantile(.95) > irr.quantile(.05) else 'separated'}")

    print(f"\n{'thr':>7} {'Se':>7} {'Sp':>7} {'PPV':>7} {'NPV':>7} {'Youden':>8}")
    best = None
    for thr in np.arange(0.02, 0.40, 0.005):
        tp = (irr >= thr).sum(); fn = (irr < thr).sum()
        tn = (reg < thr).sum();  fp = (reg >= thr).sum()
        se = tp/(tp+fn) if tp+fn else 0; sp = tn/(tn+fp) if tn+fp else 0
        ppv = tp/(tp+fp) if tp+fp else 0; npv = tn/(tn+fn) if tn+fn else 0
        j = se+sp-1
        if best is None or j > best[1]: best = (thr, j, se, sp, ppv, npv)
        if abs(thr*1000 % 25) < 1e-6:
            print(f"{thr:>7.3f} {se:>7.4f} {sp:>7.4f} {ppv:>7.4f} {npv:>7.4f} {j:>8.4f}")
    thr, j, se, sp, ppv, npv = best
    print(f"\n  BEST (Youden): threshold RR CV = {thr:.4f}")
    print(f"    sensitivity {se:.4f}  specificity {sp:.4f}  PPV {ppv:.4f}  NPV {npv:.4f}")
    print(f"\n  noise floor at 89.7 Hz (uniform grid) is CV ~0.006 -> "
          f"threshold sits {thr/0.006:.0f}x above it")
    print(f"\n{SCOPE_STATEMENT}")


if __name__ == "__main__":
    main()
