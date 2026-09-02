"""TASK 1 - held-out DATABASE validation of the provisional 0.12 threshold.

Derived on LTAFDB. Evaluated here on MITDB, which the threshold has never
seen. This is held-out-DATABASE, not held-out-record (brief 5.8, 11).

A negative result is a valid outcome and is reported as-is. Nothing is tuned
to fix it.

Two label sources, reported SEPARATELY and never pooled:

  PRIMARY   MITDB expert rhythm annotations: (AFIB -> IRREGULAR, (N -> REGULAR.
            Lead verified as MLII per record; records without MLII excluded.

  SECONDARY SVDB has only ONE rhythm annotation in the entire database and zero
            (AFIB, so it cannot provide expert rhythm labels. Its content is
            supraventricular ectopy at the BEAT level. Windows are therefore
            labelled by a PROXY - S-beat density - which is NOT an expert
            rhythm label. Reported separately, clearly marked, and never
            merged with the primary result.

NOTE: MITDB aux_note strings carry trailing NUL bytes ('(AFIB\\x00'), which
str.strip() does not remove. Unhandled, every MITDB rhythm label is silently
lost. Stripped explicitly below.
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
from rhythm.features import window_features, BEATS_PER_WINDOW
from rhythm.verdict import THRESHOLD_CV

BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")
GRID_MS = 1000.0 / DEVICE_FS
PACED = {"102", "104", "107", "217"}


def clean_aux(a: str) -> str:
    return (a or "").replace("\x00", "").strip()


def rhythm_at(ann, beats):
    marks = [(s, clean_aux(a)) for s, a in zip(ann.sample, ann.aux_note)
             if clean_aux(a).startswith("(")]
    if not marks:
        return np.array([""] * beats.size, dtype=object)
    ms = np.array([m[0] for m in marks])
    lb = np.array([m[1] for m in marks], dtype=object)
    i = np.searchsorted(ms, beats, side="right") - 1
    return np.where(i >= 0, lb[np.clip(i, 0, None)], "")


def metrics(reg, irr, thr):
    TP = int((irr >= thr).sum()); FN = int((irr < thr).sum())
    TN = int((reg < thr).sum());  FP = int((reg >= thr).sum())
    se = TP/(TP+FN) if TP+FN else float("nan")
    sp = TN/(TN+FP) if TN+FP else float("nan")
    ppv = TP/(TP+FP) if TP+FP else float("nan")
    npv = TN/(TN+FN) if TN+FN else float("nan")
    return dict(TP=TP, FN=FN, TN=TN, FP=FP, Se=se, Sp=sp, PPV=ppv, NPV=npv)


def show(name, m, n_tot):
    print(f"\n  {name}")
    print(f"                   pred IRREGULAR   pred REGULAR")
    print(f"    true IRREGULAR   TP={m['TP']:>6}        FN={m['FN']:>6}")
    print(f"    true REGULAR     FP={m['FP']:>6}        TN={m['TN']:>6}")
    print(f"    Se {m['Se']:.4f}  Sp {m['Sp']:.4f}  PPV {m['PPV']:.4f}  NPV {m['NPV']:.4f}")
    print(f"    n={n_tot:,}  prevalence {(m['TP']+m['FN'])/n_tot:.1%}")


def mitdb_windows():
    d = Path("data/raw/public/mitdb")
    rows = []
    for r in sorted(p.stem for p in d.glob("*.hea")):
        h = wfdb.rdheader(str(d / r))
        if "MLII" not in h.sig_name:          # brief 4.4 - verified, not assumed
            continue
        ann = wfdb.rdann(str(d / r), "atr")
        keep = np.array([s in BEAT_SYMBOLS for s in ann.symbol])
        bs = ann.sample[keep]
        rl = rhythm_at(ann, bs)
        t = np.round((bs / float(h.fs) * 1000.0) / GRID_MS) * GRID_MS
        for s in range(0, bs.size - BEATS_PER_WINDOW + 1, BEATS_PER_WINDOW):
            u = set(rl[s:s + BEATS_PER_WINDOW])
            if len(u) != 1:
                continue
            lab = {"(AFIB": "IRREGULAR", "(N": "REGULAR"}.get(u.pop())
            if lab is None:
                continue
            f = window_features(np.diff(t[s:s + BEATS_PER_WINDOW]))
            if not f.usable:
                continue
            rows.append(dict(db="mitdb", record=r, lead="MLII", paced=r in PACED,
                             label=lab, rr_cv=f.rr_cv, rmssd=f.rmssd_ms))
    return pd.DataFrame(rows)


def svdb_windows(s_frac_irr=0.10):
    """PROXY labels only - SVDB has no usable rhythm annotations."""
    d = Path("data/raw/public/svdb")
    rows = []
    for r in sorted(p.stem for p in d.glob("*.hea")):
        h = wfdb.rdheader(str(d / r))
        ann = wfdb.rdann(str(d / r), "atr")
        keep = np.array([s in BEAT_SYMBOLS for s in ann.symbol])
        bs = ann.sample[keep]
        sym = np.array(ann.symbol, dtype=object)[keep]
        t = np.round((bs / float(h.fs) * 1000.0) / GRID_MS) * GRID_MS
        for s in range(0, bs.size - BEATS_PER_WINDOW + 1, BEATS_PER_WINDOW):
            w = sym[s:s + BEATS_PER_WINDOW]
            frac_s = float(np.mean([x in ("S", "A", "a", "J") for x in w]))
            frac_v = float(np.mean([x in ("V", "E") for x in w]))
            if frac_v > 0.02:                 # keep the proxy about SV ectopy
                continue
            lab = "IRREGULAR" if frac_s >= s_frac_irr else ("REGULAR" if frac_s == 0 else None)
            if lab is None:
                continue
            f = window_features(np.diff(t[s:s + BEATS_PER_WINDOW]))
            if not f.usable:
                continue
            rows.append(dict(db="svdb", record=r, lead="ECG1 (UNVERIFIED)",
                             label=lab, rr_cv=f.rr_cv, s_frac=frac_s))
    return pd.DataFrame(rows)


def main():
    thr = THRESHOLD_CV
    print(f"threshold under test: RR CV >= {thr} (derived on LTAFDB, 6 records)")
    print(f"resolution: {DEVICE_FS} Hz grid for BOTH derivation and evaluation\n")

    print("=== PRIMARY: MITDB, expert rhythm annotations, lead-verified MLII ===")
    m = mitdb_windows()
    m.to_csv("reports_rhythm/s13_mitdb_windows.csv", index=False)
    print(f"windows: {len(m):,} from {m.record.nunique()} records")
    print(m.groupby(["label"]).size().to_string())
    print(f"records contributing IRREGULAR: "
          f"{sorted(m[m.label=='IRREGULAR'].record.unique())}")
    reg = m[m.label == "REGULAR"].rr_cv.dropna()
    irr = m[m.label == "IRREGULAR"].rr_cv.dropna()
    show("MITDB held-out, all records", metrics(reg, irr, thr), len(m))

    mn = m[~m.paced]
    show("MITDB held-out, paced records excluded",
         metrics(mn[mn.label == "REGULAR"].rr_cv.dropna(),
                 mn[mn.label == "IRREGULAR"].rr_cv.dropna(), thr), len(mn))

    print(f"\n  distribution comparison (RR CV median):")
    print(f"    LTAFDB derivation : REGULAR 0.0305   IRREGULAR 0.2085")
    print(f"    MITDB held-out    : REGULAR {reg.median():.4f}   IRREGULAR {irr.median():.4f}")

    print(f"\n=== SECONDARY: SVDB, PROXY labels (NOT expert rhythm labels) ===")
    s = svdb_windows()
    s.to_csv("reports_rhythm/s13_svdb_windows.csv", index=False)
    print(f"windows: {len(s):,} from {s.record.nunique()} records")
    print(s.groupby(["label"]).size().to_string())
    show("SVDB proxy (S-beat density >=10% = IRREGULAR)",
         metrics(s[s.label == "REGULAR"].rr_cv.dropna(),
                 s[s.label == "IRREGULAR"].rr_cv.dropna(), thr), len(s))
    print(f"\n{SCOPE_STATEMENT}")


if __name__ == "__main__":
    main()
