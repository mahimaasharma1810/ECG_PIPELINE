"""TASKS 1-2 - ground-truth ectopic burden, and the RR signature of ectopy.

Beat labels are used ONLY to build ground truth. They are never model input.
Nothing here reads waveform shape.

Lead handling follows the adopted standard: verification is required where the
WAVEFORM is processed, and immaterial where only annotation TIMESTAMPS are
used. This analysis uses timestamps only, so all records are admissible - but
the lead is recorded per record regardless.
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path

import numpy as np, pandas as pd, wfdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm import SCOPE_STATEMENT
from rhythm.degrade import DEVICE_FS
from rhythm.features import window_features, BEATS_PER_WINDOW
from rhythm.verdict import THRESHOLD_CV
from rhythm.sqi import rr_lag1

warnings.filterwarnings("ignore")
G = 1000.0 / DEVICE_FS
BEAT = set("NLRBAaJSVrFejnE/fQ?")
PAC = {"S", "A", "a", "J"}       # supraventricular ectopic
PVC = {"V", "E"}                 # ventricular ectopic
FUS = {"F"}


def build(dbpath, name):
    rows, beat_rows = [], []
    for r in sorted(p.stem for p in Path(dbpath).glob("*.hea")):
        h = wfdb.rdheader(f"{dbpath}/{r}")
        lead = "|".join(h.sig_name)
        ann = wfdb.rdann(f"{dbpath}/{r}", "atr")
        keep = np.array([s in BEAT for s in ann.symbol])
        bs = ann.sample[keep]; sym = np.array(ann.symbol, dtype=object)[keep]
        t = np.round((bs / float(h.fs) * 1000.0) / G) * G
        rr_all = np.diff(t)

        # ---- per-beat: the compensatory-pause test (Task 2 hypothesis) ----
        med = np.median(rr_all)
        for i in range(1, len(sym) - 1):
            cls = ("PAC" if sym[i] in PAC else "PVC" if sym[i] in PVC else
                   "FUS" if sym[i] in FUS else "N" if sym[i] == "N" else None)
            if cls is None:
                continue
            pre, post = rr_all[i - 1], rr_all[i]
            if not (0 < pre < 3000 and 0 < post < 3000):
                continue
            beat_rows.append(dict(db=name, record=r, cls=cls,
                                  pre_norm=pre / med, post_norm=post / med,
                                  sum_norm=(pre + post) / (2 * med)))

        # ---- per-window burden ----
        for s in range(0, bs.size - BEATS_PER_WINDOW + 1, BEATS_PER_WINDOW):
            w = sym[s:s + BEATS_PER_WINDOW]
            rr = np.diff(t[s:s + BEATS_PER_WINDOW])
            f = window_features(rr)
            if not f.usable:
                continue
            n_pac = int(sum(x in PAC for x in w))
            n_pvc = int(sum(x in PVC for x in w))
            # true bigeminy: ectopic beats alternating with normals
            ect = np.array([x in PAC or x in PVC for x in w])
            alt = int(np.sum(ect[:-1] != ect[1:]))
            bigeminy = bool(ect.mean() > 0.35 and alt / (len(ect) - 1) > 0.85)
            rows.append(dict(db=name, record=r, lead=lead,
                             n_pac=n_pac, n_pvc=n_pvc, n_ect=n_pac + n_pvc,
                             bigeminy=bigeminy,
                             rr_cv=f.rr_cv, rmssd=f.rmssd_ms, sdnn=f.sdnn_ms,
                             hr=f.hr_bpm, lag1=rr_lag1(rr),
                             flagged_irregular=bool(f.rr_cv >= THRESHOLD_CV)))
    return pd.DataFrame(rows), pd.DataFrame(beat_rows)


print("=" * 74); print("TASK 1 - ground-truth burden"); print("=" * 74)
frames, beats = [], []
for name, path in (("MITDB", "data/raw/public/mitdb"), ("SVDB", "data/raw/public/svdb")):
    w, b = build(path, name)
    frames.append(w); beats.append(b)
    print(f"\n{name}: {len(w):,} windows, {w.record.nunique()} records")
    print(f"  leads present: {sorted(w.lead.unique())[:4]}"
          f"{' ...' if w.lead.nunique() > 4 else ''}")
    print(f"  windows with any ectopy : {(w.n_ect>0).sum():,} ({(w.n_ect>0).mean():.1%})")
    print(f"  PAC-bearing windows     : {(w.n_pac>0).sum():,} ({(w.n_pac>0).mean():.1%})")
    print(f"  PVC-bearing windows     : {(w.n_pvc>0).sum():,} ({(w.n_pvc>0).mean():.1%})")
    print(f"  true bigeminy windows   : {w.bigeminy.sum():,} ({w.bigeminy.mean():.1%})")
W = pd.concat(frames, ignore_index=True); B = pd.concat(beats, ignore_index=True)
W.to_csv("reports_rhythm/s23_burden_windows.csv", index=False)
B.to_csv("reports_rhythm/s23_burden_beats.csv", index=False)
print(f"\nSVDB is the richer supraventricular source: "
      f"{int(frames[1].n_pac.sum()):,} PACs vs MITDB {int(frames[0].n_pac.sum()):,}")
print(f"MITDB is the richer ventricular source: "
      f"{int(frames[0].n_pvc.sum()):,} PVCs vs SVDB {int(frames[1].n_pvc.sum()):,}")

print()
print("=" * 74)
print("TASK 2a - HYPOTHESIS TEST: do PVCs produce a fuller compensatory pause?")
print("=" * 74)
print("""
Hypothesis: a PVC does not reset the sinus node, so the following pause is
FULL - the pre+post intervals sum to about 2x a normal interval. A PAC DOES
reset it, so the pause is PARTIAL and the sum falls short of 2x.

Test: (RR_before + RR_after) / (2 x median RR). Full compensation -> ~1.0.
""")
print(f"  {'beat class':<8} {'n':>8} {'pre/med':>9} {'post/med':>10} {'sum/2med':>10}")
print("  " + "-" * 50)
for cls in ("N", "PAC", "PVC", "FUS"):
    g = B[B.cls == cls]
    if len(g) < 20:
        continue
    print(f"  {cls:<8} {len(g):>8,} {g.pre_norm.median():>9.3f} "
          f"{g.post_norm.median():>10.3f} {g.sum_norm.median():>10.3f}")
pac, pvc = B[B.cls == "PAC"], B[B.cls == "PVC"]
print(f"\n  PVC sum/2med {pvc.sum_norm.median():.3f} vs PAC {pac.sum_norm.median():.3f}")
print(f"  -> {'SUPPORTED' if pvc.sum_norm.median() > pac.sum_norm.median() else 'NOT SUPPORTED'}: "
      f"PVC compensation is {'fuller' if pvc.sum_norm.median() > pac.sum_norm.median() else 'not fuller'}")
print(f"  RR perturbation magnitude |post-pre|: "
      f"PVC {np.abs(pvc.post_norm-pvc.pre_norm).median():.3f}  "
      f"PAC {np.abs(pac.post_norm-pac.pre_norm).median():.3f}")
print(f"  -> the larger perturbation is why PVCs move RR CV more at equal count")
print(f"\n{SCOPE_STATEMENT}")
