"""TASK 1 - L5 PVC morphology classifier, built fresh on public data.

Does NOT import the archived classifier. That scored V-class F1 0.830 on MITDB
and collapsed to 0.513 on SVDB, and predates the current held-out-database and
grouped-CV standards.

Evaluated at BOTH native resolution and the device's 89.70 Hz, because a model
that only works at 360 Hz can never transfer even once the device is fixed.
"""
from __future__ import annotations

import sys, warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np, pandas as pd, wfdb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm import SCOPE_STATEMENT
from rhythm.degrade import DEVICE_FS, resample_to
from rhythm.morphology.features import extract, build_template, FEATURE_NAMES

warnings.filterwarnings("ignore")
BEAT = set("NLRBAaJSVrFejnE/fQ?")
PVC = {"V", "E"}
MAX_BEATS = 500          # per record, keeps runtime sane


def per_record(args):
    dbpath, rec, degrade = args
    h = wfdb.rdheader(f"{dbpath}/{rec}")
    ch = h.sig_name.index("MLII") if "MLII" in h.sig_name else 0
    lead = h.sig_name[ch]
    sig = wfdb.rdrecord(f"{dbpath}/{rec}", channels=[ch]).p_signal[:, 0]
    fs = float(h.fs)
    ann = wfdb.rdann(f"{dbpath}/{rec}", "atr")
    keep = np.array([s in BEAT for s in ann.symbol])
    samp = ann.sample[keep]; sym = np.array(ann.symbol, dtype=object)[keep]

    if degrade:
        sig, fs_new = resample_to(sig, fs, DEVICE_FS)
        samp = np.round(samp * (fs_new / fs)).astype(int)
        fs = fs_new

    normal_idx = samp[sym == "N"]
    tmpl = build_template(sig, normal_idx, fs)
    if tmpl is None:
        return []

    if samp.size > MAX_BEATS:
        sel = np.linspace(0, samp.size - 1, MAX_BEATS).astype(int)
        samp, sym = samp[sel], sym[sel]

    rows = []
    for i, s in zip(samp, sym):
        if s not in PVC and s != "N":
            continue                        # binary task: PVC vs normal
        f = extract(sig, int(i), fs, tmpl)
        rows.append(dict(record=rec, lead=lead, y=int(s in PVC), **f))
    return rows


def build(dbpath, degrade):
    recs = sorted(p.stem for p in Path(dbpath).glob("*.hea"))
    with Pool(4) as pool:
        res = pool.map(per_record, [(dbpath, r, degrade) for r in recs])
    return pd.DataFrame([r for rr in res for r in rr]).dropna()


def evaluate(tr, te, tag):
    X, y, g = tr[list(FEATURE_NAMES)].to_numpy(), tr.y.to_numpy(), tr.record.to_numpy()
    Xt, yt = te[list(FEATURE_NAMES)].to_numpy(), te.y.to_numpy()
    m = GradientBoostingClassifier(random_state=20260830, max_depth=3, n_estimators=200)

    cv = []
    gkf = GroupKFold(n_splits=5)
    for a, b in gkf.split(X, y, groups=g):
        m.fit(X[a], y[a])
        p = m.predict(X[b])
        tp = int(((p == 1) & (y[b] == 1)).sum()); fp = int(((p == 1) & (y[b] == 0)).sum())
        fn = int(((p == 0) & (y[b] == 1)).sum())
        se = tp/(tp+fn) if tp+fn else 0; pv = tp/(tp+fp) if tp+fp else 0
        cv.append(2*se*pv/(se+pv) if se+pv else 0)

    m.fit(X, y)
    p = m.predict(Xt)
    TP = int(((p == 1) & (yt == 1)).sum()); FP = int(((p == 1) & (yt == 0)).sum())
    FN = int(((p == 0) & (yt == 1)).sum()); TN = int(((p == 0) & (yt == 0)).sum())
    se = TP/(TP+FN) if TP+FN else np.nan
    pv = TP/(TP+FP) if TP+FP else np.nan
    f1 = 2*se*pv/(se+pv) if (se and pv) else np.nan
    print(f"\n  --- {tag} ---")
    print(f"    train {len(tr):,} beats / {tr.record.nunique()} records "
          f"({tr.y.mean():.1%} PVC)   test {len(te):,} beats / "
          f"{te.record.nunique()} records ({te.y.mean():.1%} PVC)")
    print(f"    grouped-CV F1 by patient (train db): {np.mean(cv):.4f} "
          f"(+/- {np.std(cv):.4f})")
    print(f"    HELD-OUT DATABASE   TP={TP:,}  FP={FP:,}  FN={FN:,}  TN={TN:,}")
    print(f"    HELD-OUT DATABASE   Se {se:.4f}   PPV {pv:.4f}   F1 {f1:.4f}")
    return f1, np.mean(cv)


if __name__ == "__main__":
    print("=" * 74)
    print("TASK 1 - L5 PVC morphology, held-out DATABASE")
    print("=" * 74)
    print("PVC (V-class) only. PACs are NOT targeted: a PAC conducts normally so")
    print("its QRS is near-identical to a sinus beat, and the discriminator is the")
    print("P wave - small, often buried in the preceding T, unreliable single-lead.")
    print("The prior attempt scored S-class F1 0.152 for this reason.\n")

    results = {}
    import os
    arms = ([(False, "native resolution")] if os.environ.get("ARM") == "native"
            else [(True, "degraded to 89.70 Hz")] if os.environ.get("ARM") == "degraded"
            else [(False, "native resolution"), (True, "degraded to 89.70 Hz")])
    for degrade, label in arms:
        print(f"\n{'=' * 74}\n{label.upper()}\n{'=' * 74}")
        mit = build("data/raw/public/mitdb", degrade)
        svd = build("data/raw/public/svdb", degrade)
        results[("mit->svdb", degrade)] = evaluate(mit, svd, f"train MITDB -> test SVDB ({label})")
        results[("svdb->mit", degrade)] = evaluate(svd, mit, f"train SVDB -> test MITDB ({label})")
        if not degrade:
            mit.to_csv("reports_rhythm/s24_pvc_mitdb.csv", index=False)
            svd.to_csv("reports_rhythm/s24_pvc_svdb.csv", index=False)

    print(f"\n{'=' * 74}\nVERSUS THE ARCHIVED BASELINE\n{'=' * 74}")
    print(f"  archived classifier: V-class F1 0.830 on MITDB, 0.513 held-out on SVDB")
    for (direction, deg), (f1, cv) in results.items():
        res = "89.7 Hz" if deg else "native"
        verdict = "BEATS" if f1 > 0.513 else "DOES NOT BEAT"
        print(f"  {direction:<12} {res:<8} held-out F1 {f1:.4f}   {verdict} 0.513")
    print(f"\n{SCOPE_STATEMENT}")
