"""Track B - supervised model, trained and tested on PUBLIC data only.

Answers one question: does a model beat the single Track A threshold on a
HELD-OUT DATABASE?

RULES OBSERVED (from CLASSIFIER_DESIGN.md):
  - Track B NEVER sets Track A's threshold. It challenges it.
  - Trained on LTAFDB, tested on MITDB - held-out DATABASE, not record.
  - No heart-rate feature. Every feature is rate-normalised by construction.
  - NOT applied to device data: the domain gate blocks that, and device beat
    detection is known wrong by ~4.4%.
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path

import numpy as np, pandas as pd, wfdb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm import SCOPE_STATEMENT
from rhythm.degrade import DEVICE_FS
from rhythm.features import window_features, BEATS_PER_WINDOW
from rhythm.verdict import THRESHOLD_CV
from rhythm.track_b.features import extract, FEATURE_NAMES

warnings.filterwarnings("ignore")
G = 1000.0 / DEVICE_FS
BS = set("NLRBAaJSVrFejnE/fQ?")
SEED = 20260830


def clean(a): return (a or "").replace("\x00", "").strip()


def build(dbpath, require_mlii=False):
    rows = []
    for r in sorted(p.stem for p in Path(dbpath).glob("*.hea")):
        h = wfdb.rdheader(f"{dbpath}/{r}")
        if require_mlii and "MLII" not in h.sig_name:
            continue
        ann = wfdb.rdann(f"{dbpath}/{r}", "atr")
        keep = np.array([s in BS for s in ann.symbol])
        bs = ann.sample[keep]
        marks = [(s, clean(a)) for s, a in zip(ann.sample, ann.aux_note)
                 if clean(a).startswith("(")]
        if not marks:
            continue
        ms = np.array([m[0] for m in marks]); lb = np.array([m[1] for m in marks], dtype=object)
        i = np.searchsorted(ms, bs, side="right") - 1
        rl = np.where(i >= 0, lb[np.clip(i, 0, None)], "")
        t = np.round((bs / float(h.fs) * 1000.0) / G) * G
        for s in range(0, bs.size - BEATS_PER_WINDOW + 1, BEATS_PER_WINDOW):
            u = set(rl[s:s + BEATS_PER_WINDOW])
            if len(u) != 1:
                continue
            lab = {"(AFIB": 1, "(N": 0}.get(u.pop())
            if lab is None:
                continue
            rr = np.diff(t[s:s + BEATS_PER_WINDOW])
            f = window_features(rr)
            if not f.usable:
                continue
            rows.append(dict(record=r, y=lab, **extract(rr)))
    return pd.DataFrame(rows).dropna()


def metrics(y, score, thr):
    p = score >= thr
    TP = int(((p == 1) & (y == 1)).sum()); FN = int(((p == 0) & (y == 1)).sum())
    TN = int(((p == 0) & (y == 0)).sum()); FP = int(((p == 1) & (y == 0)).sum())
    return dict(TP=TP, FN=FN, TN=TN, FP=FP,
                Se=TP/(TP+FN) if TP+FN else np.nan,
                Sp=TN/(TN+FP) if TN+FP else np.nan,
                PPV=TP/(TP+FP) if TP+FP else np.nan,
                NPV=TN/(TN+FN) if TN+FN else np.nan)


def auc(score, y):
    r = pd.Series(score).rank().to_numpy(); n1 = int(y.sum()); n0 = len(y)-n1
    return (r[y == 1].sum() - n1*(n1+1)/2) / (n1*n0)


LTAF = "/ssd_scratch/mahimakopalley/data/raw/public/ltafdb"
print("building windows...")
tr = build(LTAF)
te = build("data/raw/public/mitdb", require_mlii=True)
tr.to_csv("reports_rhythm/s22_trackb_train.csv", index=False)
te.to_csv("reports_rhythm/s22_trackb_test.csv", index=False)
print(f"  TRAIN LTAFDB : {len(tr):,} windows, {tr.record.nunique()} records, "
      f"{tr.y.sum()} irregular ({tr.y.mean():.1%})")
print(f"  TEST  MITDB  : {len(te):,} windows, {te.record.nunique()} records, "
      f"{te.y.sum()} irregular ({te.y.mean():.1%})  <- HELD-OUT DATABASE")

Xtr, ytr = tr[list(FEATURE_NAMES)].to_numpy(), tr.y.to_numpy()
Xte, yte = te[list(FEATURE_NAMES)].to_numpy(), te.y.to_numpy()

print(f"\n=== TRACK A (single threshold, RR CV >= {THRESHOLD_CV}) ===")
a_tr = metrics(ytr, tr.rr_cv.to_numpy(), THRESHOLD_CV)
a_te = metrics(yte, te.rr_cv.to_numpy(), THRESHOLD_CV)
print(f"  LTAFDB : Se {a_tr['Se']:.4f}  Sp {a_tr['Sp']:.4f}  PPV {a_tr['PPV']:.4f}  NPV {a_tr['NPV']:.4f}")
print(f"  MITDB  : Se {a_te['Se']:.4f}  Sp {a_te['Sp']:.4f}  PPV {a_te['PPV']:.4f}  NPV {a_te['NPV']:.4f}")
print(f"  held-out AUC: {auc(te.rr_cv.to_numpy(), yte):.4f}")

print(f"\n=== TRACK B (models, no heart-rate feature) ===")
models = {
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    "gradient boosting": GradientBoostingClassifier(random_state=SEED, max_depth=3,
                                                    n_estimators=200),
}
results = {}
for name, m in models.items():
    # record-grouped CV on the training database, to check it is not memorising patients
    gkf = GroupKFold(n_splits=min(5, tr.record.nunique()))
    cv = []
    for trn, val in gkf.split(Xtr, ytr, groups=tr.record):
        m.fit(Xtr[trn], ytr[trn])
        cv.append(auc(m.predict_proba(Xtr[val])[:, 1], ytr[val]))
    m.fit(Xtr, ytr)
    s_te = m.predict_proba(Xte)[:, 1]
    results[name] = s_te
    print(f"\n  {name}")
    print(f"    grouped CV AUC on LTAFDB : {np.mean(cv):.4f} (+/- {np.std(cv):.4f})")
    print(f"    HELD-OUT MITDB AUC       : {auc(s_te, yte):.4f}")
    # compare at MATCHED specificity to Track A on the held-out set
    thr = np.quantile(s_te[yte == 0], a_te["Sp"])
    b = metrics(yte, s_te, thr)
    print(f"    at Track A's specificity ({a_te['Sp']:.4f}):")
    print(f"      Se {b['Se']:.4f} (Track A {a_te['Se']:.4f})  "
          f"PPV {b['PPV']:.4f} (Track A {a_te['PPV']:.4f})  "
          f"NPV {b['NPV']:.4f} (Track A {a_te['NPV']:.4f})")
    delta = b["Se"] - a_te["Se"]
    print(f"      sensitivity change vs Track A: {delta:+.4f}")

print(f"\n=== DISAGREEMENT (a finding, not a number to average) ===")
best = max(results, key=lambda k: auc(results[k], yte))
s = results[best]
thr = np.quantile(s[yte == 0], a_te["Sp"])
a_pred = te.rr_cv.to_numpy() >= THRESHOLD_CV
b_pred = s >= thr
dis = a_pred != b_pred
print(f"  best model: {best}")
print(f"  windows where Track A and Track B disagree: {dis.sum()} of {len(te)} ({dis.mean():.1%})")
if dis.sum():
    print(f"    A says irregular, B says regular: {int((a_pred & ~b_pred).sum())} "
          f"(truly irregular in {int(yte[a_pred & ~b_pred].sum())} of them)")
    print(f"    B says irregular, A says regular: {int((~a_pred & b_pred).sum())} "
          f"(truly irregular in {int(yte[~a_pred & b_pred].sum())} of them)")
print(f"\n{SCOPE_STATEMENT}")
