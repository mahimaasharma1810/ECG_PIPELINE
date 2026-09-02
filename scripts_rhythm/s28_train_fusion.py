"""BUILD_RHYTHM_MODEL.md step 6 - waveform + timing fusion, with an ABLATION.

THE QUESTION. Track B already reaches held-out-database AUC 0.9957 from RR
features alone. So the only question a waveform model has to answer is:

    does the WAVEFORM add anything the RR intervals do not already carry?

Three arms, identical splits, identical windows:
    rr    - RR features only        (reproduces Track B in a neural form)
    wave  - waveform only
    fusion- both

STOP CONDITION. If `fusion` does not beat `rr` on the HELD-OUT DATABASE, the
waveform adds nothing and the deterministic threshold plus Track B stands.
Report that plainly and stop - do not tune until it wins.

LEAD CAVEAT, stated because our own standard requires it. Our rule is that lead
verification is REQUIRED wherever the WAVEFORM is processed, and immaterial
where only annotation timestamps are used. LTAFDB headers name both channels
'ECG' and do not identify the lead. Track A and Track B used LTAFDB TIMESTAMPS
only, so they were unaffected. This model's `wave` branch processes LTAFDB
WAVEFORMS, so it inherits an UNVERIFIED lead. The `rr` arm does not, and is the
control: if the waveform adds nothing, the caveat is moot.

The device is confirmed Lead II (patch team, 2026-09-02) and MITDB is verified
MLII per record, so the TEST side is lead-verified.

Run with the GPU venv:
    /ssd_scratch/mahimakopalley/venv-gpu/bin/python scripts_rhythm/s28_train_fusion.py
"""
from __future__ import annotations

import sys, time, warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import TensorDataset, DataLoader

from rhythm import SCOPE_STATEMENT
from rhythm.fusion.dataset import build_record, WAVE_LEN
from rhythm.fusion.model import FusionNet
from rhythm.track_b.features import FEATURE_NAMES

SEED = 20260902
LTAF = "/ssd_scratch/mahimakopalley/data/raw/public/ltafdb"
MIT = "data/raw/public/mitdb"


def collect(dbpath, require_mlii, tag):
    recs = sorted(p.stem for p in Path(dbpath).glob("*.hea"))
    W, R, Y, G = [], [], [], []
    for r in recs:
        for w, rr, y, rec in build_record(dbpath, r, decimate=True,
                                          require_mlii=require_mlii):
            W.append(w); R.append(rr); Y.append(y); G.append(rec)
    print(f"  {tag}: {len(Y):,} windows from {len(set(G))} records "
          f"({np.mean(Y):.1%} irregular)")
    return (np.stack(W), np.stack(R), np.array(Y, np.float32), np.array(G))


def auc(score, y):
    import pandas as pd
    r = pd.Series(score).rank().to_numpy()
    n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def train_arm(name, tr, te, dev, epochs=30, bs=128):
    Wtr, Rtr, Ytr, _ = tr
    Wte, Rte, Yte, _ = te
    torch.manual_seed(SEED); np.random.seed(SEED)

    mu, sd = Rtr.mean(0), Rtr.std(0) + 1e-8
    Rtr_n, Rte_n = (Rtr - mu) / sd, (Rte - mu) / sd

    model = FusionNet(n_rr=len(FEATURE_NAMES),
                      use_wave=name in ("wave", "fusion"),
                      use_rr=name in ("rr", "fusion")).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = float((Ytr == 0).sum() / max((Ytr == 1).sum(), 1))
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos, device=dev))

    ds = TensorDataset(torch.from_numpy(Wtr), torch.from_numpy(Rtr_n),
                       torch.from_numpy(Ytr))
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for w, r, y in dl:
            w, r, y = w.to(dev), r.to(dev), y.to(dev)
            opt.zero_grad()
            logit, _ = model(w, r)
            lossf(logit, y).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        s = []
        for i in range(0, len(Yte), 512):
            w = torch.from_numpy(Wte[i:i+512]).to(dev)
            r = torch.from_numpy(Rte_n[i:i+512]).to(dev)
            lg, _ = model(w, r)
            s.append(torch.sigmoid(lg).cpu().numpy())
        s = np.concatenate(s)
    return auc(s, Yte), time.time() - t0, s


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    if dev.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}  "
              f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
    print(f"\nbuilding windows (device signal path: 133.83 Hz -> 2:3 decimation)")
    tr = collect(LTAF, False, "TRAIN LTAFDB (lead UNVERIFIED - see docstring)")
    te = collect(MIT, True, "TEST  MITDB  (lead-verified MLII)")

    np.save("/ssd_scratch/mahimakopalley/fusion_cache.npy",
            np.array([1], dtype=np.int8))    # marker only

    print(f"\n{'arm':<10} {'held-out MITDB AUC':>20} {'train s':>9}")
    print("-" * 42)
    res = {}
    for arm in ("rr", "wave", "fusion"):
        a, t, _ = train_arm(arm, tr, te, dev)
        res[arm] = a
        print(f"{arm:<10} {a:>20.4f} {t:>9.1f}")

    print(f"\nBASELINES on the same split:")
    print(f"  Track A threshold (RR CV >= 0.1275)   AUC 0.9511")
    print(f"  Track B gradient boosting, RR only    AUC 0.9957")
    print(f"\nSTOP CONDITION: does the waveform add anything over RR alone?")
    d = res["fusion"] - res["rr"]
    print(f"  fusion {res['fusion']:.4f} - rr {res['rr']:.4f} = {d:+.4f}")
    print(f"  -> waveform {'ADDS' if d > 0.002 else 'ADDS NOTHING'} over RR features")
    if res["fusion"] <= 0.9957:
        print(f"  -> fusion does NOT beat Track B (0.9957). The deterministic")
        print(f"     threshold plus Track B stands; a CNN is not justified.")
    print(f"\n{SCOPE_STATEMENT}")
