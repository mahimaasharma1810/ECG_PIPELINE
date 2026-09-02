"""Step 3 - cross-check pipeline RR against device-reported RR.

Compares MEDIAN RR DURATIONS per capture, which sidesteps clock alignment
entirely and needs no clinician. The device's `hrv` field is mean RR in ms
(verified: 60000/hrv tracks heart_rate at median r=0.82).

This is the only independent check available on device data, because the
ProRhythm captures carry no rhythm labels. It cannot validate rhythm - but it
can catch a detector that is counting the wrong number of beats, which is the
failure that would make every downstream number meaningless.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VIT = Path("data/raw/cliniaura_live/vitals")
w = pd.read_csv("reports_rhythm/s06_device_windows.csv")
w = w[w.mean_rr_ms.notna()]

rows = []
for (subj, f), g in w.groupby(["subject", "file"]):
    vp = VIT / str(subj) / f
    if not vp.exists():
        continue
    v = pd.read_csv(vp)
    dev_rr = v.hrv.dropna()
    dev_rr = dev_rr[(dev_rr > 300) & (dev_rr < 2000)]
    if dev_rr.empty:
        continue
    for label, sub in (("all", g), ("sqi_passed", g[g.sqi_passed == True])):
        if sub.empty:
            continue
        pipe_rr = sub.mean_rr_ms.median()
        rows.append(dict(
            subject=subj, file=f, arm=label, n_win=len(sub),
            pipe_rr_ms=round(pipe_rr, 1),
            dev_rr_ms=round(dev_rr.median(), 1),
            dev_hr=round(v.heart_rate.median(), 1),
            pipe_hr=round(60000 / pipe_rr, 1),
            rr_ratio=round(pipe_rr / dev_rr.median(), 4),
            beat_excess_pct=round((dev_rr.median() / pipe_rr - 1) * 100, 1),
        ))

df = pd.DataFrame(rows)
df.to_csv("reports_rhythm/s07_hr_crosscheck.csv", index=False)
pd.set_option("display.width", 200)

for arm in ("all", "sqi_passed"):
    a = df[df.arm == arm]
    if a.empty:
        continue
    print(f"\n=== arm: {arm}  ({len(a)} captures) ===")
    print(f"  pipeline median RR / device median RR : "
          f"median {a.rr_ratio.median():.4f}  IQR [{a.rr_ratio.quantile(.25):.4f}, "
          f"{a.rr_ratio.quantile(.75):.4f}]")
    print(f"  implied EXCESS beats detected         : "
          f"median {a.beat_excess_pct.median():+.1f}%  "
          f"range [{a.beat_excess_pct.min():+.1f}%, {a.beat_excess_pct.max():+.1f}%]")
    print(f"  captures where pipeline HR > device HR : "
          f"{(a.pipe_hr > a.dev_hr).sum()} / {len(a)}")
    print(f"  |HR difference| > 5 bpm               : "
          f"{(abs(a.pipe_hr - a.dev_hr) > 5).sum()} / {len(a)}")

print("\n--- per capture (all windows) ---")
print(df[df.arm == "all"][["subject", "file", "n_win", "pipe_rr_ms", "dev_rr_ms",
                           "pipe_hr", "dev_hr", "beat_excess_pct"]]
      .to_string(index=False))
