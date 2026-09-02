"""Survey the device vitals snapshots that pair with each ECG capture.

Establishes: pairing coverage, snapshot cadence, which columns carry real
signal vs constants, and what `hrv` actually is (candidate: mean RR in ms).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VIT = Path("data/raw/cliniaura_live/vitals")
ECG = Path("prorithm_ecg")

rows = []
for f in sorted(VIT.glob("*/*.csv")):
    df = pd.read_csv(f, parse_dates=["received_at_utc", "snapshot_timestamp"])
    subj = f.parent.name
    ecg = ECG / subj / f.name
    cad = df.snapshot_timestamp.diff().dt.total_seconds().dropna()
    hr = df.heart_rate.dropna()
    hrv = df.hrv.dropna()
    both = df[["heart_rate", "hrv"]].dropna()
    rows.append(dict(
        subject=subj, file=f.name, n_snap=len(df),
        ecg_paired=ecg.exists(),
        cad_med=round(cad.median(), 2) if len(cad) else np.nan,
        span_s=round((df.snapshot_timestamp.max() - df.snapshot_timestamp.min()).total_seconds(), 1),
        hr_med=round(hr.median(), 1) if len(hr) else np.nan,
        hr_min=hr.min() if len(hr) else np.nan, hr_max=hr.max() if len(hr) else np.nan,
        hrv_med=round(hrv.median(), 1) if len(hrv) else np.nan,
        # if hrv is mean RR in ms, 60000/hrv should track heart_rate
        hr_from_hrv_med=round((60000 / both.hrv).median(), 1) if len(both) else np.nan,
        corr=round(both.heart_rate.corr(60000 / both.hrv), 3) if len(both) > 2 else np.nan,
    ))

df = pd.DataFrame(rows)
df.to_csv("reports_rhythm/s02_vitals_survey.csv", index=False)
pd.set_option("display.width", 220, "display.max_columns", 30)
print(df.to_string(index=False))

print(f"\npaired with an ECG capture : {df.ecg_paired.sum()} / {len(df)}")
ecg_files = {f"{p.parent.name}/{p.name}" for p in ECG.glob("*/*.csv")}
vit_files = {f"{r.subject}/{r.file}" for r in df.itertuples()}
print(f"ECG captures with no vitals: {sorted(ecg_files - vit_files)}")
print(f"vitals with no ECG capture : {sorted(vit_files - ecg_files)}")
print(f"\nsnapshot cadence : median {df.cad_med.median():.2f} s "
      f"(range {df.cad_med.min():.2f}-{df.cad_med.max():.2f})")
print(f"device HR        : {df.hr_min.min():.0f}-{df.hr_max.max():.0f} bpm")
print(f"corr(heart_rate, 60000/hrv) : median {df['corr'].median():.3f}")

# which columns are constant across the whole corpus?
allv = pd.concat([pd.read_csv(f) for f in sorted(VIT.glob("*/*.csv"))], ignore_index=True)
print(f"\npooled snapshots: {len(allv):,}")
for c in ["heart_rate", "spo2", "systolic_bp", "diastolic_bp", "respiration_rate", "temperature", "hrv"]:
    s = allv[c].dropna()
    uniq = s.nunique()
    flag = "  <-- CONSTANT, not a measurement" if uniq <= 1 else ""
    print(f"  {c:<18} n={len(s):>5}  unique={uniq:>5}  "
          f"range [{s.min()}, {s.max()}]{flag}")
