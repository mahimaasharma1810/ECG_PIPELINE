"""Step 4 - run Stages 1-6 on every ProRhythm capture and apply the SQI gate.

Produces the REGULAR-side distribution on real hardware, and - equally
important - the rate at which the device data is refused.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm import SCOPE_STATEMENT
from rhythm.pipeline import process_capture

if __name__ == "__main__":
    files = sorted(Path("prorithm_ecg").glob("*/*.csv"))
    with Pool(6) as pool:
        res = pool.map(process_capture, files)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv("reports_rhythm/s06_device_windows.csv", index=False)

    err = df[df.get("error").notna()] if "error" in df else df.iloc[:0]
    ok = df[df.sqi_passed.notna()] if "sqi_passed" in df else df
    print(f"captures processed : {len(files)}")
    print(f"windows produced   : {len(ok):,} (64 beats each)")
    if len(err):
        print(f"segment errors     : {len(err)}")
    print(f"subjects           : {sorted(ok.subject.unique())}")

    print(f"\n--- SQI GATE on device data ---")
    print(f"passed : {ok.sqi_passed.sum():,} / {len(ok):,} ({ok.sqi_passed.mean():.1%})")
    print(f"refused: {(~ok.sqi_passed).sum():,} ({(~ok.sqi_passed).mean():.1%})")
    from collections import Counter
    c = Counter()
    for r in ok.loc[~ok.sqi_passed, "sqi_reason"]:
        for part in str(r).split("; "):
            c[part.split(" ")[0] + " " + part.split(" ")[1] if len(part.split(" ")) > 1 else part] += 1
    print("  refusal reasons (component, may co-occur):")
    for k, v in c.most_common(8):
        print(f"    {k:<28} {v:>6,}")

    print(f"\n--- per subject ---")
    g = ok.groupby("subject").agg(
        windows=("sqi_passed", "size"), passed=("sqi_passed", "sum"),
        pass_rate=("sqi_passed", "mean"), bsqi_med=("bsqi", "median"))
    print(g.round(3).to_string())

    p = ok[ok.sqi_passed]
    print(f"\n--- REGULAR-side distribution (device, SQI-passed, n={len(p):,}) ---")
    for col in ("hr_bpm", "mean_rr_ms", "rr_cv", "rmssd_ms", "sdnn_ms"):
        s = p[col].dropna()
        print(f"  {col:<11} median {s.median():>8.4f}  "
              f"p05 {s.quantile(.05):>8.4f}  p95 {s.quantile(.95):>8.4f}  "
              f"p99 {s.quantile(.99):>8.4f}  max {s.max():>8.4f}")
    print(f"\nflagged-RR fraction: median {p.flagged_fraction.median():.3f}, "
          f"windows with any flagged interval: {(p.n_flagged>0).mean():.1%}")
    print(f"\nwrote reports_rhythm/s06_device_windows.csv\n\n{SCOPE_STATEMENT}")
