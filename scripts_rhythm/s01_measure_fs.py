"""Step 1 - ingest every ProRhythm capture and measure its effective fs.

Independently confirms (or refutes) the timing facts asserted in the brief:
  nominal 100 Hz, measured ~89.7 Hz, ~47% duplicate timestamps.

Writes reports_rhythm/s01_capture_timing.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm import SCOPE_STATEMENT
from rhythm.ingest import load_capture, FormatError

DATA = Path("prorithm_ecg")
OUT = Path("reports_rhythm/s01_capture_timing.csv")


def main() -> None:
    rows, failures = [], []
    files = sorted(DATA.glob("*/*.csv"))
    print(f"scanning {len(files)} capture files under {DATA}/\n")

    for f in files:
        try:
            cap = load_capture(f)
        except (FormatError, ValueError) as exc:
            failures.append((str(f), str(exc)))
            continue
        rows.append(cap.timing_summary())
        print(
            f"{cap.subject}/{Path(cap.path).name:<22} "
            f"n={cap.n_samples:>8,}  span={cap.span_s:>8.1f}s  "
            f"fs_eff={cap.fs_effective:6.2f} Hz  "
            f"fs_median_dt={cap.fs_median_dt:>8.1f} Hz  "
            f"dup={cap.duplicate_fraction:5.1%}  "
            f"amp=[{cap.amp_min:8.2f},{cap.amp_max:8.2f}]  "
            f"maxgap={cap.max_gap_s:6.2f}s"
        )

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"\n--- aggregate over {len(df)} captures ---")
    tot_n, tot_span = df.n_samples.sum(), df.span_s.sum()
    print(f"pooled samples          : {tot_n:,}")
    print(f"pooled span             : {tot_span:,.0f} s ({tot_span/3600:.1f} h)")
    print(f"pooled effective fs     : {tot_n/tot_span:.2f} Hz")
    print(f"per-file fs_effective   : median {df.fs_effective.median():.2f} Hz, "
          f"range [{df.fs_effective.min():.2f}, {df.fs_effective.max():.2f}]")
    print(f"per-file duplicate frac : median {df.duplicate_fraction.median():.1%}, "
          f"range [{df.duplicate_fraction.min():.1%}, {df.duplicate_fraction.max():.1%}]")
    print(f"fs from median(dt)      : median {df.fs_median_dt.median():.1f} Hz  <- MISLEADING")
    print(f"amplitude range (pooled): [{df.amp_min.min():.2f}, {df.amp_max.max():.2f}] (units UNKNOWN)")
    print(f"non-monotonic files     : {(~df.monotonic).sum()} / {len(df)}")
    print(f"max gap across captures : {df.max_gap_s.max():.1f} s")
    if failures:
        print(f"\nfailed to parse ({len(failures)}):")
        for p, e in failures:
            print(f"  {p}: {e}")
    print(f"\nwrote {OUT}")
    print(f"\n{SCOPE_STATEMENT}")


if __name__ == "__main__":
    main()
