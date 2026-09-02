"""Step 1b - why per-file fs varies 51-167 Hz.

Two candidate explanations, both testable:
  (a) dropout gaps inflate `span`, so n/span understates the true rate
  (b) timestamps are non-monotonic, so the stream is not a clean sequence

Measures fs over CONTINUOUS segments only (dt <= GAP_S), which is the rate
that actually matters for resampling, and quantifies the backward steps.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm.ingest import load_capture

GAP_S = 1.0   # a dt above this is a dropout, not a sample interval
OUT = Path("reports_rhythm/s01b_segment_timing.csv")

rows = []
for f in sorted(Path("prorithm_ecg").glob("*/*.csv")):
    cap = load_capture(f)
    t = cap.t_ms

    # --- (b) non-monotonicity, measured on the AS-RECORDED order ---
    raw = pd.read_csv(f, usecols=["timestamp_ms"])["timestamp_ms"].to_numpy(np.float64)
    raw = raw[np.isfinite(raw)]
    d_raw = np.diff(raw)
    back = d_raw[d_raw < 0]

    # --- (a) segment-aware fs, on the sorted stream ---
    dt = np.diff(t) / 1000.0
    brk = np.flatnonzero(dt > GAP_S)
    starts = np.r_[0, brk + 1]
    ends = np.r_[brk, t.size - 1]
    seg_dur = (t[ends] - t[starts]) / 1000.0
    seg_n = ends - starts + 1
    keep = seg_dur > 0
    active_s = float(seg_dur[keep].sum())
    fs_active = float(seg_n[keep].sum() / active_s) if active_s > 0 else np.nan

    rows.append(dict(
        subject=cap.subject, file=Path(cap.path).name,
        n=cap.n_samples, span_s=round(cap.span_s, 1),
        fs_span=round(cap.fs_effective, 2),
        active_s=round(active_s, 1),
        dropout_s=round(cap.span_s - active_s, 1),
        fs_active=round(fs_active, 2),
        n_segments=int(keep.sum()),
        longest_seg_s=round(float(seg_dur[keep].max()) if keep.any() else 0.0, 1),
        dup_frac=round(cap.duplicate_fraction, 3),
        n_back=int(back.size),
        back_frac=round(float(back.size / d_raw.size), 4),
        back_med_ms=round(float(np.median(-back)), 1) if back.size else 0.0,
        back_max_ms=round(float(-back.min()), 1) if back.size else 0.0,
    ))

df = pd.DataFrame(rows)
OUT.parent.mkdir(exist_ok=True)
df.to_csv(OUT, index=False)
pd.set_option("display.width", 200, "display.max_columns", 30)
print(df.to_string(index=False))
print("\n--- pooled ---")
print(f"active recording time : {df.active_s.sum():,.0f} s "
      f"({df.active_s.sum()/3600:.1f} h) of {df.span_s.sum():,.0f} s span")
print(f"dropout time          : {df.dropout_s.sum():,.0f} s "
      f"({df.dropout_s.sum()/df.span_s.sum():.1%} of span)")
print(f"fs_active   : median {df.fs_active.median():.2f} Hz, "
      f"IQR [{df.fs_active.quantile(.25):.2f}, {df.fs_active.quantile(.75):.2f}], "
      f"range [{df.fs_active.min():.2f}, {df.fs_active.max():.2f}]")
print(f"fs_span     : median {df.fs_span.median():.2f} Hz  (gap-inflated)")
print(f"backward steps: {df.n_back.sum():,} total, "
      f"median per-file frac {df.back_frac.median():.2%}, "
      f"max magnitude {df.back_max_ms.max():,.0f} ms")
print(f"\nwrote {OUT}")
