"""Do the new axes survive contact with real patch data?

The axes were derived and validated entirely on public labelled data. This
runs them on the 37 device captures to see which of them hold up and which
inherit the known detection error.
"""
import sys, warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm.ingest import load_capture
from rhythm.resample import to_uniform_segments
from rhythm.pipeline import detect_peaks, BSQI_MIN, LAG1_MIN, SDNN_GUARD_MS
from rhythm.features import window_features, BEATS_PER_WINDOW
from rhythm.sqi import assess
from rhythm.verdict import decide
from rhythm.axes import assess_rate, assess_events, combine_axes

warnings.filterwarnings("ignore")


def run(path):
    cap = load_capture(path)
    segs, _ = to_uniform_segments(cap.t_ms, cap.amplitude)
    rows = []
    for seg in segs:
        try:
            pa, pb = detect_peaks(seg.x, seg.fs)
        except Exception:
            continue
        if pa.size < BEATS_PER_WINDOW:
            continue
        for s in range(0, pa.size - BEATS_PER_WINDOW + 1, BEATS_PER_WINDOW):
            w = pa[s:s + BEATS_PER_WINDOW]; i0, i1 = int(w[0]), int(w[-1])
            fs_loc = seg.local_fs(i0, i1)
            if not np.isfinite(fs_loc) or fs_loc <= 0:
                continue
            rr = np.diff(w).astype(float) * 1000.0 / fs_loc
            f = window_features(rr)
            pbw = pb[(pb >= i0) & (pb <= i1)]
            q = assess(seg.x[i0:i1+1], fs_loc, w.astype(float)*1000.0/fs_loc,
                       pbw.astype(float)*1000.0/fs_loc, rr_ms=rr,
                       bsqi_min=BSQI_MIN, lag1_min=LAG1_MIN)
            passed = q.passed
            if not passed and np.isfinite(f.sdnn_ms) and f.sdnn_ms <= SDNN_GUARD_MS:
                passed = not [r for r in q.reason.split("; ") if "lag-1" not in r]
            v = decide(f, passed, q.reason)
            # run the axes with the gate OPEN, to see what they WOULD say
            ra = assess_rate(f, True)
            ev = assess_events(rr, f, True)
            c = combine_axes(ra, v, ev, clinical_permitted=True)
            rows.append(dict(subject=cap.subject, hr=f.hr_bpm, rr_cv=f.rr_cv,
                             sqi_passed=passed, verdict=v.verdict,
                             rate=ra.state, rate_extreme=ra.extreme,
                             pauses=ev.pause_count, couplets=ev.ectopic_couplets,
                             burden=ev.burden, severity=c.severity,
                             escalate=c.escalate))
    return rows


if __name__ == "__main__":
    files = [str(f) for f in sorted(Path("prorithm_ecg").glob("*/*.csv"))]
    with Pool(6) as pool:
        res = pool.map(run, files)
    x = pd.DataFrame([r for rr in res for r in rr])
    x.to_csv("reports_rhythm/s21_axes_device.csv", index=False)

    print(f"device windows: {len(x):,} from {x.subject.nunique()} subjects "
          f"(all HEALTHY adults)\n")
    print("=" * 70)
    print("AXIS 1 - RATE   (does it survive?)")
    print("=" * 70)
    print(x.rate.value_counts().to_string())
    print(f"\n  HR median {x.hr.median():.1f}  p05 {x.hr.quantile(.05):.1f}  "
          f"p95 {x.hr.quantile(.95):.1f}  max {x.hr.max():.1f}")
    print(f"  rate flagged extreme (<40 or >150): {x.rate_extreme.sum()} "
          f"({x.rate_extreme.mean():.1%})")

    print()
    print("=" * 70)
    print("AXIS 2 - REGULARITY   (known broken)")
    print("=" * 70)
    print(x.verdict.value_counts().to_string())
    print(f"\n  RR CV median {x.rr_cv.median():.4f}  "
          f"(healthy public data: 0.0305)")

    print()
    print("=" * 70)
    print("AXIS 3 - EVENTS   (the new one - does IT survive?)")
    print("=" * 70)
    print(x.burden.value_counts().to_string())
    print(f"\n  ectopic couplets per 64-beat window:")
    print(f"    median {x.couplets.median():.1f}  mean {x.couplets.mean():.2f}  "
          f"p95 {x.couplets.quantile(.95):.1f}  max {x.couplets.max()}")
    print(f"  windows with ANY couplet : {(x.couplets>0).mean():.1%}")
    print(f"  windows with >=5 couplets: {(x.couplets>=5).mean():.1%}")
    print(f"\n  pauses detected: {x.pauses.sum()} across {len(x):,} windows "
          f"({(x.pauses>0).mean():.2%} of windows)")

    print()
    print("=" * 70)
    print("COMBINED SEVERITY, if the gate were OPEN")
    print("=" * 70)
    for k, v in x.severity.value_counts().items():
        print(f"  {k:<10} {v:>6,} ({v/len(x):>5.1%})")
    print(f"  escalating to CRITICAL: {x.escalate.sum()} ({x.escalate.mean():.1%})")
