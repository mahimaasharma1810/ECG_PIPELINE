"""Look at the actual device signal with detected R-peaks overlaid.

Required by brief 8 for every verdict, and used here to settle whether the
implausible RR variability is over-detection or genuine signal.
"""
import sys, warnings
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm.ingest import load_capture
from rhythm.resample import to_uniform_segments
import neurokit2 as nk
warnings.filterwarnings("ignore")

F = "prorithm_ecg/1789/2026-08-17_08.csv"
cap = load_capture(F)
segs, _ = to_uniform_segments(cap.t_ms, cap.amplitude)
seg = max(segs, key=lambda s: s.duration_s)
pk = np.asarray(nk.ecg_peaks(seg.x, sampling_rate=seg.fs)[1]["ECG_R_Peaks"], np.int64)
rr = np.diff(pk) * 1000.0 / seg.fs

# find a stretch containing several suspiciously short intervals
med = np.median(rr[(rr > 300) & (rr < 2000)])
short = np.flatnonzero(rr < 0.8 * med)
centre = short[len(short) // 2]
i0 = int(pk[max(centre - 6, 0)]); i1 = int(pk[min(centre + 8, pk.size - 1)])
pad = int(1.0 * seg.fs); i0, i1 = max(0, i0 - pad), min(seg.n - 1, i1 + pad)

fig, axes = plt.subplots(3, 1, figsize=(16, 9))
for ax, (a, b, title) in zip(axes, [
        (i0, i1, f"around a short RR interval (idx {i0}-{i1})"),
        (int(seg.fs * 60), int(seg.fs * 75), "arbitrary 15 s at t=60 s"),
        (int(seg.fs * 600), int(seg.fs * 615), "arbitrary 15 s at t=600 s")]):
    t = np.arange(a, b) / seg.fs
    ax.plot(t, seg.x[a:b], lw=0.8, color="#222")
    sel = pk[(pk >= a) & (pk < b)]
    ax.plot(sel / seg.fs, seg.x[sel], "v", ms=9, color="#d62728", label="detected R")
    for p, q in zip(sel[:-1], sel[1:]):
        ax.annotate(f"{(q-p)*1000/seg.fs:.0f}", ((p+q)/2/seg.fs, ax.get_ylim()[1]*0.85),
                    ha="center", fontsize=8, color="#1f77b4")
    ax.set_title(f"{Path(F).parent.name}/{Path(F).name} - {title}", fontsize=10)
    ax.set_ylabel("amplitude (units UNKNOWN)", fontsize=8)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=.25)
axes[-1].set_xlabel("time within segment (s)")
fig.suptitle("Rhythm regularity indicator. Not a diagnosis. Not validated for clinical use. "
             "Cannot distinguish atrial fibrillation from other causes of irregularity.",
             fontsize=8, y=0.005)
fig.tight_layout()
out = "reports_rhythm/s09_device_strip.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"wrote {out}")
print(f"segment fs={seg.fs:.2f}  peaks={pk.size:,}  median RR={med:.1f} ms")
print(f"short intervals (<0.8*median): {short.size} of {rr.size} ({short.size/rr.size:.1%})")
print(f"amplitude p5/p95 in segment: {np.percentile(seg.x,5):.1f} / {np.percentile(seg.x,95):.1f}")
