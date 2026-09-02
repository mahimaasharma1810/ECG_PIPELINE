"""Step 1e / research Q2 - the RR-timing noise floor of this device.

For a PERFECTLY REGULAR rhythm, what CV and RMSSD does the timing mechanism
alone produce? That is the floor below which no threshold can be defended.

Two reconstructions of sample time are compared, using the REAL burst
structure of a real replay-free capture:

  (A) uniform 89.7 Hz grid   - samples assumed evenly spaced, grid anchored
                               to the stream (what the hardware actually does)
  (B) per-sample timestamps  - each R-peak takes its BLE packet's arrival
                               stamp (what brief Stage 2 instructs)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm.ingest import load_capture

REF = "prorithm_ecg/1789/2026-08-20_10.csv"
SEED = 20260827
rng = np.random.default_rng(SEED)

cap = load_capture(REF)
t = cap.t_ms
fs = cap.fs_effective
dt_grid = 1000.0 / fs

# real burst windows: start stamp of each burst, and the stamp carried by it
newburst = np.r_[True, np.diff(t) > 2]
burst_t = t[newburst]

print(f"reference capture : {REF}")
print(f"measured fs       : {fs:.2f} Hz  (grid step {dt_grid:.2f} ms)")
print(f"bursts            : {burst_t.size:,} over {cap.span_s:.0f} s")
print(f"\n{'RR(ms)':>7} {'HR':>5} | {'(A) uniform grid':^26} | {'(B) packet stamps':^26}")
print(f"{'':>7} {'':>5} | {'CV':>8} {'RMSSD(ms)':>10} {'SDNN':>6} | {'CV':>8} {'RMSSD(ms)':>10} {'SDNN':>6}")
print("-" * 78)

for rr in (600, 700, 800, 1000, 1200):
    t0 = float(t[0])
    n_beats = int((cap.span_s * 1000) // rr) - 1
    true_peaks = t0 + rr * np.arange(n_beats, dtype=float)

    # (A) snap to uniform grid anchored at t0
    idx = np.round((true_peaks - t0) / dt_grid)
    peaks_a = t0 + idx * dt_grid

    # (B) assign the arrival stamp of the burst each peak falls in
    j = np.searchsorted(burst_t, true_peaks, side="right") - 1
    j = np.clip(j, 0, burst_t.size - 1)
    peaks_b = burst_t[j]

    out = []
    for peaks in (peaks_a, peaks_b):
        rr_series = np.diff(peaks)
        rr_series = rr_series[rr_series > 0]
        cv = rr_series.std(ddof=1) / rr_series.mean()
        rmssd = np.sqrt(np.mean(np.diff(rr_series) ** 2))
        sdnn = rr_series.std(ddof=1)
        out.append((cv, rmssd, sdnn))
    (cva, rma, sda), (cvb, rmb, sdb) = out
    print(f"{rr:>7} {60000/rr:>5.0f} | {cva:>8.4f} {rma:>10.2f} {sda:>6.2f} |"
          f" {cvb:>8.4f} {rmb:>10.2f} {sdb:>6.2f}")

print("\nInterpretation")
print("  (A) is the floor set by 89.7 Hz sampling alone: RR quantised to "
      f"+/-{dt_grid/2:.1f} ms.")
print("  (B) is the floor if BLE packet-arrival stamps are treated as sample times.")
print("  Any regularity threshold must sit ABOVE the floor of whichever")
print("  reconstruction the pipeline actually uses.")
