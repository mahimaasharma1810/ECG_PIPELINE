"""Stages 1-6 end to end on ProRhythm device captures.

No filtering is applied anywhere on device data: the firmware field is
literally named `ecg_clean`, and a measured second chain left only 6-7% of
the amplitude (brief 3.2 trap 3). MITDB validation confirmed the same thing
independently - the unfiltered arm scored best at device resolution
(F1 0.9890 vs 0.9831).

Rhythm regularity indicator. Not a diagnosis. Not validated for clinical
use. Cannot distinguish atrial fibrillation from other causes of irregularity.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from .ingest import load_capture
from .resample import to_uniform_segments
from .features import window_features, BEATS_PER_WINDOW, INTERVALS_PER_WINDOW
from .sqi import assess
from .verdict import decide
from .axes import assess_rate, assess_events, combine_axes
from .axes.rate import rate_trustworthy
from .domain_gate import read_gate

warnings.filterwarnings("ignore")

BSQI_MIN = 0.95      # derived on MITDB, reports_rhythm/s05b_sqi_rr.csv
LAG1_MIN = -0.5
SDNN_GUARD_MS = 15.0  # below this, RR variance is at the quantisation floor and
                      # lag-1 alternation is an artefact of the 11.15 ms grid


DETECTOR_A = "neurokit"
DETECTOR_B = "rodrigues2021"   # chosen by agreement with A ON DEVICE DATA
                               # (reports_rhythm/s11_detector_b.csv). The
                               # MITDB-inherited pantompkins1985 finds ~half
                               # the beats on this signal, so bSQI measured
                               # detector failure rather than signal quality.


def detect_peaks(x: np.ndarray, fs: float):
    """Two independent detectors, no additional filtering (device path)."""
    import neurokit2 as nk
    _, ia = nk.ecg_peaks(x, sampling_rate=fs)
    _, ib = nk.ecg_peaks(x, sampling_rate=fs, method=DETECTOR_B)
    return (np.asarray(ia["ECG_R_Peaks"], dtype=np.int64),
            np.asarray(ib["ECG_R_Peaks"], dtype=np.int64))


def process_capture(path: str | Path, step_beats: int = BEATS_PER_WINDOW) -> list[dict]:
    """Run the full device path and return one row per 64-beat window."""
    path = Path(path)
    cap = load_capture(path)
    segs, n_replayed = to_uniform_segments(cap.t_ms, cap.amplitude)
    # clinical severity is gated; the axes still compute so their evidence is
    # visible as ENGINEERING output
    gate = read_gate()

    rows: list[dict] = []
    for si, seg in enumerate(segs):
        try:
            pa, pb = detect_peaks(seg.x, seg.fs)
        except Exception as exc:
            rows.append(dict(subject=cap.subject, file=path.name, seg=si,
                             error=str(exc)[:80]))
            continue
        if pa.size < BEATS_PER_WINDOW:
            continue

        for s in range(0, pa.size - BEATS_PER_WINDOW + 1, step_beats):
            w = pa[s:s + BEATS_PER_WINDOW]
            i0, i1 = int(w[0]), int(w[-1])
            fs_loc = seg.local_fs(i0, i1)
            if not np.isfinite(fs_loc) or fs_loc <= 0:
                continue
            # RR from index differences at the locally measured rate
            rr_ms = np.diff(w).astype(float) * 1000.0 / fs_loc
            feat = window_features(rr_ms)

            pb_w = pb[(pb >= i0) & (pb <= i1)]
            q = assess(seg.x[i0:i1 + 1], fs_loc,
                       w.astype(float) * 1000.0 / fs_loc,
                       pb_w.astype(float) * 1000.0 / fs_loc,
                       rr_ms=rr_ms,
                       bsqi_min=BSQI_MIN, lag1_min=LAG1_MIN)

            # variance guard: lag-1 is meaningless at the quantisation floor
            passed, reason = q.passed, q.reason
            if not passed and np.isfinite(feat.sdnn_ms) and feat.sdnn_ms <= SDNN_GUARD_MS:
                remaining = [r for r in reason.split("; ") if "lag-1" not in r]
                passed, reason = (not remaining), ("; ".join(remaining) or "ok")

            v = decide(feat, passed, reason)
            rate_ok, rate_why = rate_trustworthy(q, feat)
            ra = assess_rate(feat, rate_ok, rate_why)
            ev = assess_events(rr_ms, feat, passed)
            comb = combine_axes(ra, v, ev, gate.may_emit_clinical_severity)

            rows.append(dict(
                subject=cap.subject, file=path.name, seg=si,
                seg_dur_s=round(seg.duration_s, 1), seg_fs=round(seg.fs, 3),
                start_idx=i0, t0_ms=float(seg.t_raw_ms[i0]),
                fs_local=round(float(fs_loc), 3),
                drop_samples=round(seg.drop_samples(i0, i1), 1),
                n_replayed_in_file=n_replayed,
                **feat.as_dict(),
                bsqi=round(q.bsqi, 4), rr_lag1=round(q.rr_lag1, 4),
                ksqi=round(q.ksqi, 2), psqi=round(q.psqi, 4),
                flatline_frac=round(q.flatline_frac, 4),
                sat_frac=round(q.sat_frac, 4),
                sqi_passed=bool(passed), sqi_reason=reason,
                # --- axis 1: rate ---
                rate_state=ra.state, rate_extreme=ra.extreme, rate_reason=ra.reason,
                # --- axis 2: regularity ---
                verdict=v.verdict, verdict_actionable=v.actionable,
                verdict_reason=v.reason,
                # --- axis 3: events ---
                pause_count=ev.pause_count, longest_rr_ms=ev.longest_rr_ms,
                ectopic_couplets=ev.ectopic_couplets, ectopy_burden=ev.burden,
                events_reason=ev.reason,
                # --- combination ---
                severity=comb.severity, escalate=comb.escalate,
                severity_reason=comb.reason,
            ))
    return rows
