"""ECG pipeline — new methodology implementation, consolidated.

This single file merges what used to be 15 separate modules
(config.py, audit.py, ingest.py, quality.py, resample.py, filters.py,
beats.py, features.py, encoder.py, classify.py, risk.py,
similar_cases.py, report.py, pipeline.py, run_pipeline.py) into one
runnable script, per request. NOTHING was removed — every function,
class, constant, and docstring below is verbatim from its original
file; only the intra-package `from .xxx import yyy` lines were dropped
(everything now shares one module namespace) and a handful of
`module.function(...)` call sites were flattened to `function(...)`
accordingly. Training/download tooling (download_datasets.py,
download_icentia11k_full.py, train_encoder.py, train_classifiers.py,
eval_classifier.py, splits.py) lives in the sibling file
`ecg_pipeline_tools.py`, which imports from this one.

The original per-stage files are preserved unchanged under
`_original_stages/` for reference/diffing.

See README.md for what changed vs. the existing baseline and why (each
change traces back to a ranked recommendation in the internal review
PDF).

Stage map (same 9 stages as the baseline):
  1. Ingest       — VitalPatch / SeNSiO / WFDB parsers
  2. Quality      — SQI gate (recommendation #6 fix)
  3. Resample     — uniform 125 Hz
  4. Filters      — 5-step filter chain + per-beat robust Z-score
  5. Beats        — XQRS R-peak detection + segmentation
  6. Features     — 56-dim handcrafted + Encoder (learned embedding, #1/#2)
  7. Classify     — beat + rhythm classification (recommendation #7)
  8. Risk         — risk scoring + conformal prediction + temporal tracking (#4/#5)
  9. Report       — MedGemma prompt/merge (recommendations #3/#10)

Run it:
    python -m ecg_pipeline.ecg_pipeline_core --source vitalpatch --limit 3
    python -m ecg_pipeline.ecg_pipeline_core --source sensio --limit 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pywt
import requests
import torch
from scipy import stats
from scipy.signal import butter, decimate, filtfilt, iirnotch, lfilter, medfilt
from torch import nn


# ============================================================================
# config.py — Shared constants for the ECG pipeline (new methodology)
# ============================================================================
"""Shared constants for the ECG pipeline (new methodology).

Every threshold here is either carried over from the existing baseline
(documented in PROJECT_OVERVIEW.md / the architecture PDF) or introduced by
one of the "Top Recommendations" in the internal review PDF. Each new
constant says which recommendation it implements.
"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

TARGET_FS = 125.0  # Hz, common resample target (VitalPatch native rate)

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


@dataclass
class SQIThresholds:
    """Stage 2 pre-filter Signal Quality Index gate.

    Recommendation #6 ("Fix the signal-quality check so it doesn't reject
    real AFib patients"): the baseline gate used a single "regularity"
    style kurtosis check that conflated *rhythm* irregularity (real AFib,
    frequent ectopy) with *noise* irregularity (motion artefact). Here the
    kurtosis/impulse check is scoped to *within-beat* QRS morphology only,
    never to RR-interval regularity, so an irregular-but-clean AFib strip
    no longer fails the gate. Window-level irregular RR is instead handed
    downstream to the rhythm classifier, not rejected here.
    """

    window_seconds: float = 5.0
    flatline_run_ms: float = 400.0  # a "stuck sensor" run of >=400ms identical samples, not scattered quantization repeats
    flatline_frac_max: float = 0.05
    clipping_run_ms: float = 200.0  # a railed run of >=200ms sitting at the window's own min/max
    clipping_frac_max: float = 0.02
    missing_frac_max: float = 0.03
    kurtosis_min: float = 1.5  # QRS impulse character, computed per-beat-window, NOT RR-interval based
    # Real device recordings arrive in arbitrary firmware-scaled raw ADC counts with no published
    # mV-per-count constant (VitalPatch and SeNSiO both do this). Absolute-mV thresholds would be
    # meaningless without that calibration constant, so baseline wander is expressed as a ratio of
    # the window's own robust dynamic range (IQR) instead of a hard mV number.
    baseline_wander_ratio_max: float = 1.2
    snr_db_min: float = 5.0
    adc_clip_value: float | None = None  # only set for a device whose true ADC full-scale is known


@dataclass
class FilterChainConfig:
    median_baseline_window_ms: float = 200.0
    highpass_hz: float = 0.5
    notch_hz: float = 50.0
    notch_q: float = 30.0
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 40.0
    bandpass_order: int = 4


@dataclass
class BeatWindowConfig:
    primary_pre_ms: float = 200.0
    primary_post_ms: float = 400.0
    wide_pre_ms: float = 500.0
    wide_post_ms: float = 500.0
    rr_min_ms: float = 300.0
    rr_max_ms: float = 2000.0
    # Amplitude floor expressed relative to the window's own sample-to-sample noise floor
    # (median absolute deviation of the first difference), not an absolute mV number — see
    # SQIThresholds.baseline_wander_ratio_max for why (unknown per-device ADC scale).
    beat_amplitude_min_noise_ratio: float = 3.0


@dataclass
class RiskThresholds:
    pvc_burden_high_pct: float = 10.0
    pvc_burden_critical_pct: float = 20.0
    pac_burden_high_pct: float = 15.0
    vt_run_beats: int = 3  # >=3 consecutive V beats = VT run
    afib_burden_high_pct: float = 30.0
    hrv_sdnn_suppressed_ms: float = 20.0  # conservative low-HRV cutoff
    news2_critical_threshold: int = 7
    qsofa_high_threshold: int = 2


@dataclass
class ConformalConfig:
    """Recommendation #4: statistically guaranteed confidence ranges.

    Split-conformal prediction over the 4 risk levels: given a calibration
    set of (softmax scores, true label), compute the (1 - alpha) quantile
    of nonconformity scores, then at inference time return the *set* of
    risk levels whose score falls within that quantile, instead of a bare
    single label. Coverage is guaranteed marginally, not per-class.
    """

    alpha: float = 0.1  # -> 90% coverage guarantee
    min_calibration_size: int = 30


@dataclass
class TemporalTrackingConfig:
    """Recommendation #5: track risk over time, not just one snapshot."""

    window_minutes: float = 15.0
    trend_slope_alert_threshold: float = 0.15  # risk-score units per minute
    history_max_windows: int = 500


SQI = SQIThresholds()
FILTER = FilterChainConfig()
BEATS = BeatWindowConfig()
RISK = RiskThresholds()
CONFORMAL = ConformalConfig()
TEMPORAL = TemporalTrackingConfig()


# ============================================================================
# audit.py — SHA-256 hash-chained audit log
# ============================================================================
"""SHA-256 hash-chained audit log.

Every pipeline decision — SQI window rejections, beat quality rejections,
classifier source (trained model vs. rule-based fallback), risk alerts,
and MedGemma accept/reject — is appended here so the whole run is
auditable end to end, per the architecture doc's Stage 9 description.
"""


@dataclass
class AuditEntry:
    seq: int
    timestamp: float
    event_type: str
    payload: dict
    prev_hash: str
    hash: str


class AuditLog:
    GENESIS_HASH = "0" * 64

    def __init__(self):
        self._entries: list[AuditEntry] = []

    def append(self, event_type: str, payload: dict) -> AuditEntry:
        prev_hash = self._entries[-1].hash if self._entries else self.GENESIS_HASH
        seq = len(self._entries)
        timestamp = time.time()
        body = json.dumps({"seq": seq, "timestamp": timestamp, "event_type": event_type,
                            "payload": payload, "prev_hash": prev_hash}, sort_keys=True, default=str)
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        entry = AuditEntry(seq, timestamp, event_type, payload, prev_hash, digest)
        self._entries.append(entry)
        return entry

    def verify_chain(self) -> bool:
        prev = self.GENESIS_HASH
        for e in self._entries:
            body = json.dumps({"seq": e.seq, "timestamp": e.timestamp, "event_type": e.event_type,
                                "payload": e.payload, "prev_hash": prev}, sort_keys=True, default=str)
            if hashlib.sha256(body.encode("utf-8")).hexdigest() != e.hash or e.prev_hash != prev:
                return False
            prev = e.hash
        return True

    def to_list(self) -> list[dict]:
        return [vars(e) for e in self._entries]

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_list(), indent=2, default=str))


# ============================================================================
# ingest.py — Stage 1: Parse and reconstruct raw device streams
# ============================================================================
"""Stage 1 — Parse and reconstruct raw device streams into a common format.

Each source gets its own parser because the raw formats are unrelated.
Every parser returns a `Recording`: a uniform-enough container the rest of
the pipeline (stages 2+) can consume regardless of which device produced it.
"""


@dataclass
class Recording:
    signal_mv: np.ndarray          # 1-D raw signal, arbitrary units at this stage
    timestamps_ms: np.ndarray      # 1-D, same length as signal, epoch ms (or synthetic)
    fs_nominal: float              # nominal sample rate in Hz
    source: str                   # "vitalpatch" | "sensio" | "wfdb"
    patient_id: str
    segment_id: str
    gaps: list = field(default_factory=list)          # list of (start_idx, end_idx, gap_ms) flagged, not dropped
    already_bandpass_filtered: bool = False             # SeNSiO pre-filtered variant
    meta: dict = field(default_factory=dict)


def _split_at_gaps(timestamps_ms: np.ndarray, values: np.ndarray, nominal_step_ms: float,
                    gap_flag_ratio: float = 1.5, gap_split_ms: float = 200.0):
    """Flag gaps > gap_flag_ratio * nominal step; split into segments at gaps > gap_split_ms.

    Small gaps are left in place for stage 3 (resample) to interpolate;
    large gaps become separate segments so filtering never bridges a real
    dropout, per the architecture doc's Stage 1 description.
    """
    deltas = np.diff(timestamps_ms)
    gap_flag_ms = nominal_step_ms * gap_flag_ratio
    flagged = np.where(deltas > gap_flag_ms)[0]
    gaps = [(int(i), int(i + 1), float(deltas[i])) for i in flagged]

    split_points = np.where(deltas > gap_split_ms)[0] + 1
    segments = np.split(np.arange(len(timestamps_ms)), split_points)
    return gaps, segments


def parse_vitalpatch_ecg(csv_path: Path, fs_nominal: float = 125.0) -> list[Recording]:
    """VitalPatch: alternating timestamp/value CSV pairs, nominal 8 ms step.

    One file == one recording chunk (already segmented upstream by the
    collector script); a file can still contain internal gaps, which are
    flagged and used to split into sub-segments here.
    """
    csv_path = Path(csv_path)
    raw = pd.read_csv(csv_path, header=None)
    flat = raw.to_numpy().reshape(-1)
    flat = flat[~pd.isna(flat)]
    if len(flat) % 2 != 0:
        flat = flat[:-1]
    timestamps_ms = flat[0::2].astype(np.int64)
    values = flat[1::2].astype(np.float64)

    order = np.argsort(timestamps_ms, kind="stable")
    timestamps_ms, values = timestamps_ms[order], values[order]
    _, keep_idx = np.unique(timestamps_ms, return_index=True)
    keep_idx = np.sort(keep_idx)
    timestamps_ms, values = timestamps_ms[keep_idx], values[keep_idx]

    nominal_step_ms = 1000.0 / fs_nominal
    gaps, segments = _split_at_gaps(timestamps_ms, values, nominal_step_ms,
                                     gap_flag_ratio=1.5, gap_split_ms=200.0)

    patient_id = csv_path.parent.name.replace("Patch_", "")
    recordings = []
    for si, seg_idx in enumerate(segments):
        if len(seg_idx) < 2:
            continue
        recordings.append(Recording(
            signal_mv=values[seg_idx],
            timestamps_ms=timestamps_ms[seg_idx],
            fs_nominal=fs_nominal,
            source="vitalpatch",
            patient_id=patient_id,
            segment_id=f"{csv_path.stem}_seg{si}",
            gaps=gaps if si == 0 else [],
            meta={"file": str(csv_path)},
        ))
    return recordings


def parse_vitalpatch_vitals(csv_path: Path) -> pd.DataFrame:
    """Companion 11-column vitals file (HR, temp, RR interval, SpO2, BP).

    VitalPatch has no SpO2/BP sensor on this hardware, so those columns are
    sentinel values if present — forward-fill only real fields.
    """
    df = pd.read_csv(csv_path)
    return df.ffill()


def parse_sensio_ecg(csv_path: Path) -> Recording:
    """SeNSiO: 11 metadata rows, blank rows, header at row 16 (0-indexed 15).

    Raw variant has column 'ECG'; pre-filtered variant has 'ECG_Raw' and
    'ECG_Filtered' — if the filtered column is present we mark the
    recording as already bandpass-filtered so stage 4 can skip steps 1-4
    and avoid double-filtering artefacts.
    """
    csv_path = Path(csv_path)
    with open(csv_path, "r") as f:
        lines = f.readlines()

    meta = {}
    header_row = None
    for i, line in enumerate(lines[:30]):
        stripped = line.strip()
        if stripped.startswith("Index,"):
            header_row = i
            break
        if "," in stripped:
            key, _, val = stripped.partition(",")
            if key:
                meta[key] = val

    if header_row is None:
        raise ValueError(f"Could not find 'Index,...' header row in {csv_path}")

    df = pd.read_csv(csv_path, skiprows=header_row)
    df = df.dropna(how="all")

    already_filtered = "ECG_Filtered" in df.columns
    if already_filtered:
        signal = df["ECG_Filtered"].to_numpy(dtype=np.float64)
    else:
        signal = df["ECG"].to_numpy(dtype=np.float64)

    fs_nominal = _sensio_fs_from_command(meta.get("Command Sent", ""))
    n = len(signal)
    timestamps_ms = np.arange(n) * (1000.0 / fs_nominal)

    return Recording(
        signal_mv=signal,
        timestamps_ms=timestamps_ms,
        fs_nominal=fs_nominal,
        source="sensio",
        patient_id=meta.get("Bluetooth Device ID", "unknown"),
        segment_id=csv_path.stem,
        already_bandpass_filtered=already_filtered,
        meta=meta,
    )


def _sensio_fs_from_command(command_sent: str, default_fs: float = 100.0) -> float:
    """Parse 'STARTECG_F:100' style command strings for sample rate."""
    if ":" in command_sent:
        try:
            return float(command_sent.rsplit(":", 1)[1])
        except ValueError:
            pass
    return default_fs


def parse_wfdb_record(record_path: Path, ann_extension: Optional[str] = "atr") -> Recording:
    """Generic WFDB loader for future public datasets (MITDB, Icentia11k, ...).

    Not exercised until those datasets are downloaded; kept here so
    stages 2+ have one call site (`load_any`) regardless of dataset.
    """
    import wfdb

    record = wfdb.rdrecord(str(record_path))
    signal = record.p_signal[:, 0]
    fs = float(record.fs)
    timestamps_ms = np.arange(len(signal)) * (1000.0 / fs)

    meta = {"units": record.units, "sig_name": record.sig_name}
    if ann_extension:
        try:
            ann = wfdb.rdann(str(record_path), ann_extension)
            meta["beat_samples"] = ann.sample.tolist()
            meta["beat_symbols"] = ann.symbol
        except FileNotFoundError:
            pass

    return Recording(
        signal_mv=signal,
        timestamps_ms=timestamps_ms,
        fs_nominal=fs,
        source="wfdb",
        patient_id=record_path.stem,
        segment_id=record_path.stem,
        meta=meta,
    )


def discover_vitalpatch_files(root: Path) -> list[Path]:
    return sorted(root.glob("Patch_*/*_ecg.csv"))


def discover_sensio_files(root: Path) -> list[Path]:
    return sorted(root.glob("ECG_*.csv"))


# ============================================================================
# quality.py — Stage 2: Pre-filter Signal Quality Index (SQI) gate
# ============================================================================
"""Stage 2 — Pre-filter Signal Quality Index (SQI) gate.

Runs on non-overlapping windows *before* any filtering. A window that
fails any threshold is rejected and logged with a quality code, metric
value, and timestamp range — never silently dropped — forming a
per-patient signal-quality audit trail (see `audit.py`).

Recommendation #6 fix ("Fix the signal-quality check so it doesn't reject
real AFib patients"): the baseline gate is documented as using a single
"regularity"-flavoured kurtosis threshold that, in practice, also caught
genuinely irregular-but-clean rhythms (AFib, frequent ectopy) because
irregular RR spacing lowers window-level kurtosis in the same direction as
motion artefact does. Here:
  1. Kurtosis is computed only as a *morphology* impulse-character check
     (peakedness of the amplitude distribution), which is legitimately
     noise-sensitive and rhythm-agnostic.
  2. RR-interval irregularity is never part of this gate at all — it is
     computed later (stage 5/7) and fed to the rhythm classifier as a
     clinical signal (AFib burden), not treated as a quality defect.
  3. RR-interval outliers detected downstream are flagged, never dropped
     (see `beats.py`), so a fast, irregular but real AFib strip survives
     end to end instead of being silently discarded as "noisy".
"""


@dataclass
class WindowVerdict:
    start_idx: int
    end_idx: int
    start_ms: float
    end_ms: float
    passed: bool
    reject_code: str | None
    metrics: dict


def _longest_equal_runs_frac(x: np.ndarray, min_run_samples: int, at_values: np.ndarray | None = None) -> float:
    """Fraction of samples that belong to a run of >= min_run_samples
    consecutive identical values (optionally restricted to runs sitting at
    specific values, e.g. the window's own min/max for railing/clipping).
    Distinguishes real "stuck sensor" flatline / ADC railing from ordinary
    quantization noise, which produces only short scattered equal-adjacent
    pairs and would otherwise false-trigger on a real, clean recording.
    """
    if len(x) < 2:
        return 0.0
    is_repeat = np.diff(x) == 0
    if at_values is not None:
        at_mask = np.isin(x[1:], at_values) & np.isin(x[:-1], at_values)
        is_repeat = is_repeat & at_mask

    flagged = np.zeros(len(x), dtype=bool)
    run_start = None
    for i, rep in enumerate(np.append(is_repeat, False)):
        if rep:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_len = i - run_start + 1  # +1: diff index i means samples[i] == samples[i+1]
                if run_len >= min_run_samples:
                    flagged[run_start:i + 1] = True
            run_start = None
    return float(np.mean(flagged))


def _flatline_frac(x: np.ndarray, fs: float, run_ms: float) -> float:
    min_run = max(2, int(round(run_ms / 1000.0 * fs)))
    return _longest_equal_runs_frac(x, min_run)


def _clipping_frac(x: np.ndarray, fs: float, run_ms: float, clip_value: float | None) -> float:
    """Railing detector: a sustained run sitting at the window's own
    observed extremes (or at an explicit device clip_value, if known)."""
    x_valid = x[~np.isnan(x)]
    if len(x_valid) < 2:
        return 0.0
    if clip_value is not None:
        rail_values = np.array([clip_value, -clip_value])
    else:
        rail_values = np.array([np.max(x_valid), np.min(x_valid)])
    min_run = max(2, int(round(run_ms / 1000.0 * fs)))
    return _longest_equal_runs_frac(x, min_run, at_values=rail_values)


def _missing_frac(x: np.ndarray) -> float:
    return float(np.mean(np.isnan(x)))


def _morphology_kurtosis(x: np.ndarray) -> float:
    """Amplitude-distribution kurtosis — QRS impulse character.

    Deliberately NOT a function of beat timing/RR spacing (see module
    docstring, recommendation #6): a perfectly regular flat noise signal
    and a perfectly irregular but clean AFib signal can both have "normal"
    RR patterns or not — this metric only asks whether the waveform shape
    has the sharp/peaky character of real QRS complexes.
    """
    x = x[~np.isnan(x)]
    if len(x) < 8:
        return 0.0
    return float(stats.kurtosis(x, fisher=True))


def _baseline_wander_ratio(x: np.ndarray, fs: float) -> float:
    """Peak-to-peak low-frequency drift, expressed relative to the
    window's own robust dynamic range (IQR) rather than an absolute mV
    number — see `SQIThresholds.baseline_wander_ratio_max` docstring for
    why (unknown per-device ADC scale)."""
    from scipy.signal import butter, filtfilt
    x = x[~np.isnan(x)]
    if len(x) < 20:
        return 0.0
    nyq = fs / 2.0
    b, a = butter(2, 0.5 / nyq, btype="low")
    try:
        low = filtfilt(b, a, x)
    except ValueError:
        return 0.0
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    scale = abs(iqr) if abs(iqr) > 1e-9 else (np.std(x) + 1e-9)
    return float(np.ptp(low)) / scale


def _snr_db(x: np.ndarray, fs: float) -> float:
    from scipy.signal import butter, filtfilt
    x = x[~np.isnan(x)]
    if len(x) < 20:
        return 0.0
    nyq = fs / 2.0
    b, a = butter(4, [0.5 / nyq, 40 / nyq], btype="band")
    try:
        signal_band = filtfilt(b, a, x)
    except ValueError:
        return 0.0
    noise = x - signal_band
    sig_power = np.mean(signal_band ** 2)
    noise_power = np.mean(noise ** 2) + 1e-12
    return float(10 * np.log10(sig_power / noise_power)) if sig_power > 0 else -999.0


def evaluate_window(x: np.ndarray, t_ms: np.ndarray, fs: float,
                     clip_value: float | None, thresholds: SQIThresholds = SQI) -> WindowVerdict:
    metrics = {
        "flatline_frac": _flatline_frac(x, fs, thresholds.flatline_run_ms),
        "clipping_frac": _clipping_frac(x, fs, thresholds.clipping_run_ms, clip_value),
        "missing_frac": _missing_frac(x),
        "morphology_kurtosis": _morphology_kurtosis(x),
        "baseline_wander_ratio": _baseline_wander_ratio(x, fs),
        "snr_db": _snr_db(x, fs),
    }

    reject_code = None
    if metrics["flatline_frac"] > thresholds.flatline_frac_max:
        reject_code = "FLATLINE"
    elif metrics["clipping_frac"] > thresholds.clipping_frac_max:
        reject_code = "CLIPPING"
    elif metrics["missing_frac"] > thresholds.missing_frac_max:
        reject_code = "MISSING_SAMPLES"
    elif metrics["morphology_kurtosis"] < thresholds.kurtosis_min:
        reject_code = "NO_QRS_IMPULSE_CHARACTER"
    elif metrics["baseline_wander_ratio"] > thresholds.baseline_wander_ratio_max:
        reject_code = "BASELINE_WANDER"
    elif metrics["snr_db"] < thresholds.snr_db_min:
        reject_code = "LOW_SNR"

    return WindowVerdict(
        start_idx=0, end_idx=len(x),
        start_ms=float(t_ms[0]) if len(t_ms) else 0.0,
        end_ms=float(t_ms[-1]) if len(t_ms) else 0.0,
        passed=reject_code is None,
        reject_code=reject_code,
        metrics=metrics,
    )


def run_sqi_gate(signal: np.ndarray, timestamps_ms: np.ndarray, fs: float,
                  clip_value: float | None = None,
                  thresholds: SQIThresholds = SQI) -> tuple[np.ndarray, list[WindowVerdict]]:
    """Split into non-overlapping windows, evaluate each, return a boolean
    keep-mask (same length as signal) plus the full per-window audit trail.
    """
    win_len = max(1, int(round(thresholds.window_seconds * fs)))
    n = len(signal)
    keep_mask = np.zeros(n, dtype=bool)
    verdicts: list[WindowVerdict] = []

    for start in range(0, n, win_len):
        end = min(start + win_len, n)
        window = signal[start:end]
        t_window = timestamps_ms[start:end]
        verdict = evaluate_window(window, t_window, fs, clip_value, thresholds)
        verdict.start_idx, verdict.end_idx = start, end
        verdicts.append(verdict)
        if verdict.passed:
            keep_mask[start:end] = True

    return keep_mask, verdicts


def rejection_rate(verdicts: list[WindowVerdict]) -> float:
    if not verdicts:
        return 0.0
    return sum(1 for v in verdicts if not v.passed) / len(verdicts)


# ============================================================================
# resample.py — Stage 3: Uniform resample to the common target rate (125 Hz)
# ============================================================================
"""Stage 3 — Uniform resample to the common target rate (125 Hz)."""


def resample_linear(signal: np.ndarray, timestamps_ms: np.ndarray,
                     target_fs: float = TARGET_FS) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation onto a strict uniform time grid.

    Linear rather than cubic, to avoid ringing near sharp QRS peaks
    (cubic splines overshoot around the steep R-wave edge).
    """
    valid = ~np.isnan(signal)
    t = timestamps_ms[valid]
    x = signal[valid]
    if len(t) < 2:
        return np.array([]), np.array([])

    step_ms = 1000.0 / target_fs
    t_uniform = np.arange(t[0], t[-1], step_ms)
    x_uniform = np.interp(t_uniform, t, x)
    return x_uniform, t_uniform


def resample_decimate(signal: np.ndarray, fs_in: float,
                       target_fs: float = TARGET_FS) -> np.ndarray:
    """FIR anti-aliased decimation for higher native rates (e.g. Icentia11k
    250 Hz -> 125 Hz). Naive every-other-sample downsampling would alias
    high-frequency content into the QRS complex; `scipy.signal.decimate`
    applies a proper anti-aliasing filter first.
    """
    ratio = fs_in / target_fs
    if abs(ratio - round(ratio)) > 1e-6:
        raise ValueError(f"decimate requires an integer ratio, got {ratio} "
                          f"({fs_in} Hz -> {target_fs} Hz)")
    return decimate(signal, int(round(ratio)), ftype="fir", zero_phase=True)


def to_target_rate(signal: np.ndarray, timestamps_ms: np.ndarray, fs_nominal: float,
                    target_fs: float = TARGET_FS) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch: decimate when downsampling from a clean integer-ratio
    source (WFDB-style fixed fs), otherwise linear-interpolate irregular
    wearable timestamps (VitalPatch/SeNSiO).
    """
    if abs(fs_nominal - target_fs) < 1e-6:
        return signal, timestamps_ms

    ratio = fs_nominal / target_fs
    if fs_nominal > target_fs and abs(ratio - round(ratio)) < 1e-6:
        out = resample_decimate(signal, fs_nominal, target_fs)
        step_ms = 1000.0 / target_fs
        t_out = timestamps_ms[0] + np.arange(len(out)) * step_ms
        return out, t_out

    return resample_linear(signal, timestamps_ms, target_fs)


# ============================================================================
# filters.py — Stage 4: Five-step filter chain, applied in order
# ============================================================================
"""Stage 4 — Five-step filter chain, applied in order.

Each step assumes the previous one has run. SeNSiO pre-filtered files skip
steps 1-4 (device already bandpass-filtered) and start at step 5, to avoid
double-filtering artefacts.
"""


def _safe_filtfilt(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """`filtfilt` needs len(x) > ~3*max(len(a),len(b)); short beat-level
    segments (e.g. a device's shortest test recordings) can be shorter
    than that. Fall back to a causal `lfilter` for those rather than
    raising, since a short segment is still worth passing through the
    rest of the pipeline (flagged for its unusually short duration
    upstream, not silently dropped here)."""
    min_len = 3 * max(len(a), len(b))
    if len(x) <= min_len:
        return lfilter(b, a, x)
    return filtfilt(b, a, x)


def remove_baseline_median(x: np.ndarray, fs: float, window_ms: float = FILTER.median_baseline_window_ms) -> np.ndarray:
    """Step 1: double median filter, subtract to remove baseline drift
    without distorting QRS shape."""
    win_samples = int(round(window_ms / 1000.0 * fs))
    win_samples = min(win_samples, len(x))
    win_samples += 1 - (win_samples % 2)  # medfilt needs an odd kernel
    win_samples = max(win_samples, 3)
    if win_samples > len(x):
        return x - np.median(x)  # segment too short for a real median-filter window
    stage1 = medfilt(x, kernel_size=win_samples)
    stage2 = medfilt(stage1, kernel_size=win_samples)
    return x - stage2


def highpass_residual(x: np.ndarray, fs: float, cutoff_hz: float = FILTER.highpass_hz) -> np.ndarray:
    """Step 2: 2nd-order zero-phase Butterworth high-pass — catches drift
    the median filter misses (motion artefact)."""
    nyq = fs / 2.0
    b, a = butter(2, cutoff_hz / nyq, btype="high")
    return _safe_filtfilt(b, a, x)


def powerline_notch(x: np.ndarray, fs: float, notch_hz: float = FILTER.notch_hz,
                     q: float = FILTER.notch_q) -> np.ndarray:
    """Step 3: IIR notch at mains frequency."""
    nyq = fs / 2.0
    if notch_hz >= nyq:
        return x
    b, a = iirnotch(notch_hz / nyq, q)
    return _safe_filtfilt(b, a, x)


def bandpass(x: np.ndarray, fs: float, low_hz: float = FILTER.bandpass_low_hz,
             high_hz: float = FILTER.bandpass_high_hz, order: int = FILTER.bandpass_order) -> np.ndarray:
    """Step 4: industry-standard wearable ECG passband."""
    nyq = fs / 2.0
    high = min(high_hz, nyq * 0.99)
    b, a = butter(order, [low_hz / nyq, high / nyq], btype="band")
    return _safe_filtfilt(b, a, x)


def emg_suppress_kalman(x: np.ndarray, process_var: float = 1e-4, meas_var: float = 1e-2) -> np.ndarray:
    """Step 5: scalar Kalman filter with adaptive gain, removes muscle
    noise that a Butterworth bandpass alone would strip the QRS to remove.

    A minimal scalar (constant-value) Kalman filter: adequate for
    suppressing high-frequency EMG bursts between beats while a genuine
    QRS transient (large innovation) still passes through because the
    Kalman gain adapts upward on large residuals.
    """
    n = len(x)
    out = np.empty(n)
    est = x[0]
    err_cov = 1.0
    for i in range(n):
        err_cov_pred = err_cov + process_var
        innovation = x[i] - est
        adaptive_meas_var = meas_var * (1.0 + 5.0 * min(abs(innovation), 3.0))
        gain = err_cov_pred / (err_cov_pred + adaptive_meas_var)
        est = est + gain * innovation
        err_cov = (1 - gain) * err_cov_pred
        out[i] = est
    return out


def robust_zscore(x: np.ndarray) -> np.ndarray:
    """Step 6: outlier-resistant normalization (median / MAD) — mean/std
    is inflated by motion artefacts."""
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    scale = mad * 1.4826 if mad > 1e-9 else (np.std(x) + 1e-9)
    return (x - median) / scale


def apply_filter_chain(x: np.ndarray, fs: float, already_bandpass_filtered: bool = False,
                        cfg: FilterChainConfig = FILTER) -> np.ndarray:
    """Run the full chain, or steps 5-6 only if the device pre-filtered."""
    if not already_bandpass_filtered:
        x = remove_baseline_median(x, fs, cfg.median_baseline_window_ms)
        x = highpass_residual(x, fs, cfg.highpass_hz)
        x = powerline_notch(x, fs, cfg.notch_hz, cfg.notch_q)
        x = bandpass(x, fs, cfg.bandpass_low_hz, cfg.bandpass_high_hz, cfg.bandpass_order)
    x = emg_suppress_kalman(x)
    return x


# ============================================================================
# beats.py — Stage 5: R-peak detection and beat segmentation
# ============================================================================
"""Stage 5 — R-peak detection and beat segmentation.

Uses WFDB's XQRS rather than Pan-Tompkins: wearable single-lead
electrodes sit at non-standard chest positions and produce morphologically
atypical QRS complexes that XQRS's adaptive threshold handles better.

RR intervals are sanity-checked against a 300-2000 ms physiological range;
outliers are FLAGGED, never silently dropped (recommendation #6 — a fast,
irregular-but-real AFib run must survive to the rhythm classifier instead
of being discarded as a "bad beat").
"""


@dataclass
class Beat:
    r_peak_idx: int
    r_peak_ms: float
    rr_pre_ms: float | None
    rr_post_ms: float | None
    rr_flagged: bool          # True if RR outside physiological range - kept, not dropped
    primary_window: np.ndarray | None   # 75 samples @125Hz: 200ms pre + R + 400ms post
    wide_window: np.ndarray | None      # 125 samples @125Hz: 500ms pre + R + 500ms post
    quality_rejected: bool
    quality_reject_reason: str | None
    # Local-rhythm-context ("timing") fields, computed once here in
    # segment_beats from the same rr_ms array used for rr_pre_ms/rr_post_ms
    # -- see beat_feature_vector(..., include_timing=True) for how these are
    # consumed. All default to neutral (not-detected) values, never to 0 in
    # a way that would read as "extremely premature"/"extremely irregular"
    # when the true answer is "no history available yet" (see the edge-case
    # notes at each computation site in segment_beats).
    rr_ratio_k8: float = 1.0
    rr_ratio_k16: float = 1.0
    rr_ratio_k32: float = 1.0
    rr_pre_post_ratio: float = 1.0
    rr_cv_local: float = 0.0
    prematurity_score: float = 0.0
    compensatory_pause_flag: float = 0.0


def detect_r_peaks(signal: np.ndarray, fs: float) -> np.ndarray:
    """WFDB XQRS adaptive-threshold R-peak detector."""
    import wfdb.processing as wp
    if len(signal) < int(fs * 2):
        return np.array([], dtype=int)
    xqrs = wp.XQRS(sig=signal, fs=fs)
    xqrs.detect(verbose=False)
    return np.asarray(xqrs.qrs_inds, dtype=int)


def _extract_window(signal: np.ndarray, center_idx: int, pre_samples: int, post_samples: int) -> np.ndarray | None:
    start, end = center_idx - pre_samples, center_idx + post_samples
    if start < 0 or end > len(signal):
        return None
    return signal[start:end]


def _beat_level_sqi(window: np.ndarray, r_local_idx: int, cfg: BeatWindowConfig) -> tuple[bool, str | None]:
    """Reject individual beats with amplitude too low, excess in-window
    baseline drift, or an R-peak that isn't the true local maximum —
    catches detection jitter the window-level gate can't see."""
    if window is None:
        return True, "WINDOW_OUT_OF_BOUNDS"

    amplitude = float(np.ptp(window))
    noise_floor = float(np.median(np.abs(np.diff(window)))) + 1e-9
    if amplitude < cfg.beat_amplitude_min_noise_ratio * noise_floor:
        return True, "LOW_AMPLITUDE"

    drift = float(np.abs(np.median(window[:5]) - np.median(window[-5:])))
    if drift > 3.0 * (np.std(window) + 1e-9):
        return True, "EXCESS_BASELINE_DRIFT"

    search_radius = max(2, len(window) // 20)
    lo = max(0, r_local_idx - search_radius)
    hi = min(len(window), r_local_idx + search_radius)
    local_patch = window[lo:hi]
    if len(local_patch) and np.argmax(np.abs(local_patch)) != (r_local_idx - lo):
        return True, "R_PEAK_NOT_LOCAL_MAX"

    return False, None


def segment_beats(signal: np.ndarray, fs: float, r_peaks: np.ndarray,
                   cfg: BeatWindowConfig = BEATS) -> list[Beat]:
    # Round pre/post independently but derive the total from their sum so the
    # window length is deterministic (e.g. exactly 125 samples for the 1s wide
    # window @ 125Hz) rather than drifting by +/-1 sample from rounding each
    # half separately (Python's round-half-to-even can round both 62.5 -> 62).
    primary_pre = int(round(cfg.primary_pre_ms / 1000.0 * fs))
    primary_total = int(round((cfg.primary_pre_ms + cfg.primary_post_ms) / 1000.0 * fs))
    primary_post = primary_total - primary_pre
    wide_pre = int(round(cfg.wide_pre_ms / 1000.0 * fs))
    wide_total = int(round((cfg.wide_pre_ms + cfg.wide_post_ms) / 1000.0 * fs))
    wide_post = wide_total - wide_pre

    beats: list[Beat] = []
    rr_ms = np.diff(r_peaks) * (1000.0 / fs) if len(r_peaks) > 1 else np.array([])

    for i, r_idx in enumerate(r_peaks):
        rr_pre = float(rr_ms[i - 1]) if i > 0 else None
        rr_post = float(rr_ms[i]) if i < len(rr_ms) else None

        flagged = False
        for rr in (rr_pre, rr_post):
            if rr is not None and not (cfg.rr_min_ms <= rr <= cfg.rr_max_ms):
                flagged = True  # out-of-physiological-range RR: flagged, never dropped

        primary = _extract_window(signal, r_idx, primary_pre, primary_post)
        wide = _extract_window(signal, r_idx, wide_pre, wide_post)
        rejected, reason = _beat_level_sqi(primary, primary_pre, cfg)

        # Local-rhythm-context features. `history` is the up-to-K completed
        # RR intervals strictly BEFORE this beat (rr_ms[i-1] is the interval
        # ending at beat i) -- never includes rr_ms[i] (the POST interval),
        # so nothing here leaks future/lookahead information about this
        # beat's own outcome. Since rr_ms is derived fresh per call from
        # this recording's own r_peaks, a window also never spans two
        # recordings. i < 8/16/32 beats into a recording simply gets a
        # shorter history (rr_ms[0:i]); i == 0 gets none at all, in which
        # case every ratio/score below falls back to its documented neutral
        # default from the Beat dataclass rather than a value that would
        # misread as "very premature" or "very irregular".
        history16 = rr_ms[max(0, i - 16):i]
        mean16 = float(np.mean(history16)) if len(history16) else 0.0

        rr_ratio_k8 = rr_ratio_k16 = rr_ratio_k32 = 1.0
        if rr_pre is not None:
            hist8 = rr_ms[max(0, i - 8):i]
            mean8 = float(np.mean(hist8)) if len(hist8) else 0.0
            rr_ratio_k8 = rr_pre / mean8 if mean8 > 1e-9 else 1.0
            hist32 = rr_ms[max(0, i - 32):i]
            mean32 = float(np.mean(hist32)) if len(hist32) else 0.0
            rr_ratio_k32 = rr_pre / mean32 if mean32 > 1e-9 else 1.0
            rr_ratio_k16 = rr_pre / mean16 if mean16 > 1e-9 else 1.0

        rr_pre_post_ratio = 1.0
        if rr_pre is not None and rr_post is not None and rr_post > 1e-9:
            rr_pre_post_ratio = rr_pre / rr_post

        rr_cv_local = 0.0
        if len(history16) >= 2 and mean16 > 1e-9:
            rr_cv_local = float(np.std(history16) / mean16)

        prematurity_score = 0.0
        if rr_pre is not None and mean16 > 1e-9:
            prematurity_score = (mean16 - rr_pre) / mean16

        compensatory_pause_flag = 0.0
        if rr_post is not None and mean16 > 1e-9 and rr_post > 1.2 * mean16:
            compensatory_pause_flag = 1.0

        beats.append(Beat(
            r_peak_idx=int(r_idx),
            r_peak_ms=float(r_idx * 1000.0 / fs),
            rr_pre_ms=rr_pre,
            rr_post_ms=rr_post,
            rr_flagged=flagged,
            primary_window=primary,
            wide_window=wide,
            quality_rejected=rejected,
            quality_reject_reason=reason,
            rr_ratio_k8=rr_ratio_k8,
            rr_ratio_k16=rr_ratio_k16,
            rr_ratio_k32=rr_ratio_k32,
            rr_pre_post_ratio=rr_pre_post_ratio,
            rr_cv_local=rr_cv_local,
            prematurity_score=prematurity_score,
            compensatory_pause_flag=compensatory_pause_flag,
        ))

    return beats


def detect_and_segment(signal: np.ndarray, fs: float, cfg: BeatWindowConfig = BEATS) -> list[Beat]:
    r_peaks = detect_r_peaks(signal, fs)
    return segment_beats(signal, fs, r_peaks, cfg)


# ============================================================================
# features.py — Stage 6: Feature extraction
# ============================================================================
"""Stage 6 — Feature extraction.

Two parallel outputs:
  1. The 56-dimensional per-beat feature vector (Zhu et al. 2021): 5
     morphological + 51 wavelet coefficients. Kept as the interpretable,
     no-training-data-required baseline feature set — SHAP-ranked and fed
     to the XGBoost classifier in `classify.py`.
  2. Recording-level HRV/frequency features (SDNN, RMSSD, pNN50, LF/HF,
     QRS-width trend) for the stage 8 risk scorer.

The learned encoder in `encoder.py` (recommendation #1/#2) is a SEPARATE,
richer representation computed from the same beat windows — it does not
replace this module, it supplements it, so the pipeline still works with
zero training data (this module) while the encoder is being pretrained.

CORRECTION (2026-07-16): a prior version of this note claimed a 7-feature
local-rhythm-context extension had already been "built and evaluated here"
with specific DS1_VAL/DS2 numbers, then reverted. That claim was found to
be unverifiable and almost certainly false: no such code existed anywhere
in the codebase (only this comment), no ABLATION_REPORT.md existed to back
the cited numbers, and this repo has zero git history to check against —
the same "claimed done, no artifact" pattern independently caught twice
elsewhere this session (a phantom five_class_xgb_timing_v1.json referenced
in a docstring, and this note itself). Treat any prior "done"/"reverted"
claim in this codebase as unverified until a real file, fresh metric, or
diff backs it up.

The 7-feature local-rhythm-context extension (rolling RR-ratio at
K=8/16/32, rr_pre/rr_post ratio, RR CV, normalized prematurity score,
compensatory-pause flag) is implemented for real below, as
`rr_ratio_k8/16/32`, `rr_pre_post_ratio`, `rr_cv_local`,
`prematurity_score`, `compensatory_pause_flag` on `Beat`, computed once in
`segment_beats` (shared by training and runtime inference, so there is one
implementation, not two) and appended by `beat_feature_vector(...,
include_timing=True)`. Defaults to `include_timing=False` everywhere so
the existing 56-dim production model (`five_class_xgb.json`) and its
runtime inference path are completely unaffected; the new 63-dim vector is
opt-in only, exercised by `--timing-features` in train-classifiers /
eval-classifier and saved as its own model artifact. See
ABLATION_REPORT.md (created for real, not just referenced) for the actual
before/after DS1_VAL and DS2 numbers from this run.
"""

# The spec's 75-sample primary window (200ms pre + R + 400ms post @ 125Hz) is
# shorter than what a level-4 db4 decomposition ideally wants, so every
# beat trips pywt's boundary-effects warning. The coefficients are still
# valid (just more boundary-influenced) and are resampled to a fixed
# length below regardless, so the warning is expected noise, not a bug.
warnings.filterwarnings("ignore", message="Level value of.*is too high", module="pywt")

N_MORPHOLOGICAL = 5
N_WAVELET = 51
N_TIMING = 7
N_FEATURES = N_MORPHOLOGICAL + N_WAVELET  # unchanged: production's dimensionality
N_FEATURES_WITH_TIMING = N_FEATURES + N_TIMING


def _feature_width(include_timing: bool, drop_compensatory_pause: bool = False,
                    timing_only: bool = False, include_r_amp: bool = False) -> int:
    timing_width = (N_TIMING - (1 if drop_compensatory_pause else 0)) if (include_timing or timing_only) else 0
    if timing_only:
        return timing_width
    base_width = N_FEATURES + (1 if include_r_amp else 0)
    return base_width + timing_width


TIMING_FEATURE_NAMES = ["rr_ratio_k8", "rr_ratio_k16", "rr_ratio_k32", "rr_pre_post_ratio",
                         "rr_cv_local", "prematurity_score", "compensatory_pause_flag"]


def _timing_features(beat: "Beat", drop_compensatory_pause: bool = False) -> np.ndarray:
    """`drop_compensatory_pause` is a single-purpose ablation knob (not a
    general feature-selection mechanism) for the timing-features
    micro-experiment in ABLATION_REPORT.md: compensatory_pause_flag was
    found to be the single most important feature in the 63-dim model
    (rank 1/63, ~32% of total gain) and fires for both premature S and
    premature V beats, the suspected driver of the S->V confusion increase
    that regressed V's DS2 F1. Drops the column entirely (6-dim timing
    block) rather than zeroing it, so the ablation removes the information,
    not just its typical value."""
    vals = [beat.rr_ratio_k8, beat.rr_ratio_k16, beat.rr_ratio_k32,
            beat.rr_pre_post_ratio, beat.rr_cv_local,
            beat.prematurity_score, beat.compensatory_pause_flag]
    if drop_compensatory_pause:
        vals = vals[:-1]
    return np.array(vals)


def _wavelet_features(window: np.ndarray, wavelet: str = "db4") -> np.ndarray:
    """Discrete wavelet decomposition -> 14 approximation (a4) + 23 detail
    (d3) + 14 detail (d4) coefficients, resampled to fixed lengths so the
    feature vector has constant dimensionality regardless of input length.
    """
    coeffs = pywt.wavedec(window, wavelet, level=4)
    # coeffs = [a4, d4, d3, d2, d1]
    a4, d4, d3 = coeffs[0], coeffs[1], coeffs[2]

    def _fixed_len(arr: np.ndarray, target_len: int) -> np.ndarray:
        if len(arr) == target_len:
            return arr
        x_old = np.linspace(0, 1, len(arr))
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_old, arr)

    return np.concatenate([
        _fixed_len(a4, 14),
        _fixed_len(d3, 23),
        _fixed_len(d4, 14),
    ])


def _morphological_features(window: np.ndarray, primary_pre_samples: int,
                             rr_pre_ms: float | None, rr_post_ms: float | None) -> np.ndarray:
    r_idx = primary_pre_samples

    rr_pre = rr_pre_ms if rr_pre_ms is not None else 0.0
    local_hrv = (rr_post_ms - rr_pre_ms) if (rr_pre_ms and rr_post_ms) else 0.0

    left = window[:r_idx]
    right = window[r_idx:]
    left_area, right_area = float(np.sum(np.abs(left))), float(np.sum(np.abs(right)))
    area_ratio = left_area / right_area if right_area > 1e-9 else 0.0

    above = window[window > 0]
    below = window[window < 0]
    above_below_ratio = (np.sum(above) / abs(np.sum(below))) if len(below) and np.sum(below) != 0 else 0.0

    amplitude_range = float(np.ptp(window))

    return np.array([rr_pre, local_hrv, area_ratio, above_below_ratio, amplitude_range])


def _r_amp_feature(window: np.ndarray, primary_pre_samples: int) -> np.ndarray:
    """R-peak amplitude (post-`robust_zscore` normalized window). This was
    originally computed inline in `_morphological_features` (as `r_amp`)
    but never returned -- dead code, silently discarded, found during the
    two-stage-classifier prerequisite work (ABLATION_REPORT.md,
    "Prerequisite 2"). Split into its own function and appended at the END
    of the full feature vector in `beat_feature_vector` (not inserted
    between the 5 morphological features and the 51 wavelet features),
    because inserting it there was tried first and shifted every wavelet
    column's index by one -- caught by the train/inference parity check,
    which is exactly what that check is for. Appending at the end keeps
    every existing index (0-55) meaning exactly what it always meant;
    this is index 56, opt-in only via include_r_amp."""
    return np.array([float(window[primary_pre_samples])])


def beat_feature_vector(beat: Beat, primary_pre_samples: int,
                         include_timing: bool = False,
                         drop_compensatory_pause: bool = False,
                         timing_only: bool = False,
                         include_r_amp: bool = False) -> np.ndarray | None:
    """Step 6 of the filter chain (robust Z-score, per beat window) is
    applied here, right before feature extraction — quality rejection
    upstream (`beats.py`) intentionally runs on the raw window instead, so
    normalization can't mask a genuinely flat/low-amplitude beat.

    `include_timing=False` (default) reproduces the exact 56-dim vector the
    production model (`five_class_xgb.json`) and the live runtime pipeline
    expect -- this function is the single source used by BOTH training
    (`_load_record_beats`/`build_dataset`) and runtime inference
    (`batch_feature_matrix`), so leaving the default unchanged means
    neither path can silently drift out of sync with the other, and
    production is unaffected by anything below. `include_timing=True`
    appends the 7 local-rhythm-context features computed in `segment_beats`
    (final layout: 5 morph + 51 wavelet + 7 timing = 63) -- opt-in only,
    for the timing-features experiment. `drop_compensatory_pause` (only
    meaningful when include_timing=True or timing_only=True) drops just
    that one column. `timing_only=True` returns JUST the timing block (7,
    or 6 if drop_compensatory_pause) with no morphology/wavelet at all --
    for the feature-family ablation (morphology-only vs timing-only vs
    combined) in ABLATION_REPORT.md; note the 56 "morphology" features
    already include 2 raw RR scalars (rr_pre, local_hrv) via
    `_morphological_features`, so "morphology-only" isn't strictly
    zero-timing-information, just missing the 7 richer local-rhythm-context
    features. `include_r_amp=True` appends R-peak amplitude (computed but
    previously discarded in `_morphological_features` -- see
    `_r_amp_feature`'s docstring) as the LAST feature in the vector
    (after wavelet, and after timing if `include_timing` is also set) so
    every previously-established index (0-55, or 0-62 with timing) keeps
    its exact original meaning -- inserting it earlier, between morphology
    and wavelet, was tried first and shifted every wavelet index by one;
    caught by the train/inference parity check.
    """
    if beat.primary_window is None or beat.quality_rejected:
        return None
    if timing_only:
        return _timing_features(beat, drop_compensatory_pause=drop_compensatory_pause)
    normalized = robust_zscore(beat.primary_window)
    morph = _morphological_features(normalized, primary_pre_samples, beat.rr_pre_ms, beat.rr_post_ms)
    wavelet = _wavelet_features(normalized)
    parts = [morph, wavelet]
    if include_timing:
        parts.append(_timing_features(beat, drop_compensatory_pause=drop_compensatory_pause))
    if include_r_amp:
        parts.append(_r_amp_feature(normalized, primary_pre_samples))
    return np.concatenate(parts)


def batch_feature_matrix(beats: list[Beat], primary_pre_samples: int,
                          include_timing: bool = False,
                          drop_compensatory_pause: bool = False,
                          timing_only: bool = False,
                          include_r_amp: bool = False) -> tuple[np.ndarray, list[int]]:
    """Returns (feature_matrix, indices_into_beats_used) — skipping
    rejected/out-of-bounds beats but preserving which original beat each
    row corresponds to."""
    rows, idxs = [], []
    for i, beat in enumerate(beats):
        vec = beat_feature_vector(beat, primary_pre_samples, include_timing=include_timing,
                                   drop_compensatory_pause=drop_compensatory_pause,
                                   timing_only=timing_only, include_r_amp=include_r_amp)
        if vec is not None:
            rows.append(vec)
            idxs.append(i)
    if not rows:
        return np.zeros((0, _feature_width(include_timing, drop_compensatory_pause,
                                            timing_only, include_r_amp))), []
    return np.vstack(rows), idxs


def recording_level_hrv(beats: list[Beat]) -> dict:
    """SDNN, RMSSD, pNN50, LF/HF power ratio, QRS-width trend — used by
    the stage 8 risk scorer, not the beat classifier."""
    rr = np.array([b.rr_post_ms for b in beats if b.rr_post_ms is not None and not b.rr_flagged])
    if len(rr) < 3:
        return {"sdnn_ms": 0.0, "rmssd_ms": 0.0, "pnn50_pct": 0.0, "lf_hf_ratio": 0.0, "qrs_width_trend": 0.0}

    sdnn = float(np.std(rr, ddof=1))
    diffs = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) else 0.0
    pnn50 = float(np.mean(np.abs(diffs) > 50.0) * 100) if len(diffs) else 0.0
    lf_hf = _lf_hf_ratio(rr)

    widths = [float(np.ptp(np.where(np.abs(b.primary_window) > np.std(b.primary_window))[0]))
              for b in beats if b.primary_window is not None and not b.quality_rejected]
    qrs_width_trend = float(np.polyfit(np.arange(len(widths)), widths, 1)[0]) if len(widths) > 2 else 0.0

    return {
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "pnn50_pct": pnn50,
        "lf_hf_ratio": lf_hf,
        "qrs_width_trend": qrs_width_trend,
    }


def _lf_hf_ratio(rr_ms: np.ndarray) -> float:
    """Rough LF (0.04-0.15Hz)/HF (0.15-0.4Hz) power ratio from the RR
    tachogram via Lomb-Scargle (unevenly sampled RR series)."""
    from scipy.signal import lombscargle
    t = np.cumsum(rr_ms) / 1000.0
    rr_detrended = rr_ms - np.mean(rr_ms)
    freqs = np.linspace(0.01, 0.5, 200) * 2 * np.pi
    try:
        power = lombscargle(t, rr_detrended, freqs, normalize=True)
    except (ZeroDivisionError, ValueError):
        return 0.0
    f_hz = freqs / (2 * np.pi)
    lf_power = float(np.sum(power[(f_hz >= 0.04) & (f_hz < 0.15)]))
    hf_power = float(np.sum(power[(f_hz >= 0.15) & (f_hz < 0.4)]))
    return lf_power / hf_power if hf_power > 1e-9 else 0.0


# ============================================================================
# encoder.py — Self-supervised ECG foundation encoder
# ============================================================================
"""Self-supervised ECG foundation encoder — the core of the new methodology.

Implements recommendation #1 ("Use a pretrained 'ECG expert' AI instead of
25 handcrafted numbers" — Very High impact) and recommendation #2
("Pre-train an AI on our own unlabeled VitalPatch data" — High impact,
Med-High effort) from the internal review, combined into one component:

  * `ECGEncoder` is a small 1D-conv encoder (Option C / "pretrained ECG
    expert model + small AI" from the review's comparison table) that
    reads the raw beat waveform directly, instead of the 25 handcrafted
    summary numbers, so morphological detail is no longer thrown away
    before it reaches a classifier or the LLM.
  * It is pretrained with a masked-reconstruction objective (mask random
    spans of the waveform, learn to reconstruct them) — a standard
    self-supervised recipe that needs zero labels, only raw ECG. This lets
    us pretrain directly on this pipeline's own unlabeled VitalPatch/SeNSiO
    recordings today, before any public labeled dataset (MITDB, Icentia11k,
    ...) is available, and later fine-tune a classification head once
    labels exist.
  * The encoder is deliberately tiny (a few hundred KB of weights) so it
    stays edge-deployable on the same Jetson-class hardware as the
    existing pipeline, per the review's "Runs on small device?" column for
    Option C.

The resulting embedding is consumed by:
  - `classify.py` (a classifier head on top, once labeled data exists)
  - `similar_cases.py` (nearest-neighbour "similar past patient" retrieval,
    recommendation #8)
  - `report.py` (embedding-derived summary handed to MedGemma alongside
    the handcrafted features, so the LLM's input is no longer text-only)
"""

EMBED_DIM = 32
INPUT_LEN = 125  # samples: 1 s wide window @ 125 Hz


class ECGEncoder(nn.Module):
    """Conv1D encoder: raw 125-sample beat window -> EMBED_DIM embedding.

    ~30K parameters — small enough to run on the same low-power edge
    hardware as the existing student model (An et al. 2024 style budget).
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.BatchNorm1d(16), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(32, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, INPUT_LEN) -> (batch, 1, INPUT_LEN)
        h = self.net(x.unsqueeze(1)).squeeze(-1)
        return self.proj(h)


class ReconstructionDecoder(nn.Module):
    """Lightweight decoder used only during self-supervised pretraining;
    discarded afterwards (only the encoder is deployed)."""

    def __init__(self, embed_dim: int = EMBED_DIM, output_len: int = INPUT_LEN):
        super().__init__()
        self.output_len = output_len
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.GELU(),
            nn.Linear(64, output_len),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


def random_mask(batch: torch.Tensor, mask_frac: float = 0.25, span: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero out random contiguous spans of each waveform; return the
    masked input and a boolean mask (True = masked, to be reconstructed)."""
    b, length = batch.shape
    mask = torch.zeros_like(batch, dtype=torch.bool)
    n_spans = max(1, int(length * mask_frac / span))
    for i in range(b):
        for _ in range(n_spans):
            start = np.random.randint(0, max(1, length - span))
            mask[i, start:start + span] = True
    masked = batch.clone()
    masked[mask] = 0.0
    return masked, mask


@dataclass
class PretrainResult:
    epochs: int
    final_loss: float
    n_windows: int


def pretrain_self_supervised(windows: np.ndarray, epochs: int = 20, batch_size: int = 64,
                              lr: float = 1e-3, mask_frac: float = 0.25,
                              device: str = "cpu") -> tuple[ECGEncoder, PretrainResult]:
    """Masked-reconstruction pretraining on raw, UNLABELED beat windows.

    `windows` must be shape (n, INPUT_LEN), already filtered/normalized
    (robust z-score) by stages 4/6. No labels required — this is exactly
    what lets us pretrain on this pipeline's own VitalPatch/SeNSiO data before
    any labeled public dataset is downloaded.
    """
    if len(windows) == 0:
        raise ValueError("No windows provided for pretraining")

    encoder = ECGEncoder().to(device)
    decoder = ReconstructionDecoder().to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    loss_fn = nn.MSELoss()

    data = torch.tensor(windows, dtype=torch.float32, device=device)
    n = len(data)
    final_loss = float("nan")

    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_losses = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch = data[idx]
            masked, mask = random_mask(batch, mask_frac=mask_frac)
            z = encoder(masked)
            recon = decoder(z)
            loss = loss_fn(recon[mask], batch[mask]) if mask.any() else loss_fn(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        final_loss = float(np.mean(epoch_losses))

    return encoder, PretrainResult(epochs=epochs, final_loss=final_loss, n_windows=n)


def save_encoder(encoder: ECGEncoder, path: Path, meta: dict | None = None) -> None:
    path = Path(path)
    torch.save(encoder.state_dict(), path)
    if meta is not None:
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def load_encoder(path: Path) -> ECGEncoder:
    encoder = ECGEncoder()
    encoder.load_state_dict(torch.load(path, map_location="cpu"))
    encoder.eval()
    return encoder


@torch.no_grad()
def embed_windows(encoder: ECGEncoder, windows: np.ndarray, device: str = "cpu") -> np.ndarray:
    if len(windows) == 0:
        return np.zeros((0, EMBED_DIM))
    encoder.eval()
    data = torch.tensor(windows, dtype=torch.float32, device=device)
    return encoder(data).cpu().numpy()


# ============================================================================
# classify.py — Stage 7: Beat and rhythm classification
# ============================================================================
"""Stage 7 — Beat and rhythm classification.

Two-pass design (fast binary triage, then a 5-class model), matching the
baseline architecture, plus a rhythm-context layer that enforces
physiological plausibility rules a single-beat classifier can't know on
its own (VT runs, bigeminy, trigeminy).

Recommendation #7 ("Replace beat-shape matching with a trained AI
classifier"): `FiveClassBeatClassifier` is a real trainable XGBoost model
that fits directly on `fit()`. Until labeled beats exist, `predict_proba`
transparently falls back to `RuleBasedBeatClassifier` — a heuristic,
no-training-data-needed stand-in for the old template-matching approach —
and every prediction is tagged with which path produced it
(`source: "trained_model" | "rule_based_fallback"`). This tagging is what
recommendation #10 ("Separate 'what the AI decided' from 'what the safety
rules changed'") requires downstream in `report.py`: nothing here is ever
silently presented as a trained model's opinion when it isn't one.
"""

BINARY_INPUT_LEN = 125  # 1 s wide window @ 125 Hz


class BinaryGateCNN(nn.Module):
    """Pass 1: fast Normal-vs-Arrhythmia triage. p > 0.85 -> Normal, skip
    Pass 2. Architecture mirrors the baseline (Conv1D -> MaxPool -> Conv1D
    -> MaxPool -> Dense -> sigmoid); untrained until fit on labeled data."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=9, padding=4), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.unsqueeze(1)).squeeze(-1)
        return torch.sigmoid(self.fc(h)).squeeze(-1)


@dataclass
class ClassificationResult:
    label: str
    probabilities: dict
    source: str  # "trained_model" | "rule_based_fallback"
    escalated_to_pass2: bool


class RuleBasedBeatClassifier:
    """No-training-data-needed heuristic fallback (transparent replacement
    for the old opaque template-matching approach). Uses RR prematurity
    and QRS width/amplitude from the primary window — classic, explainable
    criteria, not a trained model. Always reports low confidence and
    `source="rule_based_fallback"` so downstream stages never mistake this
    for a real classifier's judgement.
    """

    def classify(self, beat: Beat, mean_rr_ms: float) -> ClassificationResult:
        if beat.primary_window is None:
            return ClassificationResult("Q", {"Q": 1.0}, "rule_based_fallback", False)

        window = beat.primary_window
        qrs_width = float(np.ptp(np.where(np.abs(window) > np.std(window))[0])) if np.std(window) > 0 else 0.0
        amplitude = float(np.ptp(window))

        premature = (beat.rr_pre_ms is not None and mean_rr_ms > 0
                     and beat.rr_pre_ms < 0.85 * mean_rr_ms)
        wide_qrs = qrs_width > 0.5 * len(window)

        if premature and wide_qrs:
            label, conf = "V", 0.55
        elif premature:
            label, conf = "S", 0.5
        elif amplitude < 0.05 * (np.std(window) + 1e-9):
            label, conf = "Q", 0.4
        else:
            label, conf = "N", 0.6

        remainder = (1.0 - conf) / (len(AAMI_CLASSES) - 1)
        probs = {c: (conf if c == label else remainder) for c in AAMI_CLASSES}
        return ClassificationResult(label, probs, "rule_based_fallback", False)


class FiveClassBeatClassifier:
    """AAMI 5-class classifier (N/S/V/F/Q). Trainable via `fit()` on
    (feature_matrix, labels) once a labeled dataset (MITDB/Icentia11k/...)
    is available; falls back to `RuleBasedBeatClassifier` until then.
    """

    def __init__(self):
        self.model = None
        self._fallback = RuleBasedBeatClassifier()

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def fit(self, X: np.ndarray, y: list[str], sample_weight: np.ndarray | None = None,
            random_state: int = 42, class_weight_multiplier: dict[str, float] | None = None):
        """`random_state` is fixed (not left to XGBoost's default) so two
        runs with identical inputs produce byte-identical DS2 metrics —
        needed for the validation-set ablations in train_classifiers.py to
        be trustworthy (a result can't be "better" if it's actually just
        run-to-run noise).

        CORRECTION (2026-07-16): `random_state` alone was found NOT
        sufficient for that guarantee. A morphology-only ablation run scored
        V F1 0.866 vs production's freshly-reproduced 0.826 under an
        otherwise-identical nominal config (same seed, same data, same
        recipe) -- traced to XGBoost's histogram-based split-finding being
        thread-count-dependent: floating-point summation order in gradient
        histograms can differ across thread counts / hardware even with a
        fixed random_state, since XGBoost's determinism guarantee is
        conditional on a fixed `n_jobs` too, not just `random_state`. Pinned
        `n_jobs=1` below (verified via two back-to-back identical runs
        producing byte-identical DS2 metrics -- see ABLATION_REPORT.md's
        "Prerequisite 1" section) to remove this variable. Training is
        somewhat slower single-threaded, but reproducibility takes priority
        over speed here (AGENT_RULES.md rule 5).

        `class_weight_multiplier` (optional, e.g. {"F": 3.0}) is applied on
        top of `sample_weight` per-class — the F-specific misclassification
        cost knob used to fight F->S absorption, kept separate from the
        general ROS/balanced-weight machinery in train_classifiers.py so
        each can be ablated independently.
        """
        import xgboost as xgb
        from sklearn.preprocessing import LabelEncoder

        # Fit on whichever classes actually appear in y, not the fixed
        # 5-class AAMI_CLASSES list — if a class (e.g. Q) has been
        # deliberately dropped from training, XGBoost's sklearn wrapper
        # requires the encoded labels to be a contiguous 0..k-1 range for
        # however many classes k are actually present, or .fit() raises
        # "Invalid classes inferred from unique values of y".
        present_classes = sorted(set(y))
        self._label_encoder = LabelEncoder().fit(present_classes)
        y_enc = self._label_encoder.transform(y)

        if class_weight_multiplier:
            if sample_weight is None:
                sample_weight = np.ones(len(y), dtype=float)
            sample_weight = sample_weight.copy()
            y_arr = np.array(y)
            for cls, mult in class_weight_multiplier.items():
                sample_weight[y_arr == cls] *= mult

        self.model = xgb.XGBClassifier(n_estimators=164, max_depth=11,
                                        objective="multi:softprob", num_class=len(present_classes),
                                        random_state=random_state, n_jobs=1)
        self.model.fit(X, y_enc, sample_weight=sample_weight)
        return self

    def predict_one(self, feature_vec: np.ndarray | None, beat: Beat, mean_rr_ms: float) -> ClassificationResult:
        if self.model is not None and feature_vec is not None:
            proba = self.model.predict_proba(feature_vec.reshape(1, -1))[0]
            classes = self._label_encoder.inverse_transform(np.arange(len(proba)))
            probs = dict(zip(classes, proba.tolist()))
            label = max(probs, key=probs.get)
            return ClassificationResult(label, probs, "trained_model", True)
        return self._fallback.classify(beat, mean_rr_ms)

    def save(self, path: Path):
        if self.model is not None:
            self.model.save_model(str(path))
            classes_path = Path(path).with_suffix(".classes.json")
            classes_path.write_text(json.dumps(list(self._label_encoder.classes_)))

    def load(self, path: Path):
        import xgboost as xgb
        from sklearn.preprocessing import LabelEncoder
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))
        classes_path = Path(path).with_suffix(".classes.json")
        present_classes = json.loads(classes_path.read_text()) if classes_path.exists() else AAMI_CLASSES
        self._label_encoder = LabelEncoder().fit(present_classes)


@dataclass
class RhythmFinding:
    kind: str            # "VT_RUN" | "BIGEMINY" | "TRIGEMINY" | "AFIB_SUSPECTED"
    start_beat_idx: int
    end_beat_idx: int
    detail: dict


class RhythmContextEngine:
    """Looks at the sequence of beat labels (last N beats) and enforces
    physiological plausibility a single-beat classifier can't know on its
    own — deterministic rules, no training data required.
    """

    def __init__(self, vt_run_beats: int = RISK.vt_run_beats):
        self.vt_run_beats = vt_run_beats

    def analyze(self, labels: list[str], rr_ms: list[float | None]) -> list[RhythmFinding]:
        findings: list[RhythmFinding] = []
        findings += self._vt_runs(labels)
        findings += self._geminy(labels, pattern=["N", "V"], min_repeats=4, kind="BIGEMINY")
        findings += self._geminy(labels, pattern=["N", "N", "V"], min_repeats=3, kind="TRIGEMINY")
        findings += self._afib_suspected(rr_ms)
        return findings

    def _vt_runs(self, labels: list[str]) -> list[RhythmFinding]:
        findings, run_start, run_len = [], None, 0
        for i, lab in enumerate(labels + [None]):
            if lab == "V":
                if run_start is None:
                    run_start = i
                run_len += 1
            else:
                if run_len >= self.vt_run_beats:
                    findings.append(RhythmFinding("VT_RUN", run_start, run_start + run_len - 1,
                                                   {"beat_count": run_len}))
                run_start, run_len = None, 0
        return findings

    def _geminy(self, labels: list[str], pattern: list[str], min_repeats: int, kind: str) -> list[RhythmFinding]:
        p_len = len(pattern)
        findings = []
        i = 0
        while i + p_len * min_repeats <= len(labels):
            repeats = 0
            j = i
            while j + p_len <= len(labels) and labels[j:j + p_len] == pattern:
                repeats += 1
                j += p_len
            if repeats >= min_repeats:
                findings.append(RhythmFinding(kind, i, j - 1, {"repeats": repeats}))
                i = j
            else:
                i += 1
        return findings

    def _afib_suspected(self, rr_ms: list[float | None], window: int = 20,
                         cv_threshold: float = 0.15) -> list[RhythmFinding]:
        """AFib is a RHYTHM finding computed here from RR irregularity —
        never treated as a quality defect (recommendation #6). High RR
        coefficient-of-variation over a rolling window suggests AFib,
        distinct from motion-artefact noise which the stage-2 SQI gate
        already screened out via morphology, not RR timing.
        """
        findings = []
        clean_rr = [r for r in rr_ms if r is not None]
        for start in range(0, max(0, len(clean_rr) - window), window):
            chunk = np.array(clean_rr[start:start + window])
            if len(chunk) < window:
                continue
            cv = float(np.std(chunk) / np.mean(chunk)) if np.mean(chunk) > 0 else 0.0
            if cv > cv_threshold:
                findings.append(RhythmFinding("AFIB_SUSPECTED", start, start + window - 1, {"rr_cv": cv}))
        return findings


# ============================================================================
# risk.py — Stage 8: Risk scoring, conformal confidence ranges, temporal tracking
# ============================================================================
"""Stage 8 — Risk scoring, conformal confidence ranges, and temporal
risk tracking.

Deterministic risk metrics (PVC/PAC burden, VT runs, AFib burden, HRV
suppression, QRS-width trend) mirror the baseline. Two additions from the
review's top recommendations:

  * `ConformalRiskPredictor` (#4, "Add statistically guaranteed confidence
    ranges" — High impact, Low effort): wraps any per-window risk-level
    probability estimate in a split-conformal prediction set with a
    marginal coverage guarantee, so the system can say "90% sure this is
    at least HIGH risk" instead of a bare point estimate — important for
    medical approval / regulatory defensibility.
  * `TemporalRiskTracker` (#5, "Track risk over time, not just one
    snapshot" — High impact, Medium effort): keeps a rolling history of
    risk scores per patient and flags a sustained upward trend (patient
    slowly deteriorating over hours), not just the current instant.
"""


@dataclass
class RiskReport:
    pvc_burden_pct: float
    pac_burden_pct: float
    vt_run_count: int
    afib_burden_pct: float
    hrv_suppressed: bool
    qrs_width_trend: float
    news2_score: int | None
    qsofa_score: int | None
    alert_level: str
    alert_reasons: list[str]


def score_recording(labels: list[str], findings: list[RhythmFinding], hrv: dict,
                     news2_score: int | None = None, qsofa_score: int | None = None,
                     thresholds=RISK) -> RiskReport:
    n = max(1, len(labels))
    pvc_burden = 100.0 * sum(1 for l in labels if l == "V") / n
    pac_burden = 100.0 * sum(1 for l in labels if l == "S") / n
    vt_runs = sum(1 for f in findings if f.kind == "VT_RUN")
    afib_windows = [f for f in findings if f.kind == "AFIB_SUSPECTED"]
    afib_burden = 100.0 * len(afib_windows) / max(1, len(findings)) if findings else (
        100.0 if afib_windows else 0.0)
    sdnn_ms = hrv.get("sdnn_ms", 0.0)
    hrv_suppressed = sdnn_ms < thresholds.hrv_sdnn_suppressed_ms

    reasons = []
    level = "LOW"

    if pvc_burden > thresholds.pvc_burden_critical_pct:
        level, reasons = "CRITICAL", reasons + [
            f"PVC burden {pvc_burden:.1f}% > critical threshold ({thresholds.pvc_burden_critical_pct:.1f}%)"]
    elif vt_runs > 0:
        level, reasons = "CRITICAL", reasons + [
            f"{vt_runs} ventricular-tachycardia run(s) detected "
            f"(threshold: >={thresholds.vt_run_beats} consecutive V beats = 1 run)"]
    elif pvc_burden > thresholds.pvc_burden_high_pct:
        level, reasons = "HIGH", reasons + [
            f"PVC burden {pvc_burden:.1f}% > high threshold ({thresholds.pvc_burden_high_pct:.1f}%)"]
    elif pac_burden > thresholds.pac_burden_high_pct or afib_burden > thresholds.afib_burden_high_pct:
        level, reasons = "HIGH", reasons + [
            f"PAC burden {pac_burden:.1f}% (threshold >{thresholds.pac_burden_high_pct:.1f}%) / "
            f"AFib burden {afib_burden:.1f}% (threshold >{thresholds.afib_burden_high_pct:.1f}%)"]
    elif hrv_suppressed:
        level, reasons = "MEDIUM", reasons + [
            f"Sustained HRV suppression: SDNN {sdnn_ms:.1f}ms < threshold "
            f"({thresholds.hrv_sdnn_suppressed_ms:.1f}ms)"]
    else:
        reasons = ["No thresholds exceeded"]

    if (news2_score is not None and news2_score >= thresholds.news2_critical_threshold
            and RISK_LEVELS.index(level) < RISK_LEVELS.index("CRITICAL")):
        level, reasons = "CRITICAL", reasons + [
            f"NEWS2 {news2_score} >= critical threshold ({thresholds.news2_critical_threshold})"]
    if (qsofa_score is not None and qsofa_score >= thresholds.qsofa_high_threshold
            and RISK_LEVELS.index(level) < RISK_LEVELS.index("HIGH")):
        level = "HIGH" if RISK_LEVELS.index(level) < RISK_LEVELS.index("HIGH") else level
        reasons = reasons + [
            f"qSOFA {qsofa_score} >= high threshold ({thresholds.qsofa_high_threshold})"]

    return RiskReport(
        pvc_burden_pct=pvc_burden, pac_burden_pct=pac_burden, vt_run_count=vt_runs,
        afib_burden_pct=afib_burden, hrv_suppressed=hrv_suppressed,
        qrs_width_trend=hrv.get("qrs_width_trend", 0.0),
        news2_score=news2_score, qsofa_score=qsofa_score,
        alert_level=level, alert_reasons=reasons,
    )


# RISK_LEVELS' lowest tier is "LOW"; MedGemma-Agent's AlertLevel enum (see
# vitals.schemas.AlertLevel / ECG_TO_AGENT_LEVEL) has no "LOW" — its lowest
# tier is "NORMAL". This is the one place that mapping is defined on this
# side of the integration, so it can't drift from the agent's copy silently.
_AGENT_ALERT_LEVEL = {"LOW": "NORMAL", "MEDIUM": "MEDIUM", "HIGH": "HIGH", "CRITICAL": "CRITICAL"}


def to_agent_ecg_risk_summary(risk_report: RiskReport) -> dict:
    """Build a plain dict matching MedGemma-Agent's `ECGRiskSummary` schema
    from a `RiskReport` — the wire-format contract for the additive
    ECG-into-vitals-agent integration (agent repo: `vitals/schemas.py`,
    `guardrails/clinical_rules.escalate_with_ecg`). Deliberately a plain
    dict, not a shared class: the two systems are separate services and
    should only share a JSON contract, not a Python import dependency.
    """
    return {
        "alert_level": _AGENT_ALERT_LEVEL[risk_report.alert_level],
        "alert_reasons": list(risk_report.alert_reasons),
        "pvc_burden_pct": risk_report.pvc_burden_pct,
        "pac_burden_pct": risk_report.pac_burden_pct,
        "vt_run_count": risk_report.vt_run_count,
        "afib_burden_pct": risk_report.afib_burden_pct,
        "hrv_suppressed": risk_report.hrv_suppressed,
        "source": "ecg_pipeline",
    }


class ConformalRiskPredictor:
    """Split-conformal prediction over the 4 risk levels.

    Usage:
      1. `calibrate(scores, true_labels)` once a calibration set exists
         (held-out recordings with known outcomes/labels).
      2. `predict_set(scores)` at inference time returns the *set* of
         risk levels consistent with (1 - alpha) coverage, not a single
         point estimate.

    Until `calibrate()` has enough data (`min_calibration_size`), returns
    the full label set (maximally conservative — "not enough evidence to
    narrow this down yet") rather than a false guarantee.
    """

    def __init__(self, cfg: ConformalConfig = CONFORMAL):
        self.cfg = cfg
        self._qhat: float | None = None

    @property
    def is_calibrated(self) -> bool:
        return self._qhat is not None

    def calibrate(self, softmax_scores: np.ndarray, true_label_idx: np.ndarray) -> None:
        """softmax_scores: (n, 4) predicted probability per risk level.
        true_label_idx: (n,) integer index of the true risk level."""
        if len(softmax_scores) < self.cfg.min_calibration_size:
            raise ValueError(
                f"Need >= {self.cfg.min_calibration_size} calibration examples, got {len(softmax_scores)}")
        nonconformity = 1.0 - softmax_scores[np.arange(len(true_label_idx)), true_label_idx]
        n = len(nonconformity)
        q_level = np.ceil((n + 1) * (1 - self.cfg.alpha)) / n
        self._qhat = float(np.quantile(nonconformity, min(q_level, 1.0)))

    def predict_set(self, softmax_scores: np.ndarray) -> list[str]:
        if not self.is_calibrated:
            return list(RISK_LEVELS)  # no false guarantee: return the full set
        keep = 1.0 - softmax_scores <= self._qhat
        levels = [RISK_LEVELS[i] for i in range(len(RISK_LEVELS)) if keep[i]]
        return levels or [RISK_LEVELS[int(np.argmax(softmax_scores))]]


@dataclass
class TemporalSnapshot:
    timestamp_ms: float
    risk_score_ordinal: int  # index into RISK_LEVELS
    alert_level: str


class TemporalRiskTracker:
    """Rolling per-patient risk history + trend detection.

    Flags a sustained upward slope over `window_minutes`, catching a
    patient who is slowly deteriorating across several snapshots even
    though no single snapshot alone crosses an alert threshold.
    """

    def __init__(self, cfg: TemporalTrackingConfig = TEMPORAL):
        self.cfg = cfg
        self._history: dict[str, deque] = {}

    def record(self, patient_id: str, timestamp_ms: float, alert_level: str) -> None:
        history = self._history.setdefault(patient_id, deque(maxlen=self.cfg.history_max_windows))
        history.append(TemporalSnapshot(timestamp_ms, RISK_LEVELS.index(alert_level), alert_level))

    def trend(self, patient_id: str) -> dict:
        history = self._history.get(patient_id)
        if not history or len(history) < 3:
            return {"slope_per_minute": 0.0, "worsening": False, "n_snapshots": len(history or [])}

        window_ms = self.cfg.window_minutes * 60_000
        latest_t = history[-1].timestamp_ms
        recent = [s for s in history if latest_t - s.timestamp_ms <= window_ms]
        if len(recent) < 3:
            recent = list(history)[-3:]

        t_min = np.array([s.timestamp_ms for s in recent]) / 60_000.0
        y = np.array([s.risk_score_ordinal for s in recent], dtype=float)
        slope = float(np.polyfit(t_min - t_min[0], y, 1)[0]) if len(set(t_min)) > 1 else 0.0

        return {
            "slope_per_minute": slope,
            "worsening": slope > self.cfg.trend_slope_alert_threshold,
            "n_snapshots": len(recent),
        }


# ============================================================================
# similar_cases.py — Recommendation #8: similar past patient cases
# ============================================================================
"""Recommendation #8 — 'Show the AI similar past patient cases before it
answers' (Med-High impact, High effort in the original ranking, but made
tractable here by reusing the encoder embeddings we already compute for
recommendation #1, rather than standing up separate infrastructure).

`SimilarCaseIndex` is a lightweight nearest-neighbour store over encoder
embeddings (see `encoder.py`) plus each case's outcome metadata. It grounds
the stage-9 report in real precedent instead of the LLM guessing from the
current recording alone. Runs locally (no server dependency) via
scikit-learn's NearestNeighbors, trading the review's "no (needs a
server)" limitation for a smaller, in-process index.
"""


@dataclass
class CaseRecord:
    case_id: str
    patient_id: str
    embedding: np.ndarray
    outcome_label: str          # e.g. final alert_level for that recording/beat
    metadata: dict = field(default_factory=dict)


class SimilarCaseIndex:
    def __init__(self):
        self._records: list[CaseRecord] = []
        self._nn = None

    def add(self, record: CaseRecord) -> None:
        self._records.append(record)
        self._nn = None  # invalidate, rebuild lazily

    def add_many(self, records: list[CaseRecord]) -> None:
        self._records.extend(records)
        self._nn = None

    def _ensure_index(self):
        from sklearn.neighbors import NearestNeighbors
        if self._nn is None and self._records:
            X = np.vstack([r.embedding for r in self._records])
            self._nn = NearestNeighbors(n_neighbors=min(5, len(self._records)), metric="cosine").fit(X)

    def query(self, embedding: np.ndarray, k: int = 5) -> list[tuple[CaseRecord, float]]:
        self._ensure_index()
        if self._nn is None:
            return []
        k = min(k, len(self._records))
        dist, idx = self._nn.kneighbors(embedding.reshape(1, -1), n_neighbors=k)
        return [(self._records[i], float(d)) for i, d in zip(idx[0], dist[0])]

    def __len__(self) -> int:
        return len(self._records)

    def save(self, path: Path) -> None:
        import pickle
        Path(path).write_bytes(pickle.dumps(self._records))

    def load(self, path: Path) -> None:
        import pickle
        self._records = pickle.loads(Path(path).read_bytes())
        self._nn = None


# ============================================================================
# report.py — Stage 9: MedGemma clinical report
# ============================================================================
"""Stage 9 — MedGemma clinical report.

Recommendation #3 ("Teach the small AI to reason step-by-step, not just
answer" — High impact): the prompt requires MedGemma to emit its reasoning
chain before the final JSON verdict, not just a bare label, so its
decisions are explainable to clinicians and so any later student-model
distillation (see PROJECT_OVERVIEW.md's distillation plan) can be trained
on the reasoning trace, not just the answer.

Recommendation #10 ("Separate 'what the AI decided' from 'what the safety
rules changed'" — Medium impact, Low effort): `MergedDecision` keeps
`deterministic_decision`, `llm_decision`, and `final_decision` as three
distinct fields rather than collapsing them, so it's always possible to
see how much the LLM actually changed vs. how much came from fixed rules.

Safety (unchanged from baseline): CRITICAL alerts bypass the LLM entirely.
MedGemma may only RAISE an alert level, never lower a CRITICAL produced by
deterministic rules. If MedGemma disagrees with deterministic scoring by
more than one severity level, its output is rejected and the system falls
back to rule-based-only mode. Every decision — including any rejection —
is written to the SHA-256 hash-chained audit log.
"""

OLLAMA_URL = "http://localhost:11434/api/generate"
MEDGEMMA_MODEL = "medgemma"
CRITICAL_LATENCY_TARGET_MS = 500


PROMPT_TEMPLATE = """You are assisting clinical staff monitoring a post-operative patient's ECG.

## Deterministic ECG risk summary
{ecg_summary}

## Rolling risk trend (last {trend_window} minutes)
{trend_summary}

## Similar past cases (nearest neighbours by waveform embedding)
{similar_cases_summary}

## EHR context
{ehr_context}

## Instructions
Think through this step by step BEFORE giving your final answer:
1. Summarize what the deterministic ECG metrics indicate.
2. Note any disagreement between the metrics and the similar past cases.
3. State your reasoning for a risk level.
4. Only then give the final answer.

You may only RAISE the deterministic risk level below, never lower it:
  deterministic_alert_level = {deterministic_level}

Respond with your step-by-step reasoning, then a final JSON object of the form:
{{"risk_level": "LOW|MEDIUM|HIGH|CRITICAL", "narrative": "...", "disclaimer": "..."}}
"""


@dataclass
class MergedDecision:
    deterministic_decision: dict
    llm_decision: dict | None
    llm_rejected_reason: str | None
    final_decision: dict
    bypassed_llm: bool


def build_prompt(risk_report: RiskReport, trend: dict, similar_cases_summary: str,
                  ehr_context: str = "Not available", trend_window_minutes: float = 15.0) -> str:
    ecg_summary = (
        f"PVC burden: {risk_report.pvc_burden_pct:.1f}% | PAC burden: {risk_report.pac_burden_pct:.1f}% | "
        f"VT runs: {risk_report.vt_run_count} | AFib burden: {risk_report.afib_burden_pct:.1f}% | "
        f"HRV suppressed: {risk_report.hrv_suppressed} | QRS width trend: {risk_report.qrs_width_trend:.2f} | "
        f"NEWS2: {risk_report.news2_score} | qSOFA: {risk_report.qsofa_score}"
    )
    trend_summary = (
        f"Slope: {trend.get('slope_per_minute', 0.0):.3f} risk-levels/min | "
        f"Worsening: {trend.get('worsening', False)} | Snapshots: {trend.get('n_snapshots', 0)}"
    )
    return PROMPT_TEMPLATE.format(
        ecg_summary=ecg_summary, trend_window=trend_window_minutes, trend_summary=trend_summary,
        similar_cases_summary=similar_cases_summary, ehr_context=ehr_context,
        deterministic_level=risk_report.alert_level,
    )


def call_medgemma(prompt: str, model: str = MEDGEMMA_MODEL, timeout_s: float = 10.0) -> dict | None:
    """Calls a locally-deployed MedGemma via Ollama's HTTP API. Returns
    None (not raises) if Ollama isn't reachable, so the pipeline degrades
    to rule-based-only mode instead of crashing — this is itself logged
    to the audit trail by the caller."""
    try:
        resp = requests.post(OLLAMA_URL, json={"model": model, "prompt": prompt, "stream": False},
                              timeout=timeout_s)
        resp.raise_for_status()
        text = resp.json().get("response", "")
    except (requests.RequestException, ValueError):
        return None

    try:
        json_start = text.rindex("{")
        parsed = json.loads(text[json_start:])
        parsed["_raw_reasoning"] = text[:json_start].strip()
        return parsed
    except (ValueError, json.JSONDecodeError):
        return None


def merge_decision(risk_report: RiskReport, llm_output: dict | None, audit: AuditLog) -> MergedDecision:
    deterministic = {"risk_level": risk_report.alert_level, "reasons": risk_report.alert_reasons}

    if risk_report.alert_level == "CRITICAL":
        audit.append("MEDGEMMA_BYPASSED", {"reason": "CRITICAL alert bypasses LLM",
                                            "latency_target_ms": CRITICAL_LATENCY_TARGET_MS})
        return MergedDecision(deterministic, None, None, deterministic, bypassed_llm=True)

    if llm_output is None:
        audit.append("MEDGEMMA_UNAVAILABLE", {"fallback": "rule_based_only"})
        return MergedDecision(deterministic, None, "MedGemma unavailable", deterministic, bypassed_llm=False)

    llm_level = llm_output.get("risk_level")
    if llm_level not in RISK_LEVELS:
        audit.append("MEDGEMMA_REJECTED", {"reason": "invalid risk_level in response", "raw": llm_output})
        return MergedDecision(deterministic, llm_output, "invalid risk_level", deterministic, bypassed_llm=False)

    det_idx, llm_idx = RISK_LEVELS.index(risk_report.alert_level), RISK_LEVELS.index(llm_level)

    if abs(llm_idx - det_idx) > 1:
        audit.append("MEDGEMMA_REJECTED", {"reason": "disagreement > 1 severity level",
                                            "deterministic": risk_report.alert_level, "llm": llm_level})
        return MergedDecision(deterministic, llm_output, "disagreement > 1 severity level", deterministic,
                               bypassed_llm=False)

    final_level = RISK_LEVELS[max(det_idx, llm_idx)]  # LLM may only raise, never lower
    final = {"risk_level": final_level, "narrative": llm_output.get("narrative", ""),
              "disclaimer": llm_output.get("disclaimer", "This is not a substitute for clinical judgement."),
              "reasoning": llm_output.get("_raw_reasoning", "")}
    audit.append("MEDGEMMA_ACCEPTED", {"deterministic": risk_report.alert_level, "llm": llm_level,
                                        "final": final_level})
    return MergedDecision(deterministic, llm_output, None, final, bypassed_llm=False)


def generate_report(risk_report: RiskReport, trend: dict, similar_cases_summary: str,
                     audit: AuditLog, ehr_context: str = "Not available") -> MergedDecision:
    if risk_report.alert_level == "CRITICAL":
        return merge_decision(risk_report, None, audit)

    prompt = build_prompt(risk_report, trend, similar_cases_summary, ehr_context)
    audit.append("MEDGEMMA_PROMPT_BUILT", {"prompt_chars": len(prompt)})
    llm_output = call_medgemma(prompt)
    return merge_decision(risk_report, llm_output, audit)


# ============================================================================
# pipeline.py — Orchestrator: wires stages 1-9 together end to end
# ============================================================================
"""Orchestrator: wires stages 1-9 together end to end.

A beat or window that fails a quality check is rejected and logged rather
than silently dropped, so the whole run is auditable end to end.
"""


@dataclass
class PipelineResult:
    recording: Recording
    n_raw_samples: int
    n_kept_samples: int
    sqi_rejection_rate: float
    beats: list[Beat]
    n_beats_accepted: int
    n_beats_rejected: int
    beat_labels: list[str]
    embeddings: np.ndarray | None
    rhythm_findings: list
    risk_report: object
    temporal_trend: dict
    merged_decision: object
    audit: AuditLog


class ECGPipeline:
    def __init__(self, encoder: ECGEncoder | None = None,
                 classifier: FiveClassBeatClassifier | None = None,
                 similar_case_index: SimilarCaseIndex | None = None):
        self.encoder = encoder
        self.classifier = classifier or FiveClassBeatClassifier()
        self.rhythm_engine = RhythmContextEngine()
        self.conformal = ConformalRiskPredictor()
        self.temporal_tracker = TemporalRiskTracker()
        self.similar_case_index = similar_case_index or SimilarCaseIndex()

    def run(self, recording: Recording, clip_value: float | None = None,
            news2_score: int | None = None, qsofa_score: int | None = None) -> PipelineResult:
        audit = AuditLog()
        audit.append("STAGE1_INGEST", {"source": recording.source, "patient_id": recording.patient_id,
                                        "segment_id": recording.segment_id, "n_samples": len(recording.signal_mv),
                                        "gaps_flagged": len(recording.gaps)})

        # Stage 2: SQI gate (before any filtering)
        keep_mask, verdicts = run_sqi_gate(recording.signal_mv, recording.timestamps_ms,
                                            recording.fs_nominal, clip_value=clip_value)
        rejection_rate_value = rejection_rate(verdicts)
        audit.append("STAGE2_SQI_GATE", {"n_windows": len(verdicts), "rejection_rate": rejection_rate_value,
                                          "reject_codes": [v.reject_code for v in verdicts if not v.passed]})

        signal_clean = recording.signal_mv.copy()
        signal_clean[~keep_mask] = np.nan

        # Stage 3: resample to common rate
        resampled, t_resampled = to_target_rate(signal_clean, recording.timestamps_ms,
                                                 recording.fs_nominal, TARGET_FS)
        audit.append("STAGE3_RESAMPLE", {"fs_in": recording.fs_nominal, "fs_out": TARGET_FS,
                                          "n_out": len(resampled)})

        valid = ~np.isnan(resampled)
        if len(resampled) == 0 or not valid.any() or valid.sum() < int(TARGET_FS * 2):
            audit.append("SEGMENT_REJECTED_INSUFFICIENT_DATA",
                          {"n_resampled": len(resampled), "n_valid": int(valid.sum())})
            return self._insufficient_data_result(recording, keep_mask, rejection_rate_value, audit)
        resampled_filled = np.interp(t_resampled, t_resampled[valid], resampled[valid])

        # Stage 4: filter chain
        filtered = apply_filter_chain(resampled_filled, TARGET_FS,
                                       already_bandpass_filtered=recording.already_bandpass_filtered)
        audit.append("STAGE4_FILTER", {"already_bandpass_filtered": recording.already_bandpass_filtered})

        # Stage 5: R-peak detection + beat segmentation
        beats = detect_and_segment(filtered, TARGET_FS, BEATS)
        n_rejected = sum(1 for b in beats if b.quality_rejected)
        audit.append("STAGE5_BEATS", {"n_beats_detected": len(beats), "n_beats_rejected": n_rejected,
                                       "n_rr_flagged": sum(1 for b in beats if b.rr_flagged)})

        # Stage 6: features (handcrafted + optional learned embeddings)
        primary_pre_samples = int(round(BEATS.primary_pre_ms / 1000.0 * TARGET_FS))
        feature_matrix, feature_idxs = batch_feature_matrix(beats, primary_pre_samples)
        hrv = recording_level_hrv(beats)
        audit.append("STAGE6_FEATURES", {"n_feature_vectors": len(feature_idxs), "hrv": hrv})

        embeddings = None
        if self.encoder is not None:
            wide_windows = np.array([robust_zscore(b.wide_window) for b in beats
                                      if b.wide_window is not None and not b.quality_rejected])
            if len(wide_windows):
                embeddings = embed_windows(self.encoder, wide_windows)
            audit.append("STAGE6_ENCODER_EMBEDDING", {"n_embedded": 0 if embeddings is None else len(embeddings),
                                                        "encoder_available": True})
        else:
            audit.append("STAGE6_ENCODER_EMBEDDING", {"encoder_available": False,
                                                        "reason": "no pretrained encoder supplied"})

        # Stage 7: beat + rhythm classification
        mean_rr = float(np.mean([b.rr_post_ms for b in beats if b.rr_post_ms is not None])) \
            if any(b.rr_post_ms is not None for b in beats) else 0.0
        labels = []
        classifier_sources = set()
        feat_lookup = dict(zip(feature_idxs, feature_matrix))
        for i, beat in enumerate(beats):
            if beat.quality_rejected:
                labels.append("Q")
                continue
            result = self.classifier.predict_one(feat_lookup.get(i), beat, mean_rr)
            labels.append(result.label)
            classifier_sources.add(result.source)
        rr_list = [b.rr_post_ms for b in beats]
        rhythm_findings = self.rhythm_engine.analyze(labels, rr_list)
        audit.append("STAGE7_CLASSIFY", {"classifier_sources": sorted(classifier_sources),
                                          "classifier_trained": self.classifier.is_trained,
                                          "rhythm_findings": [f.kind for f in rhythm_findings]})

        # Stage 8: risk scoring + conformal + temporal
        risk_report = score_recording(labels, rhythm_findings, hrv, news2_score, qsofa_score)
        audit.append("STAGE8_RISK", {"alert_level": risk_report.alert_level, "reasons": risk_report.alert_reasons})

        self.temporal_tracker.record(recording.patient_id, float(t_resampled[-1]) if len(t_resampled) else 0.0,
                                      risk_report.alert_level)
        trend = self.temporal_tracker.trend(recording.patient_id)
        audit.append("STAGE8_TEMPORAL_TREND", trend)

        similar_summary = "No similar-case index available."
        if len(self.similar_case_index) and embeddings is not None and len(embeddings):
            neighbours = self.similar_case_index.query(embeddings.mean(axis=0), k=3)
            similar_summary = "; ".join(f"{r.outcome_label} (dist={d:.3f})" for r, d in neighbours) or similar_summary

        # Stage 9: MedGemma report
        merged = generate_report(risk_report, trend, similar_summary, audit)

        return PipelineResult(
            recording=recording, n_raw_samples=len(recording.signal_mv), n_kept_samples=int(keep_mask.sum()),
            sqi_rejection_rate=rejection_rate_value, beats=beats, n_beats_accepted=len(beats) - n_rejected,
            n_beats_rejected=n_rejected, beat_labels=labels, embeddings=embeddings,
            rhythm_findings=rhythm_findings, risk_report=risk_report, temporal_trend=trend,
            merged_decision=merged, audit=audit,
        )

    def _insufficient_data_result(self, recording: Recording, keep_mask: np.ndarray,
                                    rejection_rate: float, audit: AuditLog) -> PipelineResult:
        """A segment with too little surviving signal to run stages 4-9 on
        (e.g. a sub-2-second test recording, or one where the SQI gate
        rejected everything). Reported plainly rather than crashing or
        fabricating a risk score from no data."""
        risk_report = RiskReport(
            pvc_burden_pct=0.0, pac_burden_pct=0.0, vt_run_count=0, afib_burden_pct=0.0,
            hrv_suppressed=False, qrs_width_trend=0.0, news2_score=None, qsofa_score=None,
            alert_level="LOW", alert_reasons=["Insufficient signal survived quality gating / resampling"],
        )
        merged = MergedDecision(
            deterministic_decision={"risk_level": "LOW", "reasons": risk_report.alert_reasons},
            llm_decision=None, llm_rejected_reason="Insufficient data — LLM not invoked",
            final_decision={"risk_level": "LOW", "reasons": risk_report.alert_reasons}, bypassed_llm=False,
        )
        return PipelineResult(
            recording=recording, n_raw_samples=len(recording.signal_mv), n_kept_samples=int(keep_mask.sum()),
            sqi_rejection_rate=rejection_rate, beats=[], n_beats_accepted=0, n_beats_rejected=0,
            beat_labels=[], embeddings=None, rhythm_findings=[], risk_report=risk_report,
            temporal_trend={"slope_per_minute": 0.0, "worsening": False, "n_snapshots": 0},
            merged_decision=merged, audit=audit,
        )


# ============================================================================
# run_pipeline.py — CLI: run the full stage 1-9 pipeline on a real local recording
# ============================================================================
"""CLI: run the full stage 1-9 pipeline on a real local recording.

Usage:
    python -m ecg_pipeline.ecg_pipeline_core --source vitalpatch --limit 3
    python -m ecg_pipeline.ecg_pipeline_core --source sensio --limit 3
"""


def summarize(result) -> dict:
    return {
        "source": result.recording.source,
        "patient_id": result.recording.patient_id,
        "segment_id": result.recording.segment_id,
        "n_raw_samples": result.n_raw_samples,
        "n_kept_samples": result.n_kept_samples,
        "sqi_rejection_rate": round(result.sqi_rejection_rate, 3),
        "n_beats_detected": len(result.beats),
        "n_beats_accepted": result.n_beats_accepted,
        "n_beats_rejected": result.n_beats_rejected,
        "label_counts": {c: result.beat_labels.count(c) for c in set(result.beat_labels)},
        "rhythm_findings": [f.kind for f in result.rhythm_findings],
        "risk": {
            "alert_level": result.risk_report.alert_level,
            "reasons": result.risk_report.alert_reasons,
            "pvc_burden_pct": round(result.risk_report.pvc_burden_pct, 2),
            "pac_burden_pct": round(result.risk_report.pac_burden_pct, 2),
            "vt_run_count": result.risk_report.vt_run_count,
            "afib_burden_pct": round(result.risk_report.afib_burden_pct, 2),
        },
        "temporal_trend": result.temporal_trend,
        "final_decision": result.merged_decision.final_decision,
        "bypassed_llm": result.merged_decision.bypassed_llm,
        "llm_rejected_reason": result.merged_decision.llm_rejected_reason,
        "audit_chain_valid": result.audit.verify_chain(),
        "audit_n_entries": len(result.audit.to_list()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["vitalpatch", "sensio"], default="vitalpatch")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--encoder", type=Path, default=MODELS_DIR / "ecg_encoder.pt")
    parser.add_argument("--classifier", type=Path, default=MODELS_DIR / "five_class_xgb.json")
    parser.add_argument("--no-classifier", action="store_true",
                         help="bypass the trained classifier and use RuleBasedBeatClassifier instead")
    args = parser.parse_args()

    encoder = None
    if args.encoder.exists():
        encoder = load_encoder(args.encoder)
        print(f"Loaded pretrained encoder from {args.encoder}")
    else:
        print(f"No pretrained encoder at {args.encoder} — run train_encoder.py first. "
              f"Continuing without learned embeddings.")

    if args.no_classifier:
        classifier = None
        print("--no-classifier set: using RuleBasedBeatClassifier fallback")
    else:
        classifier = FiveClassBeatClassifier()
        classifier.load(args.classifier)
        if not classifier.is_trained:
            print(f"ERROR: failed to load a trained classifier from {args.classifier} "
                  f"(classifier.is_trained is False after load()). Refusing to silently "
                  f"fall through to the rule-based fallback in a production run. "
                  f"Pass --no-classifier if that's actually what you want.")
            sys.exit(1)
        print(f"Loaded trained classifier from {args.classifier} "
              f"(classes: {list(classifier._label_encoder.classes_)})")

    pipeline = ECGPipeline(encoder=encoder, classifier=classifier)

    if args.source == "vitalpatch":
        files = discover_vitalpatch_files(DATA_RAW / "vitalpatch")[:args.limit]
        recordings = [r for f in files for r in parse_vitalpatch_ecg(f)]
    else:
        files = discover_sensio_files(DATA_RAW / "sense_io")[:args.limit]
        recordings = [parse_sensio_ecg(f) for f in files]

    for rec in recordings:
        result = pipeline.run(rec)
        print(json.dumps(summarize(result), indent=2, default=str))
        print("-" * 80)


if __name__ == "__main__":
    main()
