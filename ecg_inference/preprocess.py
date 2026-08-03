"""ecg_inference.preprocess -- Stages 1-4: parse raw device streams,
signal-quality gate, resample, and the deterministic filter chain.

Extracted verbatim from ecg_pipeline/ecg_pipeline_core.py's config.py,
audit.py, ingest.py, quality.py, resample.py, and filters.py sections
(inference-safe: no code here trains, fits, or writes a model). See
EDGE_DEPLOYMENT_FIX_REPORT.md for the provenance of the source="wfdb" vs
"vitalpatch"/"prorhythm" behavior encoded in these thresholds/filters.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, decimate, filtfilt, iirnotch, lfilter, medfilt


"""Shared constants for the ECG pipeline (new methodology).

Every threshold here is either carried over from the existing baseline
(documented in PROJECT_OVERVIEW.md / the architecture PDF) or introduced by
one of the "Top Recommendations" in the internal review PDF. Each new
constant says which recommendation it implements.
"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
# Single repo-level models tree shared with ecg_pipeline -- see the matching
# comment in ecg_pipeline_core.py. Inference only ever loads production.
MODELS_DIR = PROJECT_ROOT / "models" / "production"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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
    # mV-per-count constant (VitalPatch and ProRhythm both do this). Absolute-mV thresholds would be
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
    source: str                   # "vitalpatch" | "prorhythm" | "wfdb"
    patient_id: str
    segment_id: str
    gaps: list = field(default_factory=list)          # list of (start_idx, end_idx, gap_ms) flagged, not dropped
    already_bandpass_filtered: bool = False             # ProRhythm pre-filtered variant
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
    flat_raw = raw.to_numpy().reshape(-1)
    # Some real device files contain a stray non-numeric sentinel (observed:
    # a literal '-') for an occasional missing sample, which previously
    # crashed the whole file with ValueError on the final .astype(float64)
    # (70/3570 real VitalPatch segments across all 6 patients, per
    # PROJECT_STATUS.md). Coerce to NaN instead of crashing, then drop by
    # PAIR (timestamp, value) rather than filtering the flat stream alone --
    # dropping a single element from the flat alternating stream before
    # splitting into timestamps/values would silently shift every
    # timestamp/value pairing after it by one. This also fixes that same
    # latent misalignment risk for any ordinary pre-existing NaN cell, not
    # just the '-' sentinel.
    flat = pd.to_numeric(pd.Series(flat_raw), errors="coerce").to_numpy(dtype=np.float64)
    if len(flat) % 2 != 0:
        flat = flat[:-1]
    pairs = flat.reshape(-1, 2)
    pairs = pairs[~np.isnan(pairs).any(axis=1)]
    timestamps_ms = pairs[:, 0].astype(np.int64)
    values = pairs[:, 1].astype(np.float64)

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


def parse_prorhythm_ecg(csv_path: Path) -> Recording:
    """ProRhythm: 11 metadata rows, blank rows, header at row 16 (0-indexed 15).

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

    fs_nominal = _prorhythm_fs_from_command(meta.get("Command Sent", ""))
    n = len(signal)
    timestamps_ms = np.arange(n) * (1000.0 / fs_nominal)

    return Recording(
        signal_mv=signal,
        timestamps_ms=timestamps_ms,
        fs_nominal=fs_nominal,
        source="prorhythm",
        patient_id=meta.get("Bluetooth Device ID", "unknown"),
        segment_id=csv_path.stem,
        already_bandpass_filtered=already_filtered,
        meta=meta,
    )


def _prorhythm_fs_from_command(command_sent: str, default_fs: float = 100.0) -> float:
    """Parse 'STARTECG_F:100' style command strings for sample rate."""
    if ":" in command_sent:
        try:
            return float(command_sent.rsplit(":", 1)[1])
        except ValueError:
            pass
    return default_fs


# Backward-compatible aliases. This source was originally named after the
# device's own Bluetooth identifier ("SeNSiO") while every external name in
# the repo already used the study name ("prorhythm"): data/raw/prorhythm/,
# batch_prorhythm_report.py, prorhythm_run_manifest.csv. The rename settles
# on "prorhythm" everywhere. These aliases stay because parse_sensio_ecg is
# in ecg_inference.__all__ and training/ecg_pipeline_tools.py imports it --
# removing them outright would break callers this rename did not touch.
parse_sensio_ecg = parse_prorhythm_ecg
_sensio_fs_from_command = _prorhythm_fs_from_command


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


def discover_prorhythm_files(root: Path) -> list[Path]:
    return sorted(root.glob("ECG_*.csv"))




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

    Caller passes a copy already band-limited to the physiological ECG band
    (see evaluate_window) rather than the fully raw window — on ProRhythm's raw
    "ECG" channel, a dominant near-Nyquist artifact (more spectral power
    above 40Hz than below, confirmed by FFT) swamps the raw amplitude
    distribution and reads as kurtosis ~ -1 (near-uniform) even when real
    QRS structure is present underneath; band-limiting first recovers the
    true impulse character (same window: -1.0 raw -> 1.8 band-limited).
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
    # baseline_wander_ratio and snr_db both split the window into a
    # 0.5-40Hz "signal" band and call everything outside that "wander"/
    # "noise". On raw, unfiltered input (e.g. WFDB) that scores the
    # window's own real per-window baseline offset as noise, because this
    # gate runs BEFORE Stage 4's baseline removal by design (so it can
    # catch genuinely bad signal before any filter gets a chance to hide
    # it). Score those two metrics on a locally baseline-corrected copy
    # instead -- the same 0.5Hz highpass Stage 4 already trusts --
    # while flatline/clipping/missing/kurtosis stay on the raw window,
    # which needs real railing/dropout/shape, not a detrended view.
    x_valid = x[~np.isnan(x)]
    x_scoring = highpass_residual(x_valid, fs, cutoff_hz=0.5) if len(x_valid) >= 9 else x_valid
    # Same 0.5-40Hz physiological band Stage 4 and _snr_db already trust,
    # applied once here so kurtosis judges QRS shape instead of whatever
    # out-of-band artifact happens to dominate this device's raw channel
    # (see _morphology_kurtosis docstring). filtfilt via bandpass() needs a
    # few dozen samples to be stable; too-short windows fall back to
    # x_scoring same as the wander/SNR metrics do.
    x_band = bandpass(x_valid, fs, FILTER.bandpass_low_hz, FILTER.bandpass_high_hz,
                       FILTER.bandpass_order) if len(x_valid) >= 20 else x_scoring

    # snr_db's "noise" is literally everything outside 0.5-40Hz, i.e. a
    # sliver from bandpass_high_hz up to this window's own Nyquist. For a
    # native rate close to 2x bandpass_high_hz (ProRhythm's 100Hz -> Nyquist
    # 50Hz) that sliver is only a few Hz wide and sits exactly where the
    # mains notch (50Hz) is mathematically invalid (powerline_notch()
    # no-ops once notch_hz >= Nyquist) -- so any near-Nyquist artifact in
    # that narrow band reads as overwhelming "noise" even though Stage 3
    # resamples to TARGET_FS and Stage 4's notch/bandpass remove exactly
    # this before the signal ever reaches R-peak detection. Score snr_db on
    # a copy upsampled to TARGET_FS first -- the same resample Stage 3
    # already performs -- so the noise band gets the same headroom the real
    # signal path will have, and the notch can actually run. Only kicks in
    # when the native rate itself can't support the notch; WFDB/VitalPatch
    # (already >= TARGET_FS) are untouched.
    x_snr_scoring = x_scoring
    if fs / 2.0 <= FILTER.notch_hz and len(x_valid) >= 9:
        valid_mask = ~np.isnan(x)
        t_valid = t_ms[valid_mask] if len(t_ms) == len(x) else \
            np.arange(len(x_valid)) * (1000.0 / fs)
        x_resampled, _ = resample_linear(x_valid, t_valid, TARGET_FS)
        if len(x_resampled) >= 9:
            x_snr_scoring = powerline_notch(
                highpass_residual(x_resampled, TARGET_FS, cutoff_hz=0.5), TARGET_FS)

    metrics = {
        "flatline_frac": _flatline_frac(x, fs, thresholds.flatline_run_ms),
        "clipping_frac": _clipping_frac(x, fs, thresholds.clipping_run_ms, clip_value),
        "missing_frac": _missing_frac(x),
        "morphology_kurtosis": _morphology_kurtosis(x_band),
        "baseline_wander_ratio": _baseline_wander_ratio(x_scoring, fs),
        "snr_db": _snr_db(x_snr_scoring, fs if x_snr_scoring is x_scoring else TARGET_FS),
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




"""Stage 3 — Uniform resample to the common target rate (125 Hz)."""


def resample_linear(signal: np.ndarray, timestamps_ms: np.ndarray,
                     target_fs: float = TARGET_FS, fs_nominal: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation onto a strict uniform time grid.

    Linear rather than cubic, to avoid ringing near sharp QRS peaks
    (cubic splines overshoot around the steep R-wave edge).

    When `fs_nominal` is given and this is a downsample (fs_nominal >
    target_fs), the valid samples are anti-alias low-pass filtered at the
    new Nyquist first -- otherwise high-frequency content (EMG/muscle
    artifact) aliases down into spurious QRS-like transients that XQRS
    later detects as extra beats. `resample_decimate` already does this for
    integer-ratio downsampling (see its docstring); this extends the same
    protection to the non-integer-ratio case, which previously had none.
    Root-caused as the dominant driver of R-peak over-detection on MITDB
    records 103/111 (n_detected ~1.8-1.9x n_true, ~1900 fully spurious
    peaks/record with no true beat within 150ms, not double-detections of
    real beats -- confirmed via diagnostic before this fix). VitalPatch is
    unaffected: its native rate equals TARGET_FS, so `to_target_rate`
    returns early and never reaches this function at all.
    """
    valid = ~np.isnan(signal)
    t = timestamps_ms[valid]
    x = signal[valid]
    if len(t) < 2:
        return np.array([]), np.array([])

    if fs_nominal is not None and fs_nominal > target_fs * (1.0 + 1e-6):
        nyq_native = fs_nominal / 2.0
        cutoff_hz = 0.9 * (target_fs / 2.0)
        if 0.0 < cutoff_hz < nyq_native:
            b, a = butter(4, cutoff_hz / nyq_native, btype="low")
            x = _safe_filtfilt(b, a, x)

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
    wearable timestamps (VitalPatch/ProRhythm).
    """
    if abs(fs_nominal - target_fs) < 1e-6:
        return signal, timestamps_ms

    ratio = fs_nominal / target_fs
    if fs_nominal > target_fs and abs(ratio - round(ratio)) < 1e-6:
        out = resample_decimate(signal, fs_nominal, target_fs)
        step_ms = 1000.0 / target_fs
        t_out = timestamps_ms[0] + np.arange(len(out)) * step_ms
        return out, t_out

    return resample_linear(signal, timestamps_ms, target_fs, fs_nominal=fs_nominal)




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

    CORRECTION (kept unchanged deliberately): this step's fixed absolute
    variances/clip were diagnosed as the dominant cause of R-peak
    over-detection on small-mV-scale WFDB signal (crushes genuine QRS
    amplitude down near the noise floor on noisy records, e.g. MITDB
    103/111 -- XQRS then can't tell noise from a real beat). NOT changed
    here, because this exact filtered output also feeds the frozen
    classifier's feature extraction (segment_beats/beat_feature_vector,
    both here and in ecg_pipeline_tools.py's training path) -- changing it
    would silently shift the classifier's input distribution away from
    what models/five_class_xgb.json was trained on, without retraining.
    The detection-side fix instead lives in detect_and_segment(), which
    runs R-peak detection on a separate, Kalman-skipped signal and only
    uses THIS function's output (unchanged) for windowing/features, so
    beat classification input is provably byte-identical to before.
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
                        cfg: FilterChainConfig = FILTER, skip_emg_suppress: bool = False) -> np.ndarray:
    """Run the full chain, or steps 5-6 only if the device pre-filtered.

    skip_emg_suppress=True stops after step 4 (bandpass), skipping the
    Kalman step entirely -- used only to build a detection-purpose signal
    for detect_and_segment (see emg_suppress_kalman's docstring). The
    default (False) path is byte-identical to before and is what feeds
    beat windowing/classification, so classifier input is unaffected.
    """
    if not already_bandpass_filtered:
        x = remove_baseline_median(x, fs, cfg.median_baseline_window_ms)
        x = highpass_residual(x, fs, cfg.highpass_hz)
        x = powerline_notch(x, fs, cfg.notch_hz, cfg.notch_q)
        x = bandpass(x, fs, cfg.bandpass_low_hz, cfg.bandpass_high_hz, cfg.bandpass_order)
    if skip_emg_suppress:
        return x
    x = emg_suppress_kalman(x)
    return x


# Defined at end of module: discover_prorhythm_files appears after the
# parser block above, so this alias cannot sit with the others.
discover_sensio_files = discover_prorhythm_files
