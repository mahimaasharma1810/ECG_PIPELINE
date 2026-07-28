"""Stage 1 — Parse and reconstruct raw device streams into a common format.

Each source gets its own parser because the raw formats are unrelated.
Every parser returns a `Recording`: a uniform-enough container the rest of
the pipeline (stages 2+) can consume regardless of which device produced it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


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
