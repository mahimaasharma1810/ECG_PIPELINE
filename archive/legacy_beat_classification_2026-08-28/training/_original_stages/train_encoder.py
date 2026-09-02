"""CLI: self-supervised pretrain the ECG foundation encoder on our own
unlabeled VitalPatch/SeNSiO recordings (recommendation #2).

Usage:
    python -m cliniaura_pipeline.train_encoder --max-files 20 --epochs 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import filters, quality, resample
from .beats import detect_and_segment
from .filters import robust_zscore
from .config import BEATS, DATA_RAW, MODELS_DIR, TARGET_FS
from .encoder import INPUT_LEN, PretrainResult, pretrain_self_supervised, save_encoder
from .ingest import discover_sensio_files, discover_vitalpatch_files, parse_sensio_ecg, parse_vitalpatch_ecg


def collect_wide_windows(max_files: int = 20, clip_value: float | None = None) -> np.ndarray:
    """Runs stages 1-5 on real local recordings and collects the resulting
    wide beat windows (unlabeled) as pretraining input."""
    windows = []

    vp_files = discover_vitalpatch_files(DATA_RAW / "vitalpatch")[:max_files]
    for f in vp_files:
        try:
            for rec in parse_vitalpatch_ecg(f):
                windows.extend(_windows_from_recording(rec.signal_mv, rec.timestamps_ms, rec.fs_nominal,
                                                         rec.already_bandpass_filtered, clip_value))
        except Exception as e:  # noqa: BLE001 - best-effort corpus building over many real files
            print(f"  skip {f.name}: {e}", file=sys.stderr)

    se_files = discover_sensio_files(DATA_RAW / "sense_io")[:max_files]
    for f in se_files:
        try:
            rec = parse_sensio_ecg(f)
            windows.extend(_windows_from_recording(rec.signal_mv, rec.timestamps_ms, rec.fs_nominal,
                                                     rec.already_bandpass_filtered, clip_value))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {f.name}: {e}", file=sys.stderr)

    return np.array(windows) if windows else np.zeros((0, INPUT_LEN))


def _windows_from_recording(signal, timestamps_ms, fs_nominal, already_filtered, clip_value) -> list[np.ndarray]:
    keep_mask, _ = quality.run_sqi_gate(signal, timestamps_ms, fs_nominal, clip_value=clip_value)
    clean = signal.copy().astype(float)
    clean[~keep_mask] = np.nan

    resampled, t_resampled = resample.to_target_rate(clean, timestamps_ms, fs_nominal, TARGET_FS)
    valid = ~np.isnan(resampled)
    if not valid.any():
        return []
    filled = np.interp(t_resampled, t_resampled[valid], resampled[valid])
    filtered = filters.apply_filter_chain(filled, TARGET_FS, already_bandpass_filtered=already_filtered)

    beats = detect_and_segment(filtered, TARGET_FS, BEATS)
    return [robust_zscore(b.wide_window) for b in beats
            if b.wide_window is not None and not b.quality_rejected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out", type=Path, default=MODELS_DIR / "ecg_encoder.pt")
    args = parser.parse_args()

    print("Collecting unlabeled beat windows from real local recordings...")
    windows = collect_wide_windows(max_files=args.max_files)
    print(f"Collected {len(windows)} beat windows.")
    if len(windows) < 50:
        print("Too few windows to pretrain meaningfully; point --max-files at more recordings.")
        return

    encoder, result = pretrain_self_supervised(windows, epochs=args.epochs)
    print(f"Pretraining done: {result.epochs} epochs, final loss {result.final_loss:.4f}, "
          f"{result.n_windows} windows.")

    save_encoder(encoder, args.out, meta={"epochs": result.epochs, "final_loss": result.final_loss,
                                           "n_windows": result.n_windows, "source": "vitalpatch+sensio (unlabeled)"})
    print(f"Saved encoder to {args.out}")


if __name__ == "__main__":
    main()
