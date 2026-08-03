#!/usr/bin/env python3
"""Export beats from a real VitalPatch segment into the annotation-tool
input JSON format.

    python annotation/export_beats_for_review.py \
        --input data/raw/vitalpatch/Patch_1844AC/<file>.csv \
        --output annotation/sample_beats.json \
        --n-beats 5

Runs the production preprocessing + detection + classification path
(ecg_inference) and emits one entry per selected beat, each carrying the
3-second display strip the clinician will actually see. Nothing here
writes back into the pipeline -- this is a read-only exporter.

Beat selection is deliberately NOT "the first N beats": reviewing five
consecutive normal beats teaches the model nothing. `--strategy` picks
either a class-balanced spread (default) or the lowest-confidence beats,
which is where clinician time is worth the most.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Marker-anchored, not hop-counted -- see the matching comment in
# scripts/verify_sqi_gate.py.
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ecg_pipeline").is_dir())
sys.path.insert(0, str(REPO_ROOT))

from ecg_inference.preprocess import (  # noqa: E402
    parse_vitalpatch_ecg, parse_prorhythm_ecg, parse_wfdb_record,
    TARGET_FS, run_sqi_gate, to_target_rate, apply_filter_chain, MODELS_DIR,
)
from ecg_inference.detector import detect_and_segment, BEATS  # noqa: E402
from ecg_inference.features import batch_feature_matrix  # noqa: E402
from ecg_inference import load_classifier  # noqa: E402

PRE_S = 1.0   # seconds of context before the R-peak
POST_S = 2.0  # seconds after -- 3.0 s total, 375 samples at 125 Hz


def _load_segments(source: str, path: Path) -> list:
    if source == "vitalpatch":
        return parse_vitalpatch_ecg(path)
    # "sensio" is the pre-rename spelling; accepted for back-compat.
    if source in ("prorhythm", "sensio"):
        return [parse_prorhythm_ecg(path)]
    if source == "wfdb":
        return [parse_wfdb_record(path)]
    raise ValueError(f"Unknown source: {source}")


def _run_to_beats(recording):
    """Production stages 2-7, returning the filtered display signal, the
    beats, and per-beat (label, confidence)."""
    keep_mask, _ = run_sqi_gate(recording.signal_mv, recording.timestamps_ms,
                                recording.fs_nominal, clip_value=None)
    sig = recording.signal_mv.copy()
    sig[~keep_mask] = np.nan
    resampled, t_res = to_target_rate(sig, recording.timestamps_ms,
                                      recording.fs_nominal, TARGET_FS)
    valid = ~np.isnan(resampled)
    if len(resampled) == 0 or not valid.any() or valid.sum() < int(TARGET_FS * 2):
        return None, [], [], [], None
    filled = np.interp(t_res, t_res[valid], resampled[valid])

    # Validity must be carried across the resample by mapping the RAW-grid
    # keep mask onto the target time grid -- NOT by re-reading NaNs out of
    # `resampled`. resample_linear() drops NaN samples and interpolates
    # across them (preprocess.py, `valid = ~np.isnan(signal)` then
    # `x = signal[valid]`), so its output contains no NaNs at all and a
    # NaN-derived mask silently reports "nothing removed". That path is
    # only reached when fs_nominal != TARGET_FS, because to_target_rate()
    # early-returns unchanged at 125 Hz -- which is why this was invisible
    # on VitalPatch (125 Hz native) and wrong on every ProRhythm file
    # (100 Hz), where it falsely reported 0% removed data on segments that
    # were in fact 73-85% blanked.
    valid = np.interp(t_res, recording.timestamps_ms,
                      keep_mask.astype(float)) > 0.5

    filtered = apply_filter_chain(
        filled, TARGET_FS, already_bandpass_filtered=recording.already_bandpass_filtered)
    if recording.source == "wfdb":
        det_sig = apply_filter_chain(
            filled, TARGET_FS, already_bandpass_filtered=recording.already_bandpass_filtered,
            skip_emg_suppress=True)
    else:
        det_sig = filtered
    snap = 8 if recording.source == "wfdb" else 15
    beats = detect_and_segment(filtered, TARGET_FS, BEATS,
                               detection_signal=det_sig, snap_radius=snap)
    if not beats:
        return filtered, [], [], [], valid

    pre = int(round(BEATS.primary_pre_ms / 1000.0 * TARGET_FS))
    feature_matrix, feature_idxs = batch_feature_matrix(beats, pre)
    feat_lookup = dict(zip(feature_idxs, feature_matrix))

    classifier = load_classifier(MODELS_DIR / "five_class_xgb.json")
    mean_rr = float(np.mean([b.rr_post_ms for b in beats if b.rr_post_ms is not None])) \
        if any(b.rr_post_ms is not None for b in beats) else 0.0

    labels: list[str] = []
    confs: list[float | None] = []
    for i, beat in enumerate(beats):
        if beat.quality_rejected:
            labels.append("Q")
            confs.append(None)  # not a model score -- deterministic quality gate
            continue
        fv = feat_lookup.get(i)
        res = classifier.predict_one(fv, beat, mean_rr)
        labels.append(res.label)
        confs.append(max(res.probabilities.values()) if res.probabilities else None)
    return filtered, beats, labels, confs, valid


def _select(labels, confs, n, strategy):
    """Indices worth a clinician's time."""
    idxs = list(range(len(labels)))
    if strategy == "low-confidence":
        scored = [(i, confs[i] if confs[i] is not None else -1.0) for i in idxs]
        scored.sort(key=lambda t: t[1])
        return sorted(i for i, _ in scored[:n])

    # balanced: take a spread across whatever classes are present, favouring
    # the abnormal ones (they are rarer and far more informative to label).
    order = ["V", "S", "F", "Q", "N"]
    by_class: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        by_class.setdefault(lab, []).append(i)
    picked: list[int] = []
    while len(picked) < n:
        added = False
        for cls in order:
            pool = [i for i in by_class.get(cls, []) if i not in picked]
            if not pool:
                continue
            picked.append(pool[len(picked) % len(pool)])
            added = True
            if len(picked) >= n:
                break
        if not added:
            break
    return sorted(picked)


def build_entries(recording, filtered, beats, labels, confs, selected,
                  valid_mask=None, fs=TARGET_FS):
    pre_n, post_n = int(PRE_S * fs), int(POST_S * fs)
    entries = []
    for i in selected:
        beat = beats[i]
        centre = int(beat.r_peak_idx)
        start, end = centre - pre_n, centre + post_n
        # Clamp to the signal and pad so every strip is exactly 375 samples,
        # otherwise the plot's time axis silently shifts between pages.
        lo, hi = max(0, start), min(len(filtered), end)
        strip = filtered[lo:hi].astype(float)
        pad_l, pad_r = max(0, -start), max(0, end - len(filtered))
        if pad_l or pad_r:
            strip = np.concatenate([np.full(pad_l, np.nan), strip, np.full(pad_r, np.nan)])
        r_in_strip = pre_n  # by construction, after padding

        # CRITICAL for label integrity: `filtered` has already had every
        # SQI-blanked sample replaced by a straight np.interp line, so
        # plotting it directly draws deleted data as if it were recorded
        # ECG -- on a 5 s blanked window that reads as asystole to a
        # clinician. Re-apply the pre-interpolation validity mask here so
        # those samples travel as null and the PDF can shade them "NO
        # DATA" instead of drawing a line through them.
        strip_valid = None
        if valid_mask is not None:
            vm = valid_mask[lo:hi]
            if pad_l or pad_r:
                vm = np.concatenate([np.zeros(pad_l, bool), vm, np.zeros(pad_r, bool)])
            strip_valid = vm
            strip = np.where(vm, strip, np.nan)

        neigh = [labels[i - 1] if i > 0 else None, labels[i],
                 labels[i + 1] if i + 1 < len(labels) else None]

        # Does the RR gap on either side of this beat run through blanked
        # data? If so the interval is NOT a measured beat-to-beat time and
        # must never be presented to a clinician as a pause.
        def _spans_blank(a: int, b: int) -> bool:
            if valid_mask is None or b <= a:
                return False
            lo_i, hi_i = max(0, a), min(len(valid_mask), b)
            return bool(hi_i > lo_i and (~valid_mask[lo_i:hi_i]).any())

        rr_after_samples = (int(round(beat.rr_post_ms / 1000.0 * fs))
                            if beat.rr_post_ms is not None else 0)
        rr_before_samples = (int(round(beat.rr_pre_ms / 1000.0 * fs))
                             if beat.rr_pre_ms is not None else 0)
        interp_frac = (float((~strip_valid).mean()) if strip_valid is not None else None)

        entries.append({
            "segment_id": recording.segment_id,
            "patient_id": recording.patient_id,
            "beat_index": i,
            "r_peak_sample": centre,
            "r_peak_in_strip": r_in_strip,
            "r_peak_time_s": round(float(beat.r_peak_ms) / 1000.0, 3),
            "predicted_label": labels[i],
            "confidence": (round(float(confs[i]), 4) if confs[i] is not None else None),
            "label_type": "pseudo",
            "neighboring_labels": neigh,
            "rr_before_ms": (round(float(beat.rr_pre_ms), 1) if beat.rr_pre_ms is not None else None),
            "rr_after_ms": (round(float(beat.rr_post_ms), 1) if beat.rr_post_ms is not None else None),
            "rr_flagged": bool(beat.rr_flagged),
            "quality_rejected": bool(beat.quality_rejected),
            "quality_reject_reason": beat.quality_reject_reason,
            "ecg_samples": [None if not np.isfinite(v) else round(float(v), 3) for v in strip],
            "fs": fs,
            "amplitude_units": "raw device units (not mV)",
            # Provenance of the displayed strip and of the RR intervals.
            "strip_interpolated_fraction": (round(interp_frac, 4)
                                            if interp_frac is not None else None),
            "rr_after_spans_blanked_data": _spans_blank(centre, centre + rr_after_samples),
            "rr_before_spans_blanked_data": _spans_blank(centre - rr_before_samples, centre),
            # Must reflect what apply_filter_chain ACTUALLY ran for this
            # recording. Devices that report a pre-filtered stream (ProRhythm
            # ECG_Filtered_*) set already_bandpass_filtered=True, which
            # skips steps 1-4 -- claiming baseline/notch/bandpass ran on
            # those files would be false on a clinician-facing form.
            "signal_stage": ("device pre-filtered + Kalman EMG only "
                             "(baseline/notch/bandpass skipped)"
                             if recording.already_bandpass_filtered else
                             "preprocessed (baseline-removed, notch, bandpass, Kalman EMG)"),
            "native_fs_hz": float(recording.fs_nominal),
            "resampled_to_hz": float(fs),
            "timestamps_synthetic": recording.source == "prorhythm",
        })
    return entries


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", choices=["vitalpatch", "prorhythm", "wfdb", "sensio"], default="vitalpatch")
    p.add_argument("--segment-index", type=int, default=0)
    p.add_argument("--n-beats", type=int, default=5)
    p.add_argument("--strategy", choices=["balanced", "low-confidence"], default="balanced",
                   help="which beats to put in front of the clinician (default: balanced)")
    args = p.parse_args(argv)

    if not args.input.exists() and args.source != "wfdb":
        raise SystemExit(f"--input not found: {args.input}")

    segments = _load_segments(args.source, args.input)
    if not segments:
        raise SystemExit(f"No usable segments parsed from {args.input}")
    if args.segment_index >= len(segments):
        raise SystemExit(f"--segment-index {args.segment_index} out of range "
                         f"({len(segments)} segment(s) available)")
    recording = segments[args.segment_index]

    filtered, beats, labels, confs, valid_mask = _run_to_beats(recording)
    if not beats:
        raise SystemExit(f"No beats detected in {args.input} segment {args.segment_index}")

    selected = _select(labels, confs, args.n_beats, args.strategy)
    entries = build_entries(recording, filtered, beats, labels, confs, selected,
                            valid_mask=valid_mask)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(entries, indent=2))
    dist = {c: labels.count(c) for c in sorted(set(labels))}
    print(f"Segment {recording.segment_id}: {len(beats)} beats detected, distribution {dist}")
    print(f"Selected {len(entries)} beat(s) via '{args.strategy}': "
          f"{[(e['beat_index'], e['predicted_label']) for e in entries]}")
    print(f"Wrote -> {args.output}")


if __name__ == "__main__":
    main()
