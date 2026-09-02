"""CLI: fit FiveClassBeatClassifier on real labeled public data and
calibrate ConformalRiskPredictor — the "flip .fit() on" step that
`classify.py`'s docstring says is all that's needed once labeled data
exists (recommendation #7).

Beats are segmented using the SAME `beats.segment_beats` / `features.py`
code path production inference uses, but seeded with each dataset's
ground-truth annotation sample positions instead of XQRS detection, so
training features are extracted identically to how they'll be extracted
at inference time.

Usage:
    python -m cliniaura_pipeline.train_classifiers --dataset mitdb
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import wfdb

from . import features, filters, resample
from .beats import segment_beats
from .classify import FiveClassBeatClassifier
from .config import AAMI_CLASSES, BEATS, DATA_RAW, MODELS_DIR, TARGET_FS
from .risk import ConformalRiskPredictor
from .splits import (MITDB_DS1, MITDB_DS2, SVDB_RECORDS, DS1_TRAIN, DS1_VAL,
                      INCART_RECORDS, LTAFDB_RECORDS, SDDB_RECORDS)

# Standard AAMI EC57 beat-symbol mapping (the same grouping Zhu et al. 2021
# and the rest of the review's reference papers use).
AAMI_SYMBOL_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}


def _snap_to_local_peak(signal: np.ndarray, r_peaks: np.ndarray, search_radius: int = 15) -> np.ndarray:
    """Annotation sample positions, rescaled from the original fs (e.g.
    MITDB's 360Hz) to TARGET_FS and run through resampling + the filter
    chain, land close to but not exactly on the true local extremum
    (empirically ~3 samples off on MITDB record 100, consistent enough to
    be a rounding/interpolation artefact rather than noise). Beat
    segmentation's own beat-level SQI check rejects any beat whose R-peak
    isn't the true local max — appropriate for XQRS-detected peaks, which
    are true maxima by construction, but not for rescaled annotations. So
    snap each annotation to the true nearby extremum before segmenting,
    same as any annotation-to-resampled-timeline training pipeline needs.
    """
    snapped = r_peaks.copy()
    for i, r in enumerate(r_peaks):
        lo, hi = max(0, r - search_radius), min(len(signal), r + search_radius)
        if hi <= lo:
            continue
        snapped[i] = lo + int(np.argmax(np.abs(signal[lo:hi])))
    return snapped


def _load_record_beats(record_path: Path, ann_ext: str = "atr") -> tuple[np.ndarray, list[str]]:
    """Returns (feature_matrix, labels) for every valid, non-rejected beat
    in one WFDB record, using ground-truth annotation positions."""
    record = wfdb.rdrecord(str(record_path))
    ann = wfdb.rdann(str(record_path), ann_ext)
    fs = float(record.fs)
    signal = record.p_signal[:, 0].astype(np.float64)

    keep = [i for i, sym in enumerate(ann.symbol) if sym in AAMI_SYMBOL_MAP]
    if not keep:
        return np.zeros((0, features.N_FEATURES)), []
    ann_samples = ann.sample[keep]
    ann_labels = [AAMI_SYMBOL_MAP[ann.symbol[i]] for i in keep]

    timestamps_ms = np.arange(len(signal)) * (1000.0 / fs)
    resampled, t_resampled = resample.to_target_rate(signal, timestamps_ms, fs, TARGET_FS)
    filtered = filters.apply_filter_chain(resampled, TARGET_FS, already_bandpass_filtered=False)

    scale = TARGET_FS / fs
    r_peaks_resampled = np.round(ann_samples * scale).astype(int)
    r_peaks_resampled = np.clip(r_peaks_resampled, 0, len(filtered) - 1)
    r_peaks_resampled = _snap_to_local_peak(filtered, r_peaks_resampled)

    beats = segment_beats(filtered, TARGET_FS, r_peaks_resampled, BEATS)
    primary_pre_samples = int(round(BEATS.primary_pre_ms / 1000.0 * TARGET_FS))

    rows, labels = [], []
    for beat, label in zip(beats, ann_labels):
        vec = features.beat_feature_vector(beat, primary_pre_samples)
        if vec is not None:
            rows.append(vec)
            labels.append(label)

    return (np.vstack(rows) if rows else np.zeros((0, features.N_FEATURES))), labels


def build_dataset(db_dir: Path, record_ids: list[int]) -> tuple[np.ndarray, list[str]]:
    all_X, all_y = [], []
    for rid in record_ids:
        record_path = db_dir / str(rid)
        if not record_path.with_suffix(".hea").exists():
            print(f"  skip {rid}: not downloaded")
            continue
        X, y = _load_record_beats(record_path)
        print(f"  {rid}: {len(y)} labeled beats")
        all_X.append(X)
        all_y.extend(y)
    X = np.vstack(all_X) if all_X else np.zeros((0, features.N_FEATURES))
    return X, all_y


def per_class_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=AAMI_CLASSES, zero_division=0)
    return {c: {"sensitivity": round(recall[i], 4), "precision": round(precision[i], 4),
                "f1": round(f1[i], 4), "support": int(support[i])}
            for i, c in enumerate(AAMI_CLASSES)}


def random_oversample(X: np.ndarray, y: list[str], minority_ratio: float = 1.0 / 3.0,
                       seed: int = 0,
                       per_class_ratio_overrides: dict[str, float] | None = None
                       ) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Mild Random Oversampling: any class below `minority_ratio` of the
    majority class's count is oversampled (with replacement) up to that
    floor. Classes already at or above it are left untouched — this is a
    1:3 floor, not a 1:1 rebalance, since 1:1 over-represents rare classes
    like F given how few real examples exist (Talukder et al. found ROS
    beats SMOTE/ADASYN/GAN for this exact imbalance regime).

    Also returns the row-index array used to build the output (original
    indices followed by duplicated ones), so a caller can apply the same
    resampling to a parallel array (e.g. a per-row data-source tag).

    `per_class_ratio_overrides` (e.g. {"F": 0.6}) replaces `minority_ratio`
    for just that class, so F can be given its own floor independent of
    S/V — added because lumping F under the same 1:3 floor as S/V means F
    (already the rarest real class, ~410 examples) gets the exact same
    proportional boost as S despite being an order of magnitude scarcer,
    which was one hypothesis for why F gets absorbed into S/V at
    classification time (see train_classifiers.py Task 3 ablation).
    """
    rng = np.random.default_rng(seed)
    y_arr = np.array(y)
    # sorted(), not set() iteration order: Python randomizes str hash order
    # per-process (PYTHONHASHSEED), so `for cls in set(y)` visits classes in
    # a different order each run -- since every class's oversampled rows are
    # drawn from the same shared `rng` in sequence, a different visit order
    # consumes the seeded random stream differently and silently produces a
    # different ROS result on every run despite an identical `seed`. This is
    # what task_4's reproducibility rerun caught (two seed=42 runs of the
    # exact same config gave different DS2 Macro-F1: 0.3983 vs 0.3945).
    counts = {c: int((y_arr == c).sum()) for c in sorted(set(y))}
    majority_count = max(counts.values())
    overrides = per_class_ratio_overrides or {}

    idx_parts = [np.arange(len(y))]
    for cls, count in counts.items():
        ratio = overrides.get(cls, minority_ratio)
        floor = int(majority_count * ratio)
        if count < floor and count > 0:
            idxs = np.where(y_arr == cls)[0]
            extra = rng.choice(idxs, size=floor - count, replace=True)
            idx_parts.append(extra)

    all_idx = np.concatenate(idx_parts)
    X_ros = X[all_idx]
    y_ros = y_arr[all_idx].tolist()
    return X_ros, y_ros, all_idx


def list_icentia11k_patients(icentia_dir: Path) -> list[Path]:
    """All patient directories currently on disk under the bucketed
    `pXX/pXXXXX` layout — works with however much of the dataset has been
    downloaded so far, not the full 11,000-patient set."""
    return sorted(icentia_dir.glob("p*/p*"))


def select_icentia11k_patients(patient_dirs: list[Path], n_patients: int, seed: int) -> list[Path]:
    rng = np.random.default_rng(seed)
    if n_patients >= len(patient_dirs):
        return patient_dirs
    idx = rng.choice(len(patient_dirs), size=n_patients, replace=False)
    return [patient_dirs[i] for i in sorted(idx)]


def build_icentia11k_dataset(patient_dirs: list[Path], segments_per_patient: int,
                              seed: int) -> tuple[np.ndarray, list[str]]:
    """Loads a random `segments_per_patient` ~1-hour segment(s) from each
    given Icentia11k patient. Reuses `_load_record_beats` unchanged —
    Icentia11k's raw N/S/V/Q beat symbols already map onto
    `AAMI_SYMBOL_MAP` as-is (no F symbol exists in this dataset at all),
    its `.atr` annotation extension matches the default, and its 250Hz
    native rate hits the exact-integer-ratio decimation path in
    `resample.to_target_rate` (already written with this dataset in mind).
    A patient-subsample is used, not the full download, because at
    ~5,000 beats/segment even a few hundred patients dwarfs the current
    ~220K-beat MITDB+SVDB training set.
    """
    rng = np.random.default_rng(seed)
    all_X, all_y = [], []
    for pdir in patient_dirs:
        hea_files = sorted(pdir.glob("*.hea"))
        if not hea_files:
            continue
        n_pick = min(segments_per_patient, len(hea_files))
        chosen = rng.choice(len(hea_files), size=n_pick, replace=False)
        for i in chosen:
            record_path = hea_files[i].with_suffix("")
            try:
                X, y = _load_record_beats(record_path)
            except Exception as e:
                print(f"  skip {record_path.name}: {e}")
                continue
            if len(y):
                all_X.append(X)
                all_y.extend(y)
    X = np.vstack(all_X) if all_X else np.zeros((0, features.N_FEATURES))
    return X, all_y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mitdb"], default="mitdb")
    parser.add_argument("--data-root", type=Path, default=DATA_RAW / "public")
    parser.add_argument("--out", type=Path, default=MODELS_DIR / "five_class_xgb.json")
    parser.add_argument("--include-svdb", action=argparse.BooleanOptionalAction, default=True,
                         help="add all 78 SVDB records into training (S-class boost); DS2 stays pure MITDB")
    parser.add_argument("--include-incart", action=argparse.BooleanOptionalAction, default=False,
                         help="add all 75 INCART records into training (12-lead source, channel 0 only, "
                              "same as every other WFDB record here); DS2 stays pure MITDB")
    parser.add_argument("--include-ltafdb", action=argparse.BooleanOptionalAction, default=False,
                         help="add all 84 LTAFDB records into training (24h+ Holter, real per-beat "
                              "N/V/A(->S) annotations, plus AFib rhythm context); DS2 stays pure MITDB")
    parser.add_argument("--include-sddb", action=argparse.BooleanOptionalAction, default=False,
                         help="add all 23 SDDB records into training -- notable for containing real "
                              "F-class beats, unlike Icentia11k which has none; DS2 stays pure MITDB")
    parser.add_argument("--drop-q", action=argparse.BooleanOptionalAction, default=True,
                         help="exclude Q-class beats from the training objective (too few real examples "
                              "to be learnable — 7 in MITDB DS2 alone; handle paced beats with a rule instead)")
    parser.add_argument("--include-icentia11k", action=argparse.BooleanOptionalAction, default=False,
                         help="add a random patient subsample from the downloaded Icentia11k data "
                              "(S/V boost only — it has no F-class labels)")
    parser.add_argument("--icentia11k-patients", type=int, default=750,
                         help="number of Icentia11k patients to randomly sample from whatever has "
                              "been downloaded so far")
    parser.add_argument("--icentia11k-segments-per-patient", type=int, default=1,
                         help="how many ~1-hour segments to load per sampled patient")
    parser.add_argument("--icentia11k-weight", type=float, default=0.4,
                         help="relative sample_weight multiplier applied to Icentia11k-origin training "
                              "rows, down-weighting them vs MITDB/SVDB's more rigorously verified labels")
    parser.add_argument("--icentia11k-seed", type=int, default=None,
                         help="defaults to --seed if not given")
    parser.add_argument("--train-split", choices=["ds1_train", "ds1_full"], default="ds1_train",
                         help="ds1_train (default): train on splits.DS1_TRAIN, hold out splits.DS1_VAL "
                              "for honest validation reporting. ds1_full: train on all of DS1 (no "
                              "validation report) — only for building a final model AFTER tuning "
                              "decisions have already been made on the validation set.")
    parser.add_argument("--seed", type=int, default=42,
                         help="fixed random_state for ROS, XGBoost, and Icentia11k sampling — "
                              "two runs with the same config and same seed must match exactly")
    parser.add_argument("--ros", action=argparse.BooleanOptionalAction, default=True,
                         help="apply Random Oversampling (1:3-of-majority floor) to the minority classes")
    parser.add_argument("--balanced-weights", action=argparse.BooleanOptionalAction, default=True,
                         help="apply sklearn compute_sample_weight('balanced') on top of "
                              "(whatever the --ros result is). Default True+True stacks both, which "
                              "STATUS_QA.md flagged as a possible double-correction — use --no-ros / "
                              "--no-balanced-weights to ablate each in isolation.")
    parser.add_argument("--f-ros-ratio", type=float, default=None,
                         help="dedicated ROS floor for F only (fraction of majority-class count), "
                              "overriding the shared 1:3 floor used for S/V. E.g. 0.6 gives F its own, "
                              "much higher floor without changing S/V's oversampling.")
    parser.add_argument("--f-weight-multiplier", type=float, default=1.0,
                         help="extra sample_weight multiplier applied ONLY to F-labeled rows, on top of "
                              "whatever ROS/balanced-weight combination is used — a misclassification-cost "
                              "knob distinct from the general class-imbalance handling above.")
    args = parser.parse_args()
    if args.icentia11k_seed is None:
        args.icentia11k_seed = args.seed
    if not args.out.is_absolute():
        # `-m cliniaura_pipeline.train_classifiers` resolves relative paths
        # against the caller's cwd, not this package's directory, so a
        # relative --out can silently point outside models/ (and crash at
        # save time once training has already finished).
        args.out = (MODELS_DIR.parent / args.out).resolve()

    db_dir = args.data_root / args.dataset
    ds1_record_ids = DS1_TRAIN if args.train_split == "ds1_train" else MITDB_DS1
    print(f"Building DS1 train ({args.train_split}) from {db_dir} ...")
    X_train, y_train = build_dataset(db_dir, ds1_record_ids)
    print(f"Building DS2 (held-out test, report-only) from {db_dir} ...")
    X_test, y_test = build_dataset(db_dir, MITDB_DS2)
    X_val, y_val = (None, None)
    if args.train_split == "ds1_train":
        print(f"Building DS1_VAL (patient-level validation carve-out) from {db_dir} ...")
        X_val, y_val = build_dataset(db_dir, DS1_VAL)

    if args.include_svdb:
        svdb_dir = args.data_root / "svdb"
        print(f"Building SVDB training data (S-class boost) from {svdb_dir} ...")
        X_svdb, y_svdb = build_dataset(svdb_dir, SVDB_RECORDS)
        print(f"  SVDB contributes {len(y_svdb)} beats to training")
        X_train = np.vstack([X_train, X_svdb])
        y_train = y_train + y_svdb

    if args.include_incart:
        incart_dir = args.data_root / "incartdb"
        print(f"Building INCART training data from {incart_dir} ...")
        X_incart, y_incart = build_dataset(incart_dir, INCART_RECORDS)
        print(f"  INCART contributes {len(y_incart)} beats to training")
        X_train = np.vstack([X_train, X_incart])
        y_train = y_train + y_incart

    if args.include_ltafdb:
        ltafdb_dir = args.data_root / "ltafdb"
        print(f"Building LTAFDB training data from {ltafdb_dir} ...")
        X_ltafdb, y_ltafdb = build_dataset(ltafdb_dir, LTAFDB_RECORDS)
        print(f"  LTAFDB contributes {len(y_ltafdb)} beats to training")
        X_train = np.vstack([X_train, X_ltafdb])
        y_train = y_train + y_ltafdb

    if args.include_sddb:
        sddb_dir = args.data_root / "sddb"
        print(f"Building SDDB training data (F-class boost -- real F beats) from {sddb_dir} ...")
        X_sddb, y_sddb = build_dataset(sddb_dir, SDDB_RECORDS)
        print(f"  SDDB contributes {len(y_sddb)} beats to training")
        X_train = np.vstack([X_train, X_sddb])
        y_train = y_train + y_sddb

    source_train = None
    if args.include_icentia11k:
        icentia_dir = args.data_root / "icentia11k"
        print(f"\nSelecting Icentia11k patient subsample from {icentia_dir} ...")
        all_patients = list_icentia11k_patients(icentia_dir)
        print(f"  {len(all_patients)} patient directories available on disk")
        chosen_patients = select_icentia11k_patients(all_patients, args.icentia11k_patients,
                                                       args.icentia11k_seed)
        print(f"  sampling {len(chosen_patients)} patients, "
              f"{args.icentia11k_segments_per_patient} segment(s) each")
        X_icentia, y_icentia = build_icentia11k_dataset(
            chosen_patients, args.icentia11k_segments_per_patient, args.icentia11k_seed)
        print(f"  Icentia11k contributes {len(y_icentia)} beats to training "
              f"(no F-class beats exist in this dataset)")
        source_train = ["primary"] * len(y_train) + ["icentia11k"] * len(y_icentia)
        X_train = np.vstack([X_train, X_icentia])
        y_train = y_train + y_icentia

    print(f"\nDS1(+SVDB{'+Icentia11k' if args.include_icentia11k else ''}) train: "
          f"{len(y_train)} beats, DS2 test: {len(y_test)} beats")
    if len(y_train) < 100 or len(y_test) < 100:
        print("Not enough labeled beats to train meaningfully — check --data-root.")
        return

    from collections import Counter
    from sklearn.utils.class_weight import compute_sample_weight

    if args.drop_q:
        keep = [i for i, lab in enumerate(y_train) if lab != "Q"]
        n_dropped = len(y_train) - len(keep)
        X_train = X_train[keep]
        y_train = [y_train[i] for i in keep]
        if source_train is not None:
            source_train = [source_train[i] for i in keep]
        print(f"Dropped {n_dropped} Q-class beats from training (not learnable at this sample size)")

    print(f"Train class counts before ROS: {dict(Counter(y_train))}")
    if args.ros:
        f_overrides = {"F": args.f_ros_ratio} if args.f_ros_ratio is not None else None
        X_train_ros, y_train_ros, ros_idx = random_oversample(
            X_train, y_train, minority_ratio=1.0 / 3.0, seed=args.seed,
            per_class_ratio_overrides=f_overrides)
        print(f"Train class counts after ROS (1:3 floor{', F override=' + str(args.f_ros_ratio) if f_overrides else ''}): "
              f"{dict(Counter(y_train_ros))}")
    else:
        X_train_ros, y_train_ros, ros_idx = X_train, y_train, np.arange(len(y_train))
        print("ROS disabled (--no-ros): training on raw class counts")

    if args.balanced_weights:
        sample_weight = compute_sample_weight("balanced", y_train_ros)
    else:
        sample_weight = np.ones(len(y_train_ros), dtype=float)
        print("Balanced class weights disabled (--no-balanced-weights): uniform sample_weight")

    if args.f_weight_multiplier != 1.0:
        y_ros_arr = np.array(y_train_ros)
        sample_weight = sample_weight.copy()
        sample_weight[y_ros_arr == "F"] *= args.f_weight_multiplier
        print(f"Applied extra {args.f_weight_multiplier}x sample_weight multiplier to "
              f"{int((y_ros_arr == 'F').sum())} F-labeled training rows")

    if source_train is not None:
        source_train_ros = [source_train[i] for i in ros_idx]
        icentia_mask = np.array([s == "icentia11k" for s in source_train_ros])
        sample_weight = sample_weight.copy()
        sample_weight[icentia_mask] *= args.icentia11k_weight
        print(f"Down-weighted {int(icentia_mask.sum())} Icentia11k-origin training rows "
              f"by {args.icentia11k_weight}x")

    classifier = FiveClassBeatClassifier()
    classifier.fit(X_train_ros, y_train_ros, sample_weight=sample_weight, random_state=args.seed)
    classifier.save(args.out)
    print(f"Saved trained classifier -> {args.out}  (seed={args.seed})")

    def _predict(X):
        y_pred, proba_all = [], []
        for x in X:
            proba = classifier.model.predict_proba(x.reshape(1, -1))[0]
            proba_all.append(proba)
            classes = classifier._label_encoder.inverse_transform(np.arange(len(proba)))
            y_pred.append(classes[np.argmax(proba)])
        return y_pred, proba_all

    from sklearn.metrics import f1_score

    if X_val is not None and len(y_val) > 0:
        # Validation report — this is where all tuning decisions should be
        # made. Never used to fit/oversample/weight anything above.
        y_val_pred, _ = _predict(X_val)
        val_metrics = per_class_metrics(y_val, y_val_pred)
        val_macro_f1 = f1_score(y_val, y_val_pred, labels=AAMI_CLASSES, average="macro", zero_division=0)
        print(f"\nPer-class metrics on DS1_VAL (patient-level validation, tuning signal — NOT DS2):")
        for c, m in val_metrics.items():
            print(f"  {c}: sensitivity={m['sensitivity']:.3f} precision={m['precision']:.3f} "
                  f"f1={m['f1']:.3f} support={m['support']}")
        print(f"Macro-F1 on DS1_VAL: {val_macro_f1:.4f}")

    y_pred, proba_all = _predict(X_test)

    metrics = per_class_metrics(y_test, y_pred)
    print("\nPer-class metrics on DS2 (held-out test set, REPORT-ONLY — never touched during tuning):")
    for c, m in metrics.items():
        print(f"  {c}: sensitivity={m['sensitivity']:.3f} precision={m['precision']:.3f} "
              f"f1={m['f1']:.3f} support={m['support']}")

    macro_f1 = f1_score(y_test, y_pred, labels=AAMI_CLASSES, average="macro", zero_division=0)
    overall_acc = float(np.mean(np.array(y_pred) == np.array(y_test)))
    print(f"\nMacro-F1 on DS2: {macro_f1:.4f}  (this is the number that matters here, not accuracy)")
    print(f"Overall DS2 accuracy: {overall_acc:.4f}")

    # Calibrate the conformal risk predictor on the same held-out probabilities,
    # mapping AAMI beat-class confidence to a coarse risk-level proxy so
    # ConformalRiskPredictor.calibrate() has a real (if approximate) calibration
    # set instead of remaining permanently uncalibrated (recommendation #4).
    print("\nCalibrating ConformalRiskPredictor (beat-class confidence as a risk-level proxy)...")
    conformal = ConformalRiskPredictor()
    try:
        risk_level_idx = np.array([_label_to_risk_idx(l) for l in y_test])
        risk_scores = _beat_proba_to_risk_scores(np.array(proba_all), classifier)
        conformal.calibrate(risk_scores, risk_level_idx)
        print(f"Calibrated. qhat={conformal._qhat:.4f}")
    except ValueError as e:
        print(f"Skipped calibration: {e}")


def _label_to_risk_idx(aami_label: str) -> int:
    # N -> LOW, S/F -> MEDIUM, V -> HIGH (coarse proxy; real calibration should
    # use actual recording-level risk outcomes once available)
    return {"N": 0, "S": 1, "F": 1, "V": 2, "Q": 0}[aami_label]


def _beat_proba_to_risk_scores(beat_proba: np.ndarray, classifier: FiveClassBeatClassifier) -> np.ndarray:
    classes = list(classifier._label_encoder.inverse_transform(np.arange(beat_proba.shape[1])))
    risk_groups = {0: ["N", "Q"], 1: ["S", "F"], 2: ["V"], 3: []}
    out = np.zeros((len(beat_proba), 4))
    for level, members in risk_groups.items():
        idxs = [classes.index(m) for m in members if m in classes]
        out[:, level] = beat_proba[:, idxs].sum(axis=1) if idxs else 1e-6
    out[:, 3] = 1e-6  # no beat-level class maps to CRITICAL; kept as a near-zero floor
    return out / out.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    main()
