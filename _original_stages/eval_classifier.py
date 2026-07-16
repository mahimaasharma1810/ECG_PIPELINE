"""Standalone evaluation CLI — reproduces per-class DS2 metrics for an
ALREADY-TRAINED classifier without retraining anything.

This exists because train_classifiers.py bakes training and evaluation
into one main() call; there was no way to re-check a saved model's
numbers in isolation. Used as the baseline/ablation harness for the
timing-features experiment (see ABLATION_REPORT.md).

Usage:
    python -m cliniaura_pipeline.eval_classifier \
        --model models/five_class_xgb.json --split DS2

    python -m cliniaura_pipeline.eval_classifier \
        --model models/five_class_xgb_timing_v1.json --split DS2 --split-set validation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import features
from .classify import FiveClassBeatClassifier
from .config import AAMI_CLASSES, MODELS_DIR, DATA_RAW
from .train_classifiers import build_dataset, per_class_metrics
from .splits import MITDB_DS2, DS1_TRAIN, DS1_VAL


def evaluate(model_path: Path, record_ids: list[int], db_dir: Path,
             n_features: int | None = None, label: str = "") -> dict:
    X, y = build_dataset(db_dir, record_ids)
    if n_features is not None and X.shape[1] > n_features:
        X = X[:, :n_features]

    classifier = FiveClassBeatClassifier()
    classifier.load(model_path)

    y_pred = []
    for x in X:
        proba = classifier.model.predict_proba(x.reshape(1, -1))[0]
        classes = classifier._label_encoder.inverse_transform(np.arange(len(proba)))
        y_pred.append(classes[np.argmax(proba)])

    metrics = per_class_metrics(y, y_pred)

    from sklearn.metrics import f1_score, confusion_matrix
    macro_f1 = f1_score(y, y_pred, labels=AAMI_CLASSES, average="macro", zero_division=0)
    overall_acc = float(np.mean(np.array(y_pred) == np.array(y)))
    cm = confusion_matrix(y, y_pred, labels=AAMI_CLASSES)

    print(f"\n=== {label or model_path.name} on {len(y)} beats ===")
    print(f"{'class':<6}{'sensitivity':<13}{'precision':<12}{'f1':<8}{'support':<8}")
    for c in AAMI_CLASSES:
        m = metrics[c]
        print(f"{c:<6}{m['sensitivity']:<13.3f}{m['precision']:<12.3f}{m['f1']:<8.3f}{m['support']:<8}")
    print(f"Macro-F1: {macro_f1:.4f}   Overall accuracy: {overall_acc:.4f}  (accuracy is NOT the success metric)")
    print(f"\nConfusion matrix (rows=true, cols=pred), order {AAMI_CLASSES}:")
    print(cm)

    f_idx, s_idx = AAMI_CLASSES.index("F"), AAMI_CLASSES.index("S")
    f_support = cm[f_idx].sum()
    f_to_s_rate = cm[f_idx, s_idx] / f_support if f_support else 0.0
    print(f"F->S misclassification rate: {f_to_s_rate:.3f} ({cm[f_idx, s_idx]}/{f_support} true F beats predicted S)")

    return {"metrics": metrics, "macro_f1": macro_f1, "overall_acc": overall_acc,
            "confusion_matrix": cm.tolist(), "f_to_s_rate": f_to_s_rate, "n": len(y)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "five_class_xgb.json")
    parser.add_argument("--data-root", type=Path, default=DATA_RAW / "public")
    parser.add_argument("--split-set", choices=["ds1_train", "ds1_val", "ds2"], default="ds2",
                         help="ds1_train/ds1_val are the new patient-level carve-out of DS1 "
                              "(splits.py); ds2 is the literature held-out test set")
    parser.add_argument("--n-features", type=int, default=None,
                         help="slice feature vectors to first N columns (use 56 to evaluate the "
                              "legacy production model against the extended feature extractor)")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    db_dir = args.data_root / "mitdb"
    record_ids = {"ds1_train": DS1_TRAIN, "ds1_val": DS1_VAL, "ds2": MITDB_DS2}[args.split_set]
    evaluate(args.model, record_ids, db_dir, n_features=args.n_features, label=args.label)


if __name__ == "__main__":
    main()
