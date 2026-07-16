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
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .beats import Beat
from .config import AAMI_CLASSES, RISK

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
                                        random_state=random_state)
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
