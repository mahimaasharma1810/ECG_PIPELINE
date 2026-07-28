"""ecg_inference.classifier -- Stage 7 (beat + rhythm classification)
and Stage 8 (risk scoring). Extracted verbatim from ecg_pipeline_core.py's
classify.py and risk.py sections, combined into one file per the
inference package layout.

Excludes BinaryGateCNN (classify.py's unused two-pass-triage
prerequisite -- never instantiated by ECGPipeline.run(), a torch-only
dead class; matches the "isolate ... two-stage experimental models /
CNN experiments" requirement).

FiveClassBeatClassifier.fit() is training code (calls xgboost's .fit())
that ships in the same class as .load()/.predict_one() because splitting
the class would risk changing behavior -- this inference package's entry
point (run_inference.py) only ever calls .load() and .predict_one(),
never .fit(). The classifier weights, AAMI class list, and every
risk-cascade threshold below are read-only here, imported from
.preprocess, and are never modified by this package.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .detector import Beat
from .preprocess import (
    AAMI_CLASSES, RISK, RISK_LEVELS, CONFORMAL, ConformalConfig,
    TEMPORAL, TemporalTrackingConfig,
)


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

    def analyze(self, labels: list[str], rr_ms: list[float | None]) -> tuple[list[RhythmFinding], int]:
        findings: list[RhythmFinding] = []
        findings += self._vt_runs(labels)
        findings += self._geminy(labels, pattern=["N", "V"], min_repeats=4, kind="BIGEMINY")
        findings += self._geminy(labels, pattern=["N", "N", "V"], min_repeats=3, kind="TRIGEMINY")
        afib_findings, n_afib_windows_examined = self._afib_suspected(rr_ms)
        findings += afib_findings
        return findings, n_afib_windows_examined

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
                         cv_threshold: float = 0.10) -> tuple[list[RhythmFinding], int]:
        """AFib is a RHYTHM finding computed here from RR irregularity —
        never treated as a quality defect (recommendation #6). High RR
        coefficient-of-variation over a rolling window suggests AFib,
        distinct from motion-artefact noise which the stage-2 SQI gate
        already screened out via morphology, not RR timing. The caller
        is expected to have already excluded physiologically-implausible
        RR intervals (`Beat.rr_flagged`, e.g. from a missed R-peak across
        a noisy/rejected stretch) by passing `None` in their place — the
        same filter `recording_level_hrv()` applies for SDNN — otherwise
        a missed-beat gap reads as a false rhythm irregularity here.

        Returns (findings, n_windows_examined): the caller needs the
        total window count, not just the flagged ones, to compute AFib
        burden as a genuine percentage of windows examined.

        CORRECTION (2026-07-28): this rule was flagged in RESEARCH_AUDIT.md
        as "the single biggest unvalidated clinical claim in the pipeline"
        — cv_threshold=0.15 had never been checked against a real
        AFib-labeled recording. Validated for real against LTAFDB (84
        records, 449,749 windows of 20 real annotated beats each, ground
        truth from '+' rhythm-change aux_notes): at 0.15, sensitivity=0.808
        / specificity=0.800 / F1=0.829. A full threshold sweep found 0.10
        Youden's-J-optimal on this data: sensitivity=0.971 / specificity=
        0.716 / F1=0.893 — a meaningfully better catch rate for a
        "suspected" screening flag (misses ~3% of true AFib windows vs
        ~19% at 0.15), at the cost of more false positives for a clinician
        to review. Changed 0.15 -> 0.10 on that evidence, matching
        ecg_pipeline_core.py. See ABLATION_REPORT.md's "AFib rule
        validation" section for the full sweep table and methodology.
        `window=20` was left unswept — only the threshold was validated.
        """
        findings = []
        n_windows_examined = 0
        clean_rr = [r for r in rr_ms if r is not None]
        for start in range(0, max(0, len(clean_rr) - window), window):
            chunk = np.array(clean_rr[start:start + window])
            if len(chunk) < window:
                continue
            n_windows_examined += 1
            cv = float(np.std(chunk) / np.mean(chunk)) if np.mean(chunk) > 0 else 0.0
            if cv > cv_threshold:
                findings.append(RhythmFinding("AFIB_SUSPECTED", start, start + window - 1, {"rr_cv": cv}))
        return findings, n_windows_examined



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
                     thresholds=RISK, afib_windows_examined: int | None = None) -> RiskReport:
    n = max(1, len(labels))
    pvc_burden = 100.0 * sum(1 for l in labels if l == "V") / n
    pac_burden = 100.0 * sum(1 for l in labels if l == "S") / n
    vt_runs = sum(1 for f in findings if f.kind == "VT_RUN")
    afib_windows = [f for f in findings if f.kind == "AFIB_SUSPECTED"]
    # Percentage of AFib-flagged windows out of windows actually examined. Old
    # callers that don't pass afib_windows_examined fall back to len(findings)
    # (all rhythm-finding kinds combined) -- the previous, incorrect denominator
    # -- which reproduces the exact old behavior bit-for-bit rather than silently
    # changing it for any caller this patch didn't also update.
    denominator = afib_windows_examined if afib_windows_examined is not None else len(findings)
    afib_burden = 100.0 * len(afib_windows) / max(1, denominator)
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


