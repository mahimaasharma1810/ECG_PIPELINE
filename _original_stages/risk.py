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
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .classify import RhythmFinding
from .config import CONFORMAL, RISK, RISK_LEVELS, TEMPORAL, ConformalConfig, TemporalTrackingConfig


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
    hrv_suppressed = hrv.get("sdnn_ms", 0.0) < 20.0  # ms, conservative low-HRV cutoff

    reasons = []
    level = "LOW"

    if pvc_burden > thresholds.pvc_burden_critical_pct:
        level, reasons = "CRITICAL", reasons + [f"PVC burden {pvc_burden:.1f}% > critical threshold"]
    elif vt_runs > 0:
        level, reasons = "CRITICAL", reasons + [f"{vt_runs} ventricular-tachycardia run(s) detected"]
    elif pvc_burden > thresholds.pvc_burden_high_pct:
        level, reasons = "HIGH", reasons + [f"PVC burden {pvc_burden:.1f}% > high threshold"]
    elif pac_burden > thresholds.pac_burden_high_pct or afib_burden > 30.0:
        level, reasons = "HIGH", reasons + [f"PAC burden {pac_burden:.1f}% / AFib burden {afib_burden:.1f}%"]
    elif hrv_suppressed:
        level, reasons = "MEDIUM", reasons + ["Sustained HRV suppression (SDNN < 20ms)"]
    else:
        reasons = ["No thresholds exceeded"]

    if news2_score is not None and news2_score >= 7 and RISK_LEVELS.index(level) < RISK_LEVELS.index("CRITICAL"):
        level, reasons = "CRITICAL", reasons + [f"NEWS2 {news2_score} >= 7"]
    if qsofa_score is not None and qsofa_score >= 2 and RISK_LEVELS.index(level) < RISK_LEVELS.index("HIGH"):
        level = "HIGH" if RISK_LEVELS.index(level) < RISK_LEVELS.index("HIGH") else level
        reasons = reasons + [f"qSOFA {qsofa_score} >= 2"]

    return RiskReport(
        pvc_burden_pct=pvc_burden, pac_burden_pct=pac_burden, vt_run_count=vt_runs,
        afib_burden_pct=afib_burden, hrv_suppressed=hrv_suppressed,
        qrs_width_trend=hrv.get("qrs_width_trend", 0.0),
        news2_score=news2_score, qsofa_score=qsofa_score,
        alert_level=level, alert_reasons=reasons,
    )


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
