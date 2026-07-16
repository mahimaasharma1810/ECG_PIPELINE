"""Orchestrator: wires stages 1-9 together end to end.

A beat or window that fails a quality check is rejected and logged rather
than silently dropped, so the whole run is auditable end to end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import features, filters, quality, resample
from .audit import AuditLog
from .beats import Beat, detect_and_segment
from .classify import FiveClassBeatClassifier, RhythmContextEngine
from .config import BEATS, TARGET_FS
from .encoder import ECGEncoder, embed_windows
from .ingest import Recording
from .report import MergedDecision, generate_report
from .risk import ConformalRiskPredictor, RiskReport, TemporalRiskTracker, score_recording
from .similar_cases import SimilarCaseIndex


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


class CliniauraPipeline:
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
        keep_mask, verdicts = quality.run_sqi_gate(recording.signal_mv, recording.timestamps_ms,
                                                     recording.fs_nominal, clip_value=clip_value)
        rejection_rate = quality.rejection_rate(verdicts)
        audit.append("STAGE2_SQI_GATE", {"n_windows": len(verdicts), "rejection_rate": rejection_rate,
                                          "reject_codes": [v.reject_code for v in verdicts if not v.passed]})

        signal_clean = recording.signal_mv.copy()
        signal_clean[~keep_mask] = np.nan

        # Stage 3: resample to common rate
        resampled, t_resampled = resample.to_target_rate(signal_clean, recording.timestamps_ms,
                                                           recording.fs_nominal, TARGET_FS)
        audit.append("STAGE3_RESAMPLE", {"fs_in": recording.fs_nominal, "fs_out": TARGET_FS,
                                          "n_out": len(resampled)})

        valid = ~np.isnan(resampled)
        if len(resampled) == 0 or not valid.any() or valid.sum() < int(TARGET_FS * 2):
            audit.append("SEGMENT_REJECTED_INSUFFICIENT_DATA",
                          {"n_resampled": len(resampled), "n_valid": int(valid.sum())})
            return self._insufficient_data_result(recording, keep_mask, rejection_rate, audit)
        resampled_filled = np.interp(t_resampled, t_resampled[valid], resampled[valid])

        # Stage 4: filter chain
        filtered = filters.apply_filter_chain(resampled_filled, TARGET_FS,
                                               already_bandpass_filtered=recording.already_bandpass_filtered)
        audit.append("STAGE4_FILTER", {"already_bandpass_filtered": recording.already_bandpass_filtered})

        # Stage 5: R-peak detection + beat segmentation
        beats = detect_and_segment(filtered, TARGET_FS, BEATS)
        n_rejected = sum(1 for b in beats if b.quality_rejected)
        audit.append("STAGE5_BEATS", {"n_beats_detected": len(beats), "n_beats_rejected": n_rejected,
                                       "n_rr_flagged": sum(1 for b in beats if b.rr_flagged)})

        # Stage 6: features (handcrafted + optional learned embeddings)
        primary_pre_samples = int(round(BEATS.primary_pre_ms / 1000.0 * TARGET_FS))
        feature_matrix, feature_idxs = features.batch_feature_matrix(beats, primary_pre_samples)
        hrv = features.recording_level_hrv(beats)
        audit.append("STAGE6_FEATURES", {"n_feature_vectors": len(feature_idxs), "hrv": hrv})

        embeddings = None
        if self.encoder is not None:
            wide_windows = np.array([filters.robust_zscore(b.wide_window) for b in beats
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
            sqi_rejection_rate=rejection_rate, beats=beats, n_beats_accepted=len(beats) - n_rejected,
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
