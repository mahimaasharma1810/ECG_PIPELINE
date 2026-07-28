"""ecg_inference.pipeline -- Orchestrator: wires Stages 1-9 together end
to end. Extracted verbatim from ecg_pipeline_core.py's pipeline.py section
(PipelineResult, ECGPipeline), with two edits, both removals of code already
dead in every production path (see ECGPipeline.__init__'s note below) --
never a change to an active/reachable logic path, threshold, or feature
computation:

  1. `encoder` dropped from ECGPipeline.__init__'s signature (was always
     None in production).
  2. Stage 6's `if self.encoder is not None: ... else: ...` collapsed to
     just the always-taken `else` branch (embeddings = None).

A beat or window that fails a quality check is rejected and logged rather
than silently dropped, so the whole run is auditable end to end.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocess import (
    AuditLog, Recording, TARGET_FS, run_sqi_gate, rejection_rate, to_target_rate,
    apply_filter_chain,
)
from .detector import Beat, detect_and_segment, BEATS
from .features import batch_feature_matrix, recording_level_hrv
from .classifier import (
    FiveClassBeatClassifier, RhythmContextEngine, ConformalRiskPredictor,
    TemporalRiskTracker, RiskReport, score_recording,
)
from .report import SimilarCaseIndex, MergedDecision, generate_report


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


class ECGPipeline:
    def __init__(self, classifier: FiveClassBeatClassifier | None = None,
                 similar_case_index: SimilarCaseIndex | None = None):
        # NOTE (inference package): the original ECGPipeline also accepted
        # an optional `encoder` (learned embedding, ecg_pipeline_core.py's
        # encoder.py section) used only to populate `embeddings` below. No
        # real deployment call site ever passed one (grep across
        # agent_bridge.py/demo_stream.py/report_ui.py/synthetic_ecg.py
        # confirms every production ECGPipeline(...) construction omits it,
        # i.e. self.encoder was always None in production) -- so this
        # inference package omits the parameter and hardcodes the
        # always-taken `else` branch below verbatim. This is a no-op for
        # numerical output: it removes only unreachable code and the torch
        # dependency it required, per "remove training dependencies from
        # the inference path".
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
        keep_mask, verdicts = run_sqi_gate(recording.signal_mv, recording.timestamps_ms,
                                            recording.fs_nominal, clip_value=clip_value)
        rejection_rate_value = rejection_rate(verdicts)
        audit.append("STAGE2_SQI_GATE", {"n_windows": len(verdicts), "rejection_rate": rejection_rate_value,
                                          "reject_codes": [v.reject_code for v in verdicts if not v.passed]})

        signal_clean = recording.signal_mv.copy()
        signal_clean[~keep_mask] = np.nan

        # Stage 3: resample to common rate
        resampled, t_resampled = to_target_rate(signal_clean, recording.timestamps_ms,
                                                 recording.fs_nominal, TARGET_FS)
        audit.append("STAGE3_RESAMPLE", {"fs_in": recording.fs_nominal, "fs_out": TARGET_FS,
                                          "n_out": len(resampled)})

        valid = ~np.isnan(resampled)
        if len(resampled) == 0 or not valid.any() or valid.sum() < int(TARGET_FS * 2):
            audit.append("SEGMENT_REJECTED_INSUFFICIENT_DATA",
                          {"n_resampled": len(resampled), "n_valid": int(valid.sum())})
            return self._insufficient_data_result(recording, keep_mask, rejection_rate_value, audit)
        resampled_filled = np.interp(t_resampled, t_resampled[valid], resampled[valid])

        # Stage 4: filter chain
        filtered = apply_filter_chain(resampled_filled, TARGET_FS,
                                       already_bandpass_filtered=recording.already_bandpass_filtered)
        audit.append("STAGE4_FILTER", {"already_bandpass_filtered": recording.already_bandpass_filtered})

        # Stage 5: R-peak detection + beat segmentation. Detection runs on a
        # separate signal from `filtered`; windowing/features always come
        # from `filtered` above, unchanged, so the classifier's input is
        # provably unaffected by this branch either way.
        #
        # For source="wfdb" (true calibrated-mV signal): skip Kalman for
        # detection. Root-caused via direct pre/post-Kalman XQRS A/B test:
        # emg_suppress_kalman's fixed-absolute-unit variances collapse
        # genuine QRS amplitude toward the noise floor on this signal's
        # small (~0.1-0.5 mV) scale, destroying the SNR XQRS needs and
        # causing severe over-detection (e.g. MITDB 103/111 were ~2x
        # true beat count with Kalman included).
        #
        # For source="vitalpatch"/"sensio" (arbitrary firmware-scaled raw
        # ADC counts, no published mV-per-count constant -- see
        # SQIThresholds.baseline_wander_ratio_max): the opposite holds.
        # These sources have real amplitude ~1000x the wfdb mV scale, and
        # skipping Kalman here exposes large genuine motion/EMG noise
        # excursions that XQRS then over-detects as QRS complexes (tested
        # on several VitalPatch recordings: skipping Kalman produced
        # implausible ~140bpm detection rates with >100 RR intervals
        # <0.3s, i.e. >200bpm adjacent-beat gaps that no real heart
        # produces; keeping Kalman for these sources reproduces the
        # original physiologically-plausible detection rate with <10 such
        # intervals). Kalman's smoothing is genuinely needed noise
        # suppression here, not the SNR-destroying effect seen on wfdb --
        # confirmed a plain robust-amplitude rescale of the Kalman-skipped
        # signal does NOT fix it (XQRS's threshold adapts internally; the
        # problem is real noise energy, not an absolute-unit miscalibration),
        # so this is a genuine per-source difference, not a threshold hack.
        if recording.source == "wfdb":
            detection_signal = apply_filter_chain(resampled_filled, TARGET_FS,
                                                   already_bandpass_filtered=recording.already_bandpass_filtered,
                                                   skip_emg_suppress=True)
        else:
            detection_signal = filtered
        # snap_radius: 8 is wfdb-only validated (see detect_and_segment's
        # docstring); real-device sources keep the pre-existing radius=15,
        # which a full VitalPatch batch re-run confirmed does not have the
        # over-culling failure mode radius=8 has on this data.
        snap_radius = 8 if recording.source == "wfdb" else 15
        beats = detect_and_segment(filtered, TARGET_FS, BEATS, detection_signal=detection_signal,
                                    snap_radius=snap_radius)
        n_rejected = sum(1 for b in beats if b.quality_rejected)
        audit.append("STAGE5_BEATS", {"n_beats_detected": len(beats), "n_beats_rejected": n_rejected,
                                       "n_rr_flagged": sum(1 for b in beats if b.rr_flagged)})

        # Stage 6: features (handcrafted + optional learned embeddings)
        primary_pre_samples = int(round(BEATS.primary_pre_ms / 1000.0 * TARGET_FS))
        feature_matrix, feature_idxs = batch_feature_matrix(beats, primary_pre_samples)
        hrv = recording_level_hrv(beats)
        audit.append("STAGE6_FEATURES", {"n_feature_vectors": len(feature_idxs), "hrv": hrv})

        # See __init__'s note above: production never supplies an encoder,
        # so this always took the "encoder_available: False" branch below.
        embeddings = None
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
        # rr_flagged beats (physiologically-implausible RR, e.g. a missed R-peak
        # across a noisy/rejected stretch) are excluded here, same as
        # recording_level_hrv() already does for SDNN -- otherwise a missed-beat
        # gap of several seconds reads as false rhythm irregularity below.
        rr_list = [b.rr_post_ms if not b.rr_flagged else None for b in beats]
        rhythm_findings, n_afib_windows_examined = self.rhythm_engine.analyze(labels, rr_list)
        audit.append("STAGE7_CLASSIFY", {"classifier_sources": sorted(classifier_sources),
                                          "classifier_trained": self.classifier.is_trained,
                                          "rhythm_findings": [f.kind for f in rhythm_findings]})

        # Stage 8: risk scoring + conformal + temporal
        risk_report = score_recording(labels, rhythm_findings, hrv, news2_score, qsofa_score,
                                       afib_windows_examined=n_afib_windows_examined)
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
            sqi_rejection_rate=rejection_rate_value, beats=beats, n_beats_accepted=len(beats) - n_rejected,
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

