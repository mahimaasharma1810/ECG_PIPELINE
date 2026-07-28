"""ecg_inference -- deployment-ready inference package for the VitalPatch
ECG pipeline. Contains only what's needed to load the frozen, pretrained
five-class XGBoost classifier and turn one ECG recording into a structured
risk report; no training, fitting, or dataset-download code is imported by
anything in this package (see inference_ready.md for the full audit).

Re-exports the entry points run_inference.py (and any other caller) needs,
so callers can `import ecg_inference` instead of reaching into submodules.
"""
from __future__ import annotations

from .preprocess import (
    Recording,
    parse_vitalpatch_ecg,
    parse_sensio_ecg,
    parse_wfdb_record,
    TARGET_FS,
)
from .detector import Beat
from .classifier import FiveClassBeatClassifier, RiskReport
from .pipeline import ECGPipeline, PipelineResult
from .report import load_classifier, build_risk_report_json, render_narrative, save_report

__all__ = [
    "Recording",
    "parse_vitalpatch_ecg",
    "parse_sensio_ecg",
    "parse_wfdb_record",
    "TARGET_FS",
    "Beat",
    "FiveClassBeatClassifier",
    "RiskReport",
    "ECGPipeline",
    "PipelineResult",
    "load_classifier",
    "build_risk_report_json",
    "render_narrative",
    "save_report",
]
