#!/usr/bin/env python3
"""Single entry point for the inference-only ecg_inference package.

    python run_inference.py --input path/to/ecg.csv --output output.json

Loads the frozen pretrained classifier, preprocesses the ECG, detects
R-peaks, extracts features, classifies beats, computes rhythm risk, and
writes the structured JSON report (optionally including the deterministic
narrative). Never invokes training code: every import below resolves to
ecg_inference.* only -- no ecg_pipeline_tools, no _original_stages, no
xgboost/sklearn .fit() call anywhere on this path.

This orchestration (classifier -> ECGPipeline -> build_risk_report_json)
is intentionally inlined here rather than imported as a helper from
ecg_inference.report, to avoid a circular import: report.py already needs
things from classifier.py, and pipeline.py needs generate_report from
report.py, so report.py cannot also import ECGPipeline from pipeline.py.
See ecg_inference/report.py's module docstring for the full explanation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ecg_inference import ECGPipeline, load_classifier, build_risk_report_json
from ecg_inference.preprocess import parse_vitalpatch_ecg, parse_prorhythm_ecg, parse_wfdb_record, MODELS_DIR
from ecg_inference.report import _deterministic_narrative


def _load_segments(source: str, input_path: Path) -> list:
    if source == "vitalpatch":
        return parse_vitalpatch_ecg(input_path)
    # "sensio" is the pre-rename spelling of this source (the device's own
    # Bluetooth name); accepted so existing scripts/commands keep working.
    if source in ("prorhythm", "sensio"):
        return [parse_prorhythm_ecg(input_path)]
    if source == "wfdb":
        return [parse_wfdb_record(input_path)]
    raise ValueError(f"Unknown source: {source}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="raw ECG file (VitalPatch/ProRhythm CSV, or a WFDB record path)")
    parser.add_argument("--output", type=Path, required=True, help="where to write the structured JSON report")
    parser.add_argument("--source", choices=["vitalpatch", "prorhythm", "wfdb", "sensio"],
                         default="vitalpatch",
                         help="raw device format of --input (default: vitalpatch); "
                              "'sensio' is a deprecated alias for 'prorhythm'")
    parser.add_argument("--classifier", type=Path, default=MODELS_DIR / "five_class_xgb.json",
                         help="pretrained classifier weights (default: models/production/five_class_xgb.json)")
    parser.add_argument("--segment-index", type=int, default=0,
                         help="a VitalPatch file can split into multiple gap-separated segments; "
                              "which one to report on (default: 0, the first)")
    parser.add_argument("--narrative", action="store_true",
                         help="also compute the deterministic narrative (network-independent; "
                              "does not call MedGemma) and include it in the output JSON")
    args = parser.parse_args(argv)

    classifier = load_classifier(args.classifier)
    print(f"Loaded trained classifier from {args.classifier} (is_trained={classifier.is_trained})")

    # --input not existing or being unreadable as the requested --source
    # format used to surface as a raw, uncaught pandas/OS traceback --
    # surfaced here as the same clean SystemExit style already used below
    # for other bad-input cases (out-of-range --segment-index etc.). WFDB
    # records are a <id>.hea/.dat/.atr triplet, not a literal file named
    # <id>, so the existence check has to look for the .hea file there
    # instead of --input itself.
    input_exists = (args.input.with_suffix(".hea").exists() if args.source == "wfdb"
                     else args.input.exists())
    if not input_exists:
        raise SystemExit(f"--input not found for --source {args.source}: {args.input}")
    try:
        segments = _load_segments(args.source, args.input)
    except pd.errors.EmptyDataError:
        raise SystemExit(f"--input file is empty or has no parseable columns: {args.input}")
    if not segments:
        raise SystemExit(f"No usable segments parsed from {args.input}")
    if args.segment_index >= len(segments):
        raise SystemExit(f"--segment-index {args.segment_index} out of range: "
                          f"{args.input} produced {len(segments)} segment(s)")
    recording = segments[args.segment_index]
    print(f"Parsed {len(segments)} segment(s) from {args.input}; "
          f"reporting on segment {args.segment_index} "
          f"({recording.source}/{recording.patient_id}/{recording.segment_id})")

    pipeline = ECGPipeline(classifier=classifier)
    result = pipeline.run(recording)

    report_json = build_risk_report_json(result)
    # ECGPipeline.run() always attempts its own internal Stage 9 MedGemma
    # call (see inference_ready.md's "known network-dependent side effect"
    # note) -- surface what happened there explicitly, rather than leaving
    # a report reader to guess whether/why an LLM narrative did or didn't
    # get generated internally.
    report_json["pipeline_stage9_llm_status"] = {
        "bypassed_llm": result.merged_decision.bypassed_llm,
        "llm_rejected_reason": result.merged_decision.llm_rejected_reason,
    }
    if args.narrative:
        report_json["narrative"] = _deterministic_narrative(report_json)
        report_json["narrative_source"] = "deterministic_template (network-independent, not MedGemma)"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report_json, indent=2, default=str))
    print(f"risk_level={report_json['risk_level']}  "
          f"n_beats_detected={report_json['recording']['n_beats_detected']}")
    print(f"Wrote report -> {args.output}")


if __name__ == "__main__":
    main()
