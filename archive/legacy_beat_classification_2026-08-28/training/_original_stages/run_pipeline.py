"""CLI: run the full stage 1-9 pipeline on a real local recording.

Usage:
    python -m cliniaura_pipeline.run_pipeline --source vitalpatch --limit 3
    python -m cliniaura_pipeline.run_pipeline --source sensio --limit 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .classify import FiveClassBeatClassifier
from .config import DATA_RAW, MODELS_DIR
from .encoder import load_encoder
from .ingest import discover_sensio_files, discover_vitalpatch_files, parse_sensio_ecg, parse_vitalpatch_ecg
from .pipeline import CliniauraPipeline


def summarize(result) -> dict:
    return {
        "source": result.recording.source,
        "patient_id": result.recording.patient_id,
        "segment_id": result.recording.segment_id,
        "n_raw_samples": result.n_raw_samples,
        "n_kept_samples": result.n_kept_samples,
        "sqi_rejection_rate": round(result.sqi_rejection_rate, 3),
        "n_beats_detected": len(result.beats),
        "n_beats_accepted": result.n_beats_accepted,
        "n_beats_rejected": result.n_beats_rejected,
        "label_counts": {c: result.beat_labels.count(c) for c in set(result.beat_labels)},
        "rhythm_findings": [f.kind for f in result.rhythm_findings],
        "risk": {
            "alert_level": result.risk_report.alert_level,
            "reasons": result.risk_report.alert_reasons,
            "pvc_burden_pct": round(result.risk_report.pvc_burden_pct, 2),
            "pac_burden_pct": round(result.risk_report.pac_burden_pct, 2),
            "vt_run_count": result.risk_report.vt_run_count,
            "afib_burden_pct": round(result.risk_report.afib_burden_pct, 2),
        },
        "temporal_trend": result.temporal_trend,
        "final_decision": result.merged_decision.final_decision,
        "bypassed_llm": result.merged_decision.bypassed_llm,
        "llm_rejected_reason": result.merged_decision.llm_rejected_reason,
        "audit_chain_valid": result.audit.verify_chain(),
        "audit_n_entries": len(result.audit.to_list()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["vitalpatch", "sensio"], default="vitalpatch")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--encoder", type=Path, default=MODELS_DIR / "ecg_encoder.pt")
    parser.add_argument("--classifier", type=Path, default=MODELS_DIR / "five_class_xgb.json")
    parser.add_argument("--no-classifier", action="store_true",
                         help="bypass the trained classifier and use RuleBasedBeatClassifier instead")
    args = parser.parse_args()

    encoder = None
    if args.encoder.exists():
        encoder = load_encoder(args.encoder)
        print(f"Loaded pretrained encoder from {args.encoder}")
    else:
        print(f"No pretrained encoder at {args.encoder} — run train_encoder.py first. "
              f"Continuing without learned embeddings.")

    if args.no_classifier:
        classifier = None
        print("--no-classifier set: using RuleBasedBeatClassifier fallback")
    else:
        classifier = FiveClassBeatClassifier()
        classifier.load(args.classifier)
        if not classifier.is_trained:
            print(f"ERROR: failed to load a trained classifier from {args.classifier} "
                  f"(classifier.is_trained is False after load()). Refusing to silently "
                  f"fall through to the rule-based fallback in a production run. "
                  f"Pass --no-classifier if that's actually what you want.")
            sys.exit(1)
        print(f"Loaded trained classifier from {args.classifier} "
              f"(classes: {list(classifier._label_encoder.classes_)})")

    pipeline = CliniauraPipeline(encoder=encoder, classifier=classifier)

    if args.source == "vitalpatch":
        files = discover_vitalpatch_files(DATA_RAW / "vitalpatch")[:args.limit]
        recordings = [r for f in files for r in parse_vitalpatch_ecg(f)]
    else:
        files = discover_sensio_files(DATA_RAW / "sense_io")[:args.limit]
        recordings = [parse_sensio_ecg(f) for f in files]

    for rec in recordings:
        result = pipeline.run(rec)
        print(json.dumps(summarize(result), indent=2, default=str))
        print("-" * 80)


if __name__ == "__main__":
    main()
