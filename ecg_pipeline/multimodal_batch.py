"""6-patient multimodal batch runner (analysis script, not part of the shipped package).

Replicates ecg_pipeline.agent_bridge.push_main's per-segment semantics exactly,
but as a single in-process loop instead of one subprocess per file:
  * the 15MB XGBoost classifier is loaded ONCE, not 2,375 times
  * push_main has no --input/--output flags (verified: its argparse only accepts
    --vitalpatch-root/--vitals-root/--classifier/--agent-url/--api-key/--limit/--seed),
    so a per-file subprocess call is not possible at all

Faithfulness notes:
  * combined risk is computed via a real second ECGPipeline.run(recording,
    news2_score=..., qsofa_score=...) call, NOT by calling score_recording()
    directly -- score_recording needs afib_windows_examined, which PipelineResult
    does not expose, so a direct re-score would silently change afib_burden_pct.
  * agent-provided NEWS2/qSOFA supersede the local partial approximation when the
    push succeeds (news2_source="agent_live"), same rule as push_main.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ecg_pipeline.agent_bridge import (  # noqa: E402
    DEFAULT_AGENT_URL, DEFAULT_API_KEY, MIN_BEATS_FOR_ASSESSMENT,
    REQUIRED_AGENT_VITALS_FIELDS, _AGENT_FIELD_TO_LOCAL_KEY,
    compute_partial_news2, compute_qsofa_proxy, load_classifier,
    load_real_vitals, run_ecg_segment, submit_to_agent,
)
from ecg_pipeline.ecg_pipeline_core import (  # noqa: E402
    MODELS_DIR, ECGPipeline, parse_vitalpatch_ecg,
)

ECG_ROOT = REPO_ROOT / "data" / "raw" / "vitalpatch"
VITALS_ROOT = str(REPO_ROOT / "data" / "vitals_downloads")
OUT_DIR = REPO_ROOT / "data" / "reports" / "multimodal_batch"
PATIENTS = ["Patch_183594", "Patch_1844AC", "Patch_184635",
            "Patch_1849DF", "Patch_184B27", "Patch_184B2F"]

PER_PATIENT_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 = all


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    classifier = load_classifier(MODELS_DIR / "five_class_xgb.json")

    summary: list[dict] = []
    errors: list[dict] = []
    t0 = time.time()

    for patient in PATIENTS:
        files = sorted((ECG_ROOT / patient).glob("*_ecg.csv"))
        if PER_PATIENT_LIMIT:
            files = files[:PER_PATIENT_LIMIT]
        print(f"\n{patient}: {len(files)} ECG files", flush=True)

        for fi, f in enumerate(files):
            try:
                recordings = parse_vitalpatch_ecg(f)
            except Exception as e:  # real parse failures (e.g. the '-' sentinel bug)
                errors.append({"patient": patient, "segment": f.stem,
                               "stage": "parse", "error": f"{type(e).__name__}: {e}"[:300]})
                continue

            vitals = load_real_vitals(ecg_path=str(f), vitals_root=VITALS_ROOT)
            partial_news2 = compute_partial_news2(vitals)
            qsofa_proxy = compute_qsofa_proxy(vitals)

            for recording in recordings:
                try:
                    ecg_risk, result = run_ecg_segment(recording, classifier)
                    ecg_only_risk = result.risk_report.alert_level
                    n_analyzed = result.n_beats_accepted
                    assessable = n_analyzed >= MIN_BEATS_FOR_ASSESSMENT

                    kinds = [f.kind for f in result.rhythm_findings]
                    row = {
                        "patient": patient,
                        "segment": recording.segment_id,
                        "assessable": assessable,
                        "n_beats_analyzed": n_analyzed,
                        "rhythm_findings": len(result.rhythm_findings),
                        # kinds, not just a count -- needed to tell AFIB_SUSPECTED
                        # apart from VT_RUN/BIGEMINY/TRIGEMINY in the analysis
                        "rhythm_kinds": dict(Counter(kinds)),
                        "has_afib": "AFIB_SUSPECTED" in kinds,
                        "has_vt_run": "VT_RUN" in kinds,
                        "pvc_burden_pct": ecg_risk.get("pvc_burden_pct"),
                        "pac_burden_pct": ecg_risk.get("pac_burden_pct"),
                        "afib_burden_pct": ecg_risk.get("afib_burden_pct"),
                        "vt_run_count": ecg_risk.get("vt_run_count"),
                        "hrv_suppressed": ecg_risk.get("hrv_suppressed"),
                        "alert_reasons": ecg_risk.get("alert_reasons"),
                        "ecg_only_risk": ecg_only_risk,
                        "combined_risk": ecg_only_risk,
                        "override_activated": False,
                        "local_partial_news2": partial_news2["partial_news2_score"],
                        "local_news2_coverage": partial_news2["coverage"],
                        "agent_news2_score": None,
                        "agent_news2_coverage": None,
                        "agent_qsofa_score": None,
                        "news2_source": "local_partial",
                        "hr": vitals["hr"]["value"],
                        "rr": vitals["respiratory_rate"]["value"],
                        "temp": vitals["temperature"]["value"],
                        "posture": vitals["posture"].get("most_common"),
                        "vitals_file_found": vitals["vitals_file_found"],
                        "vitals_time_offset_ms": vitals["vitals_time_offset_ms"],
                        "agent_push_status": None,
                        "agent_push_error": None,
                    }

                    if not vitals["vitals_file_found"]:
                        row["agent_push_status"] = "SKIPPED_NO_VITALS_FILE"
                        summary.append(row)
                        continue

                    missing_required = [
                        af for af in REQUIRED_AGENT_VITALS_FIELDS
                        if vitals[_AGENT_FIELD_TO_LOCAL_KEY[af]]["value"] is None
                    ]
                    if missing_required:
                        row["agent_push_status"] = "SKIPPED_MISSING_REQUIRED_FIELDS"
                        row["agent_push_error"] = f"missing: {missing_required}"
                        summary.append(row)
                        continue

                    vitals_values = {
                        "heart_rate": vitals["hr"]["value"],
                        "spo2": vitals["spo2"]["value"],
                        "systolic_bp": vitals["sbp"]["value"],
                        "diastolic_bp": vitals["dbp"]["value"],
                        "respiratory_rate": vitals["respiratory_rate"]["value"],
                        "temperature": vitals["temperature"]["value"],
                    }
                    agent_result = submit_to_agent(
                        recording.patient_id, vitals_values, ecg_risk,
                        DEFAULT_AGENT_URL, DEFAULT_API_KEY)
                    row["agent_push_status"] = agent_result["push_status"]
                    row["agent_push_error"] = agent_result.get("error")

                    resp = agent_result.get("agent_response") or {}
                    n2 = resp.get("news2") or {}
                    qs = resp.get("qsofa") or {}
                    row["agent_news2_score"] = n2.get("total_score")
                    row["agent_news2_coverage"] = n2.get("coverage")
                    row["agent_qsofa_score"] = qs.get("score")
                    # per-component breakdown, so the analysis can attribute the
                    # score to HR vs RR vs Temp without recomputing it
                    row["agent_hr_score"] = n2.get("heart_rate_score")
                    row["agent_rr_score"] = n2.get("respiratory_rate_score")
                    row["agent_temp_score"] = n2.get("temperature_score")
                    row["agent_news2_missing"] = n2.get("missing_components")
                    row["agent_qsofa_rr_flag"] = qs.get("rr_flag")
                    row["agent_qsofa_hr_flag"] = qs.get("hr_flag")

                    if row["agent_news2_score"] is not None:
                        rescored = ECGPipeline(classifier=classifier).run(
                            recording,
                            news2_score=row["agent_news2_score"],
                            qsofa_score=row["agent_qsofa_score"])
                        row["combined_risk"] = rescored.risk_report.alert_level
                        row["override_activated"] = row["combined_risk"] != ecg_only_risk
                        row["news2_source"] = "agent_live"

                    summary.append(row)

                except Exception as e:
                    errors.append({"patient": patient, "segment": recording.segment_id,
                                   "stage": "process", "error": f"{type(e).__name__}: {e}"[:300]})

            if (fi + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  {patient} {fi+1}/{len(files)} files | "
                      f"{len(summary)} segs | {len(errors)} errs | {el:.0f}s", flush=True)

    manifest = {"summary": summary, "errors": errors,
                "elapsed_s": round(time.time() - t0, 1)}
    (OUT_DIR / "multimodal_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nDONE  segments={len(summary)}  errors={len(errors)}  "
          f"elapsed={manifest['elapsed_s']}s")
    print(f"Manifest: {OUT_DIR / 'multimodal_manifest.json'}")
    print("push status:", dict(Counter(r["agent_push_status"] for r in summary)))


if __name__ == "__main__":
    main()
