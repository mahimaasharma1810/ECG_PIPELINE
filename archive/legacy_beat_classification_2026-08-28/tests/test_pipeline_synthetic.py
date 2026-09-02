"""test_pipeline_synthetic.py -- self-test suite for the ECG pipeline's
deterministic rules and guards, using KNOWN-ground-truth synthetic signals
from synthetic_ecg.py.

============================================================================
WHAT THIS TESTS (and what it does NOT test)
============================================================================
For each labeled scenario, a synthetic signal with a KNOWN injected ground
truth is generated and run through the real, frozen, EXISTING pipeline
(agent_bridge.run_full_report -> ECGPipeline.run() -> the production
five_class_xgb.json classifier + RhythmContextEngine + score_recording()),
and the result is asserted against that ground truth:

  * VT_RUN     -> a VT_RUN rhythm finding is detected AND risk_level is CRITICAL.
  * PVC_BURDEN -> the PVC-burden rule fires (rhythm/burden pushed above the
                  HIGH threshold) AND risk_level is HIGH or CRITICAL.
  * AFIB_LIKE  -> AFIB_SUSPECTED appears in rhythm_findings.
  * NORMAL     -> risk_level is LOW and no dangerous rule fired.
  * NOISY      -> either NOT_ASSESSABLE, or (if some signal did survive) a
                  result that is NOT a false-clean LOW on effectively-empty
                  data.

This tests the PIPELINE's rule engine, rhythm-finding logic, and the
NOT_ASSESSABLE guard reacting correctly to synthetic INPUT -- it is
regression protection for those rules, NOT a measurement of the beat
classifier's real-world accuracy (see synthetic_ecg.py's Hard Boundary /
Morphology Disclaimer: this classifier's real accuracy can only be judged
on real, annotated data such as DS1/DS2, never on this generator's output).

Run directly for a plain pass/fail report (no pytest dependency needed):
    python -m ecg_pipeline.test_pipeline_synthetic
or under pytest (functions are also plain `assert`-based, pytest-discoverable):
    pytest ecg_pipeline/test_pipeline_synthetic.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
import traceback

# Marker-anchored, not hop-counted -- see scripts/verify_sqi_gate.py. Needed
# because this test moved out of the ecg_pipeline package into tests/.
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ecg_pipeline").is_dir())
sys.path.insert(0, str(REPO_ROOT))

from ecg_pipeline import agent_bridge  # noqa: E402
from ecg_pipeline.ecg_pipeline_core import MODELS_DIR, RISK  # noqa: E402
from ecg_pipeline.synthetic_ecg import generate_scenario  # noqa: E402

_DANGEROUS_KINDS = {"VT_RUN", "BIGEMINY", "TRIGEMINY"}


def _load_classifier():
    return agent_bridge.load_classifier(MODELS_DIR / "five_class_xgb.json")


def _run(scenario: str, **kwargs):
    """Generates one scenario and runs it through the real, unmodified
    pipeline (via agent_bridge.run_full_report, exactly as any real
    recording would be). Returns (report_json, result, ground_truth)."""
    classifier = _load_classifier()
    recording, ground_truth = generate_scenario(scenario, **kwargs)
    report_json, result = agent_bridge.run_full_report(recording, classifier)
    return report_json, result, ground_truth


def test_normal():
    report, result, gt = _run("NORMAL")
    rhythm_kinds = {f["kind"] for f in report["rhythm_findings"]}
    assert report["assessable"], f"NORMAL scenario unexpectedly NOT_ASSESSABLE: {report.get('not_assessable_reason')}"
    assert report["risk_level"] == "LOW", (
        f"NORMAL scenario expected risk_level LOW, got {report['risk_level']!r} "
        f"(deciding rule: {report['deciding_rule']['condition']})")
    assert not (rhythm_kinds & _DANGEROUS_KINDS), (
        f"NORMAL scenario unexpectedly fired a dangerous rhythm finding: {rhythm_kinds}")


def test_pvc_burden():
    report, result, gt = _run("PVC_BURDEN")
    assert report["assessable"], f"PVC_BURDEN scenario unexpectedly NOT_ASSESSABLE: {report.get('not_assessable_reason')}"
    pvc_rule_fired = any(
        r["fired"] and "PVC burden" in r["condition"] for r in report["rule_trace"]
    )
    assert pvc_rule_fired, (
        f"PVC_BURDEN scenario (ground truth: {gt.pvc_target_pct:.1f}% injected V beats) did not trip "
        f"either PVC-burden rule. rule_trace: "
        f"{[(r['condition'], r['measured_value'], r['fired']) for r in report['rule_trace']]}")
    assert report["risk_level"] in ("HIGH", "CRITICAL"), (
        f"PVC_BURDEN scenario expected risk_level HIGH or CRITICAL, got {report['risk_level']!r}")


def test_vt_run():
    report, result, gt = _run("VT_RUN")
    assert report["assessable"], f"VT_RUN scenario unexpectedly NOT_ASSESSABLE: {report.get('not_assessable_reason')}"
    rhythm_kinds = [f["kind"] for f in report["rhythm_findings"]]
    assert "VT_RUN" in rhythm_kinds, (
        f"VT_RUN scenario (ground truth: {gt.vt_run_length} consecutive V-like beats injected at "
        f"positions {gt.ectopic_cycle_indices}) did not produce a VT_RUN rhythm finding. "
        f"rhythm_findings: {rhythm_kinds}. beat_labels V-count: {result.beat_labels.count('V')}")
    assert report["risk_level"] == "CRITICAL", (
        f"VT_RUN scenario found a VT_RUN finding but risk_level was {report['risk_level']!r}, not CRITICAL "
        f"(deciding rule: {report['deciding_rule']['condition']})")


def test_afib_like():
    report, result, gt = _run("AFIB_LIKE")
    assert report["assessable"], f"AFIB_LIKE scenario unexpectedly NOT_ASSESSABLE: {report.get('not_assessable_reason')}"
    rhythm_kinds = [f["kind"] for f in report["rhythm_findings"]]
    assert "AFIB_SUSPECTED" in rhythm_kinds, (
        f"AFIB_LIKE scenario (ground truth RR CV ~= {gt.afib_rr_cv_target:.3f}, rule threshold "
        f"{RISK.afib_rr_cv_threshold:.2f}) did not "
        f"produce an AFIB_SUSPECTED rhythm finding. rhythm_findings: {rhythm_kinds}")


def test_noisy():
    report, result, gt = _run("NOISY")
    if not report["assessable"]:
        assert report["risk_level"] == "NOT_ASSESSABLE", (
            f"NOISY scenario is not assessable but risk_level is {report['risk_level']!r}, not NOT_ASSESSABLE "
            f"-- this is exactly the false-clean-reading failure mode this guard exists to prevent.")
        assert report["recording"]["n_beats_analyzed"] < 5, (
            "NOISY scenario marked NOT_ASSESSABLE despite having enough beats analyzed -- "
            "guard fired for the wrong reason.")
    else:
        # Some signal survived the quality gate -- that's fine (heavy noise doesn't
        # guarantee total rejection), but it must not be a false-clean LOW built
        # from a near-empty analysis. A genuinely-assessed result is acceptable at
        # any risk level as long as it analyzed a real number of beats.
        assert report["recording"]["n_beats_analyzed"] >= 5, (
            f"NOISY scenario reported assessable=True with only "
            f"{report['recording']['n_beats_analyzed']} beats analyzed -- should have been NOT_ASSESSABLE "
            f"instead of a result that could be misread as a clean reading.")


TESTS = [
    ("NORMAL", test_normal),
    ("PVC_BURDEN", test_pvc_burden),
    ("VT_RUN", test_vt_run),
    ("AFIB_LIKE", test_afib_like),
    ("NOISY", test_noisy),
]


def main() -> int:
    print("=" * 78)
    print("ECG PIPELINE SYNTHETIC SELF-TEST SUITE")
    print("(tests the pipeline's rules/guards against KNOWN synthetic ground truth --")
    print(" NOT a measurement of classifier accuracy; see synthetic_ecg.py's Hard Boundary)")
    print("=" * 78)

    results = []
    for name, fn in TESTS:
        try:
            fn()
            results.append((name, True, None))
            print(f"[PASS] {name}")
        except AssertionError as e:
            results.append((name, False, str(e)))
            print(f"[FAIL] {name}: {e}")
        except Exception as e:
            results.append((name, False, f"{type(e).__name__}: {e}"))
            print(f"[ERROR] {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)

    n_pass = sum(1 for _, ok, _ in results if ok)
    print("=" * 78)
    print(f"SUMMARY: {n_pass}/{len(results)} scenarios passed")
    for name, ok, err in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {err}" if err else ""))
    print("=" * 78)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
