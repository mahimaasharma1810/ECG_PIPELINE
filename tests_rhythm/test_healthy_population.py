"""TASK D - the standing physiological check.

*** THIS TEST IS EXPECTED TO FAIL TODAY. THAT IS ITS PURPOSE. ***

All 37 ProRhythm captures are from healthy, self-recording adults. A working
pipeline should therefore return mostly REGULAR on them.

It does not. It returns:
  - gated:   38 of 39 windows UNABLE_TO_DETERMINE (bSQI below 0.95)
  - ungated: 34 of 39 windows IRREGULAR, 0 REGULAR

Because R-peak detection on this device finds ~4.4% spurious peaks, and each
false peak splits one RR interval into two, manufacturing irregularity. Device
windows show median RR CV ~0.19 against ~0.03 for genuinely healthy windows in
expert-labelled public data - 6x higher on a HEALTHIER population.

THIS IS THE SINGLE TEST THAT WILL TELL US WHEN THE DETECTOR IS FIXED.

Do not skip it. Do not weaken it. Do not lower bsqi_min to make it pass -
that produces 34/39 IRREGULAR on healthy people, which is worse than refusing.

It is marked xfail(strict=True), so:
  - while detection is broken: reports xfail, suite stays green
  - the moment it PASSES: reports XPASS and FAILS the suite, forcing someone
    to confirm the detector was genuinely fixed and update this file.

Run:  python3 -m pytest tests_rhythm/test_healthy_population.py -q -rxX
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm.pipeline import process_capture
from rhythm.verdict import REGULAR, IRREGULAR, UNABLE

# a mid-length, replay-free capture; representative and quick
CAPTURE = "prorithm_ecg/1789/2026-08-17_08.csv"
MIN_REGULAR_FRACTION = 0.50     # a healthy adult should read mostly regular


@pytest.fixture(scope="module")
def windows():
    return [r for r in process_capture(CAPTURE) if "error" not in r]


@pytest.mark.xfail(strict=True,
                   reason="BLOCKED on device R-peak detection (~4.4% spurious "
                          "peaks). Expected to fail until a controlled reference "
                          "recording allows the detector to be fixed and verified. "
                          "An XPASS here means someone should confirm the fix.")
def test_healthy_population_reads_regular(windows):
    """Healthy adults should mostly read REGULAR. Currently they do not."""
    verdicts = [w["verdict"] if "verdict" in w else None for w in windows]
    n = len(windows)
    assert n > 0
    n_regular = sum(1 for w in windows
                    if w.get("sqi_passed") and w.get("rr_cv") is not None
                    and w["rr_cv"] < 0.1275)
    assert n_regular / n >= MIN_REGULAR_FRACTION, (
        f"only {n_regular}/{n} windows read REGULAR "
        f"({n_regular/n:.1%}, need >={MIN_REGULAR_FRACTION:.0%})")


def test_current_failure_mode_is_documented(windows):
    """Pins the CURRENT behaviour so a silent change is visible.

    This test passes today. If it starts failing, the pipeline's behaviour on
    device data has changed - which may be the fix, or may be a regression.
    """
    n = len(windows)
    refused = sum(1 for w in windows if not w.get("sqi_passed"))
    assert refused / n > 0.80, (
        f"only {refused}/{n} windows refused - device behaviour has CHANGED. "
        f"If detection was fixed, update this test and "
        f"test_healthy_population_reads_regular together.")


def test_ungated_would_be_wrong(windows):
    """Removing the SQI gate would label healthy adults IRREGULAR.

    Documents WHY bsqi_min must not be loosened to increase yield.
    """
    cvs = [w["rr_cv"] for w in windows if w.get("rr_cv") is not None]
    n_irr = sum(1 for c in cvs if c > 0.1545)      # above the refusal band
    n_reg = sum(1 for c in cvs if c < 0.1005)      # below it
    assert n_irr > n_reg, (
        "ungated device data no longer skews IRREGULAR - detection may have "
        "been fixed; re-check test_healthy_population_reads_regular")
