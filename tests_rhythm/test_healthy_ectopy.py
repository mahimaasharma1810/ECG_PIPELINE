"""Standing check on the EVENTS axis.

*** THIS TEST IS EXPECTED TO FAIL TODAY. THAT IS ITS PURPOSE. ***

Companion to test_healthy_population_reads_regular. That one watches the
REGULARITY axis; this one watches the EVENTS axis, because they fail for
related but distinct reasons and one can be fixed without the other.

All 37 ProRhythm captures are healthy, self-recording adults. In a healthy
population ectopic beats are uncommon - typically well under 1% of beats, so
most 64-beat windows should contain none at all.

Measured on device data (reports_rhythm/s21_axes_device.csv, 1,210 windows):
    windows with at least one ectopic couplet : 95.8%
    median couplets per window                : 3.0
    windows classed HIGH burden               : 29.8%
    windows containing a "pause" over 2000 ms : 7.5%

Not credible for this population.

WHY THE EVENTS AXIS IS ESPECIALLY VULNERABLE. A spurious R-peak splits one
interval into a SHORT one followed by a LONGER one. That is precisely the
short-then-compensatory signature the ectopy detector searches for. The
detector is not merely degraded by detection error - it is tuned to the shape
that detection error produces. A missed beat does the mirror image: two
intervals merge into one long one, which reads as a pause.

CONSEQUENCE: the events axis needs its OWN device-domain validation criterion.
Beat-COUNT accuracy is not sufficient for it. A detector could get the count
right while still placing beats into short/long pairs, which would satisfy a
count-based check and still produce false ectopy. Ectopy burden must be
validated against the beat-level reference independently.

Run:  python3 -m pytest tests_rhythm/test_healthy_ectopy.py -q -rxX
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm.ingest import load_capture
from rhythm.resample import to_uniform_segments
from rhythm.pipeline import detect_peaks
from rhythm.features import window_features, BEATS_PER_WINDOW
from rhythm.axes.events import count_ectopic_couplets, PAUSE_MS

CAPTURE = "prorithm_ecg/1789/2026-08-17_08.csv"

# Healthy adults: ectopy is uncommon, so most windows should contain none.
# Generous bounds - they still fail by a wide margin today.
MAX_WINDOWS_WITH_ANY_ECTOPY = 0.30
MAX_WINDOWS_WITH_PAUSE = 0.02


@pytest.fixture(scope="module")
def device_windows():
    cap = load_capture(CAPTURE)
    segs, _ = to_uniform_segments(cap.t_ms, cap.amplitude)
    out = []
    for seg in segs:
        try:
            pa, _ = detect_peaks(seg.x, seg.fs)
        except Exception:
            continue
        if pa.size < BEATS_PER_WINDOW:
            continue
        for s in range(0, pa.size - BEATS_PER_WINDOW + 1, BEATS_PER_WINDOW):
            w = pa[s:s + BEATS_PER_WINDOW]
            i0, i1 = int(w[0]), int(w[-1])
            fs_loc = seg.local_fs(i0, i1)
            if not np.isfinite(fs_loc) or fs_loc <= 0:
                continue
            rr = np.diff(w).astype(float) * 1000.0 / fs_loc
            if not window_features(rr).usable:
                continue
            out.append(dict(couplets=count_ectopic_couplets(rr),
                            pauses=int(np.sum(rr > PAUSE_MS))))
    return out


@pytest.mark.xfail(strict=True,
                   reason="BLOCKED on device R-peak detection. A spurious peak "
                          "produces exactly the short-then-compensatory pattern "
                          "the ectopy detector looks for, so detection error is "
                          "read as ectopy. Expected to fail until the detector "
                          "is fixed AND the events axis is validated against a "
                          "beat-level reference in its own right. An XPASS means "
                          "someone should confirm both.")
def test_healthy_population_has_low_ectopy_burden(device_windows):
    """Healthy adults should mostly show NO ectopic beats. Currently 96% do."""
    n = len(device_windows)
    assert n > 0
    with_ectopy = sum(1 for w in device_windows if w["couplets"] > 0)
    assert with_ectopy / n <= MAX_WINDOWS_WITH_ANY_ECTOPY, (
        f"{with_ectopy}/{n} windows ({with_ectopy/n:.1%}) show ectopic beats "
        f"in a healthy population; expected <= {MAX_WINDOWS_WITH_ANY_ECTOPY:.0%}")


@pytest.mark.xfail(strict=True,
                   reason="BLOCKED on device R-peak detection. A MISSED beat "
                          "merges two intervals into one long interval, which "
                          "reads as a pause. Expected to fail until detection "
                          "is fixed.")
def test_healthy_population_has_no_pauses(device_windows):
    """Healthy adults should not show pauses over 2 s. Currently 7.5% do."""
    n = len(device_windows)
    with_pause = sum(1 for w in device_windows if w["pauses"] > 0)
    assert with_pause / n <= MAX_WINDOWS_WITH_PAUSE, (
        f"{with_pause}/{n} windows ({with_pause/n:.1%}) contain a pause over "
        f"{PAUSE_MS:.0f} ms in a healthy population")


def test_current_ectopy_failure_mode_is_documented(device_windows):
    """Pins CURRENT behaviour so a silent change is visible.

    Passes today. If it starts failing, device behaviour has changed - which
    may be the fix, or may be a regression. Either way someone must look.
    """
    n = len(device_windows)
    with_ectopy = sum(1 for w in device_windows if w["couplets"] > 0)
    assert with_ectopy / n > 0.50, (
        f"only {with_ectopy}/{n} windows show ectopy - device behaviour has "
        f"CHANGED. If detection was fixed, update the xfail tests in this file "
        f"and in test_healthy_population.py together.")
