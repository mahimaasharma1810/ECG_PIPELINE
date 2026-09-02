"""Tests for the three-axis layer.

Run:  python3 -m pytest tests_rhythm/test_axes.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm.features import window_features, INTERVALS_PER_WINDOW
from rhythm.verdict import decide, REGULAR, IRREGULAR, UNABLE
from rhythm.axes import assess_rate, assess_events, combine_axes, count_ectopic_couplets
from rhythm.axes.rate import BRADYCARDIC, TACHYCARDIC, NORMAL as RATE_NORMAL, UNDETERMINED
from rhythm.axes.events import BURDEN_NONE, BURDEN_LOW, BURDEN_HIGH, PAUSE_MS
from rhythm.axes.combine import NORMAL, ABNORMAL, CRITICAL, WITHHELD

SEED = 20260830


def steady(rr_ms, n=INTERVALS_PER_WINDOW):
    return np.full(n, float(rr_ms))


def full(rr, trustworthy=True, permitted=True):
    f = window_features(rr)
    v = decide(f, trustworthy, "ok" if trustworthy else "bSQI low")
    r = assess_rate(f, trustworthy)
    e = assess_events(rr, f, trustworthy)
    return r, v, e, combine_axes(r, v, e, permitted)


# ------------------------------------------------------------------ rate axis
@pytest.mark.parametrize("rr,expect,extreme", [
    (1714.0, BRADYCARDIC, True),    # 35 bpm
    (1200.0, BRADYCARDIC, False),   # 50 bpm
    (800.0,  RATE_NORMAL, False),   # 75 bpm
    (545.0,  TACHYCARDIC, False),   # 110 bpm
    (353.0,  TACHYCARDIC, True),    # 170 bpm
])
def test_rate_bands(rr, expect, extreme):
    r, _, _, _ = full(steady(rr))
    assert r.state == expect and r.extreme is extreme


def test_rate_refuses_independently():
    """Rate can be undetermined while other axes are not, and vice versa."""
    r, v, e, c = full(steady(800.0), trustworthy=False)
    assert r.state == UNDETERMINED and v.verdict == UNABLE and c.severity == WITHHELD


# ---------------------------------------------------------------- events axis
def test_pause_detected_and_escalates():
    rr = np.r_[steady(800.0, INTERVALS_PER_WINDOW - 1), 2500.0]
    r, v, e, c = full(rr)
    assert e.pause_count == 1 and e.longest_rr_ms == 2500.0
    assert c.severity == CRITICAL and c.escalate is True


def test_no_pause_below_threshold():
    rr = np.r_[steady(800.0, INTERVALS_PER_WINDOW - 1), PAUSE_MS - 1]
    _, _, e, _ = full(rr)
    assert e.pause_count == 0


def test_ectopic_couplet_counting():
    """A short interval followed by a compensatory one is one couplet."""
    rr = steady(800.0).copy()
    for i in (10, 20, 30):
        rr[i], rr[i + 1] = 560.0, 1000.0
    assert count_ectopic_couplets(rr) == 3


def test_burden_bands():
    rr = steady(800.0).copy()
    _, _, e, _ = full(rr)
    assert e.burden == BURDEN_NONE
    rr[10], rr[11] = 560.0, 1000.0
    _, _, e, _ = full(rr)
    assert e.burden == BURDEN_LOW
    rr = steady(800.0).copy()
    for i in range(0, 42, 6):
        rr[i], rr[i + 1] = 560.0, 1000.0
    _, _, e, _ = full(rr)
    assert e.burden == BURDEN_HIGH


def test_events_axis_never_reports_beat_type():
    """Burden only. Type is a waveform property timing cannot reach."""
    rr = steady(800.0).copy(); rr[10], rr[11] = 560.0, 1000.0
    _, _, e, _ = full(rr)
    d = e.as_dict()
    for banned in ("pac", "pvc", "type", "atrial", "ventricular"):
        assert not any(banned in str(k).lower() for k in d)


# ----------------------------------------------------------- combination layer
def test_regular_tachycardia_is_flagged_the_SVT_gap():
    """The gap a regularity-only view misses: fast but evenly spaced."""
    r, v, e, c = full(steady(462.0))          # 130 bpm, perfectly steady
    assert v.verdict == REGULAR, "precondition: regularity axis says REGULAR"
    assert r.state == TACHYCARDIC
    assert c.severity == ABNORMAL, "combination layer must not call this normal"
    assert "regularity-only" in c.reason


def test_normal_requires_regularity_to_be_determined():
    """NORMAL is never asserted while regularity is undetermined."""
    rr = steady(800.0)
    f = window_features(rr)
    v = decide(f, sqi_passed=False, sqi_reason="bSQI 0.90 < 0.95")
    r = assess_rate(f, True); e = assess_events(rr, f, True)
    c = combine_axes(r, v, e, clinical_permitted=True)
    assert v.verdict == UNABLE and c.severity == WITHHELD


def test_irregular_does_not_escalate():
    """PPV ~0.47 - prompts review, never an alarm."""
    rng = np.random.default_rng(SEED)
    rr = 706.0 * (1 + rng.normal(0, 0.30, INTERVALS_PER_WINDOW))
    r, v, e, c = full(rr)
    assert v.verdict == IRREGULAR
    assert c.severity == ABNORMAL and c.escalate is False


def test_only_rate_extremes_and_pauses_escalate():
    _, _, _, c_brady = full(steady(1714.0))
    _, _, _, c_tachy = full(steady(353.0))
    assert c_brady.escalate and c_tachy.escalate
    _, _, _, c_ok = full(steady(800.0))
    assert not c_ok.escalate


def test_gate_closed_withholds_everything():
    _, _, _, c = full(steady(353.0), permitted=False)
    assert c.severity == WITHHELD and c.escalate is False


def test_healthy_steady_window_is_normal():
    _, _, _, c = full(steady(800.0))
    assert c.severity == NORMAL and c.escalate is False
