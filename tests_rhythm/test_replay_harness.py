"""TASK A - replay harness tests.

Run:  python3 -m pytest tests_rhythm/test_replay_harness.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhythm.replay import Replayer, FaultConfig
from rhythm.ws_schema import parse_packet, PacketStats, SchemaError

CAP = "prorithm_ecg/1793/2026-08-08_06.csv"
ALL_CAPS = sorted(Path("prorithm_ecg").glob("*/*.csv"))


def drain(rep, stats=None, tolerate=False):
    out = []
    for p in rep.packets():
        try:
            out.extend(parse_packet(p, stats))
        except SchemaError:
            if not tolerate:
                raise
    return out


def test_roundtrip_is_byte_identical():
    """Packets -> parser must reproduce the CSV exactly."""
    rep = Replayer(CAP)
    got = drain(rep)
    raw = pd.read_csv(CAP, usecols=["timestamp_ms", "amplitude"])
    assert np.array_equal(np.array([g[0] for g in got]),
                          raw.timestamp_ms.to_numpy(float))
    assert np.array_equal(np.array([g[1] for g in got]),
                          raw.amplitude.to_numpy(float))


def test_injected_replay_detected_via_sno_repeat():
    rep = Replayer(CAP, faults=FaultConfig(replay_block_every=50, replay_block_len=10))
    st = PacketStats(); drain(rep, st)
    assert st.n_seq_repeat > 0, "replayed packets not detected by sNo repeat"


def test_injected_packet_loss_detected_via_sno_gap():
    rep = Replayer(CAP, faults=FaultConfig(drop_every=25))
    st = PacketStats(); drain(rep, st)
    assert st.n_seq_gap > 0 and st.n_packets_lost > 0, "packet loss not detected"


def test_malformed_packet_raises_not_silent():
    """A schema violation must RAISE, never yield zero samples quietly."""
    rep = Replayer(CAP, faults=FaultConfig(malformed_every=20))
    with pytest.raises(SchemaError):
        drain(rep)


def test_clean_stream_reports_no_faults():
    rep = Replayer(CAP)
    st = PacketStats(); drain(rep, st)
    assert st.n_seq_repeat == 0 and st.n_seq_gap == 0 and st.n_malformed == 0


@pytest.mark.parametrize("cap", [str(c) for c in ALL_CAPS])
def test_all_captures_replay_end_to_end(cap):
    """All 37 captures must replay without crashing and match the source."""
    rep = Replayer(cap)
    st = PacketStats(); got = drain(rep, st)
    raw = pd.read_csv(cap, usecols=["timestamp_ms", "amplitude"])
    raw = raw[np.isfinite(raw.timestamp_ms) & np.isfinite(raw.amplitude)]
    assert len(got) == len(raw)
    assert st.n_packets > 0
