"""TASK A - replay captured CSVs as REAL-SCHEMA device packets.

Exercises the whole live chain without a patch:

    CSV -> device-schema packets -> ws parse -> dedupe -> grid -> R-peaks
        -> RR -> features -> SQI -> classification -> JSON -> narrative

This is the integration test we would otherwise run against a live device.
It would have caught the assumed-schema bug immediately: the old client,
fed real packets, yielded zero samples without error.

Faults are injectable so the detection paths are exercised rather than
assumed: replayed blocks (sNo repeat), dropped packets (sNo gap), and
malformed packets (must raise, never silently yield nothing).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLES_PER_PACKET = 7          # device sends bursts of ~6.5


@dataclass
class FaultConfig:
    replay_block_every: int = 0     # every Nth packet, re-send a past block
    replay_block_len: int = 12
    drop_every: int = 0             # every Nth packet, drop it (sNo gap)
    malformed_every: int = 0        # every Nth packet, corrupt the schema
    seed: int = 20260829


@dataclass
class Replayer:
    """Turn a capture CSV into a stream of real-schema device packets."""
    path: str | Path
    patient_id: str | None = None
    faults: FaultConfig = field(default_factory=FaultConfig)
    samples_per_packet: int = SAMPLES_PER_PACKET

    def __post_init__(self):
        self.path = Path(self.path)
        if self.patient_id is None:
            self.patient_id = self.path.parent.name
        df = pd.read_csv(self.path, usecols=["timestamp_ms", "amplitude"])
        t = df.timestamp_ms.to_numpy(float); a = df.amplitude.to_numpy(float)
        ok = np.isfinite(t) & np.isfinite(a)
        self.t, self.a = t[ok], a[ok]
        self._rng = random.Random(self.faults.seed)

    def n_packets(self) -> int:
        return int(np.ceil(self.t.size / self.samples_per_packet))

    def packets(self):
        """Yield dicts in the REAL ProRhythm schema."""
        n = self.samples_per_packet
        sent, sno = [], 0
        total = self.n_packets()
        for k in range(total):
            block = [{"e": float(self.a[i]), "t": float(self.t[i])}
                     for i in range(k * n, min((k + 1) * n, self.t.size))]
            if not block:
                continue
            f = self.faults

            if f.drop_every and (k + 1) % f.drop_every == 0:
                sno += 1                      # burn the sequence number: loss
                continue

            sno += 1
            pkt = self._wrap(block, sno)

            if f.malformed_every and (k + 1) % f.malformed_every == 0:
                bad = dict(pkt); bad.pop("ecg_clean")
                bad["ecg"] = block            # renamed field -> schema violation
                yield bad
                continue

            sent.append(pkt)
            yield pkt

            if (f.replay_block_every and (k + 1) % f.replay_block_every == 0
                    and len(sent) > f.replay_block_len):
                for old in sent[-f.replay_block_len:]:
                    yield dict(old)           # same sNo -> replay

    def _wrap(self, block, sno) -> dict:
        t0 = block[0]["t"]
        return {
            "b": 1, "app_type": "prorhythm", "pid": self.patient_id,
            "mac": "FD:DC:A1:9C:48:40", "cid": "c1", "clid": "cl1",
            "test_id": self.path.stem, "sNo": sno,
            "start_time": t0, "end_time": block[-1]["t"],
            "duration": (block[-1]["t"] - t0) / 1000.0,
            "status": "streaming",
            "vitals": {"t": t0, "hr": None, "skt": None, "rr": None,
                       "spo2": None, "hrv": None, "sp": None, "dp": None,
                       "finger_spo2": None},
            "ecg_clean": block,
            "vitals_status": "valid", "match_id": None, "os": "android",
            "new_score": None, "new_score_freq": None,
            "alerts": [], "vital_alerts": [], "alert": None,
        }
