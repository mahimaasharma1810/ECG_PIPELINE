"""ProRhythm WebSocket packet schema - the REAL one, from device logs.

Replaces an assumed schema in the streaming client. The assumption was:

    {"samples": [{"timestamp_ms": ..., "amplitude": ...}]}

The device actually sends:

    top-level: b, app_type, pid, mac, cid, clid, test_id, sNo, start_time,
               end_time, duration, status, vitals, ecg_clean, vitals_status,
               match_id, os, new_score, new_score_freq, alerts, vital_alerts,
               alert
    vitals:    t, hr, skt, rr, spo2, hrv, sp, dp, finger_spo2
    ecg_clean: [{e, t}, ...]          <- e = amplitude, t = timestamp

The assumed client would have found no `samples` key, then looked for
`timestamp_ms` on the packet root, found nothing, and yielded ZERO samples
while connecting successfully and logging no error. Same silent-failure class
as the NUL-byte bug: a working connection producing no data, with no
complaint.

`sNo` is a per-packet sequence number. It is NOT present in any historical
capture CSV, but it IS in the live stream, so the live path can do what the
offline path cannot: detect replay by sequence repeat and count packet loss
from sequence gaps, rather than inferring both from timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass, field

ECG_FIELD = "ecg_clean"      # firmware-filtered; do NOT filter again
SAMPLE_AMP = "e"
SAMPLE_TS = "t"

VITALS_FIELDS = ("t", "hr", "skt", "rr", "spo2", "hrv", "sp", "dp", "finger_spo2")
SESSION_FIELDS = ("pid", "mac", "cid", "clid", "test_id", "sNo",
                  "start_time", "end_time", "duration", "status",
                  "vitals_status", "match_id", "os", "app_type")


class SchemaError(ValueError):
    """Packet does not match the ProRhythm schema."""


@dataclass
class PacketStats:
    n_packets: int = 0
    n_samples: int = 0
    n_seq_repeat: int = 0       # replay, detected by sNo repeat
    n_seq_gap: int = 0          # packet loss events
    n_packets_lost: int = 0     # summed sequence gap size
    n_malformed: int = 0
    seen_seq: set = field(default_factory=set)
    last_seq: int | None = None

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "seen_seq"}
        d["replay_packet_fraction"] = (self.n_seq_repeat / self.n_packets
                                       if self.n_packets else 0.0)
        d["packet_loss_fraction"] = (
            self.n_packets_lost / (self.n_packets + self.n_packets_lost)
            if (self.n_packets + self.n_packets_lost) else 0.0)
        return d


def parse_packet(pkt: dict, stats: PacketStats | None = None):
    """Yield (t_ms, amplitude) from one device packet, updating stats.

    Raises SchemaError when the packet does not carry the expected ECG field,
    so a schema change is LOUD rather than silently yielding nothing.
    """
    if not isinstance(pkt, dict):
        raise SchemaError(f"packet is {type(pkt).__name__}, expected object")
    if ECG_FIELD not in pkt:
        raise SchemaError(
            f"packet has no {ECG_FIELD!r} field (keys: {sorted(pkt)[:8]}). "
            f"Refusing to guess - a silently empty stream is worse than an error.")

    if stats is not None:
        stats.n_packets += 1
        s = pkt.get("sNo")
        if s is not None:
            try:
                s = int(s)
                if s in stats.seen_seq:
                    stats.n_seq_repeat += 1          # replayed packet
                else:
                    stats.seen_seq.add(s)
                    if stats.last_seq is not None and s > stats.last_seq + 1:
                        stats.n_seq_gap += 1
                        stats.n_packets_lost += s - stats.last_seq - 1
                    if stats.last_seq is None or s > stats.last_seq:
                        stats.last_seq = s
                if len(stats.seen_seq) > 100000:
                    stats.seen_seq.clear()
            except (TypeError, ValueError):
                pass

    block = pkt.get(ECG_FIELD) or []
    if not isinstance(block, list):
        raise SchemaError(f"{ECG_FIELD!r} is {type(block).__name__}, expected list")
    for smp in block:
        try:
            t = float(smp[SAMPLE_TS]); a = float(smp[SAMPLE_AMP])
        except (KeyError, TypeError, ValueError):
            if stats is not None:
                stats.n_malformed += 1
            continue
        if stats is not None:
            stats.n_samples += 1
        yield t, a


def parse_vitals(pkt: dict) -> dict | None:
    """Extract the vitals block, if present."""
    v = pkt.get("vitals")
    if not isinstance(v, dict):
        return None
    return {k: v.get(k) for k in VITALS_FIELDS if k in v}


def session_meta(pkt: dict) -> dict:
    """Session fields, captured rather than discarded."""
    return {k: pkt.get(k) for k in SESSION_FIELDS if k in pkt}
