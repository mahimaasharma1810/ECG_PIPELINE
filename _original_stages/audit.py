"""SHA-256 hash-chained audit log.

Every pipeline decision — SQI window rejections, beat quality rejections,
classifier source (trained model vs. rule-based fallback), risk alerts,
and MedGemma accept/reject — is appended here so the whole run is
auditable end to end, per the architecture doc's Stage 9 description.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AuditEntry:
    seq: int
    timestamp: float
    event_type: str
    payload: dict
    prev_hash: str
    hash: str


class AuditLog:
    GENESIS_HASH = "0" * 64

    def __init__(self):
        self._entries: list[AuditEntry] = []

    def append(self, event_type: str, payload: dict) -> AuditEntry:
        prev_hash = self._entries[-1].hash if self._entries else self.GENESIS_HASH
        seq = len(self._entries)
        timestamp = time.time()
        body = json.dumps({"seq": seq, "timestamp": timestamp, "event_type": event_type,
                            "payload": payload, "prev_hash": prev_hash}, sort_keys=True, default=str)
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        entry = AuditEntry(seq, timestamp, event_type, payload, prev_hash, digest)
        self._entries.append(entry)
        return entry

    def verify_chain(self) -> bool:
        prev = self.GENESIS_HASH
        for e in self._entries:
            body = json.dumps({"seq": e.seq, "timestamp": e.timestamp, "event_type": e.event_type,
                                "payload": e.payload, "prev_hash": prev}, sort_keys=True, default=str)
            if hashlib.sha256(body.encode("utf-8")).hexdigest() != e.hash or e.prev_hash != prev:
                return False
            prev = e.hash
        return True

    def to_list(self) -> list[dict]:
        return [vars(e) for e in self._entries]

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_list(), indent=2, default=str))
