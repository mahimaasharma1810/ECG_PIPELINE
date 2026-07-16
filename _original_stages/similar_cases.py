"""Recommendation #8 — 'Show the AI similar past patient cases before it
answers' (Med-High impact, High effort in the original ranking, but made
tractable here by reusing the encoder embeddings we already compute for
recommendation #1, rather than standing up separate infrastructure).

`SimilarCaseIndex` is a lightweight nearest-neighbour store over encoder
embeddings (see `encoder.py`) plus each case's outcome metadata. It grounds
the stage-9 report in real precedent instead of the LLM guessing from the
current recording alone. Runs locally (no server dependency) via
scikit-learn's NearestNeighbors, trading the review's "no (needs a
server)" limitation for a smaller, in-process index.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class CaseRecord:
    case_id: str
    patient_id: str
    embedding: np.ndarray
    outcome_label: str          # e.g. final alert_level for that recording/beat
    metadata: dict = field(default_factory=dict)


class SimilarCaseIndex:
    def __init__(self):
        self._records: list[CaseRecord] = []
        self._nn = None

    def add(self, record: CaseRecord) -> None:
        self._records.append(record)
        self._nn = None  # invalidate, rebuild lazily

    def add_many(self, records: list[CaseRecord]) -> None:
        self._records.extend(records)
        self._nn = None

    def _ensure_index(self):
        from sklearn.neighbors import NearestNeighbors
        if self._nn is None and self._records:
            X = np.vstack([r.embedding for r in self._records])
            self._nn = NearestNeighbors(n_neighbors=min(5, len(self._records)), metric="cosine").fit(X)

    def query(self, embedding: np.ndarray, k: int = 5) -> list[tuple[CaseRecord, float]]:
        self._ensure_index()
        if self._nn is None:
            return []
        k = min(k, len(self._records))
        dist, idx = self._nn.kneighbors(embedding.reshape(1, -1), n_neighbors=k)
        return [(self._records[i], float(d)) for i, d in zip(idx[0], dist[0])]

    def __len__(self) -> int:
        return len(self._records)

    def save(self, path: Path) -> None:
        import pickle
        Path(path).write_bytes(pickle.dumps(self._records))

    def load(self, path: Path) -> None:
        import pickle
        self._records = pickle.loads(Path(path).read_bytes())
        self._nn = None
