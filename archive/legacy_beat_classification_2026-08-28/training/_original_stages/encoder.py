"""Self-supervised ECG foundation encoder — the core of the new methodology.

Implements recommendation #1 ("Use a pretrained 'ECG expert' AI instead of
25 handcrafted numbers" — Very High impact) and recommendation #2
("Pre-train an AI on our own unlabeled VitalPatch data" — High impact,
Med-High effort) from the internal review, combined into one component:

  * `ECGEncoder` is a small 1D-conv encoder (Option C / "pretrained ECG
    expert model + small AI" from the review's comparison table) that
    reads the raw beat waveform directly, instead of the 25 handcrafted
    summary numbers, so morphological detail is no longer thrown away
    before it reaches a classifier or the LLM.
  * It is pretrained with a masked-reconstruction objective (mask random
    spans of the waveform, learn to reconstruct them) — a standard
    self-supervised recipe that needs zero labels, only raw ECG. This lets
    us pretrain directly on Cliniaura's own unlabeled VitalPatch/SeNSiO
    recordings today, before any public labeled dataset (MITDB, Icentia11k,
    ...) is available, and later fine-tune a classification head once
    labels exist.
  * The encoder is deliberately tiny (a few hundred KB of weights) so it
    stays edge-deployable on the same Jetson-class hardware as the
    existing pipeline, per the review's "Runs on small device?" column for
    Option C.

The resulting embedding is consumed by:
  - `classify.py` (a classifier head on top, once labeled data exists)
  - `similar_cases.py` (nearest-neighbour "similar past patient" retrieval,
    recommendation #8)
  - `report.py` (embedding-derived summary handed to MedGemma alongside
    the handcrafted features, so the LLM's input is no longer text-only)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

EMBED_DIM = 32
INPUT_LEN = 125  # samples: 1 s wide window @ 125 Hz


class ECGEncoder(nn.Module):
    """Conv1D encoder: raw 125-sample beat window -> EMBED_DIM embedding.

    ~30K parameters — small enough to run on the same low-power edge
    hardware as the existing student model (An et al. 2024 style budget).
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.BatchNorm1d(16), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(32, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, INPUT_LEN) -> (batch, 1, INPUT_LEN)
        h = self.net(x.unsqueeze(1)).squeeze(-1)
        return self.proj(h)


class ReconstructionDecoder(nn.Module):
    """Lightweight decoder used only during self-supervised pretraining;
    discarded afterwards (only the encoder is deployed)."""

    def __init__(self, embed_dim: int = EMBED_DIM, output_len: int = INPUT_LEN):
        super().__init__()
        self.output_len = output_len
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.GELU(),
            nn.Linear(64, output_len),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


def random_mask(batch: torch.Tensor, mask_frac: float = 0.25, span: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero out random contiguous spans of each waveform; return the
    masked input and a boolean mask (True = masked, to be reconstructed)."""
    b, length = batch.shape
    mask = torch.zeros_like(batch, dtype=torch.bool)
    n_spans = max(1, int(length * mask_frac / span))
    for i in range(b):
        for _ in range(n_spans):
            start = np.random.randint(0, max(1, length - span))
            mask[i, start:start + span] = True
    masked = batch.clone()
    masked[mask] = 0.0
    return masked, mask


@dataclass
class PretrainResult:
    epochs: int
    final_loss: float
    n_windows: int


def pretrain_self_supervised(windows: np.ndarray, epochs: int = 20, batch_size: int = 64,
                              lr: float = 1e-3, mask_frac: float = 0.25,
                              device: str = "cpu") -> tuple[ECGEncoder, PretrainResult]:
    """Masked-reconstruction pretraining on raw, UNLABELED beat windows.

    `windows` must be shape (n, INPUT_LEN), already filtered/normalized
    (robust z-score) by stages 4/6. No labels required — this is exactly
    what lets us pretrain on Cliniaura's own VitalPatch/SeNSiO data before
    any labeled public dataset is downloaded.
    """
    if len(windows) == 0:
        raise ValueError("No windows provided for pretraining")

    encoder = ECGEncoder().to(device)
    decoder = ReconstructionDecoder().to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    loss_fn = nn.MSELoss()

    data = torch.tensor(windows, dtype=torch.float32, device=device)
    n = len(data)
    final_loss = float("nan")

    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_losses = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch = data[idx]
            masked, mask = random_mask(batch, mask_frac=mask_frac)
            z = encoder(masked)
            recon = decoder(z)
            loss = loss_fn(recon[mask], batch[mask]) if mask.any() else loss_fn(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        final_loss = float(np.mean(epoch_losses))

    return encoder, PretrainResult(epochs=epochs, final_loss=final_loss, n_windows=n)


def save_encoder(encoder: ECGEncoder, path: Path, meta: dict | None = None) -> None:
    path = Path(path)
    torch.save(encoder.state_dict(), path)
    if meta is not None:
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def load_encoder(path: Path) -> ECGEncoder:
    encoder = ECGEncoder()
    encoder.load_state_dict(torch.load(path, map_location="cpu"))
    encoder.eval()
    return encoder


@torch.no_grad()
def embed_windows(encoder: ECGEncoder, windows: np.ndarray, device: str = "cpu") -> np.ndarray:
    if len(windows) == 0:
        return np.zeros((0, EMBED_DIM))
    encoder.eval()
    data = torch.tensor(windows, dtype=torch.float32, device=device)
    return encoder(data).cpu().numpy()
