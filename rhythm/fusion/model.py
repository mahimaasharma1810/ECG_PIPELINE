"""Waveform + timing fusion model (BUILD_RHYTHM_MODEL.md step 6).

  waveform (2048 samples, rate-normalised)  -> 1D CNN encoder --.
                                                                 +-> fusion -> head
  RR features (7, rate-normalised)          -> MLP -------------'

Deliberately small. The comparison that matters is whether the WAVEFORM adds
anything over RR features alone (Track B: held-out-database AUC 0.9957). A model
large enough to memorise 16-78 patients would answer a different question.

ABSTENTION IS BUILT IN (brief step 10). The head emits a logit plus a learned
confidence; below a calibrated threshold the output is UNDETERMINED rather than
a forced class.

THIS MODEL NEVER OVERRIDES THE DETERMINISTIC VERDICT. Per the layered design,
L3's threshold is primary and auditable; a learned model annotates. A CNN that
silently replaced the threshold would trade an auditable decision for an
opaque one.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class WaveEncoder(nn.Module):
    """Small 1D CNN. Strided convs, no pooling-to-global until the end."""

    def __init__(self, ch=(16, 32, 64, 64), k=7, dropout=0.2):
        super().__init__()
        layers, cin = [], 1
        for cout in ch:
            layers += [nn.Conv1d(cin, cout, k, stride=2, padding=k // 2),
                       nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
                       nn.Dropout(dropout)]
            cin = cout
        self.net = nn.Sequential(*layers)
        self.out_dim = ch[-1]

    def forward(self, x):                      # (B, L) -> (B, C)
        h = self.net(x.unsqueeze(1))
        return h.mean(dim=-1)                  # global average over time


class FusionNet(nn.Module):
    def __init__(self, n_rr=7, hidden=64, dropout=0.2, use_wave=True, use_rr=True):
        super().__init__()
        assert use_wave or use_rr, "at least one branch must be enabled"
        self.use_wave, self.use_rr = use_wave, use_rr
        dim = 0
        if use_wave:
            self.wave = WaveEncoder(dropout=dropout)
            dim += self.wave.out_dim
        if use_rr:
            self.rr = nn.Sequential(nn.Linear(n_rr, hidden), nn.ReLU(inplace=True),
                                    nn.Dropout(dropout),
                                    nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
            dim += hidden
        self.head = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout))
        self.logit = nn.Linear(hidden, 1)
        self.conf = nn.Linear(hidden, 1)       # abstention signal

    def forward(self, wave, rr):
        parts = []
        if self.use_wave:
            parts.append(self.wave(wave))
        if self.use_rr:
            parts.append(self.rr(rr))
        h = self.head(torch.cat(parts, dim=1))
        return self.logit(h).squeeze(1), torch.sigmoid(self.conf(h)).squeeze(1)
