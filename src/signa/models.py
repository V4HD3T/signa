"""Two sequence classifiers over landmark frames.

The BiLSTM is the baseline. The Transformer exists to be a second data point in
the write-up -- with ~20 clips per gloss it is not obviously the better model,
and saying so with a measurement is worth more than assuming it.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import FRAME_DIM


def _pool(sequence: torch.Tensor) -> torch.Tensor:
    """Mean+max over time. Concatenating both keeps the average shape of the
    sign and its most extreme moment, which is often what separates two glosses
    that share a trajectory."""
    return torch.cat([sequence.mean(dim=1), sequence.amax(dim=1)], dim=-1)


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        input_dim: int = FRAME_DIM,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(4 * hidden),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(self.input_norm(x))
        return self.head(_pool(out))


class _PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        encoding = torch.zeros(max_len, dim)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        input_dim: int = FRAME_DIM,
        hidden: int = 128,
        layers: int = 4,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden))
        self.positional = _PositionalEncoding(hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=4 * hidden,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(self.positional(self.project(x)))
        return self.head(_pool(out))


def build(cfg, num_classes: int) -> nn.Module:
    if cfg.model == "bilstm":
        return BiLSTMClassifier(
            num_classes, hidden=cfg.hidden, layers=cfg.layers, dropout=cfg.dropout
        )
    if cfg.model == "transformer":
        return TransformerClassifier(
            num_classes,
            hidden=cfg.hidden,
            layers=max(2, cfg.layers),
            heads=cfg.heads,
            dropout=cfg.dropout,
        )
    raise ValueError(f"unknown model {cfg.model!r} (expected 'bilstm' or 'transformer')")
