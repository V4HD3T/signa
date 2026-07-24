"""Three sequence classifiers over landmark frames.

The BiLSTM is the baseline. The Transformer was the second data point, and on
LSA64 it pulled well ahead (97.1% vs 88.0% top-1) -- which raises the question
the TCN answers: is that lead about attention specifically, or just about not
being a recurrent bottleneck? A temporal convolutional net is neither recurrent
nor attentional, so where it lands separates the two explanations. Measuring
that beats asserting it, the same reason the Transformer was built in the first
place.
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


class _TCNBlock(nn.Module):
    """Residual block of two dilated 1-D convolutions.

    Padding is `(kernel-1)*dilation//2` with an odd kernel, so the block is
    length-preserving and non-causal -- for a pre-trimmed isolated clip there is
    no reason to hide future frames, unlike streaming. Stacking blocks with
    doubling dilation grows the receptive field exponentially, so a handful of
    layers already sees the whole ~48-frame clip.
    """

    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(self.act(self.norm1(self.conv1(x))))
        y = self.drop(self.act(self.norm2(self.conv2(y))))
        return self.act(x + y)


class TCNClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        input_dim: int = FRAME_DIM,
        hidden: int = 128,
        layers: int = 4,
        kernel: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.project = nn.Conv1d(input_dim, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            _TCNBlock(hidden, kernel, dilation=2 ** i, dropout=dropout)
            for i in range(layers)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1d wants (B, C, T); the frames arrive as (B, T, C).
        out = self.project(self.input_norm(x).transpose(1, 2))
        for block in self.blocks:
            out = block(out)
        return self.head(_pool(out.transpose(1, 2)))


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
    if cfg.model == "tcn":
        return TCNClassifier(
            num_classes,
            hidden=cfg.hidden,
            layers=max(2, cfg.layers),
            kernel=cfg.kernel,
            dropout=cfg.dropout,
        )
    raise ValueError(
        f"unknown model {cfg.model!r} (expected 'bilstm', 'transformer' or 'tcn')"
    )
