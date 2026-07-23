"""Landmark-space augmentation.

At roughly 20-25 training clips per gloss (5 signers x ~5 repetitions) this is
not a nice-to-have. Everything here operates on normalised sequences, so the
magnitudes are in shoulder-width units and transfer across signers.
"""

from __future__ import annotations

import numpy as np

from . import config as C
from .landmarks import resample

_PARTS = (
    (C.LEFT_HAND, C.HAND_POINTS),
    (C.RIGHT_HAND, C.HAND_POINTS),
    (C.POSE, C.POSE_POINTS),
)


def _points(sequence: np.ndarray):
    """Yield each coordinate block as a (T, N, 3) view for in-place edits."""
    for part, count in _PARTS:
        yield part, sequence[:, part].reshape(len(sequence), count, C.COORDS)


def time_warp(sequence: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Re-time the clip by +/- `amount`, then restore the original length.

    Signing tempo varies between signers more than almost anything else, and
    signer-independent evaluation is exactly where that bites.
    """
    if amount <= 0:
        return sequence
    factor = 1.0 + rng.uniform(-amount, amount)
    stretched = max(2, int(round(len(sequence) * factor)))
    return resample(resample(sequence, stretched), len(sequence))


def jitter(sequence: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add gaussian noise -- a stand-in for MediaPipe's own detection wobble."""
    if sigma <= 0:
        return sequence
    out = sequence.copy()
    for part, block in _points(out):
        present = np.any(block != 0.0, axis=(1, 2))[:, None, None]
        noise = rng.normal(0.0, sigma, size=block.shape).astype(np.float32)
        out[:, part] = np.where(present, block + noise, block).reshape(len(out), -1)
    return out


def rotate(sequence: np.ndarray, degrees: float, rng: np.random.Generator) -> np.ndarray:
    """Rotate in the image plane -- camera tilt, or a signer standing askew."""
    if degrees <= 0:
        return sequence
    theta = np.deg2rad(rng.uniform(-degrees, degrees))
    cos, sin = np.cos(theta), np.sin(theta)
    out = sequence.copy()
    for part, block in _points(out):
        present = np.any(block != 0.0, axis=(1, 2))[:, None, None]
        x, y = block[..., 0].copy(), block[..., 1].copy()
        rotated = block.copy()
        rotated[..., 0] = cos * x - sin * y
        rotated[..., 1] = sin * x + cos * y
        out[:, part] = np.where(present, rotated, block).reshape(len(out), -1)
    return out


def scale(sequence: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform scale -- residual body-size variation the normaliser missed."""
    if amount <= 0:
        return sequence
    factor = 1.0 + rng.uniform(-amount, amount)
    out = sequence.copy()
    for part, block in _points(out):
        present = np.any(block != 0.0, axis=(1, 2))[:, None, None]
        out[:, part] = np.where(present, block * factor, block).reshape(len(out), -1)
    return out


def mirror(sequence: np.ndarray) -> np.ndarray:
    """Flip left/right: negate x and swap the two hand blocks.

    Off by default (`aug_mirror = 0.0`). A mirrored sign is what a left-handed
    signer produces, so it is a *plausible* sample -- but BosphorusSign22k's six
    signers are not evenly handed, and mirroring also destroys the dominant/
    non-dominant asymmetry that distinguishes some gloss pairs. Turn it on as an
    experiment and measure it; do not assume it helps.
    """
    out = sequence.copy()
    for part, count in _PARTS:
        block = out[:, part].reshape(len(out), count, C.COORDS).copy()
        block[..., 0] *= -1.0
        out[:, part] = block.reshape(len(out), -1)
    left = out[:, C.LEFT_HAND].copy()
    out[:, C.LEFT_HAND] = out[:, C.RIGHT_HAND]
    out[:, C.RIGHT_HAND] = left
    left_flag = out[:, C.LEFT_PRESENT].copy()
    out[:, C.LEFT_PRESENT] = out[:, C.RIGHT_PRESENT]
    out[:, C.RIGHT_PRESENT] = left_flag
    return out


def apply(sequence: np.ndarray, cfg, rng: np.random.Generator) -> np.ndarray:
    """Run the configured augmentation chain over one normalised sequence."""
    out = time_warp(sequence, cfg.aug_time_warp, rng)
    out = rotate(out, cfg.aug_rotate_deg, rng)
    out = scale(out, cfg.aug_scale, rng)
    out = jitter(out, cfg.aug_jitter, rng)
    if cfg.aug_mirror > 0 and rng.random() < cfg.aug_mirror:
        out = mirror(out)
    return out
