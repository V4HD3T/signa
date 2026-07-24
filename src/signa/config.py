"""Layout of a landmark frame, and the knobs that shape a training run.

The frame layout is deliberately fixed and written down here rather than
inferred anywhere else: extraction writes it, the dataset reads it, and the
live demo has to produce exactly the same thing or the model sees garbage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- Frame layout ---------------------------------------------------------
#
# One frame is a flat float32 vector. Slices, in order:
#
#   [  0:  63)  left hand   21 points x (x, y, z)
#   [ 63: 126)  right hand  21 points x (x, y, z)
#   [126: 150)  pose subset  8 points x (x, y, z)
#   [150]       left hand present   (1.0 / 0.0)
#   [151]       right hand present  (1.0 / 0.0)
#
# The presence flags matter: a missing hand is stored as zeros, which without
# a flag is indistinguishable from a hand sitting exactly at the origin.

HAND_POINTS = 21
POSE_POINTS = 8
COORDS = 3

LEFT_HAND = slice(0, HAND_POINTS * COORDS)
RIGHT_HAND = slice(HAND_POINTS * COORDS, 2 * HAND_POINTS * COORDS)
POSE = slice(2 * HAND_POINTS * COORDS, 2 * HAND_POINTS * COORDS + POSE_POINTS * COORDS)
LEFT_PRESENT = 2 * HAND_POINTS * COORDS + POSE_POINTS * COORDS
RIGHT_PRESENT = LEFT_PRESENT + 1

FRAME_DIM = RIGHT_PRESENT + 1  # 152

# Indices into MediaPipe Holistic's 33-point pose, upper body only. The face
# mesh (468 points) is skipped entirely for the MVP -- in isolated signing the
# signal lives in hand shape and trajectory, and 468 extra points would swamp
# the ~20 training clips per gloss we actually have.
POSE_SUBSET = (
    11,  # left shoulder
    12,  # right shoulder
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16,  # right wrist
    23,  # left hip
    24,  # right hip
)
# Positions *within* the pose slice, used by the normalizer.
SHOULDER_L, SHOULDER_R = 0, 1

# --- Run configuration ----------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"


@dataclass
class Config:
    """Everything a training run needs to be reproducible from a JSON file."""

    # Data
    manifest: Path = DATA_ROOT / "manifest.csv"
    landmark_root: Path = DATA_ROOT / "landmarks"
    test_signers: tuple[str, ...] = ("User_1",)
    max_glosses: int | None = 50  # MVP vocabulary size; None = all in manifest

    # Sequence shaping
    frames: int = 48  # clips are 1-3 s @ 30 fps -> 30-90 frames, resampled here
    normalize: bool = True
    use_pose: bool = True  # False zeroes the pose block for the ablation

    # Model
    model: str = "bilstm"  # "bilstm" | "transformer" | "tcn"
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.3
    heads: int = 4  # transformer only
    kernel: int = 3  # tcn only; odd, so blocks stay length-preserving

    # Optimisation
    epochs: int = 120
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-2
    label_smoothing: float = 0.1
    patience: int = 25  # early stop, in epochs without a top-1 improvement

    # Augmentation (not optional at this dataset size -- see README)
    augment: bool = True
    aug_time_warp: float = 0.2  # +/- 20% speed
    aug_jitter: float = 0.01  # gaussian sigma, in shoulder-width units
    aug_rotate_deg: float = 8.0
    aug_scale: float = 0.1  # +/- 10%
    aug_mirror: float = 0.0  # p(mirror); see augment.mirror for the caveat

    # Bookkeeping
    seed: int = 42
    out_dir: Path = REPO_ROOT / "runs"
    tag: str = "baseline"
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    threads: int = 1  # CPU intra-op threads; 0 keeps torch's default

    extra: dict = field(default_factory=dict)

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
