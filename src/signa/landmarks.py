"""MediaPipe Holistic -> the fixed frame layout in `config`.

Extraction is the slow part of this pipeline (roughly real-time per clip), so
what lands on disk is *raw* MediaPipe output in the layout above. Normalisation
happens at load time instead, which means the normaliser can be changed without
re-extracting 20k videos.

The same functions run in training extraction and in the live demo. That is the
whole point: train and inference must see identically produced features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from . import config as C


def empty_frame() -> np.ndarray:
    return np.zeros(C.FRAME_DIM, dtype=np.float32)


def frame_from_results(results) -> np.ndarray:
    """Flatten one MediaPipe Holistic result into a FRAME_DIM vector."""
    frame = empty_frame()

    if results.left_hand_landmarks is not None:
        frame[C.LEFT_HAND] = _flatten(results.left_hand_landmarks.landmark)
        frame[C.LEFT_PRESENT] = 1.0

    if results.right_hand_landmarks is not None:
        frame[C.RIGHT_HAND] = _flatten(results.right_hand_landmarks.landmark)
        frame[C.RIGHT_PRESENT] = 1.0

    if results.pose_landmarks is not None:
        pose = results.pose_landmarks.landmark
        frame[C.POSE] = np.array(
            [[pose[i].x, pose[i].y, pose[i].z] for i in C.POSE_SUBSET],
            dtype=np.float32,
        ).ravel()

    return frame


def _flatten(landmarks) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32).ravel()


class Extractor:
    """Context manager around a single Holistic instance.

    MediaPipe's Holistic graph is expensive to build, so one instance is reused
    across a whole directory of clips rather than per clip.
    """

    def __init__(self, *, static_image_mode: bool = False, complexity: int = 1):
        import mediapipe as mp

        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=complexity,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def __enter__(self) -> "Extractor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._holistic.close()

    def frame(self, bgr_image) -> np.ndarray:
        """Process one OpenCV BGR frame."""
        import cv2

        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        return frame_from_results(self._holistic.process(rgb))

    def video(self, path: str | Path, stride: int = 1) -> np.ndarray:
        """Process a whole clip. Returns (T, FRAME_DIM); T may be 0.

        `stride` keeps every nth frame. Its real use is normalising frame rate
        across sources -- LSA64 is 60 fps where AUTSL and BosphorusSign22k are
        30, and a model trained on one tempo should not have to relearn the
        other. Halving the frames also halves extraction time, which at 1080p
        is the difference between an afternoon and a coffee break.
        """
        import cv2

        capture = cv2.VideoCapture(str(path))
        try:
            frames = [
                self.frame(image)
                for index, image in enumerate(_read(capture))
                if index % stride == 0
            ]
        finally:
            capture.release()
        if not frames:
            return np.zeros((0, C.FRAME_DIM), dtype=np.float32)
        return np.stack(frames)


def _read(capture) -> Iterator:
    while True:
        ok, image = capture.read()
        if not ok:
            return
        yield image


# --- Normalisation --------------------------------------------------------


def normalize(sequence: np.ndarray) -> np.ndarray:
    """Make a sequence invariant to where the signer stands and how big they are.

    Origin moves to the shoulder midpoint and everything is divided by shoulder
    width, both computed *per frame*. Per-frame rather than per-clip so that a
    signer drifting in the frame does not smear the trajectory -- the cost is
    that whole-body translation stops being a feature, which for isolated signs
    it is not.
    """
    if sequence.size == 0:
        return sequence

    out = sequence.astype(np.float32, copy=True)
    pose = out[:, C.POSE].reshape(len(out), C.POSE_POINTS, C.COORDS)

    left, right = pose[:, C.SHOULDER_L], pose[:, C.SHOULDER_R]
    origin = ((left + right) / 2.0)[:, None, :]  # (T, 1, 3)

    width = np.linalg.norm(left[:, :2] - right[:, :2], axis=1)
    # A frame with no pose detection has zero width; leave those frames alone
    # rather than dividing by an epsilon and blowing them up.
    scale = np.where(width > 1e-6, width, 1.0)[:, None, None]

    for part, count in ((C.LEFT_HAND, C.HAND_POINTS),
                        (C.RIGHT_HAND, C.HAND_POINTS),
                        (C.POSE, C.POSE_POINTS)):
        block = out[:, part].reshape(len(out), count, C.COORDS)
        present = np.any(block != 0.0, axis=(1, 2))[:, None, None]
        block = np.where(present, (block - origin) / scale, block)
        out[:, part] = block.reshape(len(out), count * C.COORDS)

    return out


def resample(sequence: np.ndarray, frames: int) -> np.ndarray:
    """Linearly resample a clip to a fixed length.

    Resampling rather than pad/truncate: signers differ in tempo, and padding
    would make clip duration a feature the model can cheat on -- which fails
    exactly where it matters, on an unseen signer with a different rhythm.
    """
    length = len(sequence)
    if length == frames:
        return sequence.astype(np.float32, copy=False)
    if length == 0:
        return np.zeros((frames, C.FRAME_DIM), dtype=np.float32)
    if length == 1:
        return np.repeat(sequence.astype(np.float32), frames, axis=0)

    # Interpolate every dimension at once. np.interp is 1-D, so the obvious
    # loop calls it 152 times per clip, and augmentation resamples three times
    # per clip per epoch -- that loop dominated the training step until this
    # was vectorised. Source positions are uniform, so the bracketing indices
    # are arithmetic rather than a search.
    target = np.linspace(0.0, length - 1, frames, dtype=np.float64)
    lower = np.floor(target).astype(np.intp)
    upper = np.minimum(lower + 1, length - 1)
    weight = (target - lower).astype(np.float32)[:, None]
    source = sequence.astype(np.float32, copy=False)
    out = source[lower] * (1.0 - weight) + source[upper] * weight

    # Interpolating a 0/1 flag produces fractions; snap them back.
    for flag in (C.LEFT_PRESENT, C.RIGHT_PRESENT):
        out[:, flag] = (out[:, flag] > 0.5).astype(np.float32)
    return out
