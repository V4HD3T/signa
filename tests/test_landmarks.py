"""Tests for the parts that run identically in training and in the live demo.

None of these need MediaPipe or a camera -- they build frames by hand. That is
deliberate: the normaliser and the resampler are where a train/inference
mismatch would hide, and a mismatch there produces a working demo that predicts
nonsense rather than an error anyone would notice.
"""

import numpy as np
import pytest

from signa import config as C
from signa.landmarks import empty_frame, normalize, resample


def make_frame(*, shoulder_half_width=0.1, centre=(0.5, 0.4), left=True, right=True):
    """One frame with a plausible pose and optionally each hand.

    Every offset is expressed in units of shoulder half-width, so changing
    `shoulder_half_width` models the *same* body at a different distance from
    the camera rather than a differently proportioned one.
    """
    frame = empty_frame()
    cx, cy = centre
    unit = shoulder_half_width

    pose = np.zeros((C.POSE_POINTS, C.COORDS), dtype=np.float32)
    pose[C.SHOULDER_L] = (cx - unit, cy, 0.0)
    pose[C.SHOULDER_R] = (cx + unit, cy, 0.0)
    for i in range(2, C.POSE_POINTS):
        pose[i] = (cx, cy + 0.5 * i * unit, 0.0)
    frame[C.POSE] = pose.ravel()

    if left:
        frame[C.LEFT_HAND] = np.tile([cx - 1.5 * unit, cy + 2.0 * unit, 0.0], C.HAND_POINTS)
        frame[C.LEFT_PRESENT] = 1.0
    if right:
        frame[C.RIGHT_HAND] = np.tile([cx + 1.5 * unit, cy + 2.0 * unit, 0.0], C.HAND_POINTS)
        frame[C.RIGHT_PRESENT] = 1.0
    return frame


def test_frame_layout_is_self_consistent():
    assert C.FRAME_DIM == 2 * C.HAND_POINTS * C.COORDS + C.POSE_POINTS * C.COORDS + 2
    assert len(C.POSE_SUBSET) == C.POSE_POINTS


def test_normalize_is_invariant_to_position_and_scale():
    near = np.stack([make_frame(shoulder_half_width=0.10, centre=(0.5, 0.4))])
    far = np.stack([make_frame(shoulder_half_width=0.05, centre=(0.2, 0.7))])

    # Same signer, same sign, half the size and elsewhere in the frame: after
    # normalisation the two must agree, or a signer standing further back
    # becomes a different class as far as the model is concerned.
    np.testing.assert_allclose(normalize(near), normalize(far), atol=1e-5)


def test_normalize_leaves_absent_hands_at_zero():
    sequence = np.stack([make_frame(left=False)])
    out = normalize(sequence)

    assert np.all(out[:, C.LEFT_HAND] == 0.0)
    assert out[0, C.LEFT_PRESENT] == 0.0
    assert out[0, C.RIGHT_PRESENT] == 1.0
    assert np.any(out[:, C.RIGHT_HAND] != 0.0)


def test_normalize_survives_a_frame_with_no_pose():
    sequence = np.stack([empty_frame(), make_frame()])
    out = normalize(sequence)

    assert np.isfinite(out).all()
    assert np.all(out[0] == 0.0)


def test_normalize_puts_the_shoulder_midpoint_at_the_origin():
    out = normalize(np.stack([make_frame()]))
    pose = out[0, C.POSE].reshape(C.POSE_POINTS, C.COORDS)
    midpoint = (pose[C.SHOULDER_L] + pose[C.SHOULDER_R]) / 2

    np.testing.assert_allclose(midpoint[:2], [0.0, 0.0], atol=1e-6)


@pytest.mark.parametrize("length", [1, 2, 17, 48, 90, 300])
def test_resample_always_produces_the_configured_length(length):
    sequence = np.stack([make_frame(centre=(0.5, 0.3 + i * 0.001)) for i in range(length)])
    out = resample(sequence, 48)

    assert out.shape == (48, C.FRAME_DIM)
    assert out.dtype == np.float32


def test_resample_keeps_presence_flags_binary():
    # A clip where the left hand appears halfway through: interpolating the flag
    # would produce 0.5s, which is a value the model never saw in training.
    sequence = np.stack(
        [make_frame(left=False) for _ in range(10)] + [make_frame() for _ in range(10)]
    )
    out = resample(sequence, 48)

    assert set(np.unique(out[:, C.LEFT_PRESENT])) <= {0.0, 1.0}


def test_resample_preserves_the_endpoints():
    sequence = np.stack([make_frame(centre=(0.2, 0.2)), make_frame(centre=(0.8, 0.8))])
    out = resample(sequence, 48)

    np.testing.assert_allclose(out[0], sequence[0], atol=1e-5)
    np.testing.assert_allclose(out[-1], sequence[-1], atol=1e-5)


def test_resample_of_an_empty_clip_is_zeros_not_a_crash():
    out = resample(np.zeros((0, C.FRAME_DIM), dtype=np.float32), 48)

    assert out.shape == (48, C.FRAME_DIM)
    assert np.all(out == 0.0)
