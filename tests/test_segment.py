"""Tests for automatic segmentation: the energy signal and the FSM.

The finite-state machine is fed synthetic energy streams so every branch --
hysteresis, minimum and maximum length, dropout handling, the trailing flush --
is pinned without a camera. A segmenter that split signs in half or merged two
into one would fail silently: the demo would just misclassify, with nothing in
the logs to say why.
"""

import numpy as np
import pytest

from signa import config as C
from signa.landmarks import empty_frame
from signa.segment import FrameStream, Segmenter, energies, frame_energy, segment

ENTER, EXIT = 0.06, 0.03


def hand_frame(x: float):
    """A frame with both hands present, both at horizontal position x."""
    frame = empty_frame()
    frame[C.LEFT_HAND] = np.tile([x, 0.3, 0.0], C.HAND_POINTS)
    frame[C.RIGHT_HAND] = np.tile([x, 0.3, 0.0], C.HAND_POINTS)
    frame[C.LEFT_PRESENT] = 1.0
    frame[C.RIGHT_PRESENT] = 1.0
    return frame


# --- Energy ---------------------------------------------------------------

def test_energy_of_a_moving_hand_is_the_displacement():
    e = frame_energy(hand_frame(0.0), hand_frame(0.1))
    assert e == pytest.approx(0.1, abs=1e-5)


def test_energy_of_a_still_hand_is_zero():
    assert frame_energy(hand_frame(0.5), hand_frame(0.5)) == pytest.approx(0.0)


def test_energy_is_nan_when_a_hand_is_missing():
    # A dropout is missing data, not stillness -- the FSM depends on the two
    # being distinguishable.
    assert np.isnan(frame_energy(empty_frame(), hand_frame(0.5)))


def test_energies_first_frame_is_nan_and_length_matches():
    seq = np.stack([hand_frame(0.0), hand_frame(0.1), hand_frame(0.1)])
    e = energies(seq)
    assert len(e) == 3 and np.isnan(e[0])
    assert e[1] == pytest.approx(0.1, abs=1e-5)


# --- FSM ------------------------------------------------------------------

def stream(*runs):
    """Build an energy stream from (value, count) runs."""
    out = []
    for value, count in runs:
        out.extend([value] * count)
    return out


def test_a_single_burst_becomes_one_segment():
    s = stream((0.0, 5), (0.10, 20), (0.0, 10))
    segs = segment(s, enter=ENTER, exit=EXIT)
    assert len(segs) == 1
    assert segs[0].start == 5
    assert len(segs[0]) >= 15  # the burst, minus trimmed trailing stillness


def test_a_short_blip_is_not_a_sign():
    # Above enter but only 4 frames, below min_frames.
    segs = segment(stream((0.0, 5), (0.10, 4), (0.0, 15)),
                   enter=ENTER, exit=EXIT, min_frames=10)
    assert segs == []


def test_a_brief_dip_does_not_split_a_sign():
    # A 3-frame dip below exit, shorter than end_patience, must not end the sign.
    s = stream((0.10, 12), (0.0, 3), (0.10, 12), (0.0, 10))
    segs = segment(s, enter=ENTER, exit=EXIT, end_patience=6)
    assert len(segs) == 1


def test_sustained_stillness_ends_the_sign_and_trims_the_tail():
    s = stream((0.10, 20), (0.0, 10))
    segs = segment(s, enter=ENTER, exit=EXIT, end_patience=6)
    assert len(segs) == 1
    # end is trimmed back to where motion stopped, not the last still frame
    assert segs[0].end == 20


def test_two_signs_separated_by_rest_are_two_segments():
    s = stream((0.10, 20), (0.0, 12), (0.10, 20), (0.0, 12))
    segs = segment(s, enter=ENTER, exit=EXIT, end_patience=6)
    assert len(segs) == 2
    assert segs[0].end <= segs[1].start


def test_a_dropout_does_not_end_a_sign():
    # NaN (hand lost) in the middle of a sign is held, not treated as stillness.
    s = stream((0.10, 10), (float("nan"), 8), (0.10, 10), (0.0, 10))
    segs = segment(s, enter=ENTER, exit=EXIT, end_patience=6)
    assert len(segs) == 1
    assert len(segs[0]) >= 28  # spans the whole burst including the gap


def test_a_dropout_while_idle_does_not_start_a_sign():
    segs = segment(stream((float("nan"), 20), (0.0, 10)), enter=ENTER, exit=EXIT)
    assert segs == []


def test_max_frames_force_cuts_a_runaway():
    # Motion that never returns to rest is cut at max_frames.
    segs = segment(stream((0.10, 200)), enter=ENTER, exit=EXIT, max_frames=90)
    assert len(segs) >= 1
    assert len(segs[0]) == 90


def test_flush_closes_a_sign_open_at_end_of_stream():
    segs = segment(stream((0.0, 5), (0.10, 20)), enter=ENTER, exit=EXIT)
    assert len(segs) == 1  # no trailing rest, but flush still emits it


def test_exit_above_enter_is_rejected():
    with pytest.raises(ValueError, match="hysteresis"):
        Segmenter(enter=0.03, exit=0.06)


def test_a_segmenter_that_could_never_emit_is_rejected():
    # min > max can never produce a segment. Failing by silently returning
    # nothing is the worst outcome: the demo just stops recognising, with no
    # error to explain why.
    with pytest.raises(ValueError, match="no segment could ever be emitted"):
        Segmenter(enter=ENTER, exit=EXIT, min_frames=100, max_frames=20)


def test_zero_end_patience_is_rejected():
    with pytest.raises(ValueError, match="end_patience"):
        Segmenter(enter=ENTER, exit=EXIT, end_patience=0)


def test_push_returns_the_segment_exactly_when_it_closes():
    s = Segmenter(enter=ENTER, exit=EXIT, end_patience=3, min_frames=5)
    results = [s.push(v) for v in stream((0.10, 10), (0.0, 3))]
    emitted = [r for r in results if r is not None]
    assert len(emitted) == 1
    assert emitted[0].start == 0 and emitted[0].end == 10


# --- FrameStream ----------------------------------------------------------

def test_framestream_yields_the_frames_of_a_completed_sign():
    # Hand still, then moving (a burst), then still again -> one sign, and the
    # returned frames are exactly the moving span.
    seg = Segmenter(enter=ENTER, exit=EXIT, end_patience=4, min_frames=5)
    stream_buf = FrameStream(seg, normalise=lambda f: f)  # frames already comparable

    positions = ([0.5] * 4                      # still
                 + [0.5 + 0.1 * i for i in range(12)]  # moving
                 + [1.6] * 8)                   # still again (held)
    emitted = [stream_buf.push(hand_frame(x)) for x in positions]
    signs = [e for e in emitted if e is not None]

    assert len(signs) == 1
    # every returned frame is a real (present-hand) frame from the moving span
    assert signs[0].shape[1] == C.FRAME_DIM
    assert len(signs[0]) >= 5


def test_framestream_reports_no_sign_for_stillness():
    seg = Segmenter(enter=ENTER, exit=EXIT)
    stream_buf = FrameStream(seg, normalise=lambda f: f)
    assert all(stream_buf.push(hand_frame(0.5)) is None for _ in range(40))
