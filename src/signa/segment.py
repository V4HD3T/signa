"""Find where a sign begins and ends in a continuous stream — no key held.

Push-to-sign sidestepped segmentation by making the user bracket each sign with a
key. This finds the brackets itself, which is the whole difference between a
demo you drive and one you sign at.

The signal is motion energy: a sign is a burst of hand movement between spells of
relative stillness (the rest pose). Energy is the per-frame hand velocity in
shoulder-width units — the same normalisation the model uses, so a threshold set
for one signer holds for another standing closer or further from the camera.

The decision is a two-state machine with hysteresis: it takes more motion to
*start* a sign than to keep one going, so a momentary dip mid-sign does not cut
it in half, and a momentary twitch at rest does not open a phantom one. A minimum
length rejects blips; a maximum length force-cuts a runaway (a signer who never
returns to rest). All of it is pure over a stream of energy values, so the FSM is
tested exhaustively on synthetic signals without a camera; the webcam loop only
feeds it frames.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from . import config as C

# Defaults are in shoulder-width units per frame at 30 fps, chosen from the
# distribution of hand velocities on real (normalised) clips: active signing sits
# well above ENTER, a settled rest pose below EXIT. They are a starting point,
# not a law -- camera frame rate and a signer's tempo shift them, which is why
# they are parameters and why `calibrate` exists to set EXIT from a rest sample.
ENTER = 0.06   # motion above this starts a sign
EXIT = 0.03    # motion below this, held for END_PATIENCE frames, ends it
END_PATIENCE = 6   # frames of sustained stillness that close a sign (~0.2s @30fps)
MIN_FRAMES = 10    # shorter bursts are twitches, not signs
MAX_FRAMES = 90    # ~3s; force-cut a signer who never rests


def frame_energy(previous: np.ndarray, current: np.ndarray) -> float:
    """Mean hand velocity between two frames, over hands present in both.

    Uses x,y only — MediaPipe's z is noisy enough that including it would add
    jitter to the very signal stillness is measured by.

    Returns NaN, not 0, when no hand is tracked in both frames. The distinction
    matters: a detected hand holding still is genuine stillness (energy ~0), but a
    hand MediaPipe momentarily lost is missing data, and this corpus loses hands
    on roughly a fifth of frames. Reporting a dropout as zero motion would let a
    detection gap masquerade as the end of a sign — the segmenter treats NaN as
    "hold", so a sign survives the gap it would otherwise be cut in half by."""
    speeds = []
    for part, flag in ((C.LEFT_HAND, C.LEFT_PRESENT), (C.RIGHT_HAND, C.RIGHT_PRESENT)):
        if previous[flag] > 0.5 and current[flag] > 0.5:
            a = previous[part].reshape(C.HAND_POINTS, C.COORDS)[:, :2]
            b = current[part].reshape(C.HAND_POINTS, C.COORDS)[:, :2]
            speeds.append(float(np.linalg.norm(b - a, axis=1).mean()))
    return float(np.mean(speeds)) if speeds else float("nan")


def energies(sequence: np.ndarray) -> np.ndarray:
    """Per-frame energy for a whole clip; the first frame is NaN (no predecessor)."""
    if len(sequence) < 2:
        return np.full(len(sequence), np.nan, dtype=np.float32)
    values = [np.nan] + [frame_energy(sequence[i - 1], sequence[i])
                         for i in range(1, len(sequence))]
    return np.asarray(values, dtype=np.float32)


@dataclass(frozen=True)
class Segment:
    start: int  # inclusive, absolute frame index in the stream
    end: int    # exclusive

    def __len__(self) -> int:
        return self.end - self.start


class Segmenter:
    """Online start/end detector. Feed energy one frame at a time.

    `push` returns a Segment the moment a sign closes, else None; `flush` closes a
    sign still open when the stream ends. Indices are absolute frame counts since
    construction, so a caller holding a bounded buffer of recent frames can slice
    out exactly the sign that fired.
    """

    def __init__(self, *, enter: float = ENTER, exit: float = EXIT,
                 end_patience: int = END_PATIENCE, min_frames: int = MIN_FRAMES,
                 max_frames: int = MAX_FRAMES):
        if exit > enter:
            raise ValueError("exit threshold must be <= enter (hysteresis)")
        self.enter = enter
        self.exit = exit
        self.end_patience = end_patience
        self.min_frames = min_frames
        self.max_frames = max_frames
        self._index = -1        # absolute index of the last-pushed frame
        self._start = None      # start index of the open sign, or None
        self._low_run = 0       # consecutive sub-exit frames while active

    @property
    def active(self) -> bool:
        return self._start is not None

    def push(self, energy: float) -> Segment | None:
        self._index += 1
        unknown = energy != energy  # NaN: hand not tracked this frame

        if self._start is None:
            # Can't start a sign on missing data; only clear motion opens one.
            if not unknown and energy >= self.enter:
                self._start = self._index
                self._low_run = 0
            return None

        # Active. A dropout (NaN) is held, not counted as stillness, so a sign
        # survives a detection gap; clear low motion grows the stillness run,
        # clear motion resets it. max_frames still bounds a gap that never ends.
        if not unknown:
            self._low_run = self._low_run + 1 if energy < self.exit else 0

        if self._low_run >= self.end_patience:
            # The sign ended end_patience frames ago; trim the trailing stillness.
            return self._close(self._index - self.end_patience + 1)
        if self._index - self._start + 1 >= self.max_frames:
            return self._close(self._index + 1)
        return None

    def flush(self) -> Segment | None:
        """Close a sign left open at end of stream, if it is long enough."""
        if self._start is None:
            return None
        return self._close(self._index + 1)

    def _close(self, end: int) -> Segment | None:
        start = self._start
        self._start = None
        self._low_run = 0
        if end - start >= self.min_frames:
            return Segment(start, end)
        return None  # too short to be a sign; discarded


def segment(stream, **kwargs) -> list[Segment]:
    """Offline convenience: run the FSM over an energy array, collect segments."""
    seg = Segmenter(**kwargs)
    out = []
    for value in stream:
        found = seg.push(float(value))
        if found is not None:
            out.append(found)
    trailing = seg.flush()
    if trailing is not None:
        out.append(trailing)
    return out


class FrameStream:
    """A bounded buffer of recent frames driving a Segmenter, yielding the frames
    of each completed sign.

    Energy is measured on *normalised* frames (the caller passes the normaliser),
    the same normalisation training and inference use, so the thresholds are in
    the units they were chosen in. The buffer only needs to reach back one
    sign-length, so it is capped rather than growing without bound over a long
    session. This is the one place demo and learning mode share for going
    keyless, so segmentation behaves identically in both.
    """

    def __init__(self, segmenter: Segmenter, normalise, keep: int | None = None):
        self.segmenter = segmenter
        self.normalise = normalise
        self._frames = deque(maxlen=keep or (segmenter.max_frames + 8))
        self._offset = 0  # absolute index of _frames[0]
        self._prev = None

    def push(self, frame: np.ndarray) -> np.ndarray | None:
        """Feed one raw frame; return the raw frames of a sign the moment it
        closes, else None."""
        if len(self._frames) == self._frames.maxlen:
            self._offset += 1  # the append below evicts the current oldest
        self._frames.append(frame)

        current = self.normalise(frame)
        energy = frame_energy(self._prev, current) if self._prev is not None else float("nan")
        self._prev = current

        found = self.segmenter.push(energy)
        if found is None:
            return None
        lo, hi = found.start - self._offset, found.end - self._offset
        if lo < 0:
            return None  # slid out of the buffer before it fired (keep too small)
        return np.stack(list(self._frames)[lo:hi])


def calibrate(rest_energies, *, margin: float = 3.0) -> tuple[float, float]:
    """Suggest (enter, exit) from a sample of rest-pose energy.

    Rest is not perfectly still — MediaPipe jitters — so thresholds are set above
    the noise floor rather than at zero: exit a little over the rest mean, enter a
    margin above that. A few seconds of the user holding still is enough."""
    rest = np.asarray(rest_energies, dtype=np.float64)
    rest = rest[~np.isnan(rest)]  # dropouts are not rest, just missing
    floor = float(rest.mean() + rest.std())
    exit_t = max(floor, 1e-4)
    enter_t = exit_t * margin
    return round(enter_t, 4), round(exit_t, 4)
