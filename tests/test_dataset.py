"""Tests for the split -- the one piece of this project that, if wrong, makes
every reported number meaningless while everything still appears to work."""

import numpy as np
import pytest

from signa import config as C
from signa.augment import apply, mirror
from signa.config import Config
from signa.dataset import (
    Clip,
    SignDataset,
    filter_to,
    read_manifest,
    select_glosses,
    signer_independent_split,
    write_manifest,
)
from tests.test_landmarks import make_frame


def clips_for(signers, glosses, repeats=3):
    return [
        Clip(f"{signer}/{gloss}/{i}.npy", gloss, signer, str(i))
        for signer in signers
        for gloss in glosses
        for i in range(repeats)
    ]


def test_signer_independent_split_shares_no_signer():
    clips = clips_for(["User_1", "User_2", "User_3"], ["a", "b"])
    train, test = signer_independent_split(clips, ("User_1",))

    assert {c.signer for c in train} == {"User_2", "User_3"}
    assert {c.signer for c in test} == {"User_1"}
    assert len(train) + len(test) == len(clips)


def test_split_rejects_an_unknown_test_signer():
    # Silently producing an empty test set here would look like a passing run.
    clips = clips_for(["User_1", "User_2"], ["a"])
    with pytest.raises(ValueError, match="not in the manifest"):
        signer_independent_split(clips, ("User_9",))


def test_split_rejects_holding_out_everyone():
    clips = clips_for(["User_1"], ["a"])
    with pytest.raises(ValueError, match="empty"):
        signer_independent_split(clips, ("User_1",))


def test_select_glosses_takes_the_best_represented_and_is_deterministic():
    clips = clips_for(["User_1"], ["rare"], repeats=1) + clips_for(["User_1"], ["common"], 9)
    assert select_glosses(clips, 1) == ["common"]
    assert select_glosses(clips, None) == ["common", "rare"]
    assert select_glosses(clips, 5) == select_glosses(clips, 5)


def test_manifest_round_trips(tmp_path):
    clips = clips_for(["User_1", "User_2"], ["merhaba", "tesekkurler"])
    path = tmp_path / "manifest.csv"
    write_manifest(path, clips)

    assert read_manifest(path) == clips


def test_manifest_missing_a_column_is_an_error(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("path,gloss\na.npy,merhaba\n", encoding="utf-8")

    with pytest.raises(ValueError, match="signer"):
        read_manifest(path)


def test_dataset_returns_fixed_length_tensors(tmp_path):
    lengths = {"a": 20, "b": 75}
    clips = []
    for gloss, length in lengths.items():
        path = tmp_path / f"{gloss}.npy"
        np.save(path, np.stack([make_frame() for _ in range(length)]))
        clips.append(Clip(f"{gloss}.npy", gloss, "User_1"))

    cfg = Config(landmark_root=tmp_path, frames=48, augment=False)
    dataset = SignDataset(clips, ["a", "b"], cfg, train=False)

    for index in range(len(dataset)):
        sequence, label = dataset[index]
        assert tuple(sequence.shape) == (48, C.FRAME_DIM)
        assert label.item() in (0, 1)


def test_augmentation_keeps_the_shape_and_the_absent_hand_absent():
    sequence = np.stack([make_frame(left=False) for _ in range(30)])
    cfg = Config(aug_mirror=0.0)
    out = apply(sequence, cfg, np.random.default_rng(0))

    assert out.shape == sequence.shape
    assert np.isfinite(out).all()
    assert np.all(out[:, C.LEFT_HAND] == 0.0)


def test_mirror_swaps_the_hands_and_their_flags():
    sequence = np.stack([make_frame(left=False)])
    out = mirror(sequence)

    assert out[0, C.LEFT_PRESENT] == 1.0
    assert out[0, C.RIGHT_PRESENT] == 0.0
    assert np.all(out[:, C.RIGHT_HAND] == 0.0)


def test_filter_to_drops_everything_outside_the_vocabulary():
    clips = clips_for(["User_1"], ["a", "b", "c"])
    assert {c.gloss for c in filter_to(clips, ["a", "c"])} == {"a", "c"}
