"""Manifest-backed dataset and the signer-independent split.

A manifest is a CSV with one row per clip:

    path,gloss,signer,repeat

`path` is relative to the landmark root. Keeping the split metadata in a CSV
rather than parsing filenames at load time means a wrong filename convention is
a one-line fix in `extract.py`, not a bug that silently mixes signers across
the split.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from . import augment
from . import config as C
from .config import Config
from .landmarks import normalize, resample


@dataclass(frozen=True)
class Clip:
    path: str
    gloss: str
    signer: str
    repeat: str = ""


def read_manifest(path: str | Path) -> list[Clip]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = {"path", "gloss", "signer"} - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"{path}: manifest is missing column(s) {sorted(missing)}")
    return [
        Clip(r["path"], r["gloss"], r["signer"], r.get("repeat", "") or "")
        for r in rows
    ]


def write_manifest(path: str | Path, clips: list[Clip]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "gloss", "signer", "repeat"])
        for clip in clips:
            writer.writerow([clip.path, clip.gloss, clip.signer, clip.repeat])


def select_glosses(clips: list[Clip], limit: int | None) -> list[str]:
    """Pick the MVP vocabulary: the `limit` best-represented glosses.

    Most-frequent rather than random, so the smallest classes -- the ones whose
    accuracy would be noise on 4-5 test clips -- are not what the headline
    number rests on. Ties break alphabetically so a run is reproducible.
    """
    counts = Counter(clip.gloss for clip in clips)
    ordered = sorted(counts, key=lambda gloss: (-counts[gloss], gloss))
    return sorted(ordered if limit is None else ordered[:limit])


def signer_independent_split(
    clips: list[Clip], test_signers: tuple[str, ...]
) -> tuple[list[Clip], list[Clip]]:
    """Hold out entire signers, per the dataset authors' own protocol.

    This is the decision that makes the headline number defensible. A random
    clip-level split leaks the same person's idiosyncrasies into both sides and
    inflates accuracy by a wide margin -- it measures "can you recognise this
    person signing" rather than "can you recognise this sign".
    """
    held_out = set(test_signers)
    known = {clip.signer for clip in clips}
    unknown = held_out - known
    if unknown:
        raise ValueError(
            f"test signer(s) {sorted(unknown)} are not in the manifest; "
            f"available: {sorted(known)}"
        )
    train = [clip for clip in clips if clip.signer not in held_out]
    test = [clip for clip in clips if clip.signer in held_out]
    if not train or not test:
        raise ValueError("signer split left one side empty")
    return train, test


def pick_val_signers(train_clips: list[Clip], count: int = 1) -> tuple[str, ...]:
    """Hold out the train signers with the *most* clips, so validation is the
    least noisy signal available; deterministic, ties broken by name."""
    counts: dict[str, int] = {}
    for clip in train_clips:
        counts[clip.signer] = counts.get(clip.signer, 0) + 1
    ordered = sorted(counts, key=lambda signer: (-counts[signer], signer))
    return tuple(ordered[:count])


@dataclass(frozen=True)
class Splits:
    glosses: list[str]
    train: list[Clip]
    val: list[Clip]
    test: list[Clip]
    val_signers: tuple[str, ...]


def make_splits(cfg: Config, val_signers: tuple[str, ...] | None = None) -> Splits:
    """Turn a config into the train/val/test signer split, single-sourced.

    Training and evaluation must agree on this exactly -- the trustworthiness of
    every reported number rests on the split, so it is defined once here rather
    than reconstructed wherever a model is loaded. Validation scales with the
    test set: a benchmark that holds out N test signers gets N validation
    signers, so checkpoint selection is never made on a noisier sample than the
    number it is chasing.
    """
    clips = read_manifest(cfg.manifest)
    glosses = select_glosses(clips, cfg.max_glosses)
    clips = filter_to(clips, glosses)

    train_pool, test = signer_independent_split(clips, cfg.test_signers)
    val_signers = val_signers or pick_val_signers(train_pool, len(cfg.test_signers))
    train, val = signer_independent_split(train_pool, val_signers)
    return Splits(glosses, train, val, test, val_signers)


class SignDataset(Dataset):
    """Landmark clips -> fixed-length tensors.

    Normalisation and resampling happen here rather than at extraction time, so
    both can be re-tuned without re-running MediaPipe over the whole archive.
    """

    def __init__(
        self,
        clips: list[Clip],
        labels: list[str],
        cfg: Config,
        *,
        train: bool,
        seed: int | None = None,
    ):
        self.clips = clips
        self.labels = labels
        self.label_index = {gloss: i for i, gloss in enumerate(labels)}
        self.cfg = cfg
        self.train = train
        self.root = Path(cfg.landmark_root)
        self._rng = np.random.default_rng(cfg.seed if seed is None else seed)

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int):
        import torch

        clip = self.clips[index]
        sequence = np.load(self.root / clip.path).astype(np.float32)

        if self.cfg.normalize:
            sequence = normalize(sequence)
        if self.train and self.cfg.augment:
            sequence = augment.apply(sequence, self.cfg, self._rng)
        sequence = resample(sequence, self.cfg.frames)

        if not self.cfg.use_pose:
            # Ablation: hide the pose coordinates from the model. Zeroed *after*
            # normalisation, which needs the shoulders, so the hands are still
            # position- and scale-normalised -- the only thing removed is the
            # pose block as an input feature. The presence flags stay; they are
            # about the hands.
            sequence = sequence.copy()
            sequence[:, C.POSE] = 0.0

        return (
            torch.from_numpy(np.ascontiguousarray(sequence)),
            torch.tensor(self.label_index[clip.gloss], dtype=torch.long),
        )


def filter_to(clips: list[Clip], glosses: list[str]) -> list[Clip]:
    keep = set(glosses)
    return [clip for clip in clips if clip.gloss in keep]
