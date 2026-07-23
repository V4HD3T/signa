"""Quality-check an extracted corpus before training on it.

    python -m signa.audit --manifest data/manifest.csv --landmark-root data/landmarks

Landmark extraction fails quietly. MediaPipe returns *something* for every
frame, so a corpus where hands were never detected trains without error and
produces a model that has learned the pose subset and nothing else. The
accuracy is bad, the loss curve looks normal, and there is no message anywhere
saying the features were empty.

This prints what the training run cannot tell you: how often each hand was
actually found, how long the clips are, and whether any signer or gloss is
systematically worse than the rest.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import config as C
from .dataset import read_manifest, write_manifest


def summarise(values: list[float]) -> str:
    array = np.asarray(values, dtype=np.float64)
    return (
        f"mean {array.mean():6.3f}   "
        f"p10 {np.percentile(array, 10):6.3f}   "
        f"median {np.median(array):6.3f}   "
        f"p90 {np.percentile(array, 90):6.3f}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--landmark-root", type=Path, required=True)
    parser.add_argument("--worst", type=int, default=5,
                        help="how many worst signers/glosses to list")
    parser.add_argument("--prune", type=Path, default=None,
                        help="write a manifest with the unusable clips removed")
    parser.add_argument("--min-detection", type=float, default=0.0,
                        help="with --prune, drop clips whose either-hand detection rate "
                             "is at or below this (default 0.0: only clips with no hand "
                             "in any frame)")
    args = parser.parse_args(argv)

    clips = read_manifest(args.manifest)
    print(f"{len(clips)} clips | "
          f"{len({c.gloss for c in clips})} glosses | "
          f"{len({c.signer for c in clips})} signers\n")

    lengths: list[float] = []
    left: list[float] = []
    right: list[float] = []
    either: list[float] = []
    pose: list[float] = []
    by_signer: dict[str, list[float]] = defaultdict(list)
    by_gloss: dict[str, list[float]] = defaultdict(list)
    dead: list[str] = []
    rates: dict[str, float] = {}

    for clip in clips:
        sequence = np.load(args.landmark_root / clip.path)
        if len(sequence) == 0:
            dead.append(clip.path)
            rates[clip.path] = 0.0
            continue

        left_rate = float(sequence[:, C.LEFT_PRESENT].mean())
        right_rate = float(sequence[:, C.RIGHT_PRESENT].mean())
        either_rate = float(
            np.maximum(sequence[:, C.LEFT_PRESENT], sequence[:, C.RIGHT_PRESENT]).mean()
        )
        pose_rate = float((np.abs(sequence[:, C.POSE]).sum(axis=1) > 0).mean())

        lengths.append(len(sequence))
        left.append(left_rate)
        right.append(right_rate)
        either.append(either_rate)
        pose.append(pose_rate)
        by_signer[clip.signer].append(either_rate)
        by_gloss[clip.gloss].append(either_rate)
        rates[clip.path] = either_rate

        if either_rate == 0.0:
            dead.append(clip.path)

    print("clip length (frames)")
    print(f"  {summarise(lengths)}\n")

    print("detection rate, fraction of frames")
    print(f"  left hand   {summarise(left)}")
    print(f"  right hand  {summarise(right)}")
    print(f"  either hand {summarise(either)}")
    print(f"  pose        {summarise(pose)}\n")

    # A hand missing from most frames is not automatically a problem -- most
    # sign languages have one-handed signs, and LSA64 is 42 of 64. A *pose*
    # missing is always a problem: the normaliser needs shoulders.
    print(f"clips with no hand in any frame: {len(dead)}")
    for path in dead[: args.worst]:
        print(f"  {path}")
    if len(dead) > args.worst:
        print(f"  ... and {len(dead) - args.worst} more")

    print(f"\nworst {args.worst} signers by either-hand detection")
    for signer, values in sorted(by_signer.items(), key=lambda kv: np.mean(kv[1]))[: args.worst]:
        print(f"  {signer}  {np.mean(values):.3f}  ({len(values)} clips)")

    print(f"\nworst {args.worst} glosses by either-hand detection")
    for gloss, values in sorted(by_gloss.items(), key=lambda kv: np.mean(kv[1]))[: args.worst]:
        print(f"  {gloss}  {np.mean(values):.3f}  ({len(values)} clips)")

    if args.prune is not None:
        kept = [c for c in clips if rates.get(c.path, 0.0) > args.min_detection]
        dropped = len(clips) - len(kept)
        write_manifest(args.prune, kept)
        print(f"\npruned at detection > {args.min_detection}: "
              f"kept {len(kept)}, dropped {dropped} -> {args.prune}")
        # Dropping clips can empty a class entirely; a gloss present in training
        # but absent from test is a silent hole in the reported accuracy.
        lost = {c.gloss for c in clips} - {c.gloss for c in kept}
        if lost:
            print(f"warning: {len(lost)} gloss(es) lost every clip: {sorted(lost)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
