"""Turn a trained checkpoint into an analysis of *what* it gets wrong.

    python -m signa.report --checkpoint runs/lsa64-full-transformer/best.pt \
        --manifest data/manifest_train_full.csv --landmark-root data/landmarks_lsa64 \
        --test-signers 009 010 --val-signers 007 008

A single top-1 number says a model is 88% right; it does not say the 12% is
spread evenly or piled onto a handful of sign pairs, and it does not say whether
the errors are the model's fault or MediaPipe's. This produces both:

  * per-class top-1 accuracy, worst first
  * the most-confused ordered gloss pairs (the off-diagonal mass)
  * accuracy binned by hand-detection rate, and the mean detection rate of
    correct vs incorrect clips -- the link between the corpus audit and the
    result, i.e. how much of the error the features never gave the model a
    chance on

The full confusion matrix is written to CSV for the write-up. Nothing here needs
MediaPipe or a camera; it reads the same landmark .npy files training did.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from . import config as C
from .config import Config
from .dataset import SignDataset, make_splits


def _detection_rate(sequence: np.ndarray) -> float:
    """Fraction of frames with at least one hand -- the audit's either-hand rate."""
    if len(sequence) == 0:
        return 0.0
    return float(np.maximum(sequence[:, C.LEFT_PRESENT], sequence[:, C.RIGHT_PRESENT]).mean())


def predict(checkpoint: Path, cfg: Config, val_signers, device: str):
    """Run the checkpoint over its test split.

    Returns (labels, true, pred, detection) where true/pred are label indices
    and detection is the either-hand rate of each test clip, in the same order.
    """
    import torch

    from .demo import load_checkpoint

    model, labels, saved = load_checkpoint(checkpoint, device)

    # The split must be reconstructed from the same config the model trained on,
    # or "test" here is not the test the number was reported on. Vocabulary and
    # sequence shaping come from the checkpoint; the split comes from the config.
    eval_cfg = Config(
        manifest=cfg.manifest,
        landmark_root=cfg.landmark_root,
        max_glosses=cfg.max_glosses,
        test_signers=cfg.test_signers,
        frames=saved.frames,
        normalize=saved.normalize,
        use_pose=saved.use_pose,
        augment=False,
        threads=cfg.threads,
    )
    splits = make_splits(eval_cfg, val_signers)
    if splits.glosses != labels:
        raise ValueError(
            "checkpoint labels do not match the manifest's gloss set; "
            "the report would be scored against the wrong vocabulary"
        )

    dataset = SignDataset(splits.test, labels, eval_cfg, train=False)
    true = np.array([labels.index(clip.gloss) for clip in splits.test])
    detection = np.array([_detection_rate(np.load(eval_cfg.landmark_root / clip.path))
                          for clip in splits.test])

    if device == "cpu" and cfg.threads > 0:
        torch.set_num_threads(cfg.threads)

    preds = []
    model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            sequence, _ = dataset[index]
            logits = model(sequence.unsqueeze(0).to(device))
            preds.append(int(logits.argmax(dim=1).item()))
    return labels, true, np.array(preds), detection


def confusion(true: np.ndarray, pred: np.ndarray, n: int) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=np.int64)
    np.add.at(matrix, (true, pred), 1)
    return matrix


def per_class_accuracy(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=1)
    correct = np.diag(matrix)
    # A class with no test clips has no accuracy; report NaN rather than 0/0.
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(totals > 0, correct / totals, np.nan)


def most_confused(matrix: np.ndarray, labels: list[str], top: int):
    """The heaviest off-diagonal cells: (true, predicted, count)."""
    pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and matrix[i, j] > 0:
                pairs.append((matrix[i, j], labels[i], labels[j]))
    pairs.sort(reverse=True)
    return pairs[:top]


def detection_bins(detection: np.ndarray, correct: np.ndarray, edges) -> list[tuple]:
    """Accuracy within each hand-detection-rate band."""
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (detection >= low) & (detection <= high if high == edges[-1] else detection < high)
        n = int(mask.sum())
        acc = float(correct[mask].mean()) if n else float("nan")
        rows.append((low, high, n, acc))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--landmark-root", type=Path, required=True)
    parser.add_argument("--test-signers", nargs="+", required=True)
    parser.add_argument("--val-signers", nargs="+", default=None)
    parser.add_argument("--max-glosses", type=int, default=0,
                        help="0 means every gloss in the manifest")
    parser.add_argument("--worst", type=int, default=10)
    parser.add_argument("--csv", type=Path, default=None,
                        help="write the full confusion matrix here")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)

    cfg = Config(
        manifest=args.manifest,
        landmark_root=args.landmark_root,
        max_glosses=args.max_glosses if args.max_glosses > 0 else None,
        test_signers=tuple(args.test_signers),
        threads=args.threads,
    )
    val_signers = tuple(args.val_signers) if args.val_signers else None

    labels, true, pred, detection = predict(args.checkpoint, cfg, val_signers, args.device)
    correct = (true == pred)
    matrix = confusion(true, pred, len(labels))

    print(f"{len(labels)} glosses | {len(true)} test clips | "
          f"top-1 {correct.mean():.1%}\n")

    accuracy = per_class_accuracy(matrix)
    order = np.argsort(np.nan_to_num(accuracy, nan=2.0))  # worst real accuracies first
    print(f"worst {args.worst} classes by top-1")
    print(f"  {'gloss':10} {'acc':>6} {'n':>4}")
    for idx in order[: args.worst]:
        total = int(matrix[idx].sum())
        print(f"  {labels[idx]:10} {accuracy[idx]:>6.1%} {total:>4}")

    print(f"\nmost-confused pairs (true -> predicted)")
    for count, a, b in most_confused(matrix, labels, args.worst):
        print(f"  {a:10} -> {b:10} {count:>4}")

    # The number the whole audit was building toward: is the model failing where
    # the hands were there to see, or where MediaPipe never found them?
    print(f"\naccuracy by hand-detection rate")
    print(f"  {'band':>12} {'n':>5} {'acc':>7}")
    for low, high, n, acc in detection_bins(detection, correct, [0.0, 0.5, 0.7, 0.9, 1.0]):
        band = f"{low:.1f}-{high:.1f}"
        print(f"  {band:>12} {n:>5} {acc:>7.1%}" if n else f"  {band:>12} {n:>5} {'--':>7}")
    if correct.any() and (~correct).any():
        print(f"\n  mean detection, correct clips:   {detection[correct].mean():.3f}")
        print(f"  mean detection, incorrect clips: {detection[~correct].mean():.3f}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true\\pred", *labels])
            for i, row in enumerate(matrix):
                writer.writerow([labels[i], *row.tolist()])
        print(f"\nwrote confusion matrix -> {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
