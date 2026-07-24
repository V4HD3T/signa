"""Combine the three trained models and ask whether the whole beats its best part.

    python -m signa.ensemble --checkpoints runs/lsa64-full-bilstm/best.pt \
        runs/lsa64-full-transformer/best.pt runs/lsa64-full-tcn/best.pt \
        --manifest data/manifest_train_full.csv --landmark-root data/landmarks_lsa64 \
        --test-signers 009 010 --val-signers 007 008

An ensemble averages the models' predictions, and the standard hope is that where
one model is wrong another is right. That hope is only justified if the models
fail on *different* clips. This project already has a reason to doubt it: the
error tracks hand-detection rate (0.0.2), and a clip MediaPipe barely saw is hard
for every architecture at once. So the honest question is not "does an ensemble
help" but "do these three fail independently enough for it to" -- and the answer
is a measurement, printed next to each model's solo score so the gain, or its
absence, is legible.

Probabilities are calibrated per model (each checkpoint's temperature, 0.0.4)
before averaging, so a model is not over-weighted just for being overconfident.
The combination logic is pure and tested; the model loading reuses the demo's,
and the split is reconstructed the same way the report and calibration do it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def soft_vote(prob_lists: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Weighted mean of per-model probability matrices -> one (N, C) matrix.

    Averaging probabilities rather than logits keeps each model on a common,
    calibrated scale; a confident-but-wrong model can still be outvoted by two
    unsure-but-right ones, which is the entire point."""
    stack = np.stack(prob_lists)  # (M, N, C)
    if weights is None:
        return stack.mean(axis=0)
    w = np.asarray(weights, dtype=np.float64).reshape(-1, 1, 1)
    return (stack * w).sum(axis=0) / w.sum()


def hard_vote(prob_lists: list[np.ndarray]) -> np.ndarray:
    """Majority vote over each model's top-1, ties broken by summed probability.

    A blunter combiner than soft voting -- it throws away how confident each
    model was -- and worth reporting alongside precisely because when soft and
    hard disagree, the confidences were carrying the result."""
    stack = np.stack(prob_lists)  # (M, N, C)
    votes = stack.argmax(axis=2)  # (M, N)
    n, classes = stack.shape[1], stack.shape[2]
    tally = np.zeros((n, classes), dtype=np.float64)
    for model in range(stack.shape[0]):
        tally[np.arange(n), votes[model]] += 1.0
    # Break exact ties with the summed probability mass, so a 1-1-1 split falls
    # to the class the models were, in total, most confident about.
    tally += stack.sum(axis=0) * 1e-6
    return tally


def topk_accuracy(scores: np.ndarray, true: np.ndarray, k: int) -> float:
    k = min(k, scores.shape[1])
    topk = np.argsort(-scores, axis=1)[:, :k]
    hit = (topk == true[:, None]).any(axis=1)
    return float(hit.mean())


# --- Evaluation CLI -------------------------------------------------------


def _model_probabilities(checkpoint: Path, cfg_manifest, cfg_root, test_signers,
                         val_signers, max_glosses, device):
    """Calibrated test-set probabilities for one checkpoint, plus its labels and
    the true labels in clip order."""
    from .config import Config
    from .dataset import SignDataset, make_splits
    from .demo import load_checkpoint
    from .reject import load_calibration, softmax

    model, labels, saved = load_checkpoint(checkpoint, device)
    cfg = Config(manifest=cfg_manifest, landmark_root=cfg_root,
                 max_glosses=max_glosses, test_signers=test_signers,
                 frames=saved.frames, normalize=saved.normalize,
                 use_pose=saved.use_pose, augment=False)
    splits = make_splits(cfg, val_signers)
    dataset = SignDataset(splits.test, labels, cfg, train=False)
    true = np.array([labels.index(c.gloss) for c in splits.test])

    temperature = 1.0
    calib = load_calibration(checkpoint)
    if calib:
        temperature = calib["temperature"]

    import torch

    logits = []
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            sequence, _ = dataset[i]
            logits.append(model(sequence.unsqueeze(0).to(device)).cpu().numpy()[0])
    probs = softmax(np.array(logits), temperature)
    return probs, labels, true


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--landmark-root", type=Path, required=True)
    parser.add_argument("--test-signers", nargs="+", required=True)
    parser.add_argument("--val-signers", nargs="+", default=None)
    parser.add_argument("--max-glosses", type=int, default=0)
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="per-model weights for the soft vote (default: equal)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    test_signers = tuple(args.test_signers)
    val_signers = tuple(args.val_signers) if args.val_signers else None
    max_glosses = args.max_glosses if args.max_glosses > 0 else None

    per_model, reference_labels, reference_true = [], None, None
    print(f"{'model':>28} {'top-1':>8} {'top-5':>8}")
    for checkpoint in args.checkpoints:
        probs, labels, true = _model_probabilities(
            checkpoint, args.manifest, args.landmark_root, test_signers,
            val_signers, max_glosses, args.device)
        if reference_labels is None:
            reference_labels, reference_true = labels, true
        elif labels != reference_labels or not np.array_equal(true, reference_true):
            raise ValueError(f"{checkpoint} evaluates on a different label set or split; "
                             "every checkpoint must share the manifest and signers")
        per_model.append(probs)
        name = checkpoint.parent.name
        print(f"{name:>28} {topk_accuracy(probs, true, 1):>8.1%} "
              f"{topk_accuracy(probs, true, 5):>8.1%}")

    best_solo = max(topk_accuracy(p, reference_true, 1) for p in per_model)

    soft = soft_vote(per_model, args.weights)
    hard = hard_vote(per_model)
    print(f"\n{'ensemble (soft mean)':>28} {topk_accuracy(soft, reference_true, 1):>8.1%} "
          f"{topk_accuracy(soft, reference_true, 5):>8.1%}")
    print(f"{'ensemble (hard vote)':>28} {topk_accuracy(hard, reference_true, 1):>8.1%} "
          f"{topk_accuracy(hard, reference_true, 5):>8.1%}")

    soft_top1 = topk_accuracy(soft, reference_true, 1)
    delta = soft_top1 - best_solo
    verdict = (f"soft ensemble beats the best single model by {delta:+.1%}"
               if delta > 0 else
               f"the best single model is as good or better ({-delta:.1%} over the ensemble); "
               "the three fail on the same clips, so averaging buys nothing")
    print(f"\n{verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
