"""Let the model say "I don't know" instead of guessing confidently wrong.

The demo and learning mode always returned a top-1, even for a fumble, an
out-of-vocabulary sign, or noise. A softmax over 64 classes always sums to one,
so *something* always wins -- and a neural net's softmax is famously
overconfident, so the winning number looks reassuring even when it is wrong.

This is selective classification, done in two honest steps:

  1. Calibrate. Temperature scaling (Guo et al. 2017) divides the logits by a
     single scalar T fitted on validation data, so a reported 0.7 actually means
     ~70% right rather than an inflated number. One parameter, fitted by
     minimising validation NLL; it cannot change which class wins, only how
     confident the model is allowed to sound.

  2. Reject. Below a confidence threshold the prediction is withheld as
     "unknown". The threshold is chosen on validation to hit a target accuracy
     among *accepted* signs, so it is a measured decision, not a guessed 0.5.

Everything here is numpy: the model provides logits, and the calibration, the
reject rule, and the risk-coverage evaluation are all pinned by tests without a
camera or torch. The trade it buys -- accuracy on accepted signs vs how many are
accepted (coverage) -- is reported as a curve, because a reject option is only
honest if you also say how often it fires.
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax over the last axis, with temperature.

    T > 1 flattens the distribution (less confident); T < 1 sharpens it. T never
    reorders the classes, so calibration changes confidence without changing the
    prediction."""
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=-1, keepdims=True)


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """Mean NLL of the true labels under temperature-scaled softmax."""
    probs = softmax(logits, temperature)
    rows = np.arange(len(labels))
    true = np.clip(probs[rows, labels], 1e-12, 1.0)
    return float(-np.log(true).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    bounds: tuple[float, float] = (0.05, 10.0), iters: int = 60) -> float:
    """Fit T by minimising validation NLL (golden-section search).

    Temperature-scaling NLL is unimodal in T, so a 1-D bracketing search finds
    the optimum without any gradient machinery. T > 1 is the usual outcome: it
    means the raw model was overconfident and is being cooled down.
    """
    lo, hi = bounds
    invphi = (np.sqrt(5) - 1) / 2  # 1/golden ratio
    c = hi - (hi - lo) * invphi
    d = lo + (hi - lo) * invphi
    fc = negative_log_likelihood(logits, labels, c)
    fd = negative_log_likelihood(logits, labels, d)
    for _ in range(iters):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - (hi - lo) * invphi
            fc = negative_log_likelihood(logits, labels, c)
        else:
            lo, c, fc = c, d, fd
            d = lo + (hi - lo) * invphi
            fd = negative_log_likelihood(logits, labels, d)
    return round((lo + hi) / 2, 4)


def confidence(probs: np.ndarray) -> np.ndarray:
    """Top-1 probability per row."""
    return probs.max(axis=-1)


def margin(probs: np.ndarray) -> np.ndarray:
    """Gap between the top two probabilities -- how *decisively* a class won.

    A high top-1 with a close runner-up is a different kind of confident from a
    high top-1 that swept the field; margin catches the near-ties that a raw
    probability threshold misses."""
    top2 = np.sort(probs, axis=-1)[..., -2:]
    return top2[..., -1] - top2[..., -2]


def entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy per row (nats); high entropy = spread-out, unsure."""
    safe = np.clip(probs, 1e-12, 1.0)
    return -(safe * np.log(safe)).sum(axis=-1)


def decide(probs: np.ndarray, threshold: float, min_margin: float = 0.0):
    """Accepted prediction per row, or -1 for rejected ("unknown").

    Returns predicted class indices where the top-1 probability clears the
    threshold and the margin clears min_margin; -1 elsewhere."""
    probs = np.atleast_2d(probs)
    accept = (confidence(probs) >= threshold) & (margin(probs) >= min_margin)
    return np.where(accept, probs.argmax(axis=-1), -1)


def risk_coverage(probs: np.ndarray, true: np.ndarray, thresholds) -> list[tuple]:
    """For each threshold: (threshold, coverage, selective accuracy).

    Coverage is the fraction of signs accepted; selective accuracy is accuracy
    among only those. Lowering the threshold accepts more and usually costs
    accuracy -- the trade a reject option exists to tune."""
    pred = probs.argmax(axis=-1)
    conf = confidence(probs)
    correct = (pred == true)
    n = len(true)
    rows = []
    for t in thresholds:
        accepted = conf >= t
        k = int(accepted.sum())
        coverage = k / n if n else 0.0
        accuracy = float(correct[accepted].mean()) if k else float("nan")
        rows.append((float(t), coverage, accuracy))
    return rows


def choose_threshold(probs: np.ndarray, true: np.ndarray, target_accuracy: float,
                     grid=None) -> float:
    """Smallest threshold whose accepted signs hit target accuracy -- i.e. the
    most coverage for the required precision.

    If no threshold reaches the target (the model is not accurate enough to be
    that selective without rejecting almost everything), fall back to the
    threshold with the best selective accuracy, so the caller still gets the
    safest option rather than silently the wrong one."""
    if grid is None:
        grid = np.round(np.linspace(0.0, 0.99, 100), 4)
    rows = risk_coverage(probs, true, grid)
    hits = [(t, cov, acc) for t, cov, acc in rows
            if cov > 0 and not np.isnan(acc) and acc >= target_accuracy]
    if hits:
        return min(t for t, _, _ in hits)
    valid = [(t, acc) for t, cov, acc in rows if cov > 0 and not np.isnan(acc)]
    return max(valid, key=lambda ta: ta[1])[0] if valid else 0.0


# --- Calibration CLI ------------------------------------------------------
#
# Fits T and the threshold on the *validation* signers and reports the trade on
# the *test* signers, then writes a sidecar next to the checkpoint that the demo
# and learning mode load. Fitting on validation and reporting on test is the same
# discipline as the split itself: the reject option must not be tuned on the data
# it is judged against.


def sidecar_path(checkpoint) -> "Path":
    from pathlib import Path

    checkpoint = Path(checkpoint)
    return checkpoint.with_suffix(".calib.json")


def load_calibration(checkpoint) -> dict | None:
    import json

    path = sidecar_path(checkpoint)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_logits(model, dataset, device: str):
    import torch

    logits, true = [], []
    model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            sequence, label = dataset[index]
            logits.append(model(sequence.unsqueeze(0).to(device)).cpu().numpy()[0])
            true.append(int(label))
    return np.array(logits), np.array(true)


def main(argv=None) -> int:
    import argparse
    import json
    from pathlib import Path

    from .config import Config
    from .dataset import SignDataset, make_splits
    from .demo import load_checkpoint

    parser = argparse.ArgumentParser(description="calibrate a reject option and report the trade")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--landmark-root", type=Path, required=True)
    parser.add_argument("--test-signers", nargs="+", required=True)
    parser.add_argument("--val-signers", nargs="+", default=None)
    parser.add_argument("--max-glosses", type=int, default=0)
    parser.add_argument("--target-accuracy", type=float, default=0.95,
                        help="required accuracy among accepted signs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write", action="store_true",
                        help="write <checkpoint>.calib.json for demo/learn to use")
    args = parser.parse_args(argv)

    model, labels, saved = load_checkpoint(args.checkpoint, args.device)
    cfg = Config(
        manifest=args.manifest, landmark_root=args.landmark_root,
        max_glosses=args.max_glosses if args.max_glosses > 0 else None,
        test_signers=tuple(args.test_signers),
        frames=saved.frames, normalize=saved.normalize, augment=False,
    )
    val_signers = tuple(args.val_signers) if args.val_signers else None
    splits = make_splits(cfg, val_signers)

    val = SignDataset(splits.val, labels, cfg, train=False)
    test = SignDataset(splits.test, labels, cfg, train=False)
    val_logits, val_true = _collect_logits(model, val, args.device)
    test_logits, test_true = _collect_logits(model, test, args.device)

    temperature = fit_temperature(val_logits, val_true)
    before = negative_log_likelihood(val_logits, val_true, 1.0)
    after = negative_log_likelihood(val_logits, val_true, temperature)
    print(f"temperature {temperature}  (val NLL {before:.3f} -> {after:.3f})")

    val_probs = softmax(val_logits, temperature)
    threshold = choose_threshold(val_probs, val_true, args.target_accuracy)
    print(f"threshold {threshold:.3f} for >= {args.target_accuracy:.0%} accuracy on accepted "
          f"(chosen on validation)\n")

    test_probs = softmax(test_logits, temperature)
    print(f"test risk-coverage (calibrated, {len(test_true)} clips)")
    print(f"  {'thresh':>7} {'coverage':>9} {'sel.acc':>8}")
    for t, cov, acc in risk_coverage(test_probs, test_true,
                                     [0.0, 0.3, 0.5, threshold, 0.7, 0.9]):
        mark = "  <- chosen" if abs(t - threshold) < 1e-9 else ""
        acc_s = f"{acc:.1%}" if not np.isnan(acc) else "--"
        print(f"  {t:>7.3f} {cov:>9.1%} {acc_s:>8}{mark}")

    accepted = confidence(test_probs) >= threshold
    pred = test_probs.argmax(axis=-1)
    if accepted.any():
        sel_acc = float((pred[accepted] == test_true[accepted]).mean())
        print(f"\nat the chosen threshold: accept {accepted.mean():.1%} of signs, "
              f"{sel_acc:.1%} of those correct; the rest get an honest \"not sure\"")

    if args.write:
        payload = {"temperature": temperature, "threshold": round(float(threshold), 4),
                   "target_accuracy": args.target_accuracy}
        sidecar_path(args.checkpoint).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {sidecar_path(args.checkpoint)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
