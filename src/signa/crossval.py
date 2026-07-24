"""Leave-one-signer-out cross-validation.

    python -m signa.crossval --manifest data/manifest_train_full.csv \
        --landmark-root data/landmarks_lsa64 --model tcn --tag lsa64-loso

A single held-out signer pair gives one number, and one number invites the
question a defence always asks: what if that split was lucky? This holds out each
signer in turn as the test set, trains on the rest, and reports the mean and
spread across every fold. The headline stops being "98.2% on signers 009/010" and
becomes "97 ± 2% however we pick the held-out signer" -- a claim about the method,
not about one fortunate partition.

Each fold reuses the same three-way discipline as a single run: the test signer
is held out end to end, and one of the remaining signers is held out again for
validation (chosen the usual way, the train signer with the most clips). The fold
planning and the aggregation are pure and tested; the training is `train.run`
unchanged, once per fold.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path

from .config import Config
from .dataset import read_manifest
from .train import run


def signers_in(manifest) -> list[str]:
    """Every signer in the manifest, sorted, so folds are deterministic."""
    return sorted({clip.signer for clip in read_manifest(manifest)})


def fold_plan(pool: list[str], test_signers: list[str] | None = None) -> list[str]:
    """The test signer for each fold: every signer once, or a chosen subset.

    `pool` is every signer available; `test_signers` restricts which ones get a
    fold (for a faster partial run) but never restricts the training data. The
    check is on the pool, not the folds: a run testing only two signers is fine
    as long as the manifest has enough others to train and validate on. Written
    down and tested so "each fold is signer-disjoint, and there is always
    something to train on" is a named property rather than a loop invariant
    nobody checks."""
    if len(pool) < 3:
        raise ValueError("leave-one-signer-out needs at least 3 signers in the "
                         "manifest (test, validation, and something to train on)")
    folds = list(test_signers) if test_signers else list(pool)
    unknown = set(folds) - set(pool)
    if unknown:
        raise ValueError(f"test signer(s) {sorted(unknown)} are not in the manifest")
    return folds


def aggregate(summaries: list[dict]) -> dict:
    """Mean and spread of the per-fold test metrics."""
    top1 = [s["test_top1"] for s in summaries]
    top5 = [s["test_top5"] for s in summaries]

    def stats(values: list[float]) -> dict:
        return {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    return {
        "folds": len(summaries),
        "top1": stats(top1),
        "top5": stats(top5),
        "per_fold": [
            {"test_signer": s["test_signers"][0], "top1": s["test_top1"],
             "top5": s["test_top5"], "test_clips": s["test_clips"]}
            for s in summaries
        ],
    }


def cross_validate(cfg: Config, signers: list[str] | None = None) -> dict:
    """Run one fold per test signer and aggregate. Each fold's model is trained
    fresh on every signer except the one held out. `signers` restricts which
    signers are tested; training always uses the rest of the manifest."""
    pool = signers_in(cfg.manifest)
    plan = fold_plan(pool, signers)
    print(f"leave-one-signer-out: {len(plan)} folds over {plan} "
          f"(pool of {len(pool)} signers)\n")

    summaries = []
    for i, test_signer in enumerate(plan, start=1):
        print(f"=== fold {i}/{len(plan)}: test signer {test_signer} ===")
        fold_cfg = replace(cfg, test_signers=(test_signer,),
                           tag=f"{cfg.tag}-{test_signer}")
        # val_signers=None: train.run picks the most-populous remaining signer,
        # so validation is chosen the same way in every fold.
        summaries.append(run(fold_cfg))

    result = {"model": cfg.model, "glosses": summaries[0]["glosses"], **aggregate(summaries)}

    out = Path(cfg.out_dir) / f"{cfg.tag}-{cfg.model}-loso"
    out.mkdir(parents=True, exist_ok=True)
    (out / "loso.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n=== leave-one-signer-out, {cfg.model}, {result['folds']} folds ===")
    print(f"  {'signer':>8} {'top-1':>8} {'top-5':>8}")
    for fold in result["per_fold"]:
        print(f"  {fold['test_signer']:>8} {fold['top1']:>8.1%} {fold['top5']:>8.1%}")
    print(f"  {'mean':>8} {result['top1']['mean']:>8.1%} {result['top5']['mean']:>8.1%}")
    print(f"  {'std':>8} {result['top1']['std']:>8.1%} {result['top5']['std']:>8.1%}")
    print(f"\ntop-1 {result['top1']['mean']:.1%} ± {result['top1']['std']:.1%} "
          f"(min {result['top1']['min']:.1%}, max {result['top1']['max']:.1%})")
    print(f"wrote {out / 'loso.json'}")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = Config()
    parser.add_argument("--manifest", type=Path, default=defaults.manifest)
    parser.add_argument("--landmark-root", type=Path, default=defaults.landmark_root)
    parser.add_argument("--model", choices=["bilstm", "transformer", "tcn"],
                        default=defaults.model)
    parser.add_argument("--max-glosses", type=int, default=0,
                        help="0 means every gloss in the manifest")
    parser.add_argument("--signers", nargs="+", default=None,
                        help="restrict folds to these test signers (default: all)")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--tag", default="loso")
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--threads", type=int, default=defaults.threads)
    args = parser.parse_args(argv)

    cfg = Config(
        manifest=args.manifest,
        landmark_root=args.landmark_root,
        model=args.model,
        max_glosses=args.max_glosses if args.max_glosses > 0 else None,
        epochs=args.epochs,
        tag=args.tag,
        device=args.device,
        threads=args.threads,
    )
    return cfg, args.signers


if __name__ == "__main__":
    config, signer_list = parse_args()
    cross_validate(config, signer_list)
