"""Train and evaluate a gloss classifier, signer-independently.

    python -m signa.train --model bilstm --max-glosses 50 --test-signers User_1

Three-way signer split: the test signers are held out end to end, and as many
*more* signers are held out of training as validation. Early stopping and
checkpoint selection read validation only. Selecting a checkpoint on the test
signers would quietly turn the headline number into a best-of-N over the test
set, which is the same overfitting the signer-independent split exists to
prevent.

For a benchmark that already defines its own split -- AUTSL holds out 6 of 43
signers -- pass it explicitly rather than letting the defaults invent one:

    python -m signa.train --test-signers signer38 ... --val-signers signer32 ...
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from . import models
from .config import Config
from .dataset import SignDataset, make_splits


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    k = min(k, logits.size(1))
    predicted = logits.topk(k, dim=1).indices
    return (predicted == targets[:, None]).any(dim=1).sum().item()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    total = top1 = top5 = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for sequences, targets in loader:
        sequences, targets = sequences.to(device), targets.to(device)
        logits = model(sequences)
        loss_sum += criterion(logits, targets).item()
        top1 += topk_correct(logits, targets, 1)
        top5 += topk_correct(logits, targets, 5)
        total += targets.numel()
    return {
        "loss": loss_sum / max(total, 1),
        "top1": top1 / max(total, 1),
        "top5": top5 / max(total, 1),
        "n": total,
    }


def run(cfg: Config, val_signers: tuple[str, ...] | None = None) -> dict:
    seed_everything(cfg.seed)
    device = cfg.resolved_device()

    # torch defaults to one thread per core, which for a model this small is
    # actively harmful. Measured on 16 cores: one batch took 475 ms at one
    # thread, 1462 ms at two, 3900 ms at four. Splitting an op this small costs
    # more than the work it saves, and it degrades further when anything else
    # is competing for cores. Set it explicitly rather than inherit it, and
    # re-measure with --threads if the hardware or the model size changes.
    if device == "cpu" and cfg.threads > 0:
        torch.set_num_threads(cfg.threads)

    splits = make_splits(cfg, val_signers)
    glosses = splits.glosses
    train_clips, val_clips, test_clips = splits.train, splits.val, splits.test
    val_signers = splits.val_signers

    print(
        f"{len(glosses)} glosses | "
        f"train {len(train_clips)} clips ({len({c.signer for c in train_clips})} signers) | "
        f"val {len(val_clips)} ({', '.join(val_signers)}) | "
        f"test {len(test_clips)} ({', '.join(cfg.test_signers)})"
    )

    loaders = {
        "train": DataLoader(
            SignDataset(train_clips, glosses, cfg, train=True),
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=len(train_clips) > cfg.batch_size,
        ),
        "val": DataLoader(
            SignDataset(val_clips, glosses, cfg, train=False), batch_size=cfg.batch_size
        ),
        "test": DataLoader(
            SignDataset(test_clips, glosses, cfg, train=False), batch_size=cfg.batch_size
        ),
    }

    model = models.build(cfg, len(glosses)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    out_dir = Path(cfg.out_dir) / f"{cfg.tag}-{cfg.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best = {"top1": -1.0, "epoch": -1}
    history = []
    started = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum = seen = 0.0
        for sequences, targets in loaders["train"]:
            sequences, targets = sequences.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(sequences), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.item() * targets.numel()
            seen += targets.numel()
        schedule.step()

        val = evaluate(model, loaders["val"], device)
        history.append({"epoch": epoch, "train_loss": loss_sum / max(seen, 1), **val})

        if val["top1"] > best["top1"]:
            best = {"top1": val["top1"], "epoch": epoch}
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "labels": glosses,
                    "config": {k: str(v) if isinstance(v, Path) else v
                               for k, v in asdict(cfg).items()},
                },
                out_dir / "best.pt",
            )

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"epoch {epoch:3d}  train_loss {loss_sum / max(seen, 1):.3f}  "
                f"val_top1 {val['top1']:.3f}  val_top5 {val['top5']:.3f}"
            )

        if epoch - best["epoch"] >= cfg.patience:
            print(f"early stop at epoch {epoch} (best was {best['epoch']})")
            break

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device)["model_state"])
    test = evaluate(model, loaders["test"], device)

    summary = {
        "model": cfg.model,
        "glosses": len(glosses),
        "train_clips": len(train_clips),
        "val_signers": list(val_signers),
        "test_signers": list(cfg.test_signers),
        "best_val_epoch": best["epoch"],
        "best_val_top1": best["top1"],
        "test_top1": test["top1"],
        "test_top5": test["top5"],
        "test_clips": test["n"],
        "minutes": round((time.time() - started) / 60, 2),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(
        f"\nsigner-independent test ({', '.join(cfg.test_signers)}, {test['n']} clips): "
        f"top-1 {test['top1']:.1%}  top-5 {test['top5']:.1%}"
    )
    print(f"wrote {out_dir}")
    return summary


def parse_args(argv=None) -> tuple[Config, str | None]:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Config()
    parser.add_argument("--manifest", type=Path, default=defaults.manifest)
    parser.add_argument("--landmark-root", type=Path, default=defaults.landmark_root)
    parser.add_argument("--model", choices=["bilstm", "transformer"], default=defaults.model)
    parser.add_argument("--max-glosses", type=int, default=defaults.max_glosses,
                        help="0 or negative means all glosses in the manifest")
    parser.add_argument("--test-signers", nargs="+", default=list(defaults.test_signers))
    parser.add_argument("--val-signers", nargs="+", default=None,
                        help="defaults to the train signers with the most clips, "
                             "as many as there are test signers")
    parser.add_argument("--frames", type=int, default=defaults.frames)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--hidden", type=int, default=defaults.hidden)
    parser.add_argument("--layers", type=int, default=defaults.layers)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--mirror", type=float, default=defaults.aug_mirror)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--tag", default=defaults.tag)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--threads", type=int, default=defaults.threads,
                        help="CPU intra-op threads; more is slower for a model this "
                             "small (0 keeps torch's default)")
    args = parser.parse_args(argv)

    cfg = Config(
        manifest=args.manifest,
        landmark_root=args.landmark_root,
        model=args.model,
        max_glosses=args.max_glosses if args.max_glosses and args.max_glosses > 0 else None,
        test_signers=tuple(args.test_signers),
        frames=args.frames,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        augment=not args.no_augment,
        aug_mirror=args.mirror,
        seed=args.seed,
        tag=args.tag,
        device=args.device,
        threads=args.threads,
    )
    return cfg, tuple(args.val_signers) if args.val_signers else None


if __name__ == "__main__":
    config, validation_signers = parse_args()
    run(config, validation_signers)
