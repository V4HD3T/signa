"""Audit the full LSA64 extraction, prune, and run the three comparisons.

    python scripts/run_lsa64.py

Reproduces the reported LSA64 numbers from a finished extraction: build a
manifest of the complete glosses, drop clips with no detected hand, then train
the BiLSTM, the Transformer, and an augmentation-off BiLSTM on the benchmark's
signer split. Kept as a script rather than a shell one-liner so the exact
protocol behind the headline table lives in the repository.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from signa.audit import main as audit_main
from signa.config import Config
from signa.dataset import Clip, write_manifest
from signa.train import run

ROOT = Path(__file__).resolve().parents[1]
LANDMARKS = ROOT / "data" / "landmarks_lsa64"
FULL = ROOT / "data" / "manifest_full.csv"
TRAIN = ROOT / "data" / "manifest_train_full.csv"

# LSA64's own layout: 10 signers, held out two-and-two so validation is as wide
# as the test set. Signers are the directory names the extractor wrote.
TEST_SIGNERS = ("009", "010")
VAL_SIGNERS = ("007", "008")
REPEATS_PER_SIGNER = 5
SIGNERS = 10


def build_full_manifest() -> int:
    """List every extracted clip whose gloss is complete (all 50 recordings)."""
    clips = [
        Clip(str(p.relative_to(LANDMARKS)).replace("\\", "/"),
             p.parent.name, p.parent.parent.name, p.stem[-3:])
        for p in LANDMARKS.rglob("*.npy")
    ]
    counts = Counter(c.gloss for c in clips)
    complete = {g for g, n in counts.items() if n == SIGNERS * REPEATS_PER_SIGNER}
    incomplete = sorted(set(counts) - complete)
    if incomplete:
        print(f"skipping {len(incomplete)} still-incomplete gloss(es): {incomplete}")
    kept = sorted((c for c in clips if c.gloss in complete), key=lambda c: c.path)
    write_manifest(FULL, kept)
    print(f"{len(complete)} complete glosses, {len(kept)} clips -> {FULL.name}")
    return len(complete)


def main() -> int:
    build_full_manifest()

    # Prune clips with no hand in any frame; --prune writes TRAIN.
    audit_main([
        "--manifest", str(FULL),
        "--landmark-root", str(LANDMARKS),
        "--prune", str(TRAIN),
    ])

    runs = [
        dict(model="bilstm", tag="lsa64-full", augment=True),
        dict(model="transformer", tag="lsa64-full", augment=True),
        dict(model="bilstm", tag="lsa64-full-noaug", augment=False),
    ]
    table = []
    for spec in runs:
        cfg = Config(
            manifest=TRAIN,
            landmark_root=LANDMARKS,
            max_glosses=None,
            test_signers=TEST_SIGNERS,
            model=spec["model"],
            augment=spec["augment"],
            tag=spec["tag"],
        )
        summary = run(cfg, VAL_SIGNERS)
        label = f"{spec['model']}{'' if spec['augment'] else ' (no aug)'}"
        table.append((label, summary["glosses"], summary["test_top1"], summary["test_top5"]))

    print("\n=== LSA64, signer-independent (test 009/010), full vocabulary ===")
    print(f"{'model':22} {'glosses':>8} {'top-1':>8} {'top-5':>8}")
    for label, glosses, top1, top5 in table:
        print(f"{label:22} {glosses:>8} {top1:>8.1%} {top5:>8.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
