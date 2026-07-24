"""Gather every run's summary into one table, so the numbers have one home.

    python -m signa.results                 # a table of every run under runs/
    python -m signa.results --markdown      # the same, ready to paste

The findings quoted in the README were, until now, copied there by hand from
scattered `summary.json` files -- which is exactly how a number in prose drifts
from the number the code produced. This reads the artifacts directly: every
`summary.json` for single runs, every `loso.json` for cross-validation, sorted
best-first. The comparison table in the README is meant to be its output, not a
transcription of it.

Robust to older runs that predate a field (augmentation and pose flags were
added to the summary later): a missing value shows as "—" rather than a guess.
The scanning is I/O; the row-building and formatting are pure and tested.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Plain ASCII, not an em-dash: this table is read in a terminal, and a Windows
# console on a non-UTF-8 codepage renders "—" as a replacement character.
DASH = "-"


def _flag(value, true="yes", false="no") -> str:
    if value is None:
        return DASH
    if isinstance(value, str):
        value = value not in ("False", "false")
    return true if value else false


def read_summaries(runs_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            | {"_run": p.parent.name}
            for p in sorted(Path(runs_dir).glob("*/summary.json"))]


def read_loso(runs_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            | {"_run": p.parent.name}
            for p in sorted(Path(runs_dir).glob("*/loso.json"))]


def summary_row(summary: dict) -> dict:
    """Normalise one summary.json into a display row, tolerating missing fields."""
    return {
        "run": summary.get("_run", DASH),
        "model": summary.get("model", DASH),
        "pose": _flag(summary.get("use_pose")),
        "aug": _flag(summary.get("augment")),
        "glosses": summary.get("glosses", DASH),
        "top1": summary.get("test_top1"),
        "top5": summary.get("test_top5"),
        "clips": summary.get("test_clips", DASH),
        "minutes": summary.get("minutes", DASH),
    }


def loso_row(result: dict) -> dict:
    top1, top5 = result["top1"], result["top5"]
    return {
        "run": result.get("_run", DASH),
        "model": result.get("model", DASH),
        "folds": result.get("folds", DASH),
        "top1": top1["mean"],
        "top1_std": top1["std"],
        "top5": top5["mean"],
        "top5_std": top5["std"],
    }


def _pct(value) -> str:
    return f"{value:.1%}" if isinstance(value, (int, float)) else DASH


def format_markdown(rows: list[dict]) -> str:
    header = "| run | model | pose | aug | glosses | top-1 | top-5 | clips | min |"
    rule = "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, rule]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['model']} | {r['pose']} | {r['aug']} | {r['glosses']} "
            f"| {_pct(r['top1'])} | {_pct(r['top5'])} | {r['clips']} | {r['minutes']} |"
        )
    return "\n".join(lines)


def format_text(rows: list[dict]) -> str:
    lines = [f"{'run':28} {'model':12} {'pose':>4} {'aug':>4} {'gloss':>5} "
             f"{'top-1':>7} {'top-5':>7} {'clips':>6}"]
    for r in rows:
        lines.append(
            f"{str(r['run'])[:28]:28} {str(r['model']):12} {r['pose']:>4} {r['aug']:>4} "
            f"{str(r['glosses']):>5} {_pct(r['top1']):>7} {_pct(r['top5']):>7} "
            f"{str(r['clips']):>6}"
        )
    return "\n".join(lines)


def format_loso(rows: list[dict]) -> str:
    if not rows:
        return ""
    lines = ["", "leave-one-signer-out:"]
    for r in rows:
        lines.append(
            f"  {str(r['run'])[:28]:28} {str(r['model']):12} {r['folds']} folds  "
            f"top-1 {r['top1']:.1%} +/- {r['top1_std']:.1%}  "
            f"top-5 {r['top5']:.1%} +/- {r['top5_std']:.1%}"
        )
    return "\n".join(lines)


def sort_rows(rows: list[dict]) -> list[dict]:
    """Best top-1 first; rows without a top-1 sink to the bottom."""
    return sorted(rows, key=lambda r: (r["top1"] is None, -(r["top1"] or 0.0)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--markdown", action="store_true", help="emit a markdown table")
    args = parser.parse_args(argv)

    summaries = read_summaries(args.runs)
    if not summaries:
        print(f"no summary.json files under {args.runs}")
        return 1

    rows = sort_rows([summary_row(s) for s in summaries])
    print(format_markdown(rows) if args.markdown else format_text(rows))

    loso = [loso_row(r) for r in read_loso(args.runs)]
    if loso and not args.markdown:
        print(format_loso(loso))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
