"""Run MediaPipe over a directory of clips and write landmarks + a manifest.

    python -m signa.extract --videos data/raw --out data/landmarks

Two ways to recover (gloss, signer) for each clip:

  --layout dir       <videos>/<signer>/<gloss>/<clip>.mp4     (default)
  --pattern REGEX    a regex over the filename with named groups
                     `gloss`, `signer`, and optionally `repeat`

IMPORTANT: BosphorusSign22k's own naming is not assumed here. When the archive
arrives, run with `--dry-run` first and check that the printed (gloss, signer)
pairs are right. Getting this wrong does not crash anything -- it silently
produces a split where the same signer appears on both sides, and an accuracy
number that means nothing.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from .dataset import Clip, write_manifest

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def find_videos(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)


def parse_by_directory(video: Path, root: Path) -> tuple[str, str, str] | None:
    """<root>/<signer>/<gloss>/<clip>.<ext>"""
    parts = video.relative_to(root).parts
    if len(parts) < 3:
        return None
    return parts[-2], parts[-3], video.stem  # gloss, signer, repeat


def parse_by_pattern(video: Path, pattern: re.Pattern) -> tuple[str, str, str] | None:
    match = pattern.search(video.name)
    if not match:
        return None
    groups = match.groupdict()
    if not groups.get("gloss") or not groups.get("signer"):
        return None
    return groups["gloss"], groups["signer"], groups.get("repeat") or video.stem


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/landmarks"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--layout", choices=["dir"], default="dir")
    parser.add_argument("--pattern", default=None,
                        help=r"e.g. '(?P<signer>User_\d+)_(?P<gloss>\w+?)_(?P<repeat>\d+)'")
    parser.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2],
                        help="MediaPipe pose model complexity; 2 is slower and more accurate")
    parser.add_argument("--limit", type=int, default=None, help="stop after N clips")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the parsed (gloss, signer) pairs and exit")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    videos = find_videos(args.videos)
    if not videos:
        print(f"no video files under {args.videos}", file=sys.stderr)
        return 1
    if args.limit:
        videos = videos[: args.limit]

    pattern = re.compile(args.pattern) if args.pattern else None
    parsed: list[tuple[Path, str, str, str]] = []
    skipped = 0
    for video in videos:
        fields = (
            parse_by_pattern(video, pattern) if pattern
            else parse_by_directory(video, args.videos)
        )
        if fields is None:
            skipped += 1
            continue
        parsed.append((video, *fields))

    print(f"{len(parsed)} clips parsed, {skipped} unparseable")
    if parsed:
        glosses = {gloss for _, gloss, _, _ in parsed}
        signers = {signer for _, _, signer, _ in parsed}
        print(f"{len(glosses)} glosses, {len(signers)} signers: {sorted(signers)}")

    if args.dry_run:
        for video, gloss, signer, repeat in parsed[:20]:
            print(f"  {video.name}  ->  gloss={gloss!r} signer={signer!r} repeat={repeat!r}")
        if len(parsed) > 20:
            print(f"  ... and {len(parsed) - 20} more")
        return 0
    if skipped:
        print(f"warning: {skipped} clips were skipped -- check --layout/--pattern",
              file=sys.stderr)

    from .landmarks import Extractor  # imported late so --dry-run needs no mediapipe

    args.out.mkdir(parents=True, exist_ok=True)
    clips: list[Clip] = []
    empty: list[str] = []

    with Extractor(complexity=args.complexity) as extractor:
        for index, (video, gloss, signer, repeat) in enumerate(parsed, start=1):
            relative = Path(signer) / gloss / f"{video.stem}.npy"
            destination = args.out / relative
            if destination.exists() and not args.overwrite:
                clips.append(Clip(str(relative).replace("\\", "/"), gloss, signer, repeat))
                continue

            sequence = extractor.video(video)
            if len(sequence) == 0:
                empty.append(video.name)
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            np.save(destination, sequence)
            clips.append(Clip(str(relative).replace("\\", "/"), gloss, signer, repeat))

            if index % 25 == 0 or index == len(parsed):
                print(f"  {index}/{len(parsed)}", flush=True)

    write_manifest(args.manifest, clips)
    print(f"wrote {len(clips)} landmark files and {args.manifest}")
    if empty:
        print(f"warning: {len(empty)} clips produced no frames, e.g. {empty[:3]}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
