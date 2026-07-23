"""Record your own clips into the directory layout `extract` expects.

    python -m signa.record --gloss merhaba --signer me --count 5

This exists so the whole pipeline -- extraction, normalisation, split, training,
demo -- can be exercised on a handful of self-recorded signs while the dataset
EULA is still in the post. A model trained on five glosses by one signer is
worthless as a result, but it proves every seam fits before the real data lands.

SPACE starts and stops a take, ESC quits.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gloss", required=True)
    parser.add_argument("--signer", default="me")
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--count", type=int, default=5, help="takes to record")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv)

    import cv2

    directory = args.out / args.signer / args.gloss
    directory.mkdir(parents=True, exist_ok=True)
    existing = len(list(directory.glob("*.mp4")))

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(f"could not open camera {args.camera}")
        return 1

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    recorded = 0
    writer = None
    try:
        while recorded < args.count:
            ok, image = capture.read()
            if not ok:
                break
            view = cv2.flip(image, 1)  # mirror the preview only, never the saved frame

            if writer is not None:
                writer.write(image)
                cv2.circle(view, (30, 30), 12, (0, 0, 255), -1)

            status = "RECORDING - SPACE to stop" if writer else "SPACE to start take"
            cv2.putText(view, f"{args.gloss}  [{existing + recorded + 1}/{existing + args.count}]",
                        (20, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(view, status, (20, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("signa - record", view)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == 32:  # SPACE
                if writer is None:
                    path = directory / f"{args.signer}_{args.gloss}_{existing + recorded + 1:03d}.mp4"
                    writer = cv2.VideoWriter(str(path), fourcc, args.fps, (width, height))
                    print(f"recording {path.name}")
                else:
                    writer.release()
                    writer = None
                    recorded += 1
    finally:
        if writer is not None:
            writer.release()
        capture.release()
        cv2.destroyAllWindows()

    print(f"{recorded} take(s) saved to {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
