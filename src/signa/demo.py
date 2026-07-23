"""Push-to-sign webcam demo.

    python -m signa.demo --checkpoint runs/baseline-bilstm/best.pt

Hold SPACE, sign, release -- the held window is classified and the top 3 glosses
are shown. Holding a key is not a product decision, it is a scope decision: the
model is trained on pre-trimmed clips, so "when does a sign begin and end" is a
separate problem (segmentation) that continuous recognition has to solve and
isolated recognition does not. Push-to-sign removes it from the MVP entirely.

Automatic motion-energy segmentation is the v2 item.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_checkpoint(path: Path, device: str):
    import torch

    from . import models
    from .config import Config

    blob = torch.load(path, map_location=device, weights_only=False)
    labels = blob["labels"]
    saved = blob.get("config", {})
    cfg = Config(
        model=saved.get("model", "bilstm"),
        hidden=int(saved.get("hidden", 128)),
        layers=int(saved.get("layers", 2)),
        dropout=float(saved.get("dropout", 0.3)),
        heads=int(saved.get("heads", 4)),
        frames=int(saved.get("frames", 48)),
        normalize=bool(saved.get("normalize", True)),
    )
    model = models.build(cfg, len(labels))
    model.load_state_dict(blob["model_state"])
    model.to(device).eval()
    return model, labels, cfg


def classify(model, sequence: np.ndarray, cfg, device: str, top: int = 3):
    import torch

    from .landmarks import normalize, resample

    prepared = normalize(sequence) if cfg.normalize else sequence
    prepared = resample(prepared, cfg.frames)
    batch = torch.from_numpy(np.ascontiguousarray(prepared)).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = model(batch).softmax(dim=1)[0]
    scores, indices = probabilities.topk(min(top, probabilities.numel()))
    return list(zip(indices.tolist(), scores.tolist()))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-frames", type=int, default=10,
                        help="takes shorter than this are ignored as fumbles")
    args = parser.parse_args(argv)

    import cv2

    from .landmarks import Extractor

    model, labels, cfg = load_checkpoint(args.checkpoint, args.device)
    print(f"{len(labels)} glosses loaded from {args.checkpoint}")

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(f"could not open camera {args.camera}")
        return 1

    buffer: list[np.ndarray] = []
    holding = False
    results: list[tuple[str, float]] = []

    with Extractor() as extractor:
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break

                if holding:
                    buffer.append(extractor.frame(image))

                view = cv2.flip(image, 1)
                height = view.shape[0]

                if holding:
                    cv2.circle(view, (30, 30), 12, (0, 0, 255), -1)
                    cv2.putText(view, f"{len(buffer)} frames", (55, 38),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                else:
                    cv2.putText(view, "hold SPACE and sign  |  ESC to quit",
                                (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                for row, (gloss, score) in enumerate(results):
                    colour = (0, 255, 0) if row == 0 else (180, 180, 180)
                    size = 0.9 if row == 0 else 0.6
                    cv2.putText(view, f"{gloss}  {score:.0%}",
                                (20, height - 80 + row * 28),
                                cv2.FONT_HERSHEY_SIMPLEX, size, colour, 2 if row == 0 else 1)

                cv2.imshow("signa - push to sign", view)

                # cv2.waitKey cannot report key-up, so a take ends when SPACE
                # stops arriving for a couple of frames rather than on release.
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key == 32:
                    if not holding:
                        buffer, results, holding = [], [], True
                elif holding and key == 255:
                    holding = False
                    if len(buffer) >= args.min_frames:
                        sequence = np.stack(buffer)
                        results = [
                            (labels[i], score)
                            for i, score in classify(model, sequence, cfg, args.device)
                        ]
                        print("  ".join(f"{g} {s:.0%}" for g, s in results))
                    else:
                        results = [("(too short)", 0.0)]
        finally:
            capture.release()
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
