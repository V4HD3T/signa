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
        use_pose=str(saved.get("use_pose", True)) not in ("False", "false"),
    )
    model = models.build(cfg, len(labels))
    model.load_state_dict(blob["model_state"])
    model.to(device).eval()
    return model, labels, cfg


def classify(model, sequence: np.ndarray, cfg, device: str, top: int = 3,
             temperature: float = 1.0):
    import torch

    from .landmarks import normalize, resample

    prepared = normalize(sequence) if cfg.normalize else sequence
    prepared = resample(prepared, cfg.frames)
    if not cfg.use_pose:
        from . import config as C

        prepared = prepared.copy()
        prepared[:, C.POSE] = 0.0
    batch = torch.from_numpy(np.ascontiguousarray(prepared)).unsqueeze(0).to(device)
    with torch.no_grad():
        # Temperature scaling (see signa.reject): divide the logits before the
        # softmax so the confidence is calibrated, not raw. T=1 is a no-op.
        probabilities = (model(batch) / temperature).softmax(dim=1)[0]
    scores, indices = probabilities.topk(min(top, probabilities.numel()))
    return list(zip(indices.tolist(), scores.tolist()))


def _auto_loop(capture, extractor, model, labels, cfg, *, device, temperature,
               threshold, enter, exit):
    """Continuous, keyless recognition: a motion-energy segmenter (signa.segment)
    finds each sign's start and end, so nothing is held. This is what push-to-sign
    was standing in for until segmentation existed."""
    import cv2

    from .landmarks import normalize
    from .segment import FrameStream, Segmenter

    segmenter = Segmenter(enter=enter, exit=exit)
    stream = FrameStream(segmenter, normalise=lambda f: normalize(f[None])[0])
    results: list[tuple[str, float]] = []

    while True:
        ok, image = capture.read()
        if not ok:
            break

        sign = stream.push(extractor.frame(image))
        if sign is not None:
            results = [(labels[i], s)
                       for i, s in classify(model, sign, cfg, device, temperature=temperature)]
            print(("not sure: " if results[0][1] < threshold else "")
                  + "  ".join(f"{g} {s:.0%}" for g, s in results))

        view = cv2.flip(image, 1)
        height = view.shape[0]
        state = "signing..." if segmenter.active else "sign any time  |  ESC to quit"
        colour = (0, 0, 255) if segmenter.active else (200, 200, 200)
        cv2.putText(view, state, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1)

        if results and results[0][1] < threshold:
            cv2.putText(view, "not sure", (20, height - 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            for row, (gloss, score) in enumerate(results):
                cv2.putText(view, f"{gloss}?  {score:.0%}", (20, height - 52 + row * 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
        else:
            for row, (gloss, score) in enumerate(results):
                c = (0, 255, 0) if row == 0 else (180, 180, 180)
                cv2.putText(view, f"{gloss}  {score:.0%}", (20, height - 80 + row * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9 if row == 0 else 0.6, c,
                            2 if row == 0 else 1)

        cv2.imshow("signa - auto", view)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-frames", type=int, default=10,
                        help="takes shorter than this are ignored as fumbles")
    parser.add_argument("--auto", action="store_true",
                        help="keyless: segment signs automatically instead of holding SPACE")
    parser.add_argument("--enter", type=float, default=None,
                        help="motion threshold to start a sign (auto mode)")
    parser.add_argument("--exit", type=float, default=None,
                        help="stillness threshold to end a sign (auto mode)")
    args = parser.parse_args(argv)

    import cv2

    from .landmarks import Extractor

    from .reject import load_calibration

    model, labels, cfg = load_checkpoint(args.checkpoint, args.device)
    calibration = load_calibration(args.checkpoint)
    temperature = calibration["temperature"] if calibration else 1.0
    threshold = calibration["threshold"] if calibration else 0.0
    print(f"{len(labels)} glosses loaded from {args.checkpoint}"
          + (f"  |  calibrated T={temperature}, reject below {threshold:.0%}"
             if calibration else "  |  no calibration (run signa.reject --write)"))

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(f"could not open camera {args.camera}")
        return 1

    if args.auto:
        from .segment import ENTER, EXIT

        with Extractor() as extractor:
            try:
                _auto_loop(capture, extractor, model, labels, cfg, device=args.device,
                           temperature=temperature, threshold=threshold,
                           enter=args.enter or ENTER, exit=args.exit or EXIT)
            finally:
                capture.release()
                cv2.destroyAllWindows()
        return 0

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

                if results and results[0][1] < threshold:
                    # Confident enough to show a ranking, but not enough to claim
                    # a top-1: say so rather than assert the best of a bad guess.
                    cv2.putText(view, "not sure", (20, height - 82),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                    for row, (gloss, score) in enumerate(results):
                        cv2.putText(view, f"{gloss}?  {score:.0%}", (20, height - 52 + row * 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
                else:
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
                            for i, score in classify(model, sequence, cfg, args.device,
                                                     temperature=temperature)
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
