"""Learning mode: the app names a sign, you perform it, it grades you.

    python -m signa.learn --checkpoint runs/lsa64-full-transformer/best.pt

A spaced-repetition practice session over the model's vocabulary. Each round it
picks a sign to practise (`practice.next_gloss`), shows its name, and waits for a
push-to-sign attempt; the model's ranked guess is graded (`practice.grade`) and
the sign's schedule advances (SM-2). Progress -- per-sign schedule, streak, daily
goal -- persists to a JSON file between sessions.

All the judgement lives in `practice`, tested without hardware. This module is
the webcam loop around it, and reuses the demo's checkpoint loading and
classifier so training, the demo, and learning mode all read landmarks the same
way.

Gloss ids like "001" are not words. Pass `--labels names.json` (a {gloss: name}
map) to show real sign names; without it the raw gloss id is shown.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from . import practice
from .demo import classify, load_checkpoint


def display_name(gloss: str, names: dict[str, str]) -> str:
    return names.get(gloss, gloss)


def run_session(model, labels, cfg, store_path, names, *, device, camera, min_frames,
                temperature=1.0, reject_threshold=0.0, today=None):
    import cv2

    from .landmarks import Extractor

    today = today or date.today()
    progress = practice.load(store_path)
    vocabulary = list(labels)

    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        print(f"could not open camera {camera}")
        return 1

    target, is_new = practice.next_gloss(progress, vocabulary, today)
    buffer: list[np.ndarray] = []
    holding = False
    banner = "get ready"
    detail = ""

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

                tag = "NEW" if is_new else "review"
                cv2.putText(view, f"sign: {display_name(target, names)}  [{tag}]",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                streak = progress.streak(today)
                done = progress.reviews_today(today)
                cv2.putText(view, f"streak {streak}d   today {done}/{progress.daily_goal}",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

                if holding:
                    cv2.circle(view, (30, height - 30), 12, (0, 0, 255), -1)
                    cv2.putText(view, f"{len(buffer)} frames", (55, height - 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                else:
                    cv2.putText(view, banner, (20, height - 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    if detail:
                        cv2.putText(view, detail, (20, height - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
                    cv2.putText(view, "hold SPACE to sign  |  n skip  |  ESC quit",
                                (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)

                cv2.imshow("signa - learn", view)
                key = cv2.waitKey(1) & 0xFF

                if key == 27:  # ESC
                    break
                if key == ord("n") and not holding:  # skip without grading
                    target, is_new = practice.next_gloss(progress, vocabulary, today)
                    banner, detail = "skipped", ""
                elif key == 32 and not holding:  # SPACE down
                    buffer, holding = [], True
                elif holding and key == 255:  # SPACE released (no key this frame)
                    holding = False
                    if len(buffer) < min_frames:
                        banner, detail = "too short, try again", ""
                        continue
                    ranked = [(labels[i], p)
                              for i, p in classify(model, np.stack(buffer), cfg, device,
                                                   temperature=temperature)]
                    # A rejected read -- the model is not confident enough to
                    # judge -- is not evidence the learner forgot the sign, so it
                    # does not advance the schedule. The same sign is offered
                    # again rather than scored as a lapse.
                    if ranked and ranked[0][1] < reject_threshold:
                        banner, detail = "couldn't read that", "try again, more clearly"
                    else:
                        attempt = practice.grade(target, ranked)
                        progress.record(target, attempt.quality, today)
                        practice.save(progress, store_path)
                        banner, detail = _feedback(attempt, names)
                        target, is_new = practice.next_gloss(progress, vocabulary, today)
        finally:
            capture.release()
            cv2.destroyAllWindows()

    practice.save(progress, store_path)
    print(f"streak {progress.streak(today)} day(s), "
          f"{progress.reviews_today(today)} reviews today -> {store_path}")
    return 0


def run_auto_session(model, labels, cfg, store_path, names, *, device, camera, temperature,
                     reject_threshold, enter, exit, today=None):
    """Keyless practice: the segmenter (signa.segment) detects each completed
    sign, so the learner just signs the prompt instead of holding a key. Same
    grading and scheduling as the push-to-sign session, driven by motion instead
    of a keypress."""
    import cv2

    from .landmarks import Extractor, normalize
    from .segment import FrameStream, Segmenter

    today = today or date.today()
    progress = practice.load(store_path)
    vocabulary = list(labels)
    segmenter = Segmenter(enter=enter, exit=exit)

    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        print(f"could not open camera {camera}")
        return 1

    target, is_new = practice.next_gloss(progress, vocabulary, today)
    banner, detail = "sign when ready", ""

    with Extractor() as extractor:
        stream = FrameStream(segmenter, normalise=lambda f: normalize(f[None])[0])
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break

                sign = stream.push(extractor.frame(image))
                if sign is not None:
                    ranked = [(labels[i], p)
                              for i, p in classify(model, sign, cfg, device,
                                                   temperature=temperature)]
                    if ranked and ranked[0][1] < reject_threshold:
                        banner, detail = "couldn't read that", "try again, more clearly"
                    else:
                        attempt = practice.grade(target, ranked)
                        progress.record(target, attempt.quality, today)
                        practice.save(progress, store_path)
                        banner, detail = _feedback(attempt, names)
                        target, is_new = practice.next_gloss(progress, vocabulary, today)

                view = cv2.flip(image, 1)
                height = view.shape[0]
                tag = "NEW" if is_new else "review"
                cv2.putText(view, f"sign: {display_name(target, names)}  [{tag}]",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(view, f"streak {progress.streak(today)}d   "
                            f"today {progress.reviews_today(today)}/{progress.daily_goal}",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
                if segmenter.active:
                    cv2.circle(view, (30, height - 30), 12, (0, 0, 255), -1)
                cv2.putText(view, banner, (20, height - 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if detail:
                    cv2.putText(view, detail, (20, height - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
                cv2.putText(view, "sign the prompt  |  n skip  |  ESC quit",
                            (20, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
                cv2.imshow("signa - learn (auto)", view)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key == ord("n"):
                    target, is_new = practice.next_gloss(progress, vocabulary, today)
                    banner, detail = "skipped", ""
        finally:
            capture.release()
            cv2.destroyAllWindows()

    practice.save(progress, store_path)
    print(f"streak {progress.streak(today)} day(s), "
          f"{progress.reviews_today(today)} reviews today -> {store_path}")
    return 0


def _feedback(attempt: practice.Attempt, names: dict[str, str]) -> tuple[str, str]:
    guessed = display_name(attempt.predicted, names)
    if attempt.verdict == practice.CORRECT:
        return "correct!", f"{attempt.confidence:.0%} confident"
    if attempt.verdict == practice.CLOSE:
        return "close", f"read as '{guessed}' -- keep the shape crisp"
    return "not recognised", f"read as '{guessed}'"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--progress", type=Path, default=Path("data/progress.json"))
    parser.add_argument("--labels", type=Path, default=None,
                        help="JSON {gloss: name} map for human-readable sign names")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-frames", type=int, default=10)
    parser.add_argument("--auto", action="store_true",
                        help="keyless: segment signs automatically instead of holding SPACE")
    parser.add_argument("--enter", type=float, default=None, help="start-of-sign motion (auto)")
    parser.add_argument("--exit", type=float, default=None, help="end-of-sign stillness (auto)")
    args = parser.parse_args(argv)

    from .reject import load_calibration

    model, labels, cfg = load_checkpoint(args.checkpoint, args.device)
    names = json.loads(args.labels.read_text(encoding="utf-8")) if args.labels else {}
    calibration = load_calibration(args.checkpoint)
    temperature = calibration["temperature"] if calibration else 1.0
    threshold = calibration["threshold"] if calibration else 0.0
    print(f"{len(labels)} signs loaded; progress at {args.progress}"
          + (f"; calibrated (T={temperature}, reject below {threshold:.0%})"
             if calibration else "; no calibration"))

    if args.auto:
        from .segment import ENTER, EXIT

        return run_auto_session(model, labels, cfg, args.progress, names,
                                device=args.device, camera=args.camera,
                                temperature=temperature, reject_threshold=threshold,
                                enter=args.enter or ENTER, exit=args.exit or EXIT)

    return run_session(model, labels, cfg, args.progress, names,
                       device=args.device, camera=args.camera, min_frames=args.min_frames,
                       temperature=temperature, reject_threshold=threshold)


if __name__ == "__main__":
    raise SystemExit(main())
