"""The pedagogy behind learning mode, with no camera and no torch in sight.

Learning mode flips the recogniser around: instead of the user signing and the
model guessing, the app names a sign, the user performs it, and the model judges
whether they got it right. That judgement drives a spaced-repetition schedule --
signs you struggle with come back soon, signs you have mastered come back rarely
-- exactly the loop Lingua uses for vocabulary, ported to signing.

All of that is decision logic: grade an attempt, update a card, pick what to
practise next, count a streak. It lives here, pure and testable, so the webcam
loop in `learn.py` is only glue. The design mirrors the rest of the project:
anything whose correctness matters is exercised without hardware.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

# --- Grading --------------------------------------------------------------

# A learner's attempt lands in one of three buckets, from the model's ranked
# output. "close" -- the right sign was in the model's top-k but not its first
# guess -- is deliberately its own verdict: a beginner whose sign is recognisable
# but not crisp is in a different place from one whose sign was not read at all,
# and the schedule should treat them differently.
CORRECT = "correct"
CLOSE = "close"
MISSED = "missed"

CLOSE_K = 3
CONFIDENT = 0.80  # top-1 probability above which a correct sign counts as mastered


@dataclass(frozen=True)
class Attempt:
    verdict: str
    target: str
    predicted: str
    confidence: float
    quality: int  # SM-2 grade, 0-5


def grade(target: str, ranked: list[tuple[str, float]], *,
          close_k: int = CLOSE_K, confident: float = CONFIDENT) -> Attempt:
    """Turn a target and the model's ranked (label, prob) list into an Attempt.

    The quality is the SM-2 scale (0-5), which is what the scheduler consumes:
    a confident hit is 5, a plain hit 4, a top-k near-miss 3 (still a pass, so
    the interval grows), and a genuine miss 1 (a fail, so the sign is relearned).
    """
    if not ranked:
        return Attempt(MISSED, target, "", 0.0, 1)

    labels = [label for label, _ in ranked]
    top_label, top_prob = ranked[0]
    position = labels.index(target) if target in labels else None

    if position == 0:
        quality = 5 if top_prob >= confident else 4
        return Attempt(CORRECT, target, top_label, top_prob, quality)
    if position is not None and position < close_k:
        return Attempt(CLOSE, target, top_label, top_prob, 3)
    return Attempt(MISSED, target, top_label, top_prob, 1)


# --- Spaced repetition (SM-2) --------------------------------------------


@dataclass(frozen=True)
class Card:
    """One sign's schedule state. `due` is a plain date in the learner's own
    timezone -- days are counted locally, the lesson Lingua learned the hard way
    when streaks rolled over at the wrong hour for everyone outside UTC."""

    reps: int = 0
    ease: float = 2.5
    interval: int = 0  # days
    due: str = ""  # ISO date; empty means "never scheduled, due now"
    seen: int = 0
    correct: int = 0

    def is_due(self, today: date) -> bool:
        return not self.due or date.fromisoformat(self.due) <= today


def review(card: Card, quality: int, today: date) -> Card:
    """Advance a card by one review under SM-2.

    A quality below 3 is a lapse: repetitions reset and the sign is due again
    tomorrow, however well it had been going. Above that the interval steps
    1 -> 6 -> interval*ease, and the ease factor drifts with performance but
    never below 1.3, so a chronically hard sign keeps coming back rather than
    disappearing into a months-long interval.
    """
    quality = max(0, min(5, quality))
    seen = card.seen + 1
    correct = card.correct + (1 if quality >= 3 else 0)

    if quality < 3:
        reps, interval = 0, 1
    else:
        reps = card.reps + 1
        if reps == 1:
            interval = 1
        elif reps == 2:
            interval = 6
        else:
            interval = round(card.interval * card.ease)

    ease = card.ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    ease = max(1.3, ease)
    due = (today + timedelta(days=interval)).isoformat()
    return Card(reps=reps, ease=round(ease, 3), interval=interval, due=due,
                seen=seen, correct=correct)


# --- Progress store -------------------------------------------------------


@dataclass
class Progress:
    """Everything that persists between sessions: per-sign schedule, the set of
    days practised (for the streak), and the daily review goal."""

    cards: dict[str, Card] = field(default_factory=dict)
    days: list[str] = field(default_factory=list)  # ISO dates, sorted, unique
    daily_goal: int = 15

    def card(self, gloss: str) -> Card:
        return self.cards.get(gloss, Card())

    def record(self, gloss: str, quality: int, today: date) -> Card:
        updated = review(self.card(gloss), quality, today)
        self.cards[gloss] = updated
        stamp = today.isoformat()
        if stamp not in self.days:
            self.days.append(stamp)
            self.days.sort()
        return updated

    def reviews_today(self, today: date) -> int:
        # review() always sets due = today + interval, so due - interval
        # reconstructs the last-review date exactly; no per-attempt log needed.
        # This counts signs advanced today, which is what a daily goal should
        # mean -- not raw attempts, so retrying one sign is not fifteen reviews.
        stamp = today.isoformat()
        return sum(1 for c in self.cards.values() if _reviewed_on(c, today) == stamp)

    def streak(self, today: date) -> int:
        """Consecutive days practised, ending today or yesterday.

        Yesterday still counts as an unbroken streak the learner can save by
        practising today; a two-day gap ends it. Counting in local dates, not
        UTC timestamps, is deliberate."""
        practised = set(self.days)
        if not practised:
            return 0
        cursor = today if today.isoformat() in practised else today - timedelta(days=1)
        if cursor.isoformat() not in practised:
            return 0
        count = 0
        while cursor.isoformat() in practised:
            count += 1
            cursor -= timedelta(days=1)
        return count


def _reviewed_on(card: Card, today: date) -> str:
    """The date a card was last reviewed, inferred from its due date and
    interval. Kept explicit so reviews_today has one place to be wrong."""
    if not card.due or card.interval <= 0:
        return ""
    return (date.fromisoformat(card.due) - timedelta(days=card.interval)).isoformat()


def next_gloss(progress: Progress, vocabulary: list[str], today: date):
    """Choose what to practise next; returns (gloss, is_new).

    Due reviews first, most overdue then least easy, so struggling signs and
    backlog take priority. When nothing is due, introduce an unseen sign rather
    than drill ahead -- new material is more useful than early repetition. When
    the whole vocabulary is scheduled and nothing is due, return the soonest
    upcoming card so a keen learner is never blocked.
    """
    # A *seen* sign that is due again is backlog, and backlog clears before new
    # material -- introducing signs faster than you consolidate them is how a
    # deck becomes unreviewable. Unseen glosses (not yet in the store) are new
    # material, taken only when nothing is due.
    due = [(g, progress.cards[g]) for g in vocabulary
           if g in progress.cards and progress.cards[g].is_due(today)]
    if due:
        due.sort(key=lambda gc: (gc[1].due or "", gc[1].ease))  # most overdue, then hardest
        return due[0][0], False

    unseen = [g for g in vocabulary if g not in progress.cards]
    if unseen:
        return unseen[0], True

    upcoming = sorted(vocabulary, key=lambda g: progress.card(g).due)
    return upcoming[0], False


# --- Persistence ----------------------------------------------------------


def load(path: str | Path) -> Progress:
    file = Path(path)
    if not file.exists():
        return Progress()
    raw = json.loads(file.read_text(encoding="utf-8"))
    cards = {g: Card(**c) for g, c in raw.get("cards", {}).items()}
    return Progress(cards=cards,
                    days=raw.get("days", []),
                    daily_goal=raw.get("daily_goal", 15))


def save(progress: Progress, path: str | Path) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cards": {g: asdict(c) for g, c in progress.cards.items()},
        "days": progress.days,
        "daily_goal": progress.daily_goal,
    }
    file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
