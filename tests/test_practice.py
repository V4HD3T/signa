"""Tests for learning-mode pedagogy: grading, SM-2, scheduling, streaks, store.

No camera, no torch, no checkpoint -- the whole point of putting this logic in
`practice` is that its correctness can be pinned by hand. A wrong SM-2 transition
or an off-by-one streak is the kind of bug that silently degrades a learner's
experience without ever raising an error.
"""

import json
from datetime import date, timedelta

from signa import practice
from signa.practice import Card, Progress, grade, next_gloss, review


def ranked(*pairs):
    return list(pairs)


# --- Grading --------------------------------------------------------------

def test_confident_top1_is_mastery_quality_5():
    a = grade("merhaba", ranked(("merhaba", 0.95), ("gunaydin", 0.03)))
    assert a.verdict == practice.CORRECT and a.quality == 5


def test_unconfident_top1_is_a_pass_but_not_mastery():
    a = grade("merhaba", ranked(("merhaba", 0.55), ("gunaydin", 0.40)))
    assert a.verdict == practice.CORRECT and a.quality == 4


def test_target_in_topk_but_not_first_is_close():
    a = grade("merhaba", ranked(("gunaydin", 0.6), ("merhaba", 0.3), ("evet", 0.1)))
    assert a.verdict == practice.CLOSE and a.quality == 3


def test_target_absent_from_topk_is_missed():
    a = grade("merhaba", ranked(("gunaydin", 0.6), ("evet", 0.3), ("hayir", 0.1)))
    assert a.verdict == practice.MISSED and a.quality == 1


def test_grade_handles_an_empty_ranking():
    a = grade("merhaba", [])
    assert a.verdict == practice.MISSED and a.quality == 1


# --- SM-2 -----------------------------------------------------------------

def test_first_two_passes_step_one_then_six_days():
    today = date(2026, 7, 24)
    first = review(Card(), 4, today)
    assert first.reps == 1 and first.interval == 1
    second = review(first, 4, date.fromisoformat(first.due))
    assert second.reps == 2 and second.interval == 6


def test_third_pass_multiplies_by_ease():
    today = date(2026, 7, 24)
    card = Card(reps=2, ease=2.5, interval=6)
    nxt = review(card, 4, today)
    assert nxt.interval == round(6 * 2.5)  # 15


def test_a_lapse_resets_reps_and_is_due_tomorrow():
    today = date(2026, 7, 24)
    card = Card(reps=5, ease=2.6, interval=40, due="2026-08-01")
    lapsed = review(card, 1, today)
    assert lapsed.reps == 0 and lapsed.interval == 1
    assert lapsed.due == (today + timedelta(days=1)).isoformat()


def test_ease_never_falls_below_floor():
    today = date(2026, 7, 24)
    card = Card(ease=1.3, interval=1, reps=1)
    for _ in range(5):
        card = review(card, 3, today)  # repeated minimal passes push ease down
    assert card.ease >= 1.3


def test_review_counts_seen_and_correct():
    today = date(2026, 7, 24)
    card = review(Card(), 5, today)
    assert card.seen == 1 and card.correct == 1
    card = review(card, 1, today)  # a miss
    assert card.seen == 2 and card.correct == 1


# --- Scheduling -----------------------------------------------------------

def test_new_vocabulary_introduces_signs_one_at_a_time():
    today = date(2026, 7, 24)
    gloss, is_new = next_gloss(Progress(), ["a", "b", "c"], today)
    assert gloss == "a" and is_new is True


def test_due_reviews_beat_new_material():
    today = date(2026, 7, 24)
    p = Progress(cards={"b": Card(reps=1, interval=1, due=today.isoformat())})
    # "b" was seen and is due today; "a" and "c" are new. Backlog comes first.
    gloss, is_new = next_gloss(p, ["a", "b", "c"], today)
    assert gloss == "b" and is_new is False


def test_most_overdue_review_comes_first():
    today = date(2026, 7, 24)
    p = Progress(cards={
        "a": Card(reps=1, interval=1, due="2026-07-20"),
        "b": Card(reps=1, interval=1, due="2026-07-23"),
    })
    gloss, _ = next_gloss(p, ["a", "b"], today)
    assert gloss == "a"


def test_nothing_due_and_all_seen_returns_the_soonest_upcoming():
    today = date(2026, 7, 24)
    p = Progress(cards={
        "a": Card(reps=2, interval=6, due="2026-07-30"),
        "b": Card(reps=2, interval=6, due="2026-07-26"),
    })
    gloss, is_new = next_gloss(p, ["a", "b"], today)
    assert gloss == "b" and is_new is False


# --- Streak & daily goal --------------------------------------------------

def test_streak_counts_consecutive_days_ending_today():
    today = date(2026, 7, 24)
    p = Progress(days=["2026-07-22", "2026-07-23", "2026-07-24"])
    assert p.streak(today) == 3


def test_yesterday_still_counts_as_a_live_streak():
    today = date(2026, 7, 24)
    p = Progress(days=["2026-07-22", "2026-07-23"])  # practised through yesterday
    assert p.streak(today) == 2


def test_a_two_day_gap_breaks_the_streak():
    today = date(2026, 7, 24)
    p = Progress(days=["2026-07-20", "2026-07-21"])
    assert p.streak(today) == 0


def test_reviews_today_counts_signs_advanced_today():
    today = date(2026, 7, 24)
    p = Progress()
    p.record("a", 4, today)
    p.record("b", 5, today)
    p.record("a", 4, today)  # same sign again -- still one sign advanced
    assert p.reviews_today(today) == 2


# --- Persistence ----------------------------------------------------------

def test_progress_round_trips_through_disk(tmp_path):
    today = date(2026, 7, 24)
    p = Progress(daily_goal=20)
    p.record("merhaba", 5, today)
    p.record("gunaydin", 1, today)
    path = tmp_path / "progress.json"
    practice.save(p, path)

    loaded = practice.load(path)
    assert loaded.daily_goal == 20
    assert loaded.cards["merhaba"] == p.cards["merhaba"]
    assert loaded.cards["gunaydin"] == p.cards["gunaydin"]
    assert loaded.streak(today) == 1


def test_loading_a_missing_file_is_an_empty_progress(tmp_path):
    loaded = practice.load(tmp_path / "nope.json")
    assert loaded.cards == {} and loaded.streak(date(2026, 7, 24)) == 0


# --- Corrupt / outdated progress files ------------------------------------
#
# Practice history is the learner's own record with no second copy, so a
# malformed file must degrade rather than crash or silently reset.

def test_unparseable_json_starts_fresh_instead_of_crashing(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text("{not json at all", encoding="utf-8")
    assert practice.load(path).cards == {}


def test_a_card_with_an_unknown_field_still_loads(tmp_path):
    # A field written by a newer version must not lock the learner out.
    path = tmp_path / "progress.json"
    path.write_text(json.dumps({"cards": {
        "merhaba": {"reps": 2, "ease": 2.5, "interval": 6, "due": "2026-08-01",
                    "seen": 3, "correct": 2, "future_field": "???"}}}), encoding="utf-8")

    card = practice.load(path).cards["merhaba"]
    assert card.reps == 2 and card.interval == 6


def test_a_card_missing_fields_falls_back_to_defaults(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text(json.dumps({"cards": {"a": {"reps": 1}}}), encoding="utf-8")

    card = practice.load(path).cards["a"]
    assert card.reps == 1 and card.ease == 2.5  # default preserved


def test_one_unreadable_card_does_not_lose_the_others(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text(json.dumps({"cards": {
        "good": {"reps": 3, "interval": 15},
        "broken": "this should be an object",
    }}), encoding="utf-8")

    cards = practice.load(path).cards
    assert "good" in cards and "broken" not in cards


def test_a_nonsense_daily_goal_falls_back(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text(json.dumps({"daily_goal": -5}), encoding="utf-8")
    assert practice.load(path).daily_goal == 15
