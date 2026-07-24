"""Tests for the results-table formatting.

The scanning is I/O; what needs pinning is the pure part -- turning a summary
dict into a row that tolerates missing fields, sorting best-first, and rendering.
A row that silently guessed a missing flag would put a wrong claim in the README,
which is the one place these numbers are read as fact.
"""

from signa.results import (
    DASH,
    format_markdown,
    loso_row,
    sort_rows,
    summary_row,
)


def test_summary_row_reads_the_fields():
    row = summary_row({
        "_run": "lsa64-full-tcn", "model": "tcn", "use_pose": True, "augment": True,
        "glosses": 64, "test_top1": 0.982, "test_top5": 1.0, "test_clips": 625,
        "minutes": 8.7,
    })
    assert row["run"] == "lsa64-full-tcn"
    assert row["pose"] == "yes" and row["aug"] == "yes"
    assert row["top1"] == 0.982


def test_summary_row_marks_missing_flags_rather_than_guessing():
    # A run from before use_pose/augment were recorded must not read as a claim.
    row = summary_row({"_run": "old", "model": "bilstm", "test_top1": 0.88})
    assert row["pose"] == DASH and row["aug"] == DASH
    assert row["top5"] is None


def test_pose_ablation_off_shows_no():
    assert summary_row({"use_pose": False, "augment": False})["pose"] == "no"
    assert summary_row({"use_pose": False, "augment": False})["aug"] == "no"


def test_sort_puts_best_top1_first_and_missing_last():
    rows = [
        {"top1": 0.88}, {"top1": 0.982}, {"top1": None}, {"top1": 0.971},
    ]
    ordered = [r["top1"] for r in sort_rows(rows)]
    assert ordered == [0.982, 0.971, 0.88, None]


def test_markdown_has_a_header_and_one_row_per_entry():
    rows = [summary_row({"_run": "a", "model": "tcn", "test_top1": 0.98, "test_top5": 1.0})]
    table = format_markdown(rows)
    lines = table.splitlines()
    assert lines[0].startswith("| run |")
    assert lines[1].startswith("| ---")
    assert "98.0%" in lines[2] and "tcn" in lines[2]


def test_loso_row_carries_mean_and_spread():
    row = loso_row({
        "_run": "lsa64-tcn-loso", "model": "tcn", "folds": 10,
        "top1": {"mean": 0.977, "std": 0.015}, "top5": {"mean": 0.998, "std": 0.002},
    })
    assert row["folds"] == 10
    assert row["top1"] == 0.977 and row["top1_std"] == 0.015
