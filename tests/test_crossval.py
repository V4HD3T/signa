"""Tests for the cross-validation orchestration's pure parts.

The training itself is train.run, already covered; what needs pinning is the fold
plan (each signer tested exactly once, never fewer signers than a three-way split
needs) and the aggregation (mean, spread, and the per-fold table the headline
number is built from). A wrong fold plan would quietly test some signer twice and
never test another, inflating or deflating the average with nobody the wiser.
"""

import pytest

from signa.crossval import aggregate, fold_plan


def test_fold_plan_tests_every_signer_exactly_once():
    signers = ["001", "002", "003", "004"]
    plan = fold_plan(signers)
    assert sorted(plan) == signers
    assert len(plan) == len(set(plan)) == len(signers)


def test_fold_plan_can_restrict_to_a_subset_of_test_signers():
    # Testing only two signers is fine when the pool has enough others to train
    # and validate on -- the restriction is on folds, not on training data.
    pool = ["001", "002", "003", "004", "005"]
    assert fold_plan(pool, ["003", "004"]) == ["003", "004"]


def test_fold_plan_rejects_a_test_signer_outside_the_pool():
    with pytest.raises(ValueError, match="not in the manifest"):
        fold_plan(["001", "002", "003"], ["009"])


def test_fold_plan_needs_enough_signers_for_a_three_way_split():
    with pytest.raises(ValueError, match="at least 3"):
        fold_plan(["001", "002"])


def test_aggregate_reports_mean_and_spread():
    summaries = [
        {"test_signers": ["001"], "test_top1": 0.90, "test_top5": 1.00, "test_clips": 60},
        {"test_signers": ["002"], "test_top1": 0.80, "test_top5": 0.98, "test_clips": 60},
        {"test_signers": ["003"], "test_top1": 0.94, "test_top5": 0.99, "test_clips": 60},
    ]
    result = aggregate(summaries)

    assert result["folds"] == 3
    assert result["top1"]["mean"] == pytest.approx((0.90 + 0.80 + 0.94) / 3)
    assert result["top1"]["min"] == 0.80
    assert result["top1"]["max"] == 0.94
    assert result["top1"]["std"] > 0


def test_aggregate_keeps_a_per_fold_row_per_signer():
    summaries = [
        {"test_signers": ["009"], "test_top1": 0.9, "test_top5": 1.0, "test_clips": 60},
        {"test_signers": ["010"], "test_top1": 0.8, "test_top5": 0.9, "test_clips": 55},
    ]
    result = aggregate(summaries)

    assert [f["test_signer"] for f in result["per_fold"]] == ["009", "010"]
    assert result["per_fold"][1]["test_clips"] == 55


def test_aggregate_single_fold_has_zero_spread():
    summaries = [{"test_signers": ["001"], "test_top1": 0.9, "test_top5": 1.0, "test_clips": 60}]
    assert aggregate(summaries)["top1"]["std"] == 0.0
