"""Tests for the ensemble combination logic.

Pure array math over hand-built probability matrices -- no checkpoints -- so soft
and hard voting, weighting, and the top-k scoring are pinned without loading a
model. The interesting cases are the ones where a combiner changes the answer: a
confident-but-wrong model outvoted by two unsure-but-right ones.
"""

import numpy as np

from signa.ensemble import hard_vote, soft_vote, topk_accuracy


def test_soft_vote_averages_probabilities():
    a = np.array([[0.8, 0.2]])
    b = np.array([[0.4, 0.6]])
    np.testing.assert_allclose(soft_vote([a, b]), [[0.6, 0.4]])


def test_soft_vote_outvotes_a_wrong_model_when_the_others_are_confident_enough():
    # Model 0 favours class 0 and is wrong; models 1 and 2 favour class 1 by
    # enough that the average tips to 1. (A *very* confident wrong model would
    # not be outvoted here -- averaging is not majority rule, which is why hard
    # voting exists as the separate combiner below.)
    wrong = np.array([[0.7, 0.3]])
    right1 = np.array([[0.35, 0.65]])
    right2 = np.array([[0.40, 0.60]])
    assert soft_vote([wrong, right1, right2]).argmax() == 1


def test_soft_vote_weights_shift_the_balance():
    a = np.array([[0.9, 0.1]])  # class 0
    b = np.array([[0.3, 0.7]])  # class 1
    # Heavily weighting the second model flips the decision.
    assert soft_vote([a, b], weights=[1.0, 1.0]).argmax() == 0
    assert soft_vote([a, b], weights=[1.0, 5.0]).argmax() == 1


def test_hard_vote_takes_the_majority_top1():
    # Two models say class 2, one says class 0 -> class 2 wins regardless of how
    # confident the dissenter was.
    m0 = np.array([[0.05, 0.05, 0.9]])
    m1 = np.array([[0.1, 0.2, 0.7]])
    m2 = np.array([[0.99, 0.005, 0.005]])
    assert hard_vote([m0, m1, m2]).argmax() == 2


def test_hard_vote_breaks_a_three_way_tie_by_confidence():
    # Each model votes a different class; the tie falls to the class with the
    # most summed probability mass across models.
    m0 = np.array([[0.6, 0.2, 0.2]])
    m1 = np.array([[0.2, 0.5, 0.3]])
    m2 = np.array([[0.1, 0.1, 0.8]])  # class 2 also has the highest total mass
    assert hard_vote([m0, m1, m2]).argmax() == 2


def test_topk_accuracy_counts_a_hit_within_k():
    scores = np.array([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]])
    true = np.array([2, 0])
    assert topk_accuracy(scores, true, 1) == 0.5   # only the second is top-1 right
    assert topk_accuracy(scores, true, 2) == 1.0   # both are within top-2


def test_topk_clamps_k_to_the_number_of_classes():
    scores = np.array([[0.7, 0.3]])
    assert topk_accuracy(scores, np.array([0]), 5) == 1.0
