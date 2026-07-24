"""Tests for the report's analysis functions.

These are pure array math -- no checkpoint, no torch -- so the confusion matrix,
the per-class accuracy and the detection binning can be pinned by hand. The
model-loading path is exercised end to end by running the tool on a real
checkpoint; these lock down the arithmetic underneath it.
"""

import numpy as np

from signa.report import (
    confusion,
    detection_bins,
    most_confused,
    per_class_accuracy,
)


def test_confusion_counts_true_predicted_pairs():
    true = np.array([0, 0, 1, 2, 2])
    pred = np.array([0, 1, 1, 2, 0])
    matrix = confusion(true, pred, 3)

    assert matrix[0, 0] == 1 and matrix[0, 1] == 1  # one 0 right, one called a 1
    assert matrix[1, 1] == 1
    assert matrix[2, 2] == 1 and matrix[2, 0] == 1
    assert matrix.sum() == len(true)


def test_per_class_accuracy_is_the_row_normalised_diagonal():
    matrix = np.array([[8, 2], [1, 9]])
    acc = per_class_accuracy(matrix)

    np.testing.assert_allclose(acc, [0.8, 0.9])


def test_per_class_accuracy_is_nan_for_an_absent_class():
    # A class with no test clips must not read as 0% -- that would drag the
    # "worst classes" list toward classes that were simply never tested.
    matrix = np.array([[5, 0], [0, 0]])
    acc = per_class_accuracy(matrix)

    assert acc[0] == 1.0
    assert np.isnan(acc[1])


def test_most_confused_ranks_off_diagonal_mass():
    labels = ["a", "b", "c"]
    matrix = np.array([[10, 4, 0], [1, 10, 0], [0, 0, 10]])
    pairs = most_confused(matrix, labels, top=2)

    assert pairs[0] == (4, "a", "b")
    assert pairs[1] == (1, "b", "a")


def test_most_confused_ignores_the_diagonal():
    labels = ["a", "b"]
    matrix = np.array([[99, 0], [0, 99]])  # perfect classifier
    assert most_confused(matrix, labels, top=5) == []


def test_detection_bins_partition_every_clip_once():
    detection = np.array([0.1, 0.4, 0.6, 0.8, 0.95, 1.0])
    correct = np.array([0, 0, 1, 1, 1, 1], dtype=bool)
    rows = detection_bins(detection, correct, [0.0, 0.5, 0.7, 0.9, 1.0])

    assert sum(n for _, _, n, _ in rows) == len(detection)
    # the top bin is closed on the right so a rate of exactly 1.0 is counted
    assert rows[-1][2] >= 1


def test_detection_bins_report_accuracy_within_the_band():
    detection = np.array([0.2, 0.3, 0.95, 0.99])
    correct = np.array([False, False, True, True])
    rows = detection_bins(detection, correct, [0.0, 0.5, 0.9, 1.0])

    low_band = next(r for r in rows if r[0] == 0.0)
    high_band = next(r for r in rows if r[0] == 0.9)
    assert low_band[3] == 0.0   # both low-detection clips wrong
    assert high_band[3] == 1.0  # both high-detection clips right
