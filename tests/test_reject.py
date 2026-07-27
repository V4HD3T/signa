"""Tests for the reject option: calibration, the decision rule, risk-coverage.

Pure numpy -- logits are built by hand, so temperature fitting, the accept/reject
boundary, and the accuracy/coverage trade are all pinned without a model. A
silent bug here would make the model sound confident exactly when it should hedge.
"""

import numpy as np

from signa.reject import (
    choose_threshold,
    confidence,
    decide,
    entropy,
    fit_temperature,
    margin,
    negative_log_likelihood,
    risk_coverage,
    softmax,
)


def test_softmax_rows_are_distributions():
    logits = np.array([[2.0, 1.0, 0.1], [0.0, 0.0, 0.0]])
    probs = softmax(logits)
    np.testing.assert_allclose(probs.sum(axis=-1), [1.0, 1.0])
    np.testing.assert_allclose(probs[1], [1 / 3, 1 / 3, 1 / 3])


def test_temperature_flattens_without_reordering():
    logits = np.array([[3.0, 1.0, 0.0]])
    hot = softmax(logits, 1.0)
    cool = softmax(logits, 5.0)
    # argmax unchanged, but the winner is less dominant when cooled
    assert hot.argmax() == cool.argmax() == 0
    assert cool[0, 0] < hot[0, 0]
    assert entropy(cool) > entropy(hot)


def test_fit_temperature_cools_an_overconfident_model():
    # Logits far larger than the true separation: the model is overconfident, so
    # the fitted temperature should be well above 1 to calm it down.
    rng = np.random.default_rng(0)
    true = rng.integers(0, 5, size=400)
    logits = np.full((400, 5), -1.0)
    for i, label in enumerate(true):
        logits[i, label] = 8.0  # hugely peaked, but ~some are wrong below
    # flip 20% to be wrong so perfect confidence is unwarranted
    wrong = rng.random(400) < 0.2
    for i in np.where(wrong)[0]:
        logits[i] = -1.0
        logits[i, (true[i] + 1) % 5] = 8.0
    t = fit_temperature(logits, true)
    assert t > 1.0
    # NLL at the fitted T must be no worse than at T=1
    assert negative_log_likelihood(logits, true, t) <= negative_log_likelihood(logits, true, 1.0)


def test_fit_temperature_leaves_a_calibrated_model_near_one():
    # Build logits whose softmax already matches accuracy: T should stay ~1.
    rng = np.random.default_rng(1)
    true = rng.integers(0, 4, size=600)
    logits = rng.normal(0, 1.0, size=(600, 4))
    # make the argmax correct ~70% of the time at modest confidence
    for i, label in enumerate(true):
        if rng.random() < 0.7:
            logits[i, label] += 2.0
    t = fit_temperature(logits, true)
    assert 0.5 < t < 2.5


def test_confidence_and_margin():
    probs = np.array([[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]])
    np.testing.assert_allclose(confidence(probs), [0.7, 0.4])
    np.testing.assert_allclose(margin(probs), [0.5, 0.05])


def test_decide_rejects_below_threshold():
    probs = np.array([[0.9, 0.05, 0.05], [0.5, 0.3, 0.2], [0.34, 0.33, 0.33]])
    decisions = decide(probs, threshold=0.6)
    assert decisions[0] == 0      # confident -> accepted
    assert decisions[1] == -1     # below threshold -> rejected
    assert decisions[2] == -1


def test_decide_can_also_require_a_margin():
    # High top-1 but a near-tie runner-up: rejected when a margin is required.
    probs = np.array([[0.55, 0.44, 0.01]])
    assert decide(probs, threshold=0.5)[0] == 0
    assert decide(probs, threshold=0.5, min_margin=0.2)[0] == -1


def test_risk_coverage_trades_coverage_for_accuracy():
    # Two confident-correct, one unconfident-wrong.
    probs = np.array([[0.9, 0.1], [0.85, 0.15], [0.55, 0.45]])
    true = np.array([0, 0, 0])  # the 0.55 clip predicts 0 too -> actually correct
    # make the low-confidence one wrong:
    true = np.array([0, 0, 1])
    rows = {round(t, 2): (cov, acc) for t, cov, acc in
            risk_coverage(probs, true, [0.5, 0.6])}
    assert rows[0.5] == (1.0, 2 / 3)   # accept all: 2 of 3 right
    assert rows[0.6] == (2 / 3, 1.0)   # reject the wrong one: perfect on accepted


def test_choose_threshold_takes_the_most_coverage_meeting_target():
    probs = np.array([[0.95, 0.05], [0.8, 0.2], [0.6, 0.4], [0.55, 0.45]])
    true = np.array([0, 0, 0, 1])  # only the 0.55 clip is wrong
    # target 100%: the smallest threshold that excludes the wrong clip is > 0.55
    t = choose_threshold(probs, true, target_accuracy=1.0)
    assert 0.55 < t <= 0.6
    accepted = confidence(probs) >= t
    assert (probs.argmax(axis=-1)[accepted] == true[accepted]).all()


def test_a_malformed_sidecar_reads_as_no_calibration(tmp_path):
    # The demo and tutor both work uncalibrated, so a broken sidecar should cost
    # the reject option, not the session.
    from signa.reject import load_calibration, sidecar_path

    checkpoint = tmp_path / "best.pt"
    sidecar_path(checkpoint).write_text("{ broken", encoding="utf-8")
    assert load_calibration(checkpoint) is None


def test_a_sidecar_missing_its_fields_reads_as_no_calibration(tmp_path):
    import json as _json

    from signa.reject import load_calibration, sidecar_path

    checkpoint = tmp_path / "best.pt"
    sidecar_path(checkpoint).write_text(_json.dumps({"note": "hi"}), encoding="utf-8")
    assert load_calibration(checkpoint) is None


def test_a_valid_sidecar_loads(tmp_path):
    import json as _json

    from signa.reject import load_calibration, sidecar_path

    checkpoint = tmp_path / "best.pt"
    sidecar_path(checkpoint).write_text(
        _json.dumps({"temperature": 0.8, "threshold": 0.9}), encoding="utf-8")
    assert load_calibration(checkpoint)["temperature"] == 0.8


def test_choose_threshold_falls_back_to_best_accuracy_when_target_unreachable():
    # No threshold reaches 100% here except one that also keeps a wrong clip;
    # the fallback returns a real threshold rather than crashing.
    probs = np.array([[0.6, 0.4], [0.6, 0.4]])
    true = np.array([0, 1])  # identical confidence, one wrong -> best acc is 50%
    t = choose_threshold(probs, true, target_accuracy=0.99)
    assert 0.0 <= t <= 0.99
