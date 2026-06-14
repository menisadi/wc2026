import math

import numpy as np
import pytest

from wc2026.evaluate.metrics import (
    accuracy,
    brier_score,
    calibration_buckets,
    log_loss,
    rps,
)


def test_log_loss_perfect_forecast() -> None:
    probs = np.array([[0.999_999_999, 1e-15, 1e-15], [1e-15, 1e-15, 0.999_999_999]])
    outcomes = np.array([0, 2])
    assert log_loss(probs, outcomes) < 1e-6


def test_log_loss_uniform_equals_log3() -> None:
    probs = np.full((4, 3), 1.0 / 3.0)
    outcomes = np.array([0, 1, 2, 0])
    assert log_loss(probs, outcomes) == pytest.approx(math.log(3), abs=1e-9)


def test_brier_uniform_one_class_outcomes() -> None:
    # uniform forecast vs realised single class:
    #   per-row Brier = (1/3 - 1)^2 + (1/3)^2 + (1/3)^2 = 4/9 + 1/9 + 1/9 = 6/9 = 2/3
    probs = np.full((10, 3), 1.0 / 3.0)
    outcomes = np.zeros(10, dtype=int)
    assert brier_score(probs, outcomes) == pytest.approx(2.0 / 3.0, abs=1e-9)


def test_brier_perfect_is_zero() -> None:
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    outcomes = np.array([0, 1, 2])
    assert brier_score(probs, outcomes) == pytest.approx(0.0, abs=1e-12)


def test_rps_perfect_is_zero() -> None:
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    outcomes = np.array([0, 1, 2])
    assert rps(probs, outcomes) == pytest.approx(0.0, abs=1e-12)


def test_rps_penalizes_wrong_direction_more_than_draw_miss() -> None:
    # truth = home win.
    # forecast A: 0 home, 1 draw, 0 away   (one-step miss → adjacent)
    # forecast B: 0 home, 0 draw, 1 away   (two-step miss → far)
    outcomes = np.array([0])
    a = rps(np.array([[0.0, 1.0, 0.0]]), outcomes)
    b = rps(np.array([[0.0, 0.0, 1.0]]), outcomes)
    assert b > a


def test_accuracy() -> None:
    probs = np.array(
        [
            [0.6, 0.3, 0.1],  # picks home, truth = home → hit
            [0.2, 0.5, 0.3],  # picks draw, truth = away → miss
            [0.1, 0.2, 0.7],  # picks away, truth = away → hit
        ]
    )
    outcomes = np.array([0, 2, 2])
    assert accuracy(probs, outcomes) == pytest.approx(2.0 / 3.0)


def test_calibration_buckets_perfect() -> None:
    # Probabilities of 1.0 should land in the last bucket and observe 100%.
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    outcomes = np.array([0, 2])
    buckets = calibration_buckets(probs, outcomes, n_bins=10)
    last = [b for b in buckets if b["bin_high"] == pytest.approx(1.0)][0]
    assert last["mean_pred"] == pytest.approx(1.0)
    assert last["mean_obs"] == pytest.approx(1.0)
    # First bucket gets all the 0.0 predictions for non-realized classes
    first = [b for b in buckets if b["bin_low"] == pytest.approx(0.0)][0]
    assert first["mean_obs"] == pytest.approx(0.0)


def test_validation_shape_errors() -> None:
    with pytest.raises(ValueError):
        _ = log_loss(np.zeros((3, 4)), np.zeros(3, dtype=int))
    with pytest.raises(ValueError):
        _ = log_loss(np.zeros((3, 3)), np.zeros(2, dtype=int))
