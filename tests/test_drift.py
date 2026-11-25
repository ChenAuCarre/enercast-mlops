"""Tests for the champion drift monitoring (population stability index)."""

import math

from app.drift import (
    DRIFT_THRESHOLD,
    DriftTracker,
    feature_drift,
    population_stability_index,
)


def test_population_stability_index_is_zero_for_an_identical_distribution():
    deciles = [float(value) for value in range(0, 101, 10)]
    samples = [float(value) for value in range(0, 100, 10)]

    assert population_stability_index(deciles, samples) == 0.0


def test_population_stability_index_exceeds_the_threshold_for_a_shifted_distribution():
    deciles = [float(value) for value in range(0, 101, 10)]
    samples = [float(value) for value in range(100, 200, 10)]

    assert population_stability_index(deciles, samples) > DRIFT_THRESHOLD


def test_population_stability_index_handles_empty_inputs():
    assert population_stability_index([0.0, 10.0], []) == 0.0
    assert population_stability_index([], [0.0, 10.0]) == 0.0
    assert population_stability_index([0.0], [0.0, 10.0]) == 0.0


def test_feature_drift_uses_a_relative_shift_for_constant_features():
    constant_reference = {"deciles": [7.0, 7.0, 7.0, 7.0, 7.0], "mean": 7.0}

    assert feature_drift(constant_reference, [7.0, 7.0, 7.0]) == 0.0
    assert feature_drift(constant_reference, [9.0, 9.0, 9.0]) > DRIFT_THRESHOLD


def test_feature_drift_returns_zero_without_samples():
    assert feature_drift({"deciles": [0.0, 10.0], "mean": 5.0}, []) == 0.0


def test_drift_tracker_is_disabled_without_a_reference():
    tracker = DriftTracker()

    assert tracker.enabled is False
    assert tracker.current() == {}


def test_drift_tracker_recomputes_drift_every_reset_every_observations():
    tracker = DriftTracker(
        reference={"lights": {"deciles": [0.0, 10.0, 20.0], "mean": 10.0}},
        reset_every=100,
    )

    assert tracker.enabled is True
    tracker.observe({"lights": 5.0})
    assert tracker.current() == {}

    for _ in range(99):
        tracker.observe({"lights": 5.0})

    tracker.observe({"lights": 5.0})  # 100th observation -> recompute
    assert set(tracker.current()) == {"lights"}
    assert math.isfinite(tracker.current()["lights"])


def test_drift_tracker_keeps_a_window_of_the_last_predictions():
    tracker = DriftTracker(
        reference={"lights": {"deciles": [0.0, 10.0, 20.0], "mean": 10.0}},
        window_size=3,
        reset_every=2,
    )

    tracker.configure({"lights": {"deciles": [0.0, 10.0, 20.0], "mean": 10.0}})

    for value in (1.0, 2.0, 3.0, 4.0, 5.0):
        tracker.observe({"lights": value})

    # reset_every=2 -> recomputes on observations 2 and 4, with a window of 3.
    assert set(tracker.current()) == {"lights"}
    assert len(tracker.current()) == 1
