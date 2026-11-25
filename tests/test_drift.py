"""Tests for the champion drift monitoring (population stability index)."""

import math

import pytest

from app.drift import (
    DRIFT_THRESHOLD,
    DriftTracker,
    feature_drift,
    population_stability_index,
)


def _reference(values):
    import pandas as pd

    from ml.train import build_training_profile

    profile = build_training_profile(pd.DataFrame({"x": values}), "v1")
    return profile["features"]["x"]


def test_population_stability_index_is_zero_for_an_identical_distribution():
    ref = _reference([float(value) for value in range(100)])
    samples = [float(value) for value in range(100)]

    assert feature_drift(ref, samples) == pytest.approx(0.0, abs=1e-9)


def test_feature_drift_stays_low_for_a_feature_with_repeated_values():
    # Like lights: ~80% zeros. Equal-mass bins used to give PSI ~1.5 here.
    import random

    rng = random.Random(0)
    population = [
        rng.choice([0.0] * 8 + [10.0, 20.0, 30.0, 70.0]) for _ in range(20000)
    ]
    ref = _reference(population)
    samples = rng.sample(population, 500)

    assert feature_drift(ref, samples) < 0.05


def test_population_stability_index_stays_below_the_threshold_on_the_same_distribution():
    import random

    rng = random.Random(0)
    ref = _reference([rng.gauss(0, 1) for _ in range(20000)])
    samples = [rng.gauss(0, 1) for _ in range(500)]

    assert feature_drift(ref, samples) < DRIFT_THRESHOLD


def test_population_stability_index_exceeds_the_threshold_for_a_shifted_distribution():
    ref = _reference([float(value) for value in range(100)])
    samples = [float(value) for value in range(100, 200)]

    assert feature_drift(ref, samples) > DRIFT_THRESHOLD


def test_population_stability_index_handles_empty_samples():
    assert population_stability_index([10.0], [0.5, 0.5], []) == 0.0


def test_feature_drift_uses_a_relative_shift_for_constant_features():
    constant_reference = _reference([7.0, 7.0, 7.0])

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


def test_load_drift_profile_feeds_the_tracker_with_real_training_profile(monkeypatch):
    import pandas as pd

    from app import main
    from ml.train import build_training_profile

    X = pd.DataFrame({"a": range(100), "b": [float(i % 7) for i in range(100)]})
    profile = build_training_profile(X, "v1")
    tracker = DriftTracker()
    monkeypatch.setattr(main, "drift_tracker", tracker)
    monkeypatch.setattr(main.holder, "state", (object(), {"resolved_version": "1"}))
    monkeypatch.setattr(main.holder, "training_profile", lambda: profile)

    main._load_drift_profile()

    assert tracker.enabled
    tracker.observe({"a": 1.0, "b": 1.0})
    assert set(tracker._buffers) == {"a", "b"}
    assert all(len(buf) == 1 for buf in tracker._buffers.values())


def test_load_drift_profile_is_a_no_op_without_a_loaded_champion(monkeypatch):
    from app import main

    tracker = DriftTracker()
    monkeypatch.setattr(main, "drift_tracker", tracker)
    monkeypatch.setattr(main.holder, "state", None)

    main._load_drift_profile()

    assert not tracker.enabled
