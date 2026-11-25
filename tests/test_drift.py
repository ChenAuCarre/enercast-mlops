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
    assert feature_drift({"bin_edges": [5.0], "bin_shares": [0.5, 0.5]}, []) == 0.0


def test_drift_tracker_is_disabled_without_a_reference():
    tracker = DriftTracker()

    assert tracker.enabled is False
    assert tracker.current() == {}


def test_drift_tracker_recomputes_drift_once_the_window_is_full():
    tracker = DriftTracker(
        reference={
            "lights": {"bin_edges": [0.0, 10.0, 20.0], "bin_shares": [0.25] * 4}
        },
        window_size=200,
        recompute_every=100,
    )

    assert tracker.enabled is True
    for _ in range(199):  # observation 100 is on cadence but the window is not full
        assert tracker.observe({"lights": 5.0}) is None
    assert tracker.current() == {}

    drift = tracker.observe({"lights": 5.0})  # 200th observation -> recompute
    assert set(drift) == {"lights"}
    assert tracker.current() == drift
    assert math.isfinite(tracker.current()["lights"])


def test_drift_tracker_keeps_a_window_of_the_last_predictions():
    tracker = DriftTracker(
        reference={
            "lights": {"bin_edges": [0.0, 10.0, 20.0], "bin_shares": [0.25] * 4}
        },
        window_size=3,
        recompute_every=2,
    )

    tracker.configure(
        {"lights": {"bin_edges": [0.0, 10.0, 20.0], "bin_shares": [0.25] * 4}}
    )

    for value in (1.0, 2.0, 3.0, 4.0, 5.0):
        tracker.observe({"lights": value})

    # recompute_every=2, window 3 -> recomputes on observation 4 only.
    assert set(tracker.current()) == {"lights"}
    assert len(tracker.current()) == 1


def test_load_drift_reference_feeds_the_tracker_with_real_training_profile(monkeypatch):
    import pandas as pd

    from app import main
    from ml.train import build_training_profile

    X = pd.DataFrame({"a": range(100), "b": [float(i % 7) for i in range(100)]})
    profile = build_training_profile(X, "v1")
    tracker = DriftTracker()
    monkeypatch.setattr(main, "drift_tracker", tracker)
    monkeypatch.setattr(main.holder, "state", (object(), {"resolved_version": "1"}))
    monkeypatch.setattr(main.holder, "training_profile", lambda: profile)
    monkeypatch.setattr(main, "_drift_version", None)

    main._load_drift_reference()

    assert tracker.enabled
    tracker.observe({"a": 1.0, "b": 1.0})
    assert set(tracker._buffers) == {"a", "b"}
    assert all(len(buf) == 1 for buf in tracker._buffers.values())
