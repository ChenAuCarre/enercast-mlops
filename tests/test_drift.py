"""Tests for the champion drift monitoring (population stability index)."""

import math

import pytest

from app.drift import (
    DRIFT_THRESHOLD,
    DriftTracker,
    population_stability_index,
)


def _reference(values):
    import pandas as pd

    from ml.train import build_training_profile

    profile = build_training_profile(pd.DataFrame({"x": values}), "v1")
    return profile["features"]["x"]


def _psi(ref, samples):
    return population_stability_index(ref["bin_edges"], ref["bin_shares"], samples)


def test_population_stability_index_is_zero_for_an_identical_distribution():
    ref = _reference([float(value) for value in range(100)])
    samples = [float(value) for value in range(100)]

    assert _psi(ref, samples) == pytest.approx(0.0, abs=1e-9)


def test_psi_stays_low_for_a_feature_with_repeated_values():
    # Like lights: ~80% zeros. Equal-mass bins used to give PSI ~1.5 here.
    import random

    rng = random.Random(0)
    population = [
        rng.choice([0.0] * 8 + [10.0, 20.0, 30.0, 70.0]) for _ in range(20000)
    ]
    ref = _reference(population)
    samples = rng.sample(population, 500)

    assert _psi(ref, samples) < 0.05


def test_population_stability_index_stays_below_the_threshold_on_the_same_distribution():
    import random

    rng = random.Random(0)
    ref = _reference([rng.gauss(0, 1) for _ in range(20000)])
    samples = [rng.gauss(0, 1) for _ in range(500)]

    assert _psi(ref, samples) < DRIFT_THRESHOLD


def test_population_stability_index_exceeds_the_threshold_for_a_shifted_distribution():
    ref = _reference([float(value) for value in range(100)])
    samples = [float(value) for value in range(100, 200)]

    assert _psi(ref, samples) > DRIFT_THRESHOLD


def test_population_stability_index_handles_empty_samples():
    assert population_stability_index([10.0], [0.5, 0.5], []) == 0.0


def test_drift_tracker_recomputes_drift_once_the_window_is_full():
    tracker = DriftTracker(
        reference={
            "lights": {"bin_edges": [0.0, 10.0, 20.0], "bin_shares": [0.25] * 4}
        },
        window_size=200,
        recompute_every=100,
    )

    for _ in range(199):  # observation 100 is on cadence but the window is not full
        assert tracker.observe({"lights": 5.0}) is None

    drift = tracker.observe({"lights": 5.0})  # 200th observation -> recompute
    assert set(drift) == {"lights"}
    assert math.isfinite(drift["lights"])


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

    main._load_drift_reference()

    tracker.observe({"a": 1.0, "b": 1.0})
    assert set(tracker._buffers) == {"a", "b"}
    assert all(len(buf) == 1 for buf in tracker._buffers.values())


def test_population_stability_index_is_zero_for_a_constant_feature():
    assert _psi(_reference([7.0, 7.0, 7.0]), [9.0, 9.0]) == 0.0
