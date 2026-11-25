import random

import pandas as pd
import pytest

from app.drift import DRIFT_THRESHOLD, DriftTracker, population_stability_index
from ml.train import build_training_profile


def _psi(training, live):
    bins = build_training_profile(pd.DataFrame({"x": training}))["features"]["x"]
    return population_stability_index(bins["bin_edges"], bins["bin_shares"], live)


def test_psi_is_zero_on_the_training_data_itself():
    # Training and serving bin the same way.
    values = [float(v) for v in range(100)]
    assert _psi(values, values) == pytest.approx(0.0, abs=1e-9)


def test_psi_stays_low_for_a_sample_of_a_mostly_zero_feature():
    rng = random.Random(0)
    population = [
        rng.choice([0.0] * 8 + [10.0, 20.0, 30.0, 70.0]) for _ in range(20000)
    ]
    assert _psi(population, rng.sample(population, 500)) < 0.05


def test_psi_exceeds_the_threshold_for_a_shifted_distribution():
    values = [float(v) for v in range(100)]
    assert _psi(values, [v + 100 for v in values]) > DRIFT_THRESHOLD


def test_tracker_recomputes_only_once_the_window_is_full():
    tracker = DriftTracker(window_size=200, recompute_every=100)
    tracker.configure({"x": {"bin_edges": [10.0], "bin_shares": [0.5, 0.5]}})

    # observation 100 is on cadence but the window is not full
    assert all(tracker.observe({"x": 5.0}) is None for _ in range(199))
    assert set(tracker.observe({"x": 5.0})) == {"x"}
