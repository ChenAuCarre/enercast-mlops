import numpy as np
import pandas as pd
from prometheus_client import generate_latest

from app import drift, metrics
from ml.drift import psi, training_distribution


def test_psi_near_zero_on_same_data_and_high_on_shift():
    rng = np.random.default_rng(0)
    train = rng.normal(size=5000)
    reference = training_distribution(pd.DataFrame({"x": train}))["x"]
    assert sum(reference["proportions"]) == 1.0
    assert psi(reference, rng.normal(size=500)) < 0.1
    assert psi(reference, rng.normal(loc=2, size=500)) > 0.2


def test_constant_feature_does_not_break():
    reference = training_distribution(pd.DataFrame({"x": [0.0] * 100}))["x"]
    assert psi(reference, np.zeros(10)) == 0.0


def test_drift_recomputed_every_100_predictions(caplog):
    m = drift.DriftMonitor()
    m.reference = training_distribution(pd.DataFrame({"T1": np.arange(100.0)}))
    for _ in range(99):
        m.add({"T1": 500.0})
    assert "Drift on T1" not in caplog.text
    m.add({"T1": 500.0})
    assert metrics.DRIFT.labels("T1")._value.get() > drift.THRESHOLD
    assert "Drift on T1" in caplog.text
    assert len(m.window) == 100 and m.window.maxlen == 500


def test_set_reference_clears_stale_drift():
    m = drift.DriftMonitor()
    m.reference = training_distribution(pd.DataFrame({"T1": np.arange(100.0)}))
    for _ in range(100):
        m.add({"T1": 500.0})
    m.set_reference(None)
    assert not m.window and m.count == 0
    assert "feature_drift_psi{" not in generate_latest().decode()
