import numpy as np
import pandas as pd

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
