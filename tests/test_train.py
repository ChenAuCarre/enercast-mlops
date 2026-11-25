"""Tests for estimator instantiation in ``ml.train``.

The sklearn estimators are mutable: each run must get a dedicated instance,
provided by :func:`build_estimator`.
"""

import subprocess
from types import SimpleNamespace

import pandas as pd
import pytest

from ml.drift import DISTRIBUTION_ARTIFACT
from ml.train import (
    build_estimator,
    get_git_commit,
    is_git_dirty,
    run_experiment,
)


class FakeRun:
    def __init__(self):
        self.info = SimpleNamespace(run_id="run-1")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeMlflow:
    def __init__(self):
        self.params = {}
        self.metrics = {}
        self.tags = {}
        self.artifacts = []
        self.sklearn = SimpleNamespace(log_model=self._log_model)

    def start_run(self, run_name=None):
        return FakeRun()

    def set_experiment(self, name):
        pass

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics):
        self.metrics.update(metrics)

    def set_tags(self, tags):
        self.tags.update(tags)

    def log_artifact(self, path):
        self.artifacts.append(path)

    def log_dict(self, dictionary, artifact_file):
        self.artifacts.append(artifact_file)

    @staticmethod
    def _log_model(**kwargs):
        return SimpleNamespace(model_id="model-1")


class FakeSkModel:
    def fit(self, X, y):
        pass

    def predict(self, X):
        return [0.0 for _ in range(len(X))]


def training_data():
    return {
        "dataset_version": "v1",
        "train_dvc_hash": "hash-v1",
        "validation_fraction": 0.25,
        "X_train": pd.DataFrame({"feature": [1.0, 2.0, 3.0]}),
        "y_train": pd.Series([1.0, 2.0, 3.0]),
        "X_validation": pd.DataFrame({"feature": [4.0, 5.0]}),
        "y_validation": pd.Series([4.0, 5.0]),
        "train_period": ("2016-01-01 00:00:00", "2016-01-02 00:00:00"),
        "validation_period": ("2016-01-02 00:00:00", "2016-01-03 00:00:00"),
    }


def test_build_estimator_returns_a_fresh_linear_regression_instance():
    first = build_estimator("LinearRegression")
    second = build_estimator("LinearRegression")

    assert first is not second
    first.fit([[0.0]], [0.0])
    assert not hasattr(second, "coef_")


def test_build_estimator_returns_a_fresh_hist_gradient_boosting_instance():
    params = {"max_iter": 10, "random_state": 0}
    first = build_estimator("HistGradientBoostingRegressor", params)
    second = build_estimator("HistGradientBoostingRegressor", params)

    assert first is not second


def test_build_estimator_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown estimator type"):
        build_estimator("RandomForestRegressor")


def test_run_experiment_instantiates_a_fresh_estimator_per_run(monkeypatch):
    fake_mlflow = FakeMlflow()
    monkeypatch.setattr("ml.train.mlflow", fake_mlflow)
    monkeypatch.setattr("ml.train.get_git_commit", lambda: "abc123")
    monkeypatch.setattr("ml.train.is_git_dirty", lambda: "true")

    built = []

    def fake_build_estimator(model_type, params=None):
        estimator = FakeSkModel()
        built.append((model_type, params, estimator))
        return estimator

    monkeypatch.setattr("ml.train.build_estimator", fake_build_estimator)

    result = run_experiment(
        "run-x",
        {
            "role": "candidate",
            "model_type": "HistGradientBoostingRegressor",
            "params": {"max_iter": 300, "random_state": 42},
        },
        training_data(),
    )

    assert len(built) == 1
    assert built[0][0] == "HistGradientBoostingRegressor"
    assert result["run_id"] == "run-1"
    assert result["model_id"] == "model-1"
    assert set(result["metrics"]) == {"mae", "rmse", "r2"}
    assert fake_mlflow.tags["git_commit"] == "abc123"
    assert fake_mlflow.tags["git_dirty"] == "true"
    assert fake_mlflow.params["git_dirty"] == "true"
    assert DISTRIBUTION_ARTIFACT in fake_mlflow.artifacts


def test_get_git_commit_falls_back_to_unknown(monkeypatch):
    def fake_check_output(cmd, **kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr("ml.train.subprocess.check_output", fake_check_output)

    assert get_git_commit() == "unknown"


def test_is_git_dirty_returns_false_when_clean(monkeypatch):
    monkeypatch.setattr("ml.train.subprocess.check_output", lambda cmd, **kwargs: "")

    assert is_git_dirty() == "false"


def test_is_git_dirty_returns_true_when_dirty(monkeypatch):
    monkeypatch.setattr(
        "ml.train.subprocess.check_output",
        lambda cmd, **kwargs: " M ml/train.py\n",
    )

    assert is_git_dirty() == "true"


def test_is_git_dirty_returns_unknown_without_git(monkeypatch):
    def fake_check_output(cmd, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr("ml.train.subprocess.check_output", fake_check_output)

    assert is_git_dirty() == "unknown"
