"""Tests du chemin de promotion.

Deux garanties sont vérifiées : le test final n'est jamais lu pendant
l'entraînement/la promotion, et ``run_v1`` ne peut pas écraser un champion
existant sans autorisation explicite.
"""

from types import SimpleNamespace

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from ml.pipeline import (
    load_data,
    run_final_evaluation,
    run_v1,
    run_v2,
)
from ml.pipeline import (
    main as pipeline_main,
)
from ml.preprocess import FEATURE_COLUMNS, prepare_training_data
from ml.registry import NoChampionRegistered
from ml.splits import split_train_validation
from ml.train import run_experiment


class FakeRun:
    def __init__(self):
        self.info = SimpleNamespace(run_id="run-1")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeMlflow:
    """Client MLflow minimal : aucun appel réseau, aucun fichier mlruns."""

    def __init__(self):
        self.params = {}
        self.metrics = {}
        self.tags = {}
        self.artifacts = []
        self.sklearn = SimpleNamespace(log_model=self._log_model)

    def start_run(self, run_name=None):
        return FakeRun()

    def set_experiment(self, name):
        self.experiment_name = name

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics):
        self.metrics.update(metrics)

    def set_tags(self, tags):
        self.tags.update(tags)

    def log_artifact(self, path):
        self.artifacts.append(path)

    @staticmethod
    def _log_model(**kwargs):
        return SimpleNamespace(model_id="model-1")


def synthetic_pool(rows=40):
    """Petit jeu brut contenant toutes les colonnes attendues par le modèle."""
    dates = pd.date_range("2016-01-01", periods=rows, freq="10min")
    data = {
        "date": dates.astype(str),
        "Appliances": [50.0 + (index % 300) for index in range(rows)],
    }
    for column in FEATURE_COLUMNS:
        data[column] = [1.0 + (index % 5) for index in range(rows)]
    return pd.DataFrame(data)


def promotion_data():
    """Données d'entraînement/validation uniquement, sans test final."""
    pool = synthetic_pool()
    train_df, validation_df = split_train_validation(
        pool,
        validation_fraction=0.25,
    )

    X_train, y_train = prepare_training_data(train_df)
    X_validation, y_validation = prepare_training_data(validation_df)

    return {
        "dataset_version": "v1",
        "train_dvc_hash": "hash-v1",
        "validation_fraction": 0.25,
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "train_period": (
            str(pd.to_datetime(train_df["date"]).min()),
            str(pd.to_datetime(train_df["date"]).max()),
        ),
        "validation_period": (
            str(pd.to_datetime(validation_df["date"]).min()),
            str(pd.to_datetime(validation_df["date"]).max()),
        ),
    }


def forbid_final_test_read(*args, **kwargs):
    raise AssertionError(
        "Le test final ne doit pas être lu par le chemin de promotion."
    )


def forbid_promotion(*args, **kwargs):
    raise AssertionError("L'évaluation finale ne doit jamais promouvoir de modèle.")


def forbid_training(*args, **kwargs):
    raise AssertionError("run_v1 doit s'arrêter avant tout chargement ou entraînement.")


def no_champion():
    raise NoChampionRegistered("No champion model is registered yet.")


def test_load_data_never_reads_final_test(monkeypatch):
    monkeypatch.setattr("ml.pipeline.load_final_test", forbid_final_test_read)
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: synthetic_pool())

    data = load_data("v1")

    assert "X_test" not in data
    assert "y_test" not in data
    assert len(data["X_train"]) == 34
    assert len(data["X_validation"]) == 6


def test_run_experiment_does_not_require_final_test(monkeypatch):
    fake_mlflow = FakeMlflow()
    monkeypatch.setattr("ml.train.mlflow", fake_mlflow)
    monkeypatch.setattr("ml.train.get_git_commit", lambda: "abc123")

    data = promotion_data()
    config = {
        "role": "baseline",
        "model_type": "LinearRegression",
        "estimator": LinearRegression(),
        "params": {},
    }

    result = run_experiment("test-run", config, data)

    assert "validation_mae" in fake_mlflow.metrics
    assert "final_test_mae" not in fake_mlflow.metrics
    assert "final_test_metrics" not in result
    assert "final_test_mae" not in result["metrics"]


def test_run_v1_promotion_path_never_reads_final_test(monkeypatch):
    monkeypatch.setattr("ml.pipeline.get_champion", no_champion)
    monkeypatch.setattr("ml.pipeline.load_final_test", forbid_final_test_read)
    monkeypatch.setattr("ml.pipeline.load_data", lambda version: promotion_data())
    monkeypatch.setattr(
        "ml.pipeline.run_experiment",
        lambda *args, **kwargs: {
            "model_id": "model-1",
            "run_id": "run-1",
            "metrics": {"mae": 40.0, "rmse": 50.0, "r2": 0.5},
        },
    )
    monkeypatch.setattr(
        "ml.pipeline.evaluate_initial_models",
        lambda results: {"promote": True, "reason": "ok"},
    )
    monkeypatch.setattr(
        "ml.pipeline.register_model",
        lambda model_id: SimpleNamespace(version="1"),
    )
    monkeypatch.setattr(
        "ml.pipeline.promote_model",
        lambda version: SimpleNamespace(version="1"),
    )
    monkeypatch.setattr("ml.pipeline.mlflow", FakeMlflow())

    run_v1()


def test_run_v1_refuses_to_replace_an_existing_champion(monkeypatch):
    monkeypatch.setattr(
        "ml.pipeline.get_champion",
        lambda: SimpleNamespace(version="7"),
    )
    monkeypatch.setattr("ml.pipeline.load_data", forbid_training)
    monkeypatch.setattr("ml.pipeline.run_experiment", forbid_training)

    with pytest.raises(
        RuntimeError,
        match="Refusing to run initial training",
    ) as info:
        run_v1()

    assert "version 7" in str(info.value)
    assert "--force-initial-training" in str(info.value)


def test_run_v1_force_allows_replacing_an_existing_champion(monkeypatch):
    monkeypatch.setattr(
        "ml.pipeline.get_champion",
        lambda: SimpleNamespace(version="7"),
    )
    monkeypatch.setattr("ml.pipeline.load_data", lambda version: promotion_data())

    calls = {}

    def fake_run_experiment(*args, **kwargs):
        calls["training"] = True
        return {
            "model_id": "model-9",
            "run_id": "run-9",
            "metrics": {"mae": 40.0, "rmse": 50.0, "r2": 0.5},
        }

    monkeypatch.setattr("ml.pipeline.run_experiment", fake_run_experiment)
    monkeypatch.setattr(
        "ml.pipeline.evaluate_initial_models",
        lambda results: {"promote": True, "reason": "ok"},
    )
    monkeypatch.setattr(
        "ml.pipeline.register_model",
        lambda model_id: SimpleNamespace(version="8"),
    )
    monkeypatch.setattr(
        "ml.pipeline.promote_model",
        lambda version: SimpleNamespace(version="8"),
    )
    monkeypatch.setattr("ml.pipeline.mlflow", FakeMlflow())

    run_v1(force=True)

    assert calls["training"] is True


@pytest.mark.parametrize(
    ("argv", "expected_force"),
    [
        (["pipeline", "--dataset-version", "v1"], False),
        (
            ["pipeline", "--dataset-version", "v1", "--force-initial-training"],
            True,
        ),
    ],
)
def test_cli_forwards_the_force_flag(monkeypatch, argv, expected_force):
    received = {}

    monkeypatch.setattr(
        "ml.pipeline.run_v1",
        lambda force=False: received.update(force=force),
    )
    monkeypatch.setattr("sys.argv", argv)

    pipeline_main()

    assert received["force"] is expected_force


def test_run_v2_promotion_path_never_reads_final_test(monkeypatch):
    monkeypatch.setattr("ml.pipeline.load_final_test", forbid_final_test_read)
    monkeypatch.setattr("ml.pipeline.load_data", lambda version: promotion_data())
    monkeypatch.setattr(
        "ml.pipeline.get_champion",
        lambda: SimpleNamespace(version="1"),
    )
    monkeypatch.setattr("ml.pipeline.load_champion_model", lambda: object())
    monkeypatch.setattr(
        "ml.pipeline.evaluate_model",
        lambda model, X, y: (None, {"mae": 50.0, "rmse": 60.0, "r2": 0.3}),
    )
    monkeypatch.setattr(
        "ml.pipeline.run_experiment",
        lambda *args, **kwargs: {
            "model_id": "model-2",
            "run_id": "run-2",
            "metrics": {"mae": 40.0, "rmse": 50.0, "r2": 0.5},
        },
    )
    monkeypatch.setattr(
        "ml.pipeline.evaluate_candidate",
        lambda champion, candidate: {"promote": True, "reason": "ok"},
    )
    monkeypatch.setattr(
        "ml.pipeline.register_model",
        lambda model_id: SimpleNamespace(version="2"),
    )
    monkeypatch.setattr(
        "ml.pipeline.promote_model",
        lambda version: SimpleNamespace(version="2"),
    )
    monkeypatch.setattr("ml.pipeline.mlflow", FakeMlflow())

    run_v2()


def test_run_final_evaluation_never_promotes(monkeypatch):
    fake_mlflow = FakeMlflow()

    monkeypatch.setattr("ml.pipeline.mlflow", fake_mlflow)
    monkeypatch.setattr(
        "ml.pipeline.get_champion",
        lambda: SimpleNamespace(version="1"),
    )
    monkeypatch.setattr("ml.pipeline.load_champion_model", lambda: object())
    monkeypatch.setattr(
        "ml.pipeline.load_final_test_data",
        lambda: (
            pd.DataFrame({"feature": [1.0, 2.0]}),
            pd.Series([10.0, 20.0]),
            ("2016-04-30 08:20:00", "2016-05-27 18:00:00"),
        ),
    )
    monkeypatch.setattr(
        "ml.pipeline.evaluate_model",
        lambda model, X, y: (
            [11.0, 19.0],
            {"mae": 1.0, "rmse": 1.5, "r2": 0.9},
        ),
    )
    monkeypatch.setattr("ml.pipeline.register_model", forbid_promotion)
    monkeypatch.setattr("ml.pipeline.promote_model", forbid_promotion)

    run_final_evaluation("v2")

    assert fake_mlflow.tags["evaluation_split"] == "final_test"
    assert fake_mlflow.tags["final_test_used_for_selection"] == "false"
    assert fake_mlflow.tags["promotion_eligible"] == "false"
    assert fake_mlflow.metrics["final_test_mae"] == 1.0


def test_final_test_loader_is_not_imported_by_promotion_helpers():
    """Garde-fou statique : load_data/run_v1/run_v2 n'appellent pas le chargeur."""
    import inspect

    from ml import pipeline

    promotion_sources = [
        inspect.getsource(pipeline.load_data),
        inspect.getsource(pipeline.run_v1),
        inspect.getsource(pipeline.run_v2),
    ]

    for source in promotion_sources:
        assert "load_final_test" not in source
        assert "load_final_test_data" not in source
