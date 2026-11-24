"""Tests of champion reading from the MLflow registry.

The MLflow client is fully mocked: no network call, no database,
no ``mlruns`` file is needed.
"""

from types import SimpleNamespace

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    ENDPOINT_NOT_FOUND,
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    PERMISSION_DENIED,
    RESOURCE_DOES_NOT_EXIST,
    UNAUTHENTICATED,
)

from ml.registry import (
    CHAMPION_ALIAS,
    MODEL_NAME,
    NoChampionRegistered,
    get_champion,
    register_model,
)


class FakeClient:
    """Minimal MLflow client that returns a result or raises an error."""

    def __init__(self, result=None, error=None, version=None):
        self.result = result
        self.error = error
        self.version = version
        self.calls = []

    def get_model_version_by_alias(self, name, alias):
        self.calls.append((name, alias))

        if self.error is not None:
            raise self.error

        return self.result

    def get_model_version(self, name, version):
        self.calls.append((name, version))

        if self.error is not None:
            raise self.error

        return self.version


def install_client(monkeypatch, client):
    monkeypatch.setattr("ml.registry.get_client", lambda: client)


def test_get_champion_returns_the_version_pointed_by_the_alias(monkeypatch):
    expected = SimpleNamespace(name=MODEL_NAME, version="3", aliases=[CHAMPION_ALIAS])
    client = FakeClient(result=expected)
    install_client(monkeypatch, client)

    champion = get_champion()

    assert champion is expected
    assert client.calls == [(MODEL_NAME, CHAMPION_ALIAS)]


def test_missing_registered_model_means_no_champion(monkeypatch):
    error = MlflowException(
        f"Registered model '{MODEL_NAME}' not found.",
        error_code=RESOURCE_DOES_NOT_EXIST,
    )
    install_client(monkeypatch, FakeClient(error=error))

    with pytest.raises(
        NoChampionRegistered, match="No champion model is registered yet"
    ):
        get_champion()


def test_missing_alias_means_no_champion(monkeypatch):
    error = MlflowException(
        f"Registered model alias {CHAMPION_ALIAS} not found.",
        error_code=INVALID_PARAMETER_VALUE,
    )
    install_client(monkeypatch, FakeClient(error=error))

    with pytest.raises(
        NoChampionRegistered, match="No champion model is registered yet"
    ) as info:
        get_champion()

    assert info.value.__cause__ is error


def test_no_champion_error_stays_a_runtime_error():
    """Compatibility: existing callers catch RuntimeError."""
    assert issubclass(NoChampionRegistered, RuntimeError)


@pytest.mark.parametrize(
    ("error_code", "message"),
    [
        (PERMISSION_DENIED, "Permission denied while reading the registry."),
        (UNAUTHENTICATED, "Authentication failed for the tracking server."),
        (INTERNAL_ERROR, "The tracking server returned an internal error."),
        (ENDPOINT_NOT_FOUND, "The tracking endpoint does not exist."),
    ],
)
def test_registry_errors_are_not_hidden_as_missing_champion(
    monkeypatch,
    error_code,
    message,
):
    error = MlflowException(message, error_code=error_code)
    install_client(monkeypatch, FakeClient(error=error))

    with pytest.raises(MlflowException) as info:
        get_champion()

    assert info.value is error
    assert "No champion model is registered yet" not in str(info.value)
    assert info.value.__notes__
    assert "MLFLOW_TRACKING_URI" in info.value.__notes__[0]


def test_unrelated_invalid_parameter_value_is_propagated(monkeypatch):
    error = MlflowException(
        "Invalid alias format.",
        error_code=INVALID_PARAMETER_VALUE,
    )
    install_client(monkeypatch, FakeClient(error=error))

    with pytest.raises(MlflowException) as info:
        get_champion()

    assert info.value is error
    assert "No champion model is registered yet" not in str(info.value)


def test_connection_error_is_propagated_unchanged(monkeypatch):
    error = ConnectionError("Tracking server unreachable")
    install_client(monkeypatch, FakeClient(error=error))

    with pytest.raises(ConnectionError, match="Tracking server unreachable"):
        get_champion()


def test_failed_registration_includes_the_status_message(monkeypatch):
    version = SimpleNamespace(
        status="FAILED",
        status_message="Artifact validation failed",
    )
    monkeypatch.setattr(
        "ml.registry.mlflow.register_model",
        lambda **kwargs: SimpleNamespace(version="9"),
    )
    install_client(monkeypatch, FakeClient(version=version))

    with pytest.raises(RuntimeError, match="Artifact validation failed"):
        register_model("model-1")
