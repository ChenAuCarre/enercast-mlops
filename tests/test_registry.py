"""Tests de la lecture du champion dans le registre MLflow.

Le client MLflow est entièrement simulé : aucun appel réseau, aucune base
de données, aucun fichier ``mlruns`` n'est nécessaire.
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

from ml.registry import CHAMPION_ALIAS, MODEL_NAME, get_champion


class FakeClient:
    """Client MLflow minimal qui renvoie un résultat ou lève une erreur."""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def get_model_version_by_alias(self, name, alias):
        self.calls.append((name, alias))

        if self.error is not None:
            raise self.error

        return self.result


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

    with pytest.raises(RuntimeError, match="No champion model is registered yet"):
        get_champion()


def test_missing_alias_means_no_champion(monkeypatch):
    error = MlflowException(
        f"Registered model alias {CHAMPION_ALIAS} not found.",
        error_code=INVALID_PARAMETER_VALUE,
    )
    install_client(monkeypatch, FakeClient(error=error))

    with pytest.raises(
        RuntimeError, match="No champion model is registered yet"
    ) as info:
        get_champion()

    assert info.value.__cause__ is error


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
