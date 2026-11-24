"""Tests for reading the champion from the MLflow registry.

The MLflow client is fully simulated: no network calls, no database, no
``mlruns`` file is needed.
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
    CHAMPION_APPROVED_TAG,
    MODEL_NAME,
    AliasAuditFailed,
    ChampionAliasRestoreFailed,
    NoChampionRegistered,
    PromotionAuthorization,
    PromotionAuthorizationError,
    RollbackAuthorizationError,
    _current_champion_version,
    get_champion,
    promote_model,
    register_model,
    rollback_champion,
)


class FakeRun:
    def __init__(self):
        self.info = SimpleNamespace(run_id="run-alias")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeAuditMlflow:
    """Minimal MLflow client used to capture the alias audit logging."""

    def __init__(self, fail=False):
        self.params = {}
        self.tags = {}
        self.run_names = []
        self.fail = fail

    def start_run(self, run_name=None):
        self.run_names.append(run_name)

        if self.fail:
            raise RuntimeError("Simulated audit logging failure.")

        return FakeRun()

    def log_params(self, params):
        self.params.update(params)

    def set_tags(self, tags):
        self.tags.update(tags)


class FakeClient:
    """Minimal MLflow client that returns a result or raises an error."""

    def __init__(self, result=None, error=None, version=None):
        self.result = result
        self.error = error
        self.version = version
        self.calls = []
        self.aliases = {}
        self.tags = {}
        self.fail_set = False
        self.fail_restore = False
        self.fail_tag = False
        self.set_count = 0
        self.restore_verification_version = None
        self.deleted_alias_still_visible = False
        self.forced_alias_version = None

    def get_model_version_by_alias(self, name, alias):
        self.calls.append((name, alias))

        if self.error is not None:
            raise self.error

        if self.forced_alias_version is not None:
            return SimpleNamespace(version=self.forced_alias_version, status="READY")

        if self.restore_verification_version is not None and self.set_count > 1:
            return SimpleNamespace(
                version=self.restore_verification_version, status="READY"
            )

        if alias in self.aliases:
            return SimpleNamespace(version=self.aliases[alias], status="READY")

        return self.result

    def get_model_version(self, name, version):
        self.calls.append((name, version))

        if self.error is not None:
            raise self.error

        return self.version

    def set_registered_model_alias(self, name, alias, version):
        self.calls.append(("alias", name, alias, version))

        if self.fail_set:
            raise RuntimeError("Simulated alias move failure")

        self.set_count += 1

        if self.fail_restore and self.set_count > 1:
            raise RuntimeError("Simulated alias restore failure")

        self.aliases[alias] = version

        return SimpleNamespace(version=version)

    def delete_registered_model_alias(self, name, alias):
        self.calls.append(("alias-delete", name, alias))

        if self.deleted_alias_still_visible:
            return

        self.aliases.pop(alias, None)

    def get_model_version_tag(self, name, version, key):
        self.calls.append(("tag-get", name, version, key))

        if self.error is not None:
            raise self.error

        return SimpleNamespace(value=self.tags.get((str(version), key)))

    def set_model_version_tag(self, name, version, key, value):
        self.calls.append(("tag-set", name, version, key, value))

        if self.fail_tag:
            raise RuntimeError("Simulated approval tag failure")

        self.tags[(str(version), key)] = str(value)

        return SimpleNamespace(value=str(value))


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


def test_promote_model_rejects_a_fabricated_authorization(monkeypatch):
    client = FakeClient(version=SimpleNamespace(version="2", status="READY"))
    install_client(monkeypatch, client)

    with pytest.raises(PromotionAuthorizationError):
        promote_model("2", authorization={"promote": True, "reason": "fabricated"})

    assert client.aliases == {}


def test_promote_model_rejects_a_fabricated_negative_decision(monkeypatch):
    client = FakeClient(version=SimpleNamespace(version="2", status="READY"))
    install_client(monkeypatch, client)

    authorization = PromotionAuthorization.from_decision(
        {
            "promote": False,
            "reason": "Candidate rejected: MAE exceeds the quality threshold.",
        },
        "2",
        candidate="candidate-hist-gradient-boosting-v2",
    )

    with pytest.raises(PromotionAuthorizationError):
        promote_model(
            "2",
            authorization=authorization,
            candidate="candidate-hist-gradient-boosting-v2",
        )

    assert client.aliases == {}


def test_promotion_authorization_cannot_be_constructed_directly():
    with pytest.raises(TypeError):
        PromotionAuthorization(
            "2",
            "Candidate approved: MAE improvement=0.10.",
            "candidate-hist-gradient-boosting-v2",
        )


def test_promote_model_rejects_an_authorization_without_candidate(monkeypatch):
    client = FakeClient(version=SimpleNamespace(version="2", status="READY"))
    install_client(monkeypatch, client)

    authorization = PromotionAuthorization.from_decision(
        {
            "promote": True,
            "reason": "Candidate approved: MAE improvement=0.10.",
            "version": "2",
        },
        "2",
        candidate=None,
    )

    with pytest.raises(PromotionAuthorizationError):
        promote_model("2", authorization=authorization, candidate=None)

    assert client.aliases == {}


def test_promote_model_accepts_a_policy_issued_authorization_and_audits_it(monkeypatch):
    decision = {
        "promote": True,
        "reason": "Candidate approved: MAE improvement=0.10.",
        "version": "2",
    }
    authorization = PromotionAuthorization.from_decision(
        decision,
        "2",
        candidate="candidate-hist-gradient-boosting-v2",
    )
    client = FakeClient(version=SimpleNamespace(version="2", status="READY"))
    audit_mlflow = FakeAuditMlflow()
    install_client(monkeypatch, client)
    monkeypatch.setattr("ml.registry.mlflow", audit_mlflow)

    champion = promote_model(
        "2",
        authorization=authorization,
        candidate="candidate-hist-gradient-boosting-v2",
    )

    assert client.aliases[CHAMPION_ALIAS] == "2"
    assert champion.version == "2"
    assert audit_mlflow.run_names == ["champion-alias-promote"]
    assert audit_mlflow.tags["champion_audit_result"] == "success"
    assert client.tags[("2", CHAMPION_APPROVED_TAG)] == "true"


def test_promote_model_rejects_an_authorization_bound_to_another_version(monkeypatch):
    client = FakeClient(version=SimpleNamespace(version="2", status="READY"))
    install_client(monkeypatch, client)

    authorization = PromotionAuthorization.from_decision(
        {
            "promote": True,
            "reason": "Candidate approved: MAE improvement=0.10.",
            "version": "2",
        },
        "9",
        candidate="candidate-hist-gradient-boosting-v2",
    )

    with pytest.raises(PromotionAuthorizationError):
        promote_model(
            "2",
            authorization=authorization,
            candidate="candidate-hist-gradient-boosting-v2",
        )

    assert client.aliases == {}


def test_failed_promotion_does_not_move_the_alias(monkeypatch):
    client = FakeClient(version=SimpleNamespace(version="2", status="READY"))
    install_client(monkeypatch, client)
    client.fail_set = True

    with pytest.raises(RuntimeError, match="Simulated alias move failure"):
        promote_model(
            "2",
            authorization=PromotionAuthorization.from_decision(
                {
                    "promote": True,
                    "reason": "Candidate approved: MAE improvement=0.10.",
                    "version": "2",
                },
                "2",
                candidate="candidate-hist-gradient-boosting-v2",
            ),
            candidate="candidate-hist-gradient-boosting-v2",
        )

    assert client.aliases == {}


def test_promote_model_raises_clearly_when_the_audit_fails(monkeypatch):
    client = FakeClient(version=SimpleNamespace(version="2", status="READY"))
    audit_mlflow = FakeAuditMlflow(fail=True)
    install_client(monkeypatch, client)
    monkeypatch.setattr("ml.registry.mlflow", audit_mlflow)

    with pytest.raises(AliasAuditFailed):
        promote_model(
            "2",
            authorization=PromotionAuthorization.from_decision(
                {
                    "promote": True,
                    "reason": "Candidate approved: MAE improvement=0.10.",
                    "version": "2",
                },
                "2",
                candidate="candidate-hist-gradient-boosting-v2",
            ),
            candidate="candidate-hist-gradient-boosting-v2",
        )

    assert CHAMPION_ALIAS not in client.aliases


def test_rollback_requires_an_explicit_reason(monkeypatch):
    client = FakeClient(version=SimpleNamespace(version="1", status="READY"))
    install_client(monkeypatch, client)

    with pytest.raises(RollbackAuthorizationError):
        rollback_champion("1")

    assert client.aliases == {}


def test_rollback_without_persistent_proof_is_refused(monkeypatch):
    client = FakeClient(version=SimpleNamespace(version="1", status="READY"))
    install_client(monkeypatch, client)

    with pytest.raises(RollbackAuthorizationError):
        rollback_champion(
            "1",
            reason="The champion no longer satisfies the quality threshold.",
        )

    assert client.aliases == {}


def test_rollback_with_valid_persistent_proof_is_accepted_and_audited(monkeypatch):
    client = FakeClient(version=SimpleNamespace(version="1", status="READY"))
    client.set_model_version_tag(MODEL_NAME, "1", CHAMPION_APPROVED_TAG, "true")
    audit_mlflow = FakeAuditMlflow()
    install_client(monkeypatch, client)
    monkeypatch.setattr("ml.registry.mlflow", audit_mlflow)

    champion = rollback_champion(
        "1",
        reason="The champion no longer satisfies the quality threshold.",
    )

    assert client.aliases[CHAMPION_ALIAS] == "1"
    assert champion.version == "1"
    assert audit_mlflow.run_names == ["champion-alias-rollback"]
    assert audit_mlflow.tags["champion_audit_result"] == "success"


def test_rollback_to_a_missing_version_is_refused(monkeypatch):
    error = MlflowException(
        "Model version 42 not found.",
        error_code=RESOURCE_DOES_NOT_EXIST,
    )
    install_client(monkeypatch, FakeClient(error=error))

    with pytest.raises(MlflowException):
        rollback_champion(
            "42",
            reason="The champion no longer satisfies the quality threshold.",
        )


def test_rollback_to_a_non_ready_version_is_refused(monkeypatch):
    client = FakeClient(version=SimpleNamespace(version="3", status="FAILED"))
    install_client(monkeypatch, client)

    with pytest.raises(RuntimeError, match="Cannot rollback"):
        rollback_champion(
            "3",
            reason="The champion no longer satisfies the quality threshold.",
        )

    assert client.aliases == {}


def test_verification_failure_with_successful_restore(monkeypatch):
    client = FakeClient(version=SimpleNamespace(version="2", status="READY"))
    client.aliases[CHAMPION_ALIAS] = "1"
    client.forced_alias_version = "1"
    install_client(monkeypatch, client)

    authorization = PromotionAuthorization.from_decision(
        {
            "promote": True,
            "reason": "Candidate approved: MAE improvement=0.10.",
            "version": "2",
        },
        "2",
        candidate="candidate-hist-gradient-boosting-v2",
    )

    with pytest.raises(RuntimeError, match="Champion alias verification failed."):
        promote_model(
            "2",
            authorization=authorization,
            candidate="candidate-hist-gradient-boosting-v2",
        )

    assert client.aliases[CHAMPION_ALIAS] == "1"


def test_verification_failure_with_failed_restore_raises_a_specific_error(monkeypatch):
    client = FakeClient(version=SimpleNamespace(version="2", status="READY"))
    client.aliases[CHAMPION_ALIAS] = "1"
    client.forced_alias_version = "1"
    install_client(monkeypatch, client)

    def failing_restore(name, alias, version):
        if str(version) == "1":
            raise RuntimeError("Simulated alias restore failure.")

        return SimpleNamespace(version=version)

    client.set_registered_model_alias = failing_restore

    authorization = PromotionAuthorization.from_decision(
        {
            "promote": True,
            "reason": "Candidate approved: MAE improvement=0.10.",
            "version": "2",
        },
        "2",
        candidate="candidate-hist-gradient-boosting-v2",
    )

    with pytest.raises(ChampionAliasRestoreFailed):
        promote_model(
            "2",
            authorization=authorization,
            candidate="candidate-hist-gradient-boosting-v2",
        )


def promotion_authorization(
    version="2",
    candidate="candidate-hist-gradient-boosting-v2",
):
    return PromotionAuthorization.from_decision(
        {
            "promote": True,
            "reason": "Candidate approved: MAE improvement=0.10.",
            "version": version,
        },
        version,
        candidate=candidate,
    )


def test_current_champion_version_returns_none_when_no_champion():
    error = MlflowException(
        f"Registered model alias {CHAMPION_ALIAS} not found.",
        error_code=INVALID_PARAMETER_VALUE,
    )
    client = FakeClient(error=error)

    assert _current_champion_version(client, MODEL_NAME) is None


@pytest.mark.parametrize(
    ("error_code", "message"),
    [
        (UNAUTHENTICATED, "Authentication failed for the tracking server."),
        (INTERNAL_ERROR, "The tracking server returned an internal error."),
    ],
)
def test_current_champion_version_propagates_authentication_and_server_errors(
    error_code,
    message,
):
    error = MlflowException(message, error_code=error_code)
    client = FakeClient(error=error)

    with pytest.raises(MlflowException) as info:
        _current_champion_version(client, MODEL_NAME)

    assert info.value is error


def test_failed_audit_restores_the_previous_champion(monkeypatch):
    client = FakeClient(version=SimpleNamespace(version="2", status="READY"))
    client.aliases[CHAMPION_ALIAS] = "1"
    install_client(monkeypatch, client)
    monkeypatch.setattr("ml.registry.mlflow", FakeAuditMlflow(fail=True))

    with pytest.raises(AliasAuditFailed) as info:
        promote_model(
            "2",
            authorization=promotion_authorization(),
            candidate="candidate-hist-gradient-boosting-v2",
        )

    assert client.aliases[CHAMPION_ALIAS] == "1"
    assert info.value.__notes__
    assert "restored" in info.value.__notes__[0]


def test_failed_audit_without_previous_champion_removes_the_new_alias(monkeypatch):
    client = FakeClient(version=SimpleNamespace(version="2", status="READY"))
    install_client(monkeypatch, client)
    monkeypatch.setattr("ml.registry.mlflow", FakeAuditMlflow(fail=True))

    with pytest.raises(AliasAuditFailed):
        promote_model(
            "2",
            authorization=promotion_authorization(),
            candidate="candidate-hist-gradient-boosting-v2",
        )

    assert CHAMPION_ALIAS not in client.aliases


def test_failed_audit_and_failed_restore_raise_champion_alias_restore_failed(
    monkeypatch,
):
    client = FakeClient(version=SimpleNamespace(version="2", status="READY"))
    client.aliases[CHAMPION_ALIAS] = "1"
    client.fail_restore = True
    install_client(monkeypatch, client)
    monkeypatch.setattr("ml.registry.mlflow", FakeAuditMlflow(fail=True))

    with pytest.raises(ChampionAliasRestoreFailed) as info:
        promote_model(
            "2",
            authorization=promotion_authorization(),
            candidate="candidate-hist-gradient-boosting-v2",
        )

    message = str(info.value)
    assert "audit logging failed" in message
    assert "Simulated alias restore failure" in message
    assert isinstance(info.value.__cause__, RuntimeError)


def test_failed_approval_tags_restore_the_previous_champion(monkeypatch):
    client = FakeClient(version=SimpleNamespace(version="2", status="READY"))
    client.aliases[CHAMPION_ALIAS] = "1"
    client.fail_tag = True
    install_client(monkeypatch, client)
    monkeypatch.setattr("ml.registry.mlflow", FakeAuditMlflow())

    with pytest.raises(RuntimeError, match="Simulated approval tag failure"):
        promote_model(
            "2",
            authorization=promotion_authorization(),
            candidate="candidate-hist-gradient-boosting-v2",
        )

    assert client.aliases[CHAMPION_ALIAS] == "1"
    assert client.tags == {}


def test_missing_delete_api_raises_a_champion_alias_restore_failed(monkeypatch):
    client = FakeClient(version=SimpleNamespace(version="2", status="READY"))
    install_client(monkeypatch, client)
    monkeypatch.delattr(FakeClient, "delete_registered_model_alias")
    monkeypatch.setattr("ml.registry.mlflow", FakeAuditMlflow(fail=True))

    with pytest.raises(
        ChampionAliasRestoreFailed, match="delete_registered_model_alias"
    ):
        promote_model(
            "2",
            authorization=promotion_authorization(),
            candidate="candidate-hist-gradient-boosting-v2",
        )

    assert client.aliases[CHAMPION_ALIAS] == "2"


def test_failed_restore_verification_raises_champion_alias_restore_failed(
    monkeypatch,
):
    client = FakeClient(version=SimpleNamespace(version="2", status="READY"))
    client.aliases[CHAMPION_ALIAS] = "1"
    client.restore_verification_version = "9"
    install_client(monkeypatch, client)
    monkeypatch.setattr("ml.registry.mlflow", FakeAuditMlflow(fail=True))

    with pytest.raises(
        ChampionAliasRestoreFailed, match="restore verification failed"
    ) as info:
        promote_model(
            "2",
            authorization=promotion_authorization(),
            candidate="candidate-hist-gradient-boosting-v2",
        )

    assert client.aliases[CHAMPION_ALIAS] == "1"
    assert "expected version 1" in str(info.value)
    assert "got 9" in str(info.value)


def test_failed_alias_deletion_verification_raises_champion_alias_restore_failed(
    monkeypatch,
):
    client = FakeClient(version=SimpleNamespace(version="2", status="READY"))
    client.deleted_alias_still_visible = True
    install_client(monkeypatch, client)
    monkeypatch.setattr("ml.registry.mlflow", FakeAuditMlflow(fail=True))

    with pytest.raises(
        ChampionAliasRestoreFailed, match="deletion verification failed"
    ):
        promote_model(
            "2",
            authorization=promotion_authorization(),
            candidate="candidate-hist-gradient-boosting-v2",
        )

    assert client.aliases[CHAMPION_ALIAS] == "2"
