import time

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from ml.policy import PromotionAuthorization

MODEL_NAME = "wattwise-energy-forecaster-final"
CHAMPION_ALIAS = "champion"
CHAMPION_APPROVED_TAG = "champion_approved"
CHAMPION_APPROVAL_REASON_TAG = "champion_approval_reason"

# MLflow error codes that may indicate a missing champion. The code alone is
# not sufficient: INVALID_PARAMETER_VALUE is also used for other errors, hence
# the message check in `_is_missing_champion_error`.
_MISSING_CHAMPION_ERROR_CODES = {
    "RESOURCE_DOES_NOT_EXIST",
    "INVALID_PARAMETER_VALUE",
}


class NoChampionRegistered(RuntimeError):
    """No champion is registered yet in the MLflow registry."""


class PromotionAuthorizationError(RuntimeError):
    """Raised when no validated promotion policy decision authorizes a move."""


class RollbackAuthorizationError(RuntimeError):
    """Raised when a rollback is not explicitly authorized."""


class AliasAuditFailed(RuntimeError):
    """Raised when an alias move succeeded but its MLflow audit failed."""


class ChampionAliasRestoreFailed(RuntimeError):
    """Raised when the alias restore or its verification fails."""


def get_client():
    """Return the MLflow client using the configured tracking server."""
    return MlflowClient()


def _is_missing_champion_error(exc):
    """Indicate whether the MLflow error really means "no champion".

    A missing alias and a missing model must be distinguished from
    authentication, network or server errors: the latter must never be
    presented as "no champion registered".
    """
    if str(getattr(exc, "error_code", "")) not in _MISSING_CHAMPION_ERROR_CODES:
        return False

    message = str(exc).lower()

    return "not found" in message and any(
        keyword in message for keyword in ("alias", "registered model", "model version")
    )


def get_champion():
    """Return the currently approved champion model version.

    Only the effective absence of the model or the alias is converted to a
    ``RuntimeError``. Any other MLflow error is propagated as is.
    """
    client = get_client()

    try:
        return client.get_model_version_by_alias(
            MODEL_NAME,
            CHAMPION_ALIAS,
        )
    except MlflowException as exc:
        if _is_missing_champion_error(exc):
            raise NoChampionRegistered(
                f"No champion model is registered yet: alias "
                f"'{CHAMPION_ALIAS}' is missing on model '{MODEL_NAME}'."
            ) from exc

        # An authentication, network or server error does not mean that no
        # champion exists: let it propagate.
        exc.add_note(
            "Unable to read the MLflow champion. Check "
            "MLFLOW_TRACKING_URI, credentials and the availability of the "
            "tracking server."
        )
        raise


def _current_champion_version(client, model_name):
    """Return the current champion version, or ``None`` when no champion exists."""
    try:
        champion = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except MlflowException as exc:
        if _is_missing_champion_error(exc):
            return None

        raise

    return getattr(champion, "version", None)


def get_champion_version():
    """Return the numeric version of the current champion."""
    return get_champion().version


def register_model(model_id, model_name=MODEL_NAME):
    model_uri = f"models:/{model_id}"

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    client = get_client()

    for _ in range(60):
        version = client.get_model_version(
            name=model_name,
            version=registered.version,
        )

        if version.status == "READY":
            print(f"Registered model: {model_name}, version: {version.version}")
            return version

        if version.status == "FAILED":
            raise RuntimeError(
                f"Model registration failed: {version.status} - "
                f"{version.status_message}"
            )

        time.sleep(2)

    raise TimeoutError("Model registration timed out.")


def log_alias_change(
    model_name,
    version,
    operation,
    reason,
    authorized_by,
    previous_version,
):
    """Audit a successful champion alias move in MLflow.

    Only call this for a move that is still in effect: a failed audit restores
    the previous alias, so a failed audit must not be replayed with this
    function.
    """
    version = str(version)

    with mlflow.start_run(run_name=f"champion-alias-{operation}"):
        mlflow.log_params(
            {
                "model_name": model_name,
                "operation": operation,
                "previous_champion_version": str(previous_version or "none"),
                "new_champion_version": version,
                "decision_reason": reason or "",
            }
        )
        mlflow.set_tags(
            {
                "champion_operation": operation,
                "champion_previous_version": str(previous_version or "none"),
                "champion_new_version": version,
                "champion_authorized_by": authorized_by,
                "champion_audit_result": "success",
            }
        )


def _restore_champion_alias(client, model_name, previous_version):
    """Restore the champion alias and verify the restored state."""
    if previous_version is not None:
        client.set_registered_model_alias(
            name=model_name,
            alias=CHAMPION_ALIAS,
            version=str(previous_version),
        )

        restored = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
        restored_version = getattr(restored, "version", None)

        if str(restored_version) != str(previous_version):
            raise RuntimeError(
                "Champion alias restore verification failed: expected version "
                f"{previous_version}, got {restored_version}."
            )

        return

    delete_alias = getattr(client, "delete_registered_model_alias", None)

    if delete_alias is None:
        raise RuntimeError(
            "No previous champion existed and the installed MLflow client does "
            "not expose delete_registered_model_alias: the alias created by the "
            "failed move cannot be removed."
        )

    delete_alias(name=model_name, alias=CHAMPION_ALIAS)

    try:
        remaining = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except MlflowException as exc:
        if _is_missing_champion_error(exc):
            return
        raise

    if remaining is None:
        return

    raise RuntimeError(
        "Champion alias deletion verification failed: the alias still points "
        f"to version {getattr(remaining, 'version', None)}."
    )


def _restore_alias_then_raise(client, model_name, previous_version, error, cause=None):
    """Restore the alias after a failed move, then raise the initial error.

    If the restore fails, ``ChampionAliasRestoreFailed`` keeps both the initial
    error and the restore error.
    """
    try:
        _restore_champion_alias(client, model_name, previous_version)
    except Exception as restore_error:
        raise ChampionAliasRestoreFailed(
            f"{error} Restoring the previous champion alias also failed: "
            f"{restore_error}"
        ) from restore_error

    error.add_note(
        "The champion alias was restored to its state before the failed move "
        f"(previous champion version: {previous_version or 'none'})."
    )

    if cause is not None:
        raise error from cause

    raise error


def _move_champion_alias(model_name, version, operation, reason, authorization):
    """Move, verify, audit and (for promotions) tag the champion alias.

    Any failure after the alias move restores the previous alias. If no
    champion existed before the move, the newly created alias is deleted.
    """
    client = get_client()

    model_version = client.get_model_version(
        name=model_name,
        version=str(version),
    )

    if model_version.status != "READY":
        raise RuntimeError(
            f"Cannot {operation} model version {version}: status is {model_version.status}"
        )

    previous_version = _current_champion_version(client, model_name)

    client.set_registered_model_alias(
        name=model_name,
        alias=CHAMPION_ALIAS,
        version=str(version),
    )

    try:
        champion = client.get_model_version_by_alias(
            model_name,
            CHAMPION_ALIAS,
        )
    except Exception as exc:  # noqa: BLE001 - any failure must restore the alias
        _restore_alias_then_raise(
            client,
            model_name,
            previous_version,
            RuntimeError(f"Champion alias verification failed: {exc}"),
            cause=exc,
        )

    if champion.version != str(version):
        _restore_alias_then_raise(
            client,
            model_name,
            previous_version,
            RuntimeError("Champion alias verification failed."),
        )

    authorized_by = (
        str(authorization.reason)
        if isinstance(authorization, PromotionAuthorization)
        else str(reason)
    )

    try:
        log_alias_change(
            model_name=model_name,
            version=str(version),
            operation=operation,
            reason=reason,
            authorized_by=authorized_by,
            previous_version=previous_version,
        )
    except Exception as exc:  # noqa: BLE001 - any failure must restore the alias
        _restore_alias_then_raise(
            client,
            model_name,
            previous_version,
            AliasAuditFailed(
                f"Champion alias was moved to version {version} on model "
                f"{model_name}, but the MLflow audit logging failed: {exc}"
            ),
            cause=exc,
        )

    if operation == "promote":
        try:
            client.set_model_version_tag(
                name=model_name,
                version=str(version),
                key=CHAMPION_APPROVED_TAG,
                value="true",
            )
            client.set_model_version_tag(
                name=model_name,
                version=str(version),
                key=CHAMPION_APPROVAL_REASON_TAG,
                value=str(reason),
            )
        except Exception as exc:  # noqa: BLE001 - any failure must restore the alias
            _restore_alias_then_raise(
                client,
                model_name,
                previous_version,
                RuntimeError(
                    f"Champion alias was moved to version {version} on model "
                    f"{model_name}, but recording the approval tags failed: {exc}"
                ),
                cause=exc,
            )

    return champion


def promote_model(version, model_name=MODEL_NAME, authorization=None, candidate=None):
    """
    Move the champion alias to an approved registered model version.

    The promotion is authorized only through a typed promotion authorization
    issued by the promotion policy layer after a positive decision: this
    function must only be called after the promotion policy passes.

    Raises:
        PromotionAuthorizationError: if no policy-issued authorization is
            available or if it does not cover the promoted version.
        RuntimeError: if the version is not READY or the alias verification
            fails.
        AliasAuditFailed: if the alias move succeeded but the audit failed and
            the previous alias was restored.
        ChampionAliasRestoreFailed: if a failure occurred after the alias move
            and the best-effort restore failed.
    """
    if not isinstance(authorization, PromotionAuthorization):
        raise PromotionAuthorizationError(
            f"Promotion of version {version} on model {model_name} refused: no "
            "policy-issued promotion authorization is available."
        )

    if not authorization.authorizes(str(version), candidate=candidate):
        raise PromotionAuthorizationError(
            f"Promotion of version {version} on model {model_name} refused: the "
            "promotion authorization is not bound to this version."
        )

    champion = _move_champion_alias(
        model_name=model_name,
        version=version,
        operation="promote",
        reason=authorization.reason,
        authorization=authorization,
    )

    print(f"Champion updated: {model_name} version {champion.version}")

    return champion


def load_champion_model(model_name=MODEL_NAME):
    """Load the model currently referenced by the champion alias."""
    model_uri = f"models:/{model_name}@{CHAMPION_ALIAS}"
    return mlflow.sklearn.load_model(model_uri)


def load_model_version(version, model_name=MODEL_NAME):
    """Load a specific registered model version."""
    model_uri = f"models:/{model_name}/{version}"
    return mlflow.sklearn.load_model(model_uri)


def list_registered_versions(model_name=MODEL_NAME):
    """List registered model versions, newest first."""
    client = get_client()

    return client.search_model_versions(
        filter_string=f"name = '{model_name}'",
        order_by=["version_number DESC"],
        max_results=100,
    )


def rollback_champion(version, model_name=MODEL_NAME, reason=None):
    """
    Restore the champion alias to a previously approved registered version.

    The rollback uses a dedicated authorization path: an explicit reason is
    required and the target version must be registered, READY and previously
    approved through the durable approval proof stored in MLflow on the
    promoted version.

    Raises:
        RollbackAuthorizationError: if the rollback is not explicitly authorized.
        RuntimeError: if the version is not READY or the alias verification
            fails.
        AliasAuditFailed: if the alias move succeeded but the audit failed.
        ChampionAliasRestoreFailed: if the best-effort restore fails.
    """
    if not reason or not str(reason).strip():
        raise RollbackAuthorizationError(
            f"Rollback to version {version} on model {model_name} refused: an "
            "explicit reason is required."
        )

    client = get_client()

    model_version = client.get_model_version(
        name=model_name,
        version=str(version),
    )

    if model_version.status != "READY":
        raise RuntimeError(
            f"Cannot rollback model version {version}: status is {model_version.status}"
        )

    approved = client.get_model_version_tag(
        name=model_name,
        version=str(version),
        key=CHAMPION_APPROVED_TAG,
    )

    if str(getattr(approved, "value", approved)) != "true":
        raise RollbackAuthorizationError(
            f"Rollback to version {version} on model {model_name} refused: version "
            f"{version} was not previously approved."
        )

    return _move_champion_alias(
        model_name=model_name,
        version=version,
        operation="rollback",
        reason=reason,
        authorization=None,
    )
