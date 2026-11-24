import time

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = "wattwise-energy-forecaster-final"
CHAMPION_ALIAS = "champion"

# Codes MLflow qui peuvent signaler un champion absent. Le code seul ne
# suffit pas : INVALID_PARAMETER_VALUE sert aussi à d'autres erreurs, d'où
# la vérification du message dans `_is_missing_champion_error`.
_MISSING_CHAMPION_ERROR_CODES = {
    "RESOURCE_DOES_NOT_EXIST",
    "INVALID_PARAMETER_VALUE",
}


class NoChampionRegistered(RuntimeError):
    """Aucun champion n'est encore enregistré dans le registre MLflow."""


def get_client():
    """Return the MLflow client using the configured tracking server."""
    return MlflowClient()


def _is_missing_champion_error(exc):
    """Indique si l'erreur MLflow signifie réellement « aucun champion ».

    Un alias absent et un modèle absent doivent être distingués des erreurs
    d'authentification, de réseau ou de serveur : ces dernières ne doivent
    jamais être présentées comme « aucun champion enregistré ».
    """
    if str(getattr(exc, "error_code", "")) not in _MISSING_CHAMPION_ERROR_CODES:
        return False

    message = str(exc).lower()

    return "not found" in message and any(
        keyword in message for keyword in ("alias", "registered model", "model version")
    )


def get_champion():
    """Return the currently approved champion model version.

    Seule l'absence effective du modèle ou de l'alias est convertie en
    ``RuntimeError``. Toute autre erreur MLflow est propagée telle quelle.
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

        # Une erreur d'authentification, de réseau ou de serveur ne signifie
        # pas qu'aucun champion existe : on la laisse remonter intacte.
        exc.add_note(
            "Lecture du champion MLflow impossible. Vérifier "
            "MLFLOW_TRACKING_URI, les identifiants et la disponibilité du "
            "serveur de tracking."
        )
        raise


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


def promote_model(version, model_name=MODEL_NAME):
    """
    Move the champion alias to an approved registered model version.
    This function must only be called after the promotion policy passes.
    """
    client = get_client()

    model_version = client.get_model_version(
        name=model_name,
        version=str(version),
    )

    if model_version.status != "READY":
        raise RuntimeError(
            f"Cannot promote model version {version}: status is {model_version.status}"
        )

    client.set_registered_model_alias(
        name=model_name,
        alias=CHAMPION_ALIAS,
        version=str(version),
    )

    champion = client.get_model_version_by_alias(
        model_name,
        CHAMPION_ALIAS,
    )

    if champion.version != str(version):
        raise RuntimeError("Champion alias verification failed.")

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


def rollback_champion(version, model_name=MODEL_NAME):
    """Restore the champion alias to a previously registered version."""
    return promote_model(version, model_name)
