import logging
from datetime import UTC, datetime

import mlflow

from app import drift, metrics
from ml import registry
from ml.drift import DISTRIBUTION_ARTIFACT

log = logging.getLogger("app")


class ChampionHolder:
    """Holds the loaded @champion model; a failed reload keeps the old one."""

    def __init__(self):
        # (model, info) swapped as one tuple so readers never see a mismatched pair.
        self.state = None

    @property
    def model(self):
        return self.state[0] if self.state else None

    @property
    def info(self):
        return self.state[1] if self.state else None

    @property
    def version(self):
        return self.state[1]["resolved_version"] if self.state else None

    def load(self):
        mv = registry.get_champion()  # raises NoChampionRegistered
        model = registry.load_model_version(mv.version)
        params = mlflow.get_run(mv.run_id).data.params
        info = {
            "model_name": registry.MODEL_NAME,
            "alias": registry.CHAMPION_ALIAS,
            "resolved_version": str(mv.version),
            "mlflow_run_id": mv.run_id,
            "dataset_version": params.get("dataset_version"),
            "loaded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        # Listing first: downloading a missing artifact retries for minutes.
        artifacts = registry.get_client().list_artifacts(mv.run_id)
        if any(a.path == DISTRIBUTION_ARTIFACT for a in artifacts):
            reference = mlflow.artifacts.load_dict(
                f"runs:/{mv.run_id}/{DISTRIBUTION_ARTIFACT}"
            )
        else:  # champions trained before the artifact existed
            log.warning("No training distribution for version %s", mv.version)
            reference = None
        self.state = (model, info)
        metrics.SERVED_VERSION.set(int(mv.version))
        drift.monitor.reference = reference
