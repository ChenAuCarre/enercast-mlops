import json
import logging
from datetime import UTC, datetime

import mlflow

from ml import registry

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
        self.state = (model, info)

    def training_profile(self) -> dict | None:
        """The champion run's ``training-profile.json``; None when absent (older runs)."""
        info = self.info
        if info is None:
            return None
        try:
            artifact_path = mlflow.artifacts.download_artifacts(
                run_id=info["mlflow_run_id"], artifact_path="training-profile.json"
            )
            with open(artifact_path, encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:  # drift is optional, never fatal
            log.warning("training_profile_unavailable", exc_info=True)
            return None
