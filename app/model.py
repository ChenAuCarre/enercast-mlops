import threading
from datetime import UTC, datetime

import mlflow

from ml import registry


class ChampionHolder:
    """Holds the loaded @champion model; a failed reload keeps the old one."""

    def __init__(self):
        self._lock = threading.Lock()
        self.model = None
        self.info = None

    @property
    def version(self):
        return self.info["resolved_version"] if self.info else None

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
        with self._lock:
            self.model, self.info = model, info
