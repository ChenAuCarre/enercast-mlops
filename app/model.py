from datetime import UTC, datetime

import mlflow

from app import drift, metrics
from ml import registry


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
        reference = drift.load_reference(mv.run_id, mv.version)  # before the swap
        self.state = (model, info)
        metrics.SERVED_VERSION.set(int(mv.version))
        drift.monitor.set_reference(reference)
