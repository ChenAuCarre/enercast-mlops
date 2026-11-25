import logging
import threading
from collections import deque

import mlflow
import pandas as pd

from app import metrics
from ml import registry
from ml.drift import DISTRIBUTION_ARTIFACT, psi

WINDOW, EVERY, THRESHOLD = 500, 100, 0.20

log = logging.getLogger("app.drift")


def load_reference(run_id, version):
    """Training distribution logged with the champion's run, None if it has none."""
    # Listing first: downloading a missing artifact retries for minutes.
    artifacts = registry.get_client().list_artifacts(run_id)
    if any(a.path == DISTRIBUTION_ARTIFACT for a in artifacts):
        return mlflow.artifacts.load_dict(f"runs:/{run_id}/{DISTRIBUTION_ARTIFACT}")
    # champions trained before the artifact existed
    log.warning("No training distribution for version %s", version)
    return None


class DriftMonitor:
    """PSI of the last WINDOW predictions' features, recomputed every EVERY."""

    def __init__(self):
        self.lock = threading.Lock()
        self.window = deque(maxlen=WINDOW)
        self.count = 0
        self.reference = None  # champion's training distribution, None disables

    def set_reference(self, reference):
        """Swap the reference and drop state computed against the old one."""
        with self.lock:
            self.reference = reference
            self.window.clear()
            self.count = 0
        metrics.DRIFT.clear()

    def add(self, features):
        with self.lock:
            self.window.append(features)
            self.count += 1
            reference = self.reference
            if self.count % EVERY or reference is None:
                return
            frame = pd.DataFrame(list(self.window))
        for feature, ref in reference.items():
            value = psi(ref, frame[feature].to_numpy(dtype=float))
            metrics.DRIFT.labels(feature).set(value)
            if value > THRESHOLD:
                log.warning("Drift on %s: PSI %.3f > %.2f", feature, value, THRESHOLD)


monitor = DriftMonitor()
