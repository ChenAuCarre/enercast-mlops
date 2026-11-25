import logging
import threading
from collections import deque

import pandas as pd

from app import metrics
from ml.drift import psi

WINDOW, EVERY, THRESHOLD = 500, 100, 0.20

log = logging.getLogger("app.drift")


class DriftMonitor:
    """PSI of the last WINDOW predictions' features, recomputed every EVERY."""

    def __init__(self):
        self.lock = threading.Lock()
        self.window = deque(maxlen=WINDOW)
        self.count = 0
        self.reference = None  # champion's training distribution, None disables

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
