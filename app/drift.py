"""PSI drift of live prediction features against the champion training profile.

The reference is the ``training-profile.json`` artifact logged by ``ml.train``:
per feature, the bin edges and the training share of each bin.
"""

import bisect
import math
import threading
from collections import deque

DRIFT_THRESHOLD = 0.20
# shortcut: 500 time-ordered readings (10 min apart) cover ~3.5 days, compared
# against months of training data, so seasonal weather reads as drift; compare
# against a same-season reference window if that becomes a real alert source.
DRIFT_WINDOW_SIZE = 500
DRIFT_RECOMPUTE_EVERY = 100
PSI_FLOOR = 1e-4  # empty bins would make log() infinite


def population_stability_index(bin_edges, bin_shares, samples):
    """PSI of ``samples`` against the training share of each bin.

    Same bin rule (``bisect_left``) as ``ml.train.build_training_profile``.
    """
    counts = [0] * len(bin_shares)
    for sample in samples:
        counts[bisect.bisect_left(bin_edges, sample)] += 1

    psi = 0.0
    for count, expected in zip(counts, bin_shares, strict=True):
        observed = max(count / len(samples), PSI_FLOOR)
        expected = max(expected, PSI_FLOOR)
        psi += (observed - expected) * math.log(observed / expected)
    return psi


class DriftTracker:
    """Sliding window of live features; PSI per feature once the window is full."""

    def __init__(
        self, window_size=DRIFT_WINDOW_SIZE, recompute_every=DRIFT_RECOMPUTE_EVERY
    ):
        self._window_size = window_size
        self._recompute_every = recompute_every
        self._lock = threading.Lock()
        self.configure({})

    def configure(self, reference):
        """Start a new window against ``reference`` (feature -> bins); {} disables."""
        with self._lock:
            self._reference = reference
            self._buffers = {
                name: deque(maxlen=self._window_size) for name in reference
            }
            self._observations = 0

    def observe(self, values):
        """Add one prediction's features; return the PSI per feature when recomputed."""
        with self._lock:
            self._observations += 1
            for name, buffer in self._buffers.items():
                buffer.append(values[name])
            if (
                self._observations >= self._window_size
                and self._observations % self._recompute_every == 0
            ):
                return {
                    name: population_stability_index(
                        self._reference[name]["bin_edges"],
                        self._reference[name]["bin_shares"],
                        buffer,
                    )
                    for name, buffer in self._buffers.items()
                }
            return None
