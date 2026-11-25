"""Champion model drift monitoring.

The deployed API compares the distribution of recent predictions with the
training distribution of the champion, using a Population Stability Index
(PSI).

The reference distribution comes from the ``training-profile.json`` artifact
logged by the training run of the champion version (see ``ml.train``); the API
downloads it each time a champion is loaded (startup and ``/refresh``).

The live distribution is the sliding window of the last predictions per
feature. Once the window is full, the drift is recomputed every
``DRIFT_RECOMPUTE_EVERY`` predictions; features above ``DRIFT_THRESHOLD`` are
logged by the API and alerted on by Grafana.
"""

from __future__ import annotations

import bisect
import math
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

DRIFT_THRESHOLD = 0.20
# shortcut: 500 time-ordered readings (10 min apart) cover ~3.5 days, compared
# against months of training data, so seasonal weather reads as drift; compare
# against a same-season reference window if that becomes a real alert source.
DRIFT_WINDOW_SIZE = 500
DRIFT_RECOMPUTE_EVERY = 100
PSI_FLOOR = 1e-4  # empty bins would make log() infinite


def population_stability_index(
    bin_edges: Sequence[float],
    bin_shares: Sequence[float],
    samples: Sequence[float],
) -> float:
    """Compute the PSI of ``samples`` against the training share of each bin.

    Bin ``i`` holds the values in ``(bin_edges[i-1], bin_edges[i]]``, the same
    rule (``bisect_left``) ``ml.train.build_training_profile`` used to compute
    ``bin_shares``.
    """
    if not samples:
        return 0.0

    counts = [0] * len(bin_shares)
    for sample in samples:
        counts[bisect.bisect_left(bin_edges, float(sample))] += 1

    psi_value = 0.0
    for count, expected in zip(counts, bin_shares, strict=True):
        observed = max(count / len(samples), PSI_FLOOR)
        expected = max(float(expected), PSI_FLOOR)
        psi_value += (observed - expected) * math.log(observed / expected)

    return psi_value


class DriftTracker:
    """Sliding-window drift tracker fed with live prediction features."""

    def __init__(
        self,
        reference: Mapping[str, Any] | None = None,
        window_size: int = DRIFT_WINDOW_SIZE,
        recompute_every: int = DRIFT_RECOMPUTE_EVERY,
    ) -> None:
        self._window_size = window_size
        self._recompute_every = recompute_every
        self._lock = threading.Lock()
        self._reset(reference)

    def configure(self, reference: Mapping[str, Any] | None) -> None:
        """(Re)configure the tracker with a champion training profile."""
        with self._lock:
            self._reset(reference)

    def _reset(self, reference: Mapping[str, Any] | None) -> None:
        self._reference = dict(reference) if reference else None
        self._buffers: dict[str, deque] = {
            name: deque(maxlen=self._window_size) for name in (self._reference or {})
        }
        self._observations = 0

    def observe(self, values: Mapping[str, float]) -> dict[str, float] | None:
        """Append one prediction's features; return the drifts when recomputed."""
        with self._lock:
            self._observations += 1
            for name, value in values.items():
                buffer = self._buffers.get(name)
                if buffer is None:
                    continue
                buffer.append(float(value))

            # A partial window gives noisy PSI: wait until it is full.
            if (
                self._observations >= self._window_size
                and self._observations % self._recompute_every == 0
            ):
                return self._compute()
            return None

    def _compute(self) -> dict[str, float]:
        # Buffers exist only for features of the reference (none when disabled).
        return {
            name: population_stability_index(
                self._reference[name]["bin_edges"],
                self._reference[name]["bin_shares"],
                list(buffer),
            )
            for name, buffer in self._buffers.items()
        }
