"""Champion model drift monitoring.

The deployed API compares the distribution of recent predictions with the
training distribution of the champion, using a Population Stability Index
(PSI).

The reference distribution comes from the ``training-profile.json`` artifact
logged by the training run of the champion version (see ``ml.train``); the API
downloads it on the first prediction of each champion version.

The live distribution is the sliding window of the last predictions per
feature. The drift is recomputed every ``DRIFT_WINDOW_RESET_EVERY`` predictions
and compared to ``DRIFT_THRESHOLD`` by the alerting rules.
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
DRIFT_WINDOW_RESET_EVERY = 100
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


def _relative_mean_shift(reference_mean: float, samples: Sequence[float]) -> float:
    """Drift for degenerate (constant) features: relative mean shift."""
    if not samples:
        return 0.0
    reference_mean = float(reference_mean)
    if reference_mean == 0.0:
        return 0.0
    sample_mean = sum(float(value) for value in samples) / len(samples)
    return abs(sample_mean - reference_mean) / abs(reference_mean)


def feature_drift(
    reference: Mapping[str, Any],
    samples: Sequence[float],
) -> float:
    """Return the drift of one feature against the champion training profile."""
    values = [float(value) for value in samples]
    if not values:
        return 0.0
    if not reference.get("bin_edges"):  # constant feature in training
        return _relative_mean_shift(float(reference.get("mean") or 0.0), values)
    return population_stability_index(
        reference["bin_edges"], reference["bin_shares"], values
    )


class DriftTracker:
    """Sliding-window drift tracker fed with live prediction features."""

    def __init__(
        self,
        reference: Mapping[str, Any] | None = None,
        window_size: int = DRIFT_WINDOW_SIZE,
        reset_every: int = DRIFT_WINDOW_RESET_EVERY,
    ) -> None:
        self._reference = dict(reference) if reference else None
        self._window_size = window_size
        self._reset_every = reset_every
        self._buffers: dict[str, deque] = {
            name: deque(maxlen=window_size) for name in (self._reference or {})
        }
        self._observations = 0
        self._drift: dict[str, float] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        """Drift monitoring is on when a champion training profile is set."""
        return self._reference is not None and bool(self._buffers)

    def configure(self, reference: Mapping[str, Any] | None) -> None:
        """(Re)configure the tracker with a champion training profile."""
        with self._lock:
            self._reference = dict(reference) if reference else None
            self._buffers = {
                name: deque(maxlen=self._window_size)
                for name in (self._reference or {})
            }
            self._drift = {}
            self._observations = 0

    def observe(self, values: Mapping[str, float]) -> None:
        """Append one prediction's feature vector and recompute drift on cadence."""
        with self._lock:
            self._observations += 1
            for name, value in values.items():
                buffer = self._buffers.get(name)
                if buffer is None:
                    continue
                buffer.append(float(value))

            if self._observations % self._reset_every == 0:
                self._drift = self._compute()

    def current(self) -> dict[str, float]:
        """Return the last computed per-feature drifts (empty before the first recompute)."""
        with self._lock:
            return dict(self._drift)

    def _compute(self) -> dict[str, float]:
        # Buffers exist only for features of the reference (none when disabled).
        return {
            name: feature_drift(self._reference[name], list(buffer))
            for name, buffer in self._buffers.items()
        }
