"""Champion model drift monitoring.

The deployed API compares the distribution of recent predictions with the
training distribution of the champion, using a Population Stability Index
(PSI).

The reference distribution comes from the ``training-profile.json`` artifact
logged by the training run of the champion version (see ``ml.train``); the API
downloads it when the champion is loaded.

The live distribution is the sliding window of the last predictions per
feature. The drift is recomputed every ``DRIFT_WINDOW_RESET_EVERY`` predictions
and compared to ``DRIFT_THRESHOLD``.
"""

from __future__ import annotations

import bisect
import math
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

DRIFT_THRESHOLD = 0.20
DRIFT_WINDOW_SIZE = 500
DRIFT_WINDOW_RESET_EVERY = 100
DRIFT_BIN_COUNT = 10
PSI_EPSILON = 1e-12


def population_stability_index(
    reference_deciles: Sequence[float],
    samples: Sequence[float],
    bin_count: int = DRIFT_BIN_COUNT,
) -> float:
    """Compute the PSI between a reference distribution and a sample.

    ``reference_deciles`` summarises the reference distribution with its sorted
    deciles. The comparison uses ``bin_count`` equal-probability bins delimited
    by the interior deciles (min and max excluded) and the ``-inf``/``+inf`` sentinels.
    The reference mass of every bin is ``1 / bin_count``.
    """
    values = sorted({float(value) for value in reference_deciles})
    if len(values) < 2 or not samples:
        return 0.0

    # Interior cut points only (p10..p90): the first/last deciles are min/max.
    internal = values[1:-1][: max(bin_count - 1, 0)]
    bounds = [float("-inf")] + internal + [float("inf")]
    bins = len(bounds) - 1
    if bins < 1:
        return 0.0

    counts = [0] * bins
    n = len(samples)
    for sample in samples:
        index = bisect.bisect_left(bounds, float(sample)) - 1
        index = max(index, 0)
        index = min(index, bins - 1)
        counts[index] += 1

    psi_value = 0.0
    reference_mass = 1.0 / bins
    for count in counts:
        observed_mass = count / n
        if observed_mass <= 0.0 or reference_mass <= 0.0:
            continue
        psi_value += (observed_mass - reference_mass) * math.log(
            observed_mass / reference_mass
        )

    if not math.isfinite(psi_value):
        return DRIFT_THRESHOLD

    return float(psi_value)


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
    deciles = [float(value) for value in reference.get("deciles") or []]
    values = [float(value) for value in samples]
    if len(deciles) < 2 or not values:
        return 0.0

    unique_deciles = sorted({value for value in deciles})
    if len(unique_deciles) <= 1:
        return float(_relative_mean_shift(float(reference.get("mean") or 0.0), values))

    return float(population_stability_index(deciles, values))


class DriftTracker:
    """Sliding-window drift tracker fed with live prediction features."""

    def __init__(
        self,
        reference: Mapping[str, Any] | None = None,
        threshold: float = DRIFT_THRESHOLD,
        window_size: int = DRIFT_WINDOW_SIZE,
        reset_every: int = DRIFT_WINDOW_RESET_EVERY,
    ) -> None:
        self._reference = dict(reference) if reference else None
        self._threshold = threshold
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
        if not self.enabled or self._reference is None:
            return {}

        drifts: dict[str, float] = {}
        for name, buffer in self._buffers.items():
            samples = list(buffer)
            drifts[name] = feature_drift(self._reference.get(name) or {}, samples)

        return drifts
