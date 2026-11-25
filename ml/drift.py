import numpy as np

DISTRIBUTION_ARTIFACT = "training_distribution.json"


def _proportions(edges, values):
    bins = np.searchsorted(edges, values, side="right")
    return np.bincount(bins, minlength=len(edges) + 1) / len(values)


def training_distribution(X):
    """Per-feature decile bin edges and training proportions, the PSI reference."""
    distribution = {}
    for column in X.columns:
        values = X[column].to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9)))
        distribution[column] = {
            "edges": edges.tolist(),
            "proportions": _proportions(edges, values).tolist(),
        }
    return distribution


def psi(reference, values, eps=1e-4):
    """Population Stability Index of ``values`` against one feature's reference."""
    expected = np.clip(reference["proportions"], eps, None)
    actual = np.clip(_proportions(np.asarray(reference["edges"]), values), eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))
