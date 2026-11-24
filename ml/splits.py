"""Chronological split of training and validation data.

The final test set is never used to choose a model: it is
reserved for the independent final evaluation. Validation is used to
compare champion and candidate during a Continuous Training cycle.
"""

import pandas as pd

# Fraction of the training pool reserved for validation.
# Validation is always the last observations of the pool.
VALIDATION_FRACTION = 0.15


def split_train_validation(
    dataset: pd.DataFrame,
    validation_fraction: float = VALIDATION_FRACTION,
    date_column: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a chronological pool into training then validation.

    Validation is the end of the period: it is later than all
    training observations, which prevents any leakage from the future into
    the past. The final test set is not concerned by this function.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")

    if date_column not in dataset.columns:
        raise ValueError(f"Date column missing: {date_column}")

    if len(dataset) < 2:
        raise ValueError("The dataset must contain at least two rows.")

    dates = pd.to_datetime(dataset[date_column], errors="raise")

    if dates.isna().any():
        raise ValueError("The dataset contains missing dates.")

    if not dates.is_monotonic_increasing:
        raise ValueError("The dataset must be sorted by ascending date.")

    if dates.duplicated().any():
        raise ValueError("The dataset contains duplicated dates.")

    # Round down, with at least one validation row and one
    # training row so that both sets remain usable.
    validation_rows = int(len(dataset) * validation_fraction)
    validation_rows = max(1, min(validation_rows, len(dataset) - 1))

    train = dataset.iloc[:-validation_rows].copy()
    validation = dataset.iloc[-validation_rows:].copy()

    if (
        pd.to_datetime(train[date_column]).max()
        >= pd.to_datetime(validation[date_column]).min()
    ):
        raise ValueError("The training/validation split is not chronological.")

    return train, validation


def load_train_validation(
    train_path,
    validation_fraction: float = VALIDATION_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the training pool and split into training / validation.

    The final test set is deliberately not read here: it must never
    be consulted during training or during a CT cycle.
    """
    train_pool = pd.read_csv(train_path)

    return split_train_validation(
        train_pool,
        validation_fraction=validation_fraction,
    )


def load_final_test(test_path) -> pd.DataFrame:
    """Read the final test set.

    This function is reserved for the independent final evaluation, run
    after model selection. It must not be called by the
    training, comparison or promotion path.
    """
    return pd.read_csv(test_path)
