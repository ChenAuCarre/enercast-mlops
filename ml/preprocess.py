import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "lights",
    "T1",
    "RH_1",
    "T2",
    "RH_2",
    "T3",
    "RH_3",
    "T4",
    "RH_4",
    "T5",
    "RH_5",
    "T6",
    "RH_6",
    "T7",
    "RH_7",
    "T8",
    "RH_8",
    "T9",
    "RH_9",
    "To",
    "Pressure",
    "RH_out",
    "Windspeed",
    "Visibility",
    "Tdewpoint",
    "hour",
    "day_of_week",
]

TARGET_COLUMN = "Appliances"


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw energy observations into model-ready features.

    Supports both the original UCI column names and the API names
    for outdoor temperature and atmospheric pressure.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    data = df.copy()

    if "date" not in data.columns:
        raise ValueError("Missing required column: date")

    # Normalize column names to match the API contract.
    rename_map = {}

    if "T_out" in data.columns:
        if "To" in data.columns:
            raise ValueError("Both T_out and To are present")
        rename_map["T_out"] = "To"

    if "Press_mm_hg" in data.columns:
        if "Pressure" in data.columns:
            raise ValueError("Both Press_mm_hg and Pressure are present")
        rename_map["Press_mm_hg"] = "Pressure"

    data = data.rename(columns=rename_map)

    dates = pd.to_datetime(
        data["date"],
        format="%Y-%m-%d %H:%M:%S",
        errors="raise",
    )

    if dates.isna().any():
        raise ValueError("Missing or invalid date values")

    data["hour"] = dates.dt.hour
    data["day_of_week"] = dates.dt.day_of_week

    # Check that all expected model features are available.
    missing_columns = [
        column for column in FEATURE_COLUMNS if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required feature columns: {missing_columns}")

    # Keep only the expected features and enforce a fixed order.
    X = data[FEATURE_COLUMNS].copy()

    # Convert values to numeric and reject invalid values.
    for column in FEATURE_COLUMNS:
        X[column] = pd.to_numeric(X[column], errors="raise")

    values = X.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Features contain missing or non-finite values")

    X = X.astype("float64")
    return X


def prepare_training_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare model features and target for supervised training.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    X = preprocess_features(df)

    y = pd.to_numeric(df[TARGET_COLUMN], errors="raise")

    if y.isna().any() or not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError("Target contains missing or non-finite values")

    y = y.astype(float)

    if (y < 0).any():
        raise ValueError("Target contains negative energy consumption")

    return X, y
