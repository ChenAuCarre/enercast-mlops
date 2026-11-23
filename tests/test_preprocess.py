import pandas as pd
import pytest

from ml.preprocess import (
    FEATURE_COLUMNS,
    prepare_training_data,
    preprocess_features,
)


@pytest.fixture
def raw_data():
    return pd.DataFrame(
        {
            "date": ["2016-01-11 17:00:00", "2016-01-11 17:10:00"],
            "Appliances": [60, 70],
            "lights": [10, 20],
            "T1": [20.0, 20.1],
            "RH_1": [30.0, 31.0],
            "T2": [19.0, 19.1],
            "RH_2": [40.0, 41.0],
            "T3": [21.0, 21.1],
            "RH_3": [35.0, 36.0],
            "T4": [20.0, 20.1],
            "RH_4": [32.0, 33.0],
            "T5": [19.0, 19.1],
            "RH_5": [45.0, 46.0],
            "T6": [18.0, 18.1],
            "RH_6": [50.0, 51.0],
            "T7": [20.0, 20.1],
            "RH_7": [35.0, 36.0],
            "T8": [22.0, 22.1],
            "RH_8": [40.0, 41.0],
            "T9": [19.0, 19.1],
            "RH_9": [45.0, 46.0],
            "T_out": [10.0, 10.1],
            "Press_mm_hg": [760.0, 760.1],
            "RH_out": [60.0, 61.0],
            "Windspeed": [2.0, 2.1],
            "Visibility": [40.0, 40.1],
            "Tdewpoint": [5.0, 5.1],
            "rv1": [0.1, 0.2],
            "rv2": [0.3, 0.4],
        }
    )


def test_feature_count_and_order(raw_data):
    X = preprocess_features(raw_data)

    assert len(FEATURE_COLUMNS) == 27
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.shape == (2, 27)


def test_temporal_features_are_extracted(raw_data):
    X = preprocess_features(raw_data)

    assert X["hour"].tolist() == [17, 17]
    assert X["day_of_week"].tolist() == [0, 0]


def test_uci_and_api_column_names_are_equivalent(raw_data):
    X_uci = preprocess_features(raw_data)

    api_data = raw_data.drop(
        columns=["T_out", "Press_mm_hg", "Appliances", "rv1", "rv2"]
    ).rename(
        columns={
            "T_out": "To",
            "Press_mm_hg": "Pressure",
        }
    )

    # Recreate the API input with the expected names.
    api_data["To"] = raw_data["T_out"]
    api_data["Pressure"] = raw_data["Press_mm_hg"]

    X_api = preprocess_features(api_data)

    pd.testing.assert_frame_equal(X_uci, X_api)


def test_random_variables_are_excluded(raw_data):
    X = preprocess_features(raw_data)

    assert "rv1" not in X.columns
    assert "rv2" not in X.columns
    assert "Appliances" not in X.columns


def test_missing_feature_is_rejected(raw_data):
    invalid_data = raw_data.drop(columns=["T1"])

    with pytest.raises(ValueError, match="Missing required feature"):
        preprocess_features(invalid_data)


def test_invalid_date_is_rejected(raw_data):
    invalid_data = raw_data.copy()
    invalid_data.loc[0, "date"] = "invalid-date"

    with pytest.raises(ValueError):
        preprocess_features(invalid_data)


def test_non_finite_feature_is_rejected(raw_data):
    invalid_data = raw_data.copy()
    invalid_data["lights"] = invalid_data["lights"].astype(float)
    invalid_data.loc[0, "lights"] = float("inf")

    with pytest.raises(ValueError, match="non-finite"):
        preprocess_features(invalid_data)


def test_training_data_is_separated(raw_data):
    X, y = prepare_training_data(raw_data)

    assert X.shape == (2, 27)
    assert y.tolist() == [60.0, 70.0]
    assert y.name == "Appliances"


def test_missing_target_is_rejected(raw_data):
    invalid_data = raw_data.drop(columns=["Appliances"])

    with pytest.raises(ValueError, match="Missing target"):
        prepare_training_data(invalid_data)


def test_negative_target_is_rejected(raw_data):
    invalid_data = raw_data.copy()
    invalid_data.loc[0, "Appliances"] = -10

    with pytest.raises(ValueError, match="negative"):
        prepare_training_data(invalid_data)
