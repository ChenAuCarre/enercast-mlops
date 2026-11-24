"""Tests for the chronological training / validation / final test split."""

import pandas as pd
import pytest

from ml.splits import (
    VALIDATION_FRACTION,
    load_final_test,
    load_train_validation,
    split_train_validation,
)


@pytest.fixture
def chronological_data():
    return pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=100, freq="10min").astype(str),
            "value": range(100),
        }
    )


def test_default_split_keeps_all_rows(chronological_data):
    train, validation = split_train_validation(chronological_data)

    expected_validation_rows = int(len(chronological_data) * VALIDATION_FRACTION)

    assert len(validation) == expected_validation_rows
    assert len(train) + len(validation) == len(chronological_data)


def test_validation_is_the_chronological_tail(chronological_data):
    train, validation = split_train_validation(chronological_data)

    assert train["value"].max() < validation["value"].min()
    assert (
        pd.to_datetime(train["date"]).max() < pd.to_datetime(validation["date"]).min()
    )


def test_split_preserves_row_order(chronological_data):
    train, validation = split_train_validation(chronological_data)

    assert train["value"].tolist() == list(range(len(train)))
    assert validation["value"].tolist() == list(
        range(len(train), len(chronological_data))
    )


def test_split_is_deterministic(chronological_data):
    first_train, first_validation = split_train_validation(chronological_data)
    second_train, second_validation = split_train_validation(chronological_data)

    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_validation, second_validation)


def test_split_returns_independent_copies(chronological_data):
    train, _ = split_train_validation(chronological_data)

    train.loc[train.index[0], "value"] = -1

    assert chronological_data.loc[0, "value"] == 0


@pytest.mark.parametrize("validation_fraction", [0, 1, -0.1, 1.5])
def test_invalid_validation_fraction_is_rejected(
    chronological_data,
    validation_fraction,
):
    with pytest.raises(ValueError, match="validation_fraction"):
        split_train_validation(
            chronological_data,
            validation_fraction=validation_fraction,
        )


def test_missing_date_column_is_rejected(chronological_data):
    without_date = chronological_data.drop(columns=["date"])

    with pytest.raises(ValueError, match="date"):
        split_train_validation(without_date)


def test_too_few_rows_is_rejected():
    single_row = pd.DataFrame({"date": ["2016-01-01 00:00:00"]})

    with pytest.raises(ValueError, match="two rows"):
        split_train_validation(single_row)


def test_unsorted_dates_are_rejected(chronological_data):
    unsorted_data = chronological_data.copy()
    unsorted_data.loc[[0, 5], "date"] = unsorted_data.loc[[5, 0], "date"].to_numpy()

    with pytest.raises(ValueError, match="sorted"):
        split_train_validation(unsorted_data)


def test_duplicate_dates_are_rejected(chronological_data):
    duplicated_data = chronological_data.copy()
    duplicated_data.loc[10, "date"] = duplicated_data.loc[11, "date"]

    with pytest.raises(ValueError, match="duplicated"):
        split_train_validation(duplicated_data)


def test_missing_dates_are_rejected(chronological_data):
    missing_data = chronological_data.copy()
    missing_data.loc[3, "date"] = None

    with pytest.raises(ValueError, match="missing dates"):
        split_train_validation(missing_data)


def test_at_least_one_validation_row():
    tiny_dataset = pd.DataFrame(
        {
            "date": [
                "2016-01-01 00:00:00",
                "2016-01-01 00:10:00",
                "2016-01-01 00:20:00",
            ],
            "value": [0, 1, 2],
        }
    )

    train, validation = split_train_validation(
        tiny_dataset,
        validation_fraction=0.01,
    )

    assert len(train) == 2
    assert len(validation) == 1


def test_at_least_one_training_row():
    tiny_dataset = pd.DataFrame(
        {
            "date": ["2016-01-01 00:00:00", "2016-01-01 00:10:00"],
            "value": [0, 1],
        }
    )

    train, validation = split_train_validation(
        tiny_dataset,
        validation_fraction=0.99,
    )

    assert len(train) == 1
    assert len(validation) == 1


def test_load_train_validation_reads_only_the_training_pool(tmp_path):
    train_path = tmp_path / "train.csv"

    pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=40, freq="10min").astype(str),
            "value": range(40),
        }
    ).to_csv(train_path, index=False)

    train, validation = load_train_validation(train_path)

    assert len(train) == 34
    assert len(validation) == 6
    assert (
        pd.to_datetime(train["date"]).max() < pd.to_datetime(validation["date"]).min()
    )


def test_load_final_test_returns_the_test_file_unchanged(tmp_path):
    test_path = tmp_path / "test.csv"

    test_frame = pd.DataFrame(
        {
            "date": ["2016-02-01 00:00:00", "2016-02-01 00:10:00"],
            "value": [99, 100],
        }
    )
    test_frame.to_csv(test_path, index=False)

    loaded_test = load_final_test(test_path)

    pd.testing.assert_frame_equal(loaded_test, pd.read_csv(test_path))
