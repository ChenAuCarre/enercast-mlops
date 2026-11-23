from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path("data/raw/energydata_complete.csv")

V1_OUTPUT_PATH = Path("data/versions/v1/train.csv")
V2_OUTPUT_PATH = Path("data/versions/v2/train.csv")
TEST_OUTPUT_PATH = Path("data/versions/test/final_test.csv")

EXPECTED_COLUMNS = 29


def main():
    # Load and validate the raw dataset
    df = pd.read_csv(RAW_DATA_PATH)

    if df.shape[1] != EXPECTED_COLUMNS:
        raise ValueError(
            f"Expected {EXPECTED_COLUMNS} columns, found {df.shape[1]}"
        )

    if "date" not in df.columns or "Appliances" not in df.columns:
        raise ValueError("Required columns 'date' and 'Appliances' are missing")

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)

    if df["date"].isna().any():
        raise ValueError("Missing timestamps detected")

    if df["date"].duplicated().any():
        raise ValueError("Duplicate timestamps detected")

    intervals = df["date"].diff().dropna()
    if not (intervals == pd.Timedelta(minutes=10)).all():
        raise ValueError("The dataset does not have regular 10-minute intervals")

    if df["Appliances"].isna().any():
        raise ValueError("Missing target values detected")

    # Chronological split: 60% / 20% / 20%
    n = len(df)
    idx_v1 = int(n * 0.60)
    idx_v2 = int(n * 0.80)

    train_v1 = df.iloc[:idx_v1].copy()
    train_v2 = df.iloc[:idx_v2].copy()
    final_test = df.iloc[idx_v2:].copy()

    # Create output directories
    for path in [V1_OUTPUT_PATH, V2_OUTPUT_PATH, TEST_OUTPUT_PATH]:
        path.parent.mkdir(parents=True, exist_ok=True)

    # Save raw partitions
    train_v1.to_csv(V1_OUTPUT_PATH, index=False)
    train_v2.to_csv(V2_OUTPUT_PATH, index=False)
    final_test.to_csv(TEST_OUTPUT_PATH, index=False)

    # Verify split integrity
    assert len(train_v1) == idx_v1
    assert len(train_v2) == idx_v2
    assert len(final_test) == n - idx_v2
    assert train_v1["date"].max() < final_test["date"].min()
    assert train_v2["date"].max() < final_test["date"].min()

    print("Dataset partitioning completed successfully.")
    print(f"Total observations: {n}")
    print(
        f"v1 training: {len(train_v1)} rows | "
        f"{train_v1['date'].min()} -> {train_v1['date'].max()}"
    )
    print(
        f"v2 training: {len(train_v2)} rows | "
        f"{train_v2['date'].min()} -> {train_v2['date'].max()}"
    )
    print(
        f"Final test: {len(final_test)} rows | "
        f"{final_test['date'].min()} -> {final_test['date'].max()}"
    )
    print("Chronological separation: PASSED")
    print("Final test isolation: PASSED")


if __name__ == "__main__":
    main()
