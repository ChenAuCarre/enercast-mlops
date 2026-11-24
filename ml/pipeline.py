import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import yaml

from ml.policy import evaluate_candidate, evaluate_initial_models
from ml.preprocess import FEATURE_COLUMNS, prepare_training_data
from ml.registry import (
    NoChampionRegistered,
    get_champion,
    load_champion_model,
    promote_model,
    register_model,
)
from ml.splits import (
    VALIDATION_FRACTION,
    load_final_test,
    load_train_validation,
)
from ml.train import (
    EXPERIMENT_NAME,
    MODELS,
    evaluate_model,
    run_experiment,
)

FINAL_TEST_METADATA_PATH = "data/versions/test/final_test.csv.dvc"
FINAL_TEST_PATH = "data/versions/test/final_test.csv"

DATASET_METADATA = {
    "v1": {
        "metadata_path": "data/versions/v1/train.csv.dvc",
        "file": "data/versions/v1/train.csv",
    },
    "v2": {
        "metadata_path": "data/versions/v2/train.csv.dvc",
        "file": "data/versions/v2/train.csv",
    },
}


def _period(frame):
    """Return the period covered by a dataset (start, end)."""
    dates = pd.to_datetime(frame["date"])
    return str(dates.min()), str(dates.max())


def verify_dvc_hash(path, expected_hash):
    """Verify the DVC (MD5) checksum of a versioned file before any use.

    DVC-versioned files are consumed by the pipeline: before reading them
    to train or evaluate, the expected checksum is rebuilt and
    any run is refused if it differs.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Versioned file not found: {path}")

    # DVC 3 "hash: md5" is the plain MD5 of the file bytes.
    with file_path.open("rb") as stream:
        actual_hash = hashlib.file_digest(stream, "md5").hexdigest()

    if actual_hash != expected_hash:
        raise ValueError(
            f"Hash DVC mismatch for {path}: expected {expected_hash}, got {actual_hash}. "
            "The file has changed since DVC versioning; no run is "
            "allowed until the checksum matches."
        )

    return actual_hash


def read_dvc_metadata(metadata_path, expected_path):
    """Read the MD5 checksum and output path from the DVC metadata.

    The metadata file (YAML produced by DVC) contains an ``outs`` key
    listing the outputs. Exactly one output is expected, with an
    ``md5`` checksum: the values there are the source of truth for versioning.
    The declared path must match the expected file.
    """
    metadata_file = Path(metadata_path)

    if not metadata_file.exists():
        raise FileNotFoundError(f"DVC metadata not found: {metadata_path}")

    with metadata_file.open(encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream) or {}

    outs = metadata.get("outs") or []

    if len(outs) != 1:
        raise ValueError(
            f"Invalid DVC metadata for {metadata_path}: "
            f"expected a single 'outs', found {len(outs)}."
        )

    out = outs[0]
    algorithm = out.get("hash") or "md5"
    checksum = out.get("md5")

    if algorithm != "md5" or not checksum:
        raise ValueError(
            f"Invalid DVC metadata for {metadata_path}: "
            f"expected MD5 checksum, got hash={algorithm!r} md5={checksum!r}."
        )

    declared_path = out.get("path")

    if not declared_path:
        raise ValueError(
            f"Invalid DVC metadata for {metadata_path}: output path missing."
        )

    declared_file = (
        Path(declared_path)
        if Path(declared_path).is_absolute()
        else metadata_file.parent / declared_path
    )

    if declared_file != Path(expected_path):
        raise ValueError(
            f"Inconsistent output path in the metadata of {metadata_path}: "
            f"expected {expected_path}, declared {declared_file}."
        )

    return str(declared_file), checksum


def load_data(dataset_version):
    """Load training and validation for a training cycle.

    The final test is deliberately not read here. This path is used for
    training, champion/candidate comparison and promotion:
    it must never access ``final_test``. The final evaluation goes through
    exclusivement par :func:`run_final_evaluation`.
    """
    dataset_metadata = DATASET_METADATA[dataset_version]

    train_path, train_hash = read_dvc_metadata(
        dataset_metadata["metadata_path"],
        dataset_metadata["file"],
    )

    verify_dvc_hash(train_path, train_hash)

    train_df, validation_df = load_train_validation(train_path)

    X_train, y_train = prepare_training_data(train_df)
    X_validation, y_validation = prepare_training_data(validation_df)

    feature_columns = list(X_train.columns)

    if list(X_validation.columns) != feature_columns:
        raise ValueError("Validation columns do not match training columns.")

    return {
        "dataset_version": dataset_version,
        "train_dvc_hash": train_hash,
        "validation_fraction": VALIDATION_FRACTION,
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "train_period": _period(train_df),
        "validation_period": _period(validation_df),
    }


def load_final_test_data():
    """Load the final test set.

    This function is reserved for the independent final evaluation, run
    after model selection. It is called by no promotion or
    Continuous Training path.
    """
    final_test_path, final_test_hash = read_dvc_metadata(
        FINAL_TEST_METADATA_PATH,
        FINAL_TEST_PATH,
    )

    verify_dvc_hash(final_test_path, final_test_hash)

    test_df = load_final_test(final_test_path)
    X_test, y_test = prepare_training_data(test_df)

    if list(X_test.columns) != FEATURE_COLUMNS:
        raise ValueError("Final test columns do not match the expected schema.")

    return X_test, y_test, _period(test_df), final_test_hash


def run_v1(force=False):
    """Initial training (v1), then promotion of the reference model.

    By default, refuses to run if a champion already exists: rerunning
    v1 would move the ``champion`` alias to the v1 baseline and overwrite a
    more recent champion (typically the v2 candidate). The flag
    ``force=True`` (CLI : ``--force-initial-training``) autorise
    explicitly allows this overwrite.
    """
    print("\n=== INITIAL TRAINING: VERSION 1 ===")

    if not force:
        try:
            existing_champion = get_champion()
        except NoChampionRegistered:
            existing_champion = None

        if existing_champion is not None:
            raise RuntimeError(
                "Refusing to run initial training: champion version "
                f"{existing_champion.version} already exists. Re-running v1 "
                "would replace it. Use --force-initial-training to override."
            )
    else:
        print(
            "Warning: --force-initial-training enabled; "
            "the current champion may be replaced by the v1 baseline."
        )

    data = load_data("v1")
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = "baseline-linear-regression-v1"
    result = run_experiment(run_name, MODELS[run_name], data)

    decision = evaluate_initial_models({run_name: result})
    print("\nInitial decision:", decision)

    if not decision["promote"]:
        raise RuntimeError(decision["reason"])

    registered = register_model(result["model_id"])
    champion = promote_model(registered.version)

    print(f"\nInitial champion: version {champion.version}")


def run_v2():
    print("\n=== CONTINUOUS TRAINING: VERSION 2 ===")

    mlflow.set_experiment(EXPERIMENT_NAME)

    current_champion = get_champion()
    print(f"Current champion: version {current_champion.version}")

    data = load_data("v2")

    champion_model = load_champion_model()
    _, champion_metrics = evaluate_model(
        champion_model,
        data["X_validation"],
        data["y_validation"],
    )

    candidate_configs = {
        "candidate-hist-gradient-boosting-v2": MODELS[
            "candidate-hist-gradient-boosting-v2"
        ],
    }

    results = {}

    for run_name, config in candidate_configs.items():
        results[run_name] = run_experiment(run_name, config, data)

    decisions = {}

    for run_name, result in results.items():
        decisions[run_name] = evaluate_candidate(
            champion_metrics,
            result["metrics"],
        )

    print("\nChampion validation MAE:", round(champion_metrics["mae"], 3))

    for run_name, result in results.items():
        print(
            f"{run_name} validation MAE:",
            round(result["metrics"]["mae"], 3),
        )
        print("Decision:", decisions[run_name])

    eligible = [
        run_name for run_name, decision in decisions.items() if decision["promote"]
    ]

    selected_name = (
        min(
            eligible,
            key=lambda name: results[name]["metrics"]["mae"],
        )
        if eligible
        else None
    )

    decision_run_name = f"ct-decision-{data['dataset_version']}"

    with mlflow.start_run(run_name=decision_run_name):
        mlflow.log_params(
            {
                "dataset_version": data["dataset_version"],
                "previous_champion_version": current_champion.version,
                "candidate_count": len(results),
                "selection_split": "validation",
            }
        )

        mlflow.log_metrics(
            {
                "champion_validation_mae": champion_metrics["mae"],
                **{
                    f"{name}_validation_mae": result["metrics"]["mae"]
                    for name, result in results.items()
                },
            }
        )

        mlflow.set_tags(
            {
                "promotion_decision": str(selected_name is not None),
                "selected_candidate": selected_name or "none",
                "selection_split": "validation",
                "final_test_used_for_selection": "false",
                "decision_reason": (
                    decisions[selected_name]["reason"]
                    if selected_name
                    else "No candidate passed the promotion policy."
                ),
            }
        )

    if selected_name:
        selected_result = results[selected_name]
        registered = register_model(selected_result["model_id"])
        champion = promote_model(registered.version)

        print(f"\nNew champion: {selected_name}, version {champion.version}")
    else:
        print(
            "\nNo candidate passed. "
            f"Champion version {current_champion.version} is retained."
        )


def run_final_evaluation(dataset_version="v2"):
    """Evaluate the current champion on the final test.

    This path is strictly separate from training and promotion:
    it modifies neither the champion nor the registry. It is meant to be run
    manually after model selection, to produce the
    rapport.
    """
    print("\n=== FINAL EVALUATION (REPORTING ONLY) ===")

    mlflow.set_experiment(EXPERIMENT_NAME)

    champion = get_champion()
    champion_model = load_champion_model()

    X_test, y_test, test_period, final_test_hash = load_final_test_data()

    predictions, metrics = evaluate_model(champion_model, X_test, y_test)

    print(f"Evaluated champion: version {champion.version}")
    print(f"Final test rows: {len(X_test)}")
    print(
        f"Final test MAE: {metrics['mae']:.3f} | "
        f"RMSE: {metrics['rmse']:.3f} | "
        f"R²: {metrics['r2']:.3f}"
    )

    with mlflow.start_run(run_name=f"final-evaluation-{dataset_version}"):
        mlflow.log_params(
            {
                "dataset_version": dataset_version,
                "evaluated_model_version": champion.version,
                "test_dvc_hash": final_test_hash,
                "test_rows": len(X_test),
                "test_period_start": test_period[0],
                "test_period_end": test_period[1],
            }
        )

        mlflow.log_metrics(
            {
                "final_test_mae": metrics["mae"],
                "final_test_rmse": metrics["rmse"],
                "final_test_r2": metrics["r2"],
            }
        )

        mlflow.set_tags(
            {
                "evaluation_split": "final_test",
                "final_test_usage": "reporting_only",
                "final_test_used_for_selection": "false",
                "promotion_eligible": "false",
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            metrics_path = temp_path / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "dataset_version": dataset_version,
                        "evaluated_model_version": champion.version,
                        "evaluation_split": "final_test",
                        "final_test": metrics,
                    },
                    indent=4,
                ),
                encoding="utf-8",
            )

            predictions_path = temp_path / "final_test_predictions.csv"
            pd.DataFrame(
                {
                    "actual": y_test.to_numpy(),
                    "predicted": predictions,
                }
            ).to_csv(predictions_path, index=False)

            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact(str(predictions_path))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Training/promotion (validation) or independent final "
            "evaluation (final test)."
        )
    )

    parser.add_argument(
        "--dataset-version",
        choices=["v1", "v2"],
        help="Training pool version (training/promotion path).",
    )
    parser.add_argument(
        "--final-evaluation",
        action="store_true",
        help=(
            "Evaluate the champion on the final test, without promotion. "
            "Run after model selection."
        ),
    )
    parser.add_argument(
        "--eval-dataset-version",
        choices=["v1", "v2"],
        default="v2",
        help="Reference version logged during the final evaluation.",
    )
    parser.add_argument(
        "--force-initial-training",
        action="store_true",
        help=(
            "Allow run_v1 to replace an existing champion. "
            "Use only after a deliberate reset."
        ),
    )

    args = parser.parse_args()

    if args.final_evaluation:
        run_final_evaluation(args.eval_dataset_version)
        return

    if args.dataset_version == "v1":
        run_v1(force=args.force_initial_training)
        return

    if args.dataset_version == "v2":
        run_v2()
        return

    parser.error("Specify --dataset-version {v1,v2} or --final-evaluation.")


if __name__ == "__main__":
    main()
