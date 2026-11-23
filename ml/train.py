import json
import subprocess
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.preprocess import prepare_training_data

TRAIN_PATH = Path("data/versions/v1/train.csv")
TEST_PATH = Path("data/versions/test/final_test.csv")

DATASET_VERSION = "v1"
TRAIN_DVC_HASH = "b2fc482fe85cd1779a0faa5f7d6de625"
TEST_DVC_HASH = "082db6d3abab0442e8721d732e1db116"

EXPERIMENT_NAME = "WattWise-energy-forecasting-final"

MODELS = {
    "baseline-linear-regression-v1": {
        "role": "baseline",
        "model_type": "LinearRegression",
        "estimator": LinearRegression(),
        "params": {},
    },
    "candidate-hist-gradient-boosting-v2": {
        "role": "candidate",
        "model_type": "HistGradientBoostingRegressor",
        "estimator": HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=300,
            max_leaf_nodes=7,
            learning_rate=0.05,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=42,
        ),
        "params": {
            "loss": "absolute_error",
            "max_iter": 300,
            "max_leaf_nodes": 7,
            "learning_rate": 0.05,
            "min_samples_leaf": 30,
            "l2_regularization": 1.0,
            "random_state": 42,
        },
    },
}


def get_git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
    }

    return predictions, metrics


def run_experiment(
    run_name,
    config,
    X_train,
    y_train,
    X_test,
    y_test,
    dataset_version=DATASET_VERSION,
    train_dvc_hash=TRAIN_DVC_HASH,
    test_dvc_hash=TEST_DVC_HASH,
):
    model = config["estimator"]

    with mlflow.start_run(run_name=run_name) as run:
        git_commit = get_git_commit()

        mlflow.log_params(config["params"])
        mlflow.log_params(
            {
                "dataset_version": dataset_version,
                "train_dvc_hash": train_dvc_hash,
                "test_dvc_hash": test_dvc_hash,
                "model_type": config["model_type"],
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "feature_count": X_train.shape[1],
            }
        )

        mlflow.set_tags(
            {
                "git_commit": git_commit,
                "model_role": config["role"],
                "evaluation_split": "final_test_exploratory",
                "promotion_eligible": "false",
            }
        )

        model.fit(X_train, y_train)
        predictions, metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_metrics(metrics)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_train.head(3),
            skops_trusted_types=[
                "sklearn.tree._tree.Tree",
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            metrics_path = temp_path / "metrics.json"
            metrics_path.write_text(
                json.dumps(metrics, indent=4),
                encoding="utf-8",
            )

            predictions_path = temp_path / "predictions.csv"
            pd.DataFrame(
                {
                    "actual": y_test.to_numpy(),
                    "predicted": predictions,
                }
            ).to_csv(predictions_path, index=False)

            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact(str(predictions_path))

        print(f"\n--- {run_name} ---")
        print(f"Run ID: {run.info.run_id}")
        print(f"Model ID: {model_info.model_id}")
        print(f"MAE: {metrics['mae']:.3f}")
        print(f"RMSE: {metrics['rmse']:.3f}")
        print(f"R²: {metrics['r2']:.3f}")

        return {
            "run_id": run.info.run_id,
            "model_id": model_info.model_id,
            "metrics": metrics,
        }


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train, y_train = prepare_training_data(train_df)
    X_test, y_test = prepare_training_data(test_df)

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and test feature columns do not match")

    mlflow.set_experiment(EXPERIMENT_NAME)

    results = {}

    for run_name, config in MODELS.items():
        results[run_name] = run_experiment(
            run_name,
            config,
            X_train,
            y_train,
            X_test,
            y_test,
        )

    print("\n--- MODEL COMPARISON ---")

    for run_name, result in results.items():
        metrics = result["metrics"]
        print(
            f"{run_name}: "
            f"MAE={metrics['mae']:.3f}, "
            f"RMSE={metrics['rmse']:.3f}, "
            f"R²={metrics['r2']:.3f}"
        )

    best_model = min(
        results,
        key=lambda name: results[name]["metrics"]["mae"],
    )

    print(f"\nBest model according to MAE: {best_model}")
    print(f"Run ID: {results[best_model]['run_id']}")

    print(
        "\nNote: this comparison identifies the lowest-MAE model. "
        "It does not automatically approve or promote it."
    )


if __name__ == "__main__":
    main()
