import json
import subprocess
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

EXPERIMENT_NAME = "WattWise-energy-forecasting-final"

# Version used by the fallback entry point `python -m ml.train`.
DATASET_VERSION = "v1"

MODELS = {
    "baseline-linear-regression-v1": {
        "role": "baseline",
        "model_type": "LinearRegression",
        "params": {},
    },
    "candidate-hist-gradient-boosting-v2": {
        "role": "candidate",
        "model_type": "HistGradientBoostingRegressor",
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


def build_estimator(model_type: str, params: dict | None = None):
    """Instantiate a fresh estimator on each call.

    The sklearn estimators are mutable: the same object reused across runs
    would be reused after ``fit`` (state kept), skewing the scores. Each
    experiment must therefore get its own dedicated instance.
    """
    if model_type == "LinearRegression":
        return LinearRegression()

    if model_type == "HistGradientBoostingRegressor":
        return HistGradientBoostingRegressor(**(params or {}))

    raise ValueError(f"Unknown estimator type: {model_type}")


def get_git_commit():
    """Return the Git commit hash, or "unknown" if Git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def is_git_dirty():
    """Indicate whether the working tree contains uncommitted modifications."""
    try:
        status_output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

    return "true" if status_output.strip() else "false"


def evaluate_model(model, X_eval, y_eval):
    """Compute predictions and metrics on an evaluation set."""
    predictions = model.predict(X_eval)

    metrics = {
        "mae": float(mean_absolute_error(y_eval, predictions)),
        "rmse": float(mean_squared_error(y_eval, predictions) ** 0.5),
        "r2": float(r2_score(y_eval, predictions)),
    }

    return predictions, metrics


def build_training_profile(X_train, dataset_version):
    """Summarize the training features for production drift monitoring.

    The API downloads this artifact from the champion's run and compares the
    distribution of live predictions with it. Bin edges and training shares
    are stored so the API can compute a population stability index (PSI) without access to the
    training data, which is never shipped in the container.
    """
    quantiles = [i / 10 for i in range(11)]
    features = {}

    for name in X_train.columns:
        series = X_train[name].astype(float)
        deciles = series.quantile(quantiles).tolist()
        # Bins cut at the unique interior deciles, with the real training share of
        # each bin: repeated values (lights is mostly 0) make the bins unequal.
        # Bin rule must match app.drift.population_stability_index (bisect_left).
        edges = sorted(set(deciles[1:-1])) if series.min() < series.max() else []
        shares = np.bincount(
            np.searchsorted(edges, series, side="left"), minlength=len(edges) + 1
        ) / len(series)
        features[name] = {
            "bin_edges": [float(value) for value in edges],
            "bin_shares": [float(value) for value in shares],
        }

    return {
        "dataset_version": dataset_version,
        "features": features,
    }


def run_experiment(run_name, config, data):
    """Train a model and log the experiment to MLflow.

    Only training and validation are used here. The final test is neither
    read nor evaluated: it remains reserved for independent final evaluation
    (`python -m ml.pipeline --final-evaluation`).
    """
    model = build_estimator(config["model_type"], config.get("params"))
    dataset_version = data["dataset_version"]

    with mlflow.start_run(run_name=run_name) as run:
        git_commit = get_git_commit()
        git_dirty = is_git_dirty()

        mlflow.log_params(config["params"])
        mlflow.log_params(
            {
                "dataset_version": dataset_version,
                "train_dvc_hash": data["train_dvc_hash"],
                "git_commit": git_commit,
                "git_dirty": git_dirty,
                "model_type": config["model_type"],
                "train_rows": len(data["X_train"]),
                "validation_rows": len(data["X_validation"]),
                "feature_count": data["X_train"].shape[1],
                "validation_fraction": data["validation_fraction"],
                "train_period_start": data["train_period"][0],
                "train_period_end": data["train_period"][1],
                "validation_period_start": data["validation_period"][0],
                "validation_period_end": data["validation_period"][1],
            }
        )

        mlflow.set_tags(
            {
                "git_commit": git_commit,
                "git_dirty": git_dirty,
                "model_role": config["role"],
                "selection_split": "validation",
                "final_test_usage": "separate_final_evaluation",
                "final_test_used_for_selection": "false",
                "promotion_eligible": "true",
            }
        )

        model.fit(data["X_train"], data["y_train"])

        validation_predictions, validation_metrics = evaluate_model(
            model,
            data["X_validation"],
            data["y_validation"],
        )

        mlflow.log_metrics(
            {
                "validation_mae": validation_metrics["mae"],
                "validation_rmse": validation_metrics["rmse"],
                "validation_r2": validation_metrics["r2"],
            }
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=data["X_train"].head(3),
            skops_trusted_types=[
                "sklearn.tree._tree.Tree",
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            metrics_path = temp_path / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "dataset_version": dataset_version,
                        "evaluation_split": "validation",
                        "validation": validation_metrics,
                    },
                    indent=4,
                ),
                encoding="utf-8",
            )

            validation_predictions_path = temp_path / "validation_predictions.csv"
            pd.DataFrame(
                {
                    "actual": data["y_validation"].to_numpy(),
                    "predicted": validation_predictions,
                }
            ).to_csv(validation_predictions_path, index=False)

            profile_path = temp_path / "training-profile.json"
            profile_path.write_text(
                json.dumps(
                    build_training_profile(data["X_train"], dataset_version),
                    indent=2,
                ),
                encoding="utf-8",
            )

            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact(str(validation_predictions_path))
            mlflow.log_artifact(str(profile_path))

        print(f"\n--- {run_name} ---")
        print(f"Run ID: {run.info.run_id}")
        print(f"Model ID: {model_info.model_id}")
        print(
            f"Validation MAE: {validation_metrics['mae']:.3f} | "
            f"RMSE: {validation_metrics['rmse']:.3f} | "
            f"R²: {validation_metrics['r2']:.3f}"
        )

        return {
            "run_id": run.info.run_id,
            "model_id": model_info.model_id,
            "metrics": validation_metrics,
        }


def main():
    # Local import to avoid an import cycle between ml.pipeline and ml.train.
    from ml.pipeline import load_data

    data = load_data(DATASET_VERSION)

    mlflow.set_experiment(EXPERIMENT_NAME)

    results = {}

    for run_name, config in MODELS.items():
        results[run_name] = run_experiment(run_name, config, data)

    print("\n--- MODEL COMPARISON (validation) ---")

    for run_name, result in results.items():
        metrics = result["metrics"]
        print(
            f"{run_name}: "
            f"validation MAE={metrics['mae']:.3f}, "
            f"validation RMSE={metrics['rmse']:.3f}, "
            f"validation R²={metrics['r2']:.3f}"
        )

    best_model = min(
        results,
        key=lambda name: results[name]["metrics"]["mae"],
    )

    print(f"\nBest model according to validation MAE: {best_model}")
    print(f"Run ID: {results[best_model]['run_id']}")

    print(
        "\nNote: this comparison identifies the lowest validation-MAE model. "
        "It does not automatically approve or promote it."
    )


if __name__ == "__main__":
    main()
