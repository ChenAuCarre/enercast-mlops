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

EXPERIMENT_NAME = "WattWise-energy-forecasting-final"

# Version utilisée par l'entrée de secours `python -m ml.train`.
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
    """Instancie un estimateur frais à chaque appel.

    Les estimateurs sklearn sont mutables : un même objet réutilisé entre deux
    runs serait réutilisé après ``fit`` (état conservé), faussant les scores.
    Chaque expérience doit donc disposer de son instance dédiée.
    """
    if model_type == "LinearRegression":
        return LinearRegression()

    if model_type == "HistGradientBoostingRegressor":
        return HistGradientBoostingRegressor(**(params or {}))

    raise ValueError(f"Type d'estimateur inconnu : {model_type}")


def get_git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def evaluate_model(model, X_eval, y_eval):
    """Calcule les prédictions et les métriques sur un jeu d'évaluation."""
    predictions = model.predict(X_eval)

    metrics = {
        "mae": float(mean_absolute_error(y_eval, predictions)),
        "rmse": float(mean_squared_error(y_eval, predictions) ** 0.5),
        "r2": float(r2_score(y_eval, predictions)),
    }

    return predictions, metrics


def run_experiment(run_name, config, data):
    """Entraîne un modèle et journalise l'expérience dans MLflow.

    Seuls l'entraînement et la validation sont utilisés ici. Le jeu de test
    final n'est ni lu, ni évalué : il reste réservé à l'évaluation finale
    indépendante (`python -m ml.pipeline --final-evaluation`).
    """
    model = build_estimator(config["model_type"], config.get("params"))
    dataset_version = data["dataset_version"]

    with mlflow.start_run(run_name=run_name) as run:
        git_commit = get_git_commit()

        mlflow.log_params(config["params"])
        mlflow.log_params(
            {
                "dataset_version": dataset_version,
                "train_dvc_hash": data["train_dvc_hash"],
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

            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact(str(validation_predictions_path))

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
    # Import local pour éviter un cycle d'imports entre ml.pipeline et ml.train.
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
