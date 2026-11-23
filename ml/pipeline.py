import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

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

TEST_PATH = "data/versions/test/final_test.csv"

DATASETS = {
    "v1": {
        "train_path": "data/versions/v1/train.csv",
        "train_hash": "b2fc482fe85cd1779a0faa5f7d6de625",
    },
    "v2": {
        "train_path": "data/versions/v2/train.csv",
        "train_hash": "d08e0d2745ace0677a6330d8cc20795a",
    },
}

TEST_HASH = "082db6d3abab0442e8721d732e1db116"


def _period(frame):
    """Retourne la période couverte par un jeu de données (début, fin)."""
    dates = pd.to_datetime(frame["date"])
    return str(dates.min()), str(dates.max())


def load_data(dataset_version):
    """Charge entraînement et validation pour un cycle d'entraînement.

    Le test final n'est volontairement pas lu ici. Ce chemin sert à
    l'entraînement, à la comparaison champion/candidat et à la promotion :
    il ne doit jamais consulter ``final_test``. L'évaluation finale passe
    exclusivement par :func:`run_final_evaluation`.
    """
    dataset = DATASETS[dataset_version]

    train_df, validation_df = load_train_validation(dataset["train_path"])

    X_train, y_train = prepare_training_data(train_df)
    X_validation, y_validation = prepare_training_data(validation_df)

    feature_columns = list(X_train.columns)

    if list(X_validation.columns) != feature_columns:
        raise ValueError(
            "Les colonnes de validation ne correspondent pas "
            "aux colonnes d'entraînement."
        )

    return {
        "dataset_version": dataset_version,
        "train_dvc_hash": dataset["train_hash"],
        "validation_fraction": VALIDATION_FRACTION,
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "train_period": _period(train_df),
        "validation_period": _period(validation_df),
    }


def load_final_test_data():
    """Charge le jeu de test final.

    Cette fonction est réservée à l'évaluation finale indépendante, exécutée
    après la sélection du modèle. Elle n'est appelée par aucun chemin de
    promotion ni de Continuous Training.
    """
    test_df = load_final_test(TEST_PATH)
    X_test, y_test = prepare_training_data(test_df)

    if list(X_test.columns) != FEATURE_COLUMNS:
        raise ValueError(
            "Les colonnes du test final ne correspondent pas au schéma attendu."
        )

    return X_test, y_test, _period(test_df)


def run_v1(force=False):
    """Entraînement initial (v1), puis promotion du modèle de référence.

    Par défaut, refuse de s'exécuter si un champion existe déjà : relancer
    v1 déplacerait l'alias ``champion`` vers le baseline v1 et écraserait un
    champion plus récent (typiquement le candidat v2). Le drapeau
    ``force=True`` (CLI : ``--force-initial-training``) autorise
    explicitement cet écrasement.
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

    with mlflow.start_run(run_name="ct-decision-v2"):
        mlflow.log_params(
            {
                "dataset_version": "v2",
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
    """Évalue le champion courant sur le test final.

    Ce chemin est strictement séparé de l'entraînement et de la promotion :
    il ne modifie ni le champion, ni le registre. Il est à exécuter
    manuellement après la sélection du modèle, pour produire le score de
    rapport.
    """
    print("\n=== FINAL EVALUATION (REPORTING ONLY) ===")

    mlflow.set_experiment(EXPERIMENT_NAME)

    champion = get_champion()
    champion_model = load_champion_model()

    X_test, y_test, test_period = load_final_test_data()

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
                "test_dvc_hash": TEST_HASH,
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
            "Entraînement/promotion (validation) ou évaluation finale "
            "indépendante (test final)."
        )
    )

    parser.add_argument(
        "--dataset-version",
        choices=["v1", "v2"],
        help="Version du pool d'entraînement (chemin entraînement/promotion).",
    )
    parser.add_argument(
        "--final-evaluation",
        action="store_true",
        help=(
            "Évalue le champion sur le test final, sans promotion. "
            "À exécuter après la sélection du modèle."
        ),
    )
    parser.add_argument(
        "--eval-dataset-version",
        choices=["v1", "v2"],
        default="v2",
        help="Version de référence journalisée lors de l'évaluation finale.",
    )
    parser.add_argument(
        "--force-initial-training",
        action="store_true",
        help=(
            "Autorise run_v1 à remplacer un champion existant. "
            "À n'utiliser qu'après une réinitialisation volontaire."
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

    parser.error("Indiquez --dataset-version {v1,v2} ou --final-evaluation.")


if __name__ == "__main__":
    main()
