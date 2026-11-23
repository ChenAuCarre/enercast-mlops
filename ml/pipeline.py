import argparse

import mlflow
import pandas as pd

from ml.policy import evaluate_candidate, evaluate_initial_models
from ml.preprocess import prepare_training_data
from ml.registry import (
    get_champion,
    load_champion_model,
    promote_model,
    register_model,
)
from ml.train import (
    EXPERIMENT_NAME,
    MODELS,
    TEST_PATH,
    evaluate_model,
    run_experiment,
)

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


def load_data(dataset_version):
    dataset = DATASETS[dataset_version]

    train_df = pd.read_csv(dataset["train_path"])
    test_df = pd.read_csv(TEST_PATH)

    X_train, y_train = prepare_training_data(train_df)
    X_test, y_test = prepare_training_data(test_df)

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and evaluation features do not match.")

    return X_train, y_train, X_test, y_test, dataset["train_hash"]


def run_v1():
    print("\n=== INITIAL TRAINING: VERSION 1 ===")

    X_train, y_train, X_test, y_test, train_hash = load_data("v1")
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = "baseline-linear-regression-v1"
    config = MODELS[run_name]

    result = run_experiment(
        run_name,
        config,
        X_train,
        y_train,
        X_test,
        y_test,
        dataset_version="v1",
        train_dvc_hash=train_hash,
        test_dvc_hash=TEST_HASH,
    )

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

    X_train, y_train, X_test, y_test, train_hash = load_data("v2")

    champion_model = load_champion_model()
    _, champion_metrics = evaluate_model(
        champion_model,
        X_test,
        y_test,
    )

    candidate_configs = {
        "candidate-hist-gradient-boosting-v2":
            MODELS["candidate-hist-gradient-boosting-v2"],
    }

    results = {}

    for run_name, config in candidate_configs.items():
        results[run_name] = run_experiment(
            run_name,
            config,
            X_train,
            y_train,
            X_test,
            y_test,
            dataset_version="v2",
            train_dvc_hash=train_hash,
            test_dvc_hash=TEST_HASH,
        )

    decisions = {}

    for run_name, result in results.items():
        decisions[run_name] = evaluate_candidate(
            champion_metrics,
            result["metrics"],
        )

    print("\nChampion MAE:", round(champion_metrics["mae"], 3))

    for run_name, result in results.items():
        print(
            f"{run_name} MAE:",
            round(result["metrics"]["mae"], 3),
        )
        print("Decision:", decisions[run_name])

    eligible = [
        run_name
        for run_name, decision in decisions.items()
        if decision["promote"]
    ]

    selected_name = min(
        eligible,
        key=lambda name: results[name]["metrics"]["mae"],
    ) if eligible else None

    with mlflow.start_run(run_name="ct-decision-v2"):
        mlflow.log_params(
            {
                "dataset_version": "v2",
                "previous_champion_version": current_champion.version,
                "candidate_count": len(results),
            }
        )

        mlflow.log_metrics({
            "champion_mae": champion_metrics["mae"],
            **{
                f"{name}_mae": result["metrics"]["mae"]
                for name, result in results.items()
            },
        })

        mlflow.set_tags({
            "promotion_decision": str(selected_name is not None),
            "selected_candidate": selected_name or "none",
            "decision_reason": (
                decisions[selected_name]["reason"]
                if selected_name
                else "No candidate passed the promotion policy."
            ),
        })

    if selected_name:
        selected_result = results[selected_name]
        registered = register_model(selected_result["model_id"])
        champion = promote_model(registered.version)

        print(
            f"\nNew champion: {selected_name}, "
            f"version {champion.version}"
        )
    else:
        print(
            "\nNo candidate passed. "
            f"Champion version {current_champion.version} is retained."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-version",
        choices=["v1", "v2"],
        required=True,
    )
    args = parser.parse_args()

    if args.dataset_version == "v1":
        run_v1()
    else:
        run_v2()


if __name__ == "__main__":
    main()
