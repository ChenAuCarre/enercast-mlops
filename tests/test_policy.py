import pytest

from ml.policy import (
    MAX_MAE,
    MIN_IMPROVEMENT,
    calculate_improvement,
    evaluate_candidate,
    evaluate_initial_models,
    is_quality_acceptable,
)


@pytest.mark.parametrize(
    "mae, expected",
    [
        (40.0, True),
        (MAX_MAE, True),
        (MAX_MAE + 1, False),
        (-1.0, False),
        (float("nan"), False),
        (float("inf"), False),
    ],
)
def test_quality_threshold(mae, expected):
    assert is_quality_acceptable({"mae": mae}) is expected


def test_calculate_improvement():
    improvement = calculate_improvement(50.0, 45.0)

    assert improvement == pytest.approx(0.10)


@pytest.mark.parametrize(
    "champion_mae, candidate_mae",
    [
        (0.0, 10.0),
        (-1.0, 10.0),
        (float("inf"), 10.0),
        (50.0, -1.0),
        (50.0, float("nan")),
    ],
)
def test_invalid_improvement_inputs(champion_mae, candidate_mae):
    with pytest.raises(ValueError):
        calculate_improvement(champion_mae, candidate_mae)


def test_initial_selection_chooses_best_eligible_model():
    results = {
        "model_a": {"metrics": {"mae": 50.0}},
        "model_b": {"metrics": {"mae": 40.0}},
        "model_c": {"metrics": {"mae": 70.0}},
    }

    decision = evaluate_initial_models(results)

    assert decision["promote"] is True
    assert decision["model"] == "model_b"


def test_initial_selection_rejects_all_models():
    results = {
        "model_a": {"metrics": {"mae": 70.0}},
        "model_b": {"metrics": {"mae": 80.0}},
    }

    decision = evaluate_initial_models(results)

    assert decision["promote"] is False
    assert decision["model"] is None


def test_candidate_is_promoted_when_both_gates_pass():
    decision = evaluate_candidate(
        {"mae": 50.0},
        {"mae": 45.0},
    )

    assert decision["promote"] is True
    assert decision["improvement"] == pytest.approx(0.10)


def test_candidate_is_rejected_when_quality_gate_fails():
    decision = evaluate_candidate(
        {"mae": 50.0},
        {"mae": 65.0},
    )

    assert decision["promote"] is False
    assert decision["improvement"] is None


def test_candidate_is_rejected_when_improvement_is_insufficient():
    decision = evaluate_candidate(
        {"mae": 50.0},
        {"mae": 49.8},
    )

    assert decision["promote"] is False
    assert decision["improvement"] < MIN_IMPROVEMENT
