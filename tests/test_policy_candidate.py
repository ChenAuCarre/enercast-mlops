"""Tests for candidate decisions with invalid MAE values.

These tests complement ``tests/test_policy.py`` (kept as is, with its local
modifications) to cover the candidate MAE validation branches: non-finite /
negative values.
"""

from ml.policy import evaluate_candidate


def test_candidate_with_nan_mae_is_rejected_with_invalid_reason():
    decision = evaluate_candidate(
        {"mae": 50.0},
        {"mae": float("nan")},
    )

    assert decision["promote"] is False
    assert decision["improvement"] is None
    assert "not a valid" in decision["reason"]
    assert "exceeds the quality threshold" not in decision["reason"]


def test_candidate_with_negative_mae_is_rejected_with_invalid_reason():
    decision = evaluate_candidate(
        {"mae": 50.0},
        {"mae": -1.0},
    )

    assert decision["promote"] is False
    assert decision["improvement"] is None
    assert "not a valid" in decision["reason"]
    assert "exceeds the quality threshold" not in decision["reason"]
