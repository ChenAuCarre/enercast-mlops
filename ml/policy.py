import math

# Initial project acceptance criteria (MAE expressed in Wh).
MAX_MAE = 65.0
MIN_IMPROVEMENT = 0.01  # 1%


def is_quality_acceptable(metrics):
    """Check whether a model satisfies the minimum quality threshold."""
    mae = float(metrics["mae"])

    if not math.isfinite(mae):
        return False

    return 0 <= mae <= MAX_MAE


def calculate_improvement(champion_mae, candidate_mae):
    """Calculate relative MAE improvement over the current champion."""
    champion_mae = float(champion_mae)
    candidate_mae = float(candidate_mae)

    if not math.isfinite(champion_mae) or champion_mae <= 0:
        raise ValueError("Champion MAE must be finite and positive.")

    if not math.isfinite(candidate_mae) or candidate_mae < 0:
        raise ValueError("Candidate MAE must be finite and non-negative.")

    return (champion_mae - candidate_mae) / champion_mae


def evaluate_initial_models(results):
    """
    Select the best model satisfying the quality threshold
    during the initial v1 release.
    """
    eligible = {
        name: result
        for name, result in results.items()
        if is_quality_acceptable(result["metrics"])
    }

    if not eligible:
        return {
            "promote": False,
            "model": None,
            "reason": (f"No model satisfies the maximum MAE of {MAX_MAE} Wh."),
        }

    best_name = min(
        eligible,
        key=lambda name: eligible[name]["metrics"]["mae"],
    )

    return {
        "promote": True,
        "model": best_name,
        "reason": (
            f"{best_name} satisfies the quality threshold "
            f"with MAE={eligible[best_name]['metrics']['mae']:.3f} Wh."
        ),
    }


def evaluate_candidate(champion_metrics, candidate_metrics):
    """
    Decide whether a new candidate should replace the current champion.
    """
    champion_mae = float(champion_metrics["mae"])
    candidate_mae = float(candidate_metrics["mae"])

    if not math.isfinite(candidate_mae) or candidate_mae < 0:
        return {
            "promote": False,
            "improvement": None,
            "reason": (
                f"Candidate rejected: MAE={candidate_mae} is not a valid "
                f"finite, non-negative value."
            ),
        }

    if not is_quality_acceptable(candidate_metrics):
        return {
            "promote": False,
            "improvement": None,
            "reason": (
                f"Candidate rejected: MAE={candidate_mae:.3f} Wh "
                f"exceeds the quality threshold of {MAX_MAE} Wh."
            ),
        }

    improvement = calculate_improvement(
        champion_mae,
        candidate_mae,
    )

    # Round so an exact 1% improvement is not rejected by float error.
    if round(improvement, 9) < MIN_IMPROVEMENT:
        return {
            "promote": False,
            "improvement": improvement,
            "reason": (
                f"Candidate rejected: improvement={improvement:.2%}, "
                f"minimum required={MIN_IMPROVEMENT:.2%}."
            ),
        }

    return {
        "promote": True,
        "improvement": improvement,
        "reason": (f"Candidate approved: MAE improvement={improvement:.2%}."),
    }


class PromotionAuthorization:
    """Typed promotion authorization issued by the promotion policy layer.

    Authorizations are created only through :meth:`from_decision`, which the
    promotion policy uses after a validated positive decision; direct
    construction is refused.
    """

    def __new__(cls, version, reason, candidate=None):
        raise TypeError(
            "PromotionAuthorization must be created through the promotion "
            "policy via PromotionAuthorization.from_decision."
        )

    def __init__(self, version, reason, candidate=None):
        self.version = str(version)
        self.reason = str(reason)
        self.candidate = candidate

    @classmethod
    def from_decision(cls, decision, version, candidate=None):
        """Build a typed authorization from a validated positive decision."""
        if not isinstance(decision, dict) or decision.get("promote") is not True:
            return None

        reason = decision.get("reason")

        if not isinstance(reason, str) or not reason.strip():
            return None

        authorization = object.__new__(cls)
        authorization.version = str(version)
        authorization.reason = str(reason)
        authorization.candidate = candidate

        return authorization

    def authorizes(self, version, candidate):
        """Return whether this authorization covers the promoted version."""
        if version is None or candidate is None:
            return False

        return str(self.version) == str(version) and str(self.candidate) == str(
            candidate
        )
