"""slm_decision - build, evaluate and ship small language models as decision models.

The core (specs, policies, metrics, calibration, evaluation) is numpy-only. The
model wrappers import torch and transformers lazily, so they are optional.
"""
from .calibration import ConformalCalibrator, IsotonicCalibrator, TemperatureCalibrator, calibrate
from .evaluation import Report, compare, evaluate_policy
from .metrics import (
                       accuracy,
                       action_distribution,
                       brier,
                       coverage_at_risk,
                       ece,
                       expected_cost,
                       log_loss,
                       macro_f1,
                       per_class_recall,
                       risk_coverage,
)
from .models import (
                       DecisionResult,
                       DistilledDecisionModel,
                       FineTunedDecisionModel,
                       HybridPolicy,
                       MissingDependency,
                       Retriever,
                       SLMDecisionModel,
)
from .policies import CostOptimalPolicy, Policy, RulePolicy, ThresholdPolicy, as_policy, cost_optimal_policy
from .spec import DecisionSpec, SpecError

__version__ = "0.1.0"

__all__ = [
    "DecisionSpec", "SpecError",
    "Policy", "RulePolicy", "ThresholdPolicy", "CostOptimalPolicy",
    "cost_optimal_policy", "as_policy",
    "ece", "brier", "log_loss", "macro_f1", "accuracy", "expected_cost",
    "risk_coverage", "coverage_at_risk", "per_class_recall", "action_distribution",
    "TemperatureCalibrator", "IsotonicCalibrator", "ConformalCalibrator", "calibrate",
    "evaluate_policy", "Report", "compare",
    "SLMDecisionModel", "DistilledDecisionModel", "FineTunedDecisionModel",
    "HybridPolicy", "Retriever", "DecisionResult", "MissingDependency",
    "__version__",
]
