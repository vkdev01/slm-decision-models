import json

import pytest
from slm_decision import (
    CostOptimalPolicy,
    RulePolicy,
    compare,
    evaluate_policy,
)


def test_report_has_the_expected_fields(spec, toy_scores):
    p, y = toy_scores
    rep = evaluate_policy(spec, CostOptimalPolicy(spec), y_true=y, probs=p)
    d = rep.to_dict()
    for key in ("policy", "n", "accuracy", "macro_f1", "cost_per_case",
                "abstain_rate", "coverage", "ece", "brier"):
        assert key in d
    assert d["n"] == len(y)
    assert 0.0 <= d["abstain_rate"] <= 1.0


def test_abstain_rate_and_coverage_sum_to_one(spec, toy_scores):
    p, y = toy_scores
    rep = evaluate_policy(spec, CostOptimalPolicy(spec), y_true=y, probs=p)
    assert rep.abstain_rate + rep.coverage == pytest.approx(1.0)


def test_cost_optimal_beats_a_fixed_threshold_on_cost(spec, toy_scores):
    p, y = toy_scores
    cost_policy = evaluate_policy(spec, CostOptimalPolicy(spec), y_true=y, probs=p)
    rule = evaluate_policy(spec, RulePolicy(spec, threshold=0.5), y_true=y, probs=p)
    assert cost_policy.cost_per_case <= rule.cost_per_case


def test_report_is_json_serialisable(spec, toy_scores):
    p, y = toy_scores
    rep = evaluate_policy(spec, CostOptimalPolicy(spec), y_true=y, probs=p)
    assert json.loads(json.dumps(rep.to_dict()))["n"] == rep.n


def test_evaluate_from_precomputed_actions(spec):
    actions = ["approve", "review", "decline", "approve"]
    y = [1, 0, 0, 1]
    rep = evaluate_policy(spec, CostOptimalPolicy(spec), y_true=y, actions=actions)
    assert rep.n == 4
    assert rep.abstain_rate == pytest.approx(0.25)


def test_evaluate_requires_a_signal(spec):
    with pytest.raises(ValueError):
        evaluate_policy(spec, CostOptimalPolicy(spec), y_true=[0, 1])


def test_compare_renders_markdown(spec, toy_scores):
    p, y = toy_scores
    reps = [evaluate_policy(spec, CostOptimalPolicy(spec), y_true=y, probs=p),
            evaluate_policy(spec, RulePolicy(spec, threshold=0.5), y_true=y, probs=p)]
    table = compare(reps)
    assert table.startswith("| policy")
    assert table.count("\n") == 3
