import numpy as np
import pytest
from slm_decision import (
    CostOptimalPolicy,
    RulePolicy,
    SpecError,
    ThresholdPolicy,
    as_policy,
    cost_optimal_policy,
)


def test_rule_policy_is_a_binary_threshold(spec):
    pol = RulePolicy(spec, threshold=0.6)
    assert pol.act(0.71) == "approve"
    assert pol.act(0.59) == "decline"


def test_rule_policy_is_monotone_in_score(spec):
    pol = RulePolicy(spec, threshold=0.5)
    actions = pol.act_batch(np.linspace(0, 1, 101))
    rank = {"decline": 0, "review": 1, "approve": 2}
    assert all(rank[actions[i]] <= rank[actions[i + 1]] for i in range(len(actions) - 1))


def test_rule_policy_rejects_unknown_action(spec):
    with pytest.raises(SpecError):
        RulePolicy(spec, threshold=0.5, positive="nope")


def test_cost_optimal_policy_uses_the_matrix(spec):
    pol = CostOptimalPolicy(spec)
    assert pol.act(0.999) == "approve"
    assert pol.act(0.5) == "review"
    assert pol.act(0.001) == "decline"


def test_cost_optimal_policy_respects_abstain_cap(spec):
    pol = CostOptimalPolicy(spec, max_abstain_rate=0.10)
    signals = np.linspace(0.02, 0.98, 1000)      # almost all land in the review band
    actions = pol.act_batch(signals)
    abstain_rate = sum(1 for a in actions if a == "review") / len(actions)
    assert abstain_rate <= 0.10 + 1e-9


def test_abstain_cap_prefers_lowest_cost_abstentions(spec):
    pol = CostOptimalPolicy(spec, max_abstain_rate=0.5)
    # both cases fall in the review band; the one with the least to lose is
    # reassigned to its next-best action
    actions = pol.act_batch([0.99, 0.5])
    assert actions[0] == "review"                 # kept: review beats approve by 0.2
    assert actions[1] == "decline"                # reassigned: review beat decline by 11.0


def test_threshold_policy_picks_highest_satisfied_threshold(spec):
    pol = ThresholdPolicy(spec, {"approve": 0.8}, default="review")
    assert pol.act(0.9) == "approve"
    assert pol.act(0.5) == "review"               # no threshold met -> default


def test_as_policy_adapts_a_callable(spec):
    pol = as_policy(lambda s: "approve" if s > 0.5 else "decline", spec)
    assert pol.act(0.9) == "approve"
    assert pol.act(0.1) == "decline"


def test_cost_optimal_policy_helper_matches_class(spec):
    a = cost_optimal_policy(spec)
    b = CostOptimalPolicy(spec)
    assert a.act(0.7) == b.act(0.7)
