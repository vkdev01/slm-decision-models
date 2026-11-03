import json
import os

import pytest
from slm_decision import DecisionSpec, SpecError


def test_missing_cost_cell_raises():
    with pytest.raises(ValueError):
        DecisionSpec(actions=["a", "b"], costs={("a", "x"): 1.0}, abstain="a")


def test_unknown_action_in_costs_raises():
    with pytest.raises(SpecError):
        DecisionSpec(actions=["a"], costs={("zzz", "x"): 1.0})


def test_abstain_must_be_in_action_set():
    with pytest.raises(SpecError):
        DecisionSpec(actions=["a", "b"], costs={("a", "x"): 1.0, ("b", "x"): 2.0}, abstain="c")


def test_negative_cost_raises():
    with pytest.raises(SpecError):
        DecisionSpec(actions=["a"], costs={("a", "x"): -1.0})


def test_outcomes_are_inferred_and_sorted(spec):
    assert spec.outcomes == ["bad", "good"]


def test_cost_matrix_shape(spec):
    m = spec.cost_matrix()
    assert len(m) == 3 and len(m[0]) == 2
    assert m[0] == [120.0, 0.0]


def test_expected_cost_of_each_action(spec):
    # p(good) = 0.9
    costs = spec.expected_cost([0.1, 0.9])
    assert costs["approve"] == pytest.approx(12.0)
    assert costs["review"] == pytest.approx(1.0)
    assert costs["decline"] == pytest.approx(21.6)


def test_best_action_follows_the_cost_matrix(spec):
    assert spec.best_action([0.001, 0.999]) == "approve"
    assert spec.best_action([0.5, 0.5]) == "review"
    assert spec.best_action([0.999, 0.001]) == "decline"


def test_threshold_follows_cost_ratio(binary_spec):
    # 120 : 24 -> approve only above 120 / 144 = 0.8333 of being good
    th = binary_spec.thresholds()
    assert th["approve"] == pytest.approx(0.8333, abs=1e-3)
    assert th["decline"] == pytest.approx(0.0, abs=1e-9)


def test_thresholds_requires_two_outcomes():
    s = DecisionSpec(actions=["a"], costs={("a", "x"): 0.0, ("a", "y"): 1.0, ("a", "z"): 2.0})
    with pytest.raises(SpecError):
        s.thresholds()


def test_roundtrip_through_json(spec, tmp_path):
    path = os.path.join(str(tmp_path), "spec.json")
    spec.save(path)
    loaded = DecisionSpec.load(path)
    assert loaded.actions == spec.actions
    assert loaded.costs == spec.costs
    assert loaded.abstain == spec.abstain
    assert json.load(open(path))["name"] == "triage"
