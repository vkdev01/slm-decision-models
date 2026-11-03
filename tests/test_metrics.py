import numpy as np
import pytest
from slm_decision import (
    accuracy,
    brier,
    coverage_at_risk,
    ece,
    expected_cost,
    log_loss,
    macro_f1,
    per_class_recall,
    risk_coverage,
)


def test_ece_of_a_perfectly_calibrated_predictor_is_small(toy_scores):
    p, y = toy_scores
    assert ece(p, y) < 0.05


def test_ece_detects_overconfidence():
    rng = np.random.default_rng(0)
    n = 20000
    y = rng.integers(0, 2, n)
    p = np.full(n, 0.95)                      # says 95%, is right 50%
    assert ece(p, y, n_bins=10) > 0.4


def test_ece_is_zero_for_constant_correct_predictions():
    assert ece(np.ones(100), np.ones(100)) == pytest.approx(0.0, abs=1e-9)


def test_brier_bounds():
    assert brier([0.0, 1.0], [0, 1]) == pytest.approx(0.0)
    assert brier([1.0, 0.0], [0, 1]) == pytest.approx(1.0)


def test_log_loss_prefers_confident_correct():
    assert log_loss([0.9, 0.9], [1, 1]) < log_loss([0.6, 0.6], [1, 1])


def test_accuracy_and_macro_f1_on_a_known_case():
    preds = ["a", "b", "a", "a"]
    y = ["a", "b", "b", "a"]
    assert accuracy(preds, y) == pytest.approx(0.75)
    # class a: tp=2 fp=1 fn=0 -> F1 0.8 ; class b: tp=1 fp=0 fn=1 -> F1 0.667
    assert macro_f1(preds, y) == pytest.approx(0.7333, abs=1e-3)


def test_expected_cost_matches_hand_calculation(spec):
    # approve on a bad case costs 120, decline on a good case costs 24 -> mean 72
    assert expected_cost(spec, ["approve", "decline"], [0, 1]) == pytest.approx(72.0)


def test_expected_cost_accepts_outcome_names(spec):
    assert expected_cost(spec, ["approve"], ["bad"]) == pytest.approx(120.0)


def test_expected_cost_rejects_length_mismatch(spec):
    with pytest.raises(ValueError):
        expected_cost(spec, ["approve"], [0, 1])


def test_risk_coverage_is_monotone(toy_scores):
    p, y = toy_scores
    cov, risk = risk_coverage(p, y, n_points=50)
    assert np.all(np.diff(cov) > 0)
    assert risk[0] <= risk[-1]


def test_coverage_at_risk_is_a_fraction(toy_scores):
    p, y = toy_scores
    c = coverage_at_risk(p, y, max_risk=0.10)
    assert 0.0 <= c <= 1.0


def test_per_class_recall_includes_support():
    out = per_class_recall(["a", "a", "b"], ["a", "b", "b"])
    assert out["a"]["support"] == 1.0
    assert out["b"]["recall"] == pytest.approx(0.5)
