import numpy as np
import pytest
from slm_decision import (
    ConformalCalibrator,
    IsotonicCalibrator,
    TemperatureCalibrator,
    calibrate,
    ece,
)


def _overconfident(n=5000, seed=0, sharpness=3.0):
    """Scores that are systematically sharper than the truth.

    The true posterior is sigmoid(u); the reported score is sigmoid(sharpness * u),
    so the reported logits are ``sharpness`` times too extreme and the ideal
    temperature is ``sharpness``.
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 1.0, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-u))).astype(int)
    p = 1 / (1 + np.exp(-sharpness * u))
    return p, y


def test_temperature_scaling_reduces_ece():
    p, y = _overconfident()
    cal = TemperatureCalibrator().fit(p, y)
    assert ece(cal.transform(p), y) < ece(p, y)
    assert cal.temperature > 1.0                          # over-confident -> soften
    assert 2.0 < cal.temperature < 4.5                     # true value is 3.0


def test_temperature_scaling_preserves_ranking():
    p, y = _overconfident(n=500)
    cal = TemperatureCalibrator().fit(p, y)
    before = np.argsort(p)
    after = np.argsort(cal.transform(p))
    assert np.array_equal(before, after)


def test_temperature_is_identity_when_already_calibrated():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.random(20000) < p).astype(int)               # calibrated by construction
    cal = TemperatureCalibrator().fit(p, y)
    assert 0.7 < cal.temperature < 1.4


def test_isotonic_is_monotone():
    p, y = _overconfident(n=2000)
    iso = IsotonicCalibrator().fit(p, y)
    grid = np.linspace(0, 1, 200)
    out = iso.transform(grid)
    assert np.all(np.diff(out) >= -1e-9)


def test_isotonic_reduces_ece():
    p, y = _overconfident(n=5000)
    iso = IsotonicCalibrator().fit(p, y)
    assert ece(iso.transform(p), y) < ece(p, y)


def test_conformal_coverage_is_at_least_one_minus_alpha():
    rng = np.random.default_rng(7)
    n = 4000
    p = rng.uniform(0.01, 0.99, n)
    y = (rng.random(n) < p).astype(int)
    conf = ConformalCalibrator(alpha=0.1).fit(p[:2000], y[:2000])
    cov = conf.coverage(p[2000:], y[2000:])
    assert cov >= 0.90 - 0.02                             # finite-sample slack


def test_conformal_sets_shrink_with_confidence():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.01, 0.99, 1000)
    y = (rng.random(1000) < p).astype(int)
    conf = ConformalCalibrator(alpha=0.2).fit(p, y)
    sets = conf.predict_set([0.01, 0.5, 0.99])
    assert sets[0] == [0]                                 # confident negative
    assert sets[2] == [1]                                 # confident positive


def test_calibrate_helper_dispatches():
    p, y = _overconfident(n=1000)
    assert isinstance(calibrate(p, y, "temperature"), TemperatureCalibrator)
    assert isinstance(calibrate(p, y, "isotonic"), IsotonicCalibrator)
    assert isinstance(calibrate(p, y, "conformal"), ConformalCalibrator)
    with pytest.raises(ValueError):
        calibrate(p, y, "nope")
