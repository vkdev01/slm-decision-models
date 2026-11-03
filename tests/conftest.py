"""Shared fixtures. Everything here runs without torch installed."""
import numpy as np
import pytest


@pytest.fixture(scope="session")
def spec():
    from slm_decision import DecisionSpec
    return DecisionSpec(
        name="triage",
        actions=["approve", "review", "decline"],
        costs={("approve", "bad"): 120.0, ("approve", "good"): 0.0,
               ("review", "bad"): 1.0, ("review", "good"): 1.0,
               ("decline", "bad"): 0.0, ("decline", "good"): 24.0},
        abstain="review",
        max_abstain_rate=0.25,
    )


@pytest.fixture(scope="session")
def binary_spec():
    from slm_decision import DecisionSpec
    return DecisionSpec(
        name="binary",
        actions=["approve", "decline"],
        costs={("approve", "bad"): 120.0, ("approve", "good"): 0.0,
               ("decline", "bad"): 0.0, ("decline", "good"): 24.0},
    )


@pytest.fixture(scope="session")
def toy_scores():
    rng = np.random.default_rng(42)
    p = rng.beta(2, 5, 4000)
    y = (rng.random(4000) < p).astype(int)
    return p, y
