"""Policies: anything that turns a signal into an action under a decision spec."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .spec import DecisionSpec, SpecError

ArrayLike = Union[Sequence[float], np.ndarray]


class Policy:
    """Base class. A policy maps a signal to an action from the spec's action set."""

    def __init__(self, spec: DecisionSpec) -> None:
        self.spec = spec

    def act(self, signal: float) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def act_batch(self, signals: ArrayLike) -> List[str]:
        return [self.act(float(s)) for s in np.asarray(signals, dtype=float).ravel()]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "%s(spec=%r)" % (type(self).__name__, self.spec.name)


class RulePolicy(Policy):
    """Binary threshold rule: ``positive`` above the threshold, ``negative`` below.

    Deliberately ignores the abstain action - it is the baseline you must beat,
    not a decision policy in its own right.
    """

    def __init__(self, spec: DecisionSpec, threshold: float = 0.5,
                 positive: Optional[str] = None, negative: Optional[str] = None) -> None:
        super().__init__(spec)
        self.threshold = float(threshold)
        self.positive = positive or spec.actions[0]
        self.negative = negative or spec.actions[-1]
        for a in (self.positive, self.negative):
            if a not in spec.actions:
                raise SpecError("action %r not in spec" % a)

    def act(self, signal: float) -> str:
        return self.positive if float(signal) >= self.threshold else self.negative


class ThresholdPolicy(Policy):
    """Multi-action policy driven by an explicit threshold per action.

    ``thresholds`` maps action -> minimum signal for that action; the action with
    the highest satisfied threshold wins. Useful when the operating points were
    agreed with the business rather than derived from costs.
    """

    def __init__(self, spec: DecisionSpec, thresholds: Dict[str, float], default: Optional[str] = None) -> None:
        super().__init__(spec)
        self.thresholds = {a: float(t) for a, t in thresholds.items()}
        for a in self.thresholds:
            if a not in spec.actions:
                raise SpecError("action %r not in spec" % a)
        self.default = default or spec.abstain or spec.actions[0]

    def act(self, signal: float) -> str:
        s = float(signal)
        eligible = [(t, a) for a, t in self.thresholds.items() if s >= t]
        if not eligible:
            return self.default
        return max(eligible)[1]


class CostOptimalPolicy(Policy):
    """Minimise expected cost, then enforce an optional abstain cap.

    The cap is applied at batch level: if the unconstrained argmin abstains more
    often than ``max_abstain_rate``, the abstentions with the smallest expected
    advantage are reassigned to their next-best non-abstain action.
    """

    def __init__(self, spec: DecisionSpec, max_abstain_rate: Optional[float] = None) -> None:
        super().__init__(spec)
        self.max_abstain_rate = (spec.max_abstain_rate if max_abstain_rate is None
                                 else float(max_abstain_rate))

    def _expected(self, p_positive: float) -> Dict[str, float]:
        p = np.array([1.0 - p_positive, p_positive])
        return self.spec.expected_cost(p)

    def act(self, signal: float) -> str:
        return self.spec.best_action([1.0 - float(signal), float(signal)])

    def act_batch(self, signals: ArrayLike) -> List[str]:
        sig = np.asarray(signals, dtype=float).ravel()
        actions = [self.act(s) for s in sig]
        if self.max_abstain_rate is None or self.spec.abstain is None or len(actions) == 0:
            return actions
        n_abstain = sum(1 for a in actions if a == self.spec.abstain)
        budget = int(np.floor(self.max_abstain_rate * len(actions)))
        if n_abstain <= budget:
            return actions
        # rank abstentions by how little they gain over the best alternative
        margins: list[tuple[float, int, str]] = []
        for i, a in enumerate(actions):
            if a != self.spec.abstain:
                continue
            costs = self._expected(float(sig[i]))
            alt = min((c, act) for act, c in costs.items() if act != self.spec.abstain)
            margins.append((costs[self.spec.abstain] - alt[0], i, alt[1]))
        margins.sort()
        for _margin, i, alt_action in margins[: n_abstain - budget]:
            actions[i] = alt_action
        return actions


def cost_optimal_policy(spec: DecisionSpec, max_abstain_rate: Optional[float] = None) -> CostOptimalPolicy:
    """Convenience constructor mirroring the docs."""
    return CostOptimalPolicy(spec, max_abstain_rate=max_abstain_rate)


def as_policy(obj, spec: DecisionSpec) -> Policy:
    """Adapt a callable or an object with ``decide`` into a :class:`Policy`."""
    if isinstance(obj, Policy):
        if obj.spec is not spec:
            obj.spec = spec
        return obj
    if callable(obj):
        return _CallablePolicy(spec, obj)
    if hasattr(obj, "decide"):
        return _CallablePolicy(spec, lambda s: obj.decide(s).action)
    raise TypeError("cannot adapt %r to a Policy" % type(obj))


class _CallablePolicy(Policy):
    def __init__(self, spec: DecisionSpec, fn) -> None:
        super().__init__(spec)
        self.fn = fn

    def act(self, signal: float) -> str:
        return str(self.fn(float(signal)))
