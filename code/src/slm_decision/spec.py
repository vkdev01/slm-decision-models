"""Decision specifications: the action set, the cost matrix and the abstain policy.

A :class:`DecisionSpec` is the single source of truth for a decision. Everything
else in this library - policies, metrics, reports - takes one as input.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


class SpecError(ValueError):
    """Raised when a decision specification is not internally consistent."""


@dataclass
class DecisionSpec:
    """A decision: what you may do, what each mistake costs, and what to do when unsure.

    Parameters
    ----------
    name:
        Human readable identifier, used in reports and logs.
    actions:
        The closed set of actions the policy may choose.
    costs:
        Mapping ``(action, outcome) -> cost``. Every action must have a cost for
        every outcome; a missing cell is a specification bug, not a default.
    abstain:
        The action used when the policy declines to decide automatically.
    max_abstain_rate:
        Optional cap on the fraction of cases routed to ``abstain``.
    """

    actions: List[str]
    costs: Dict[Tuple[str, str], float]
    name: str = "decision"
    abstain: Optional[str] = None
    max_abstain_rate: Optional[float] = None
    outcomes: List[str] = field(default_factory=list)
    #: outcome -> actions considered correct for that outcome. Defaults to the
    #: single cost-minimising action (see :meth:`acceptable_actions`).
    acceptable: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.actions = list(self.actions)
        self.costs = {(str(a), str(o)): float(c) for (a, o), c in self.costs.items()}
        if not self.outcomes:
            seen: List[str] = []
            for (_a, o) in self.costs:
                if o not in seen:
                    seen.append(o)
            self.outcomes = sorted(seen)
        self.validate()

    # ------------------------------------------------------------------ validation
    def validate(self) -> "DecisionSpec":
        if not self.actions:
            raise SpecError("at least one action is required")
        if len(set(self.actions)) != len(self.actions):
            raise SpecError("actions must be unique")
        if not self.outcomes:
            raise SpecError("at least one outcome is required")
        missing = [(a, o) for a in self.actions for o in self.outcomes if (a, o) not in self.costs]
        if missing:
            raise SpecError("missing cost cells: %s" % missing[:5])
        unknown = [k for k in self.costs if k[0] not in self.actions or k[1] not in self.outcomes]
        if unknown:
            raise SpecError("cost cells reference unknown action/outcome: %s" % unknown[:5])
        for (a, o), c in self.costs.items():
            if c < 0:
                raise SpecError("negative cost for (%s, %s): %r" % (a, o, c))
        if self.abstain is not None and self.abstain not in self.actions:
            raise SpecError("abstain action %r is not in the action set" % self.abstain)
        if self.max_abstain_rate is not None and not (0.0 <= self.max_abstain_rate <= 1.0):
            raise SpecError("max_abstain_rate must be in [0, 1]")
        for outcome, acts in self.acceptable.items():
            if outcome not in self.outcomes:
                raise SpecError("acceptable references unknown outcome %r" % outcome)
            bad = [a for a in acts if a not in self.actions]
            if bad:
                raise SpecError("acceptable references unknown actions: %s" % bad)
        return self

    # ------------------------------------------------------------------ arithmetic
    def cost_matrix(self) -> List[List[float]]:
        """Cost matrix with shape ``(n_actions, n_outcomes)``."""
        return [[self.costs[(a, o)] for o in self.outcomes] for a in self.actions]

    def expected_cost(self, p_outcome: Iterable[float]) -> Dict[str, float]:
        """Expected cost of every action given a distribution over outcomes."""
        p = [float(x) for x in p_outcome]
        if len(p) != len(self.outcomes):
            raise SpecError("expected %d outcome probabilities, got %d" % (len(self.outcomes), len(p)))
        return {a: sum(c * pi for c, pi in zip(row, p)) for a, row in zip(self.actions, self.cost_matrix())}

    def best_action(self, p_outcome: Iterable[float]) -> str:
        """Action minimising expected cost. Ties break toward ``abstain``, then action order."""
        costs = self.expected_cost(p_outcome)
        best = min(costs.values())
        candidates = [a for a in self.actions if costs[a] <= best + 1e-12]
        if self.abstain is not None and self.abstain in candidates:
            return self.abstain
        return candidates[0]

    def thresholds(self) -> Dict[str, float]:
        """Lower edge of each action's optimal region, in ``P(outcomes[1])``.

        Only defined for two-outcome specs (the common binary decision). The value
        is the probability of ``outcomes[1]`` above which that action becomes
        optimal. With outcomes sorted alphabetically, ``outcomes[1]`` is the
        "positive" or "good" outcome.
        """
        if len(self.outcomes) != 2:
            raise SpecError("thresholds() is defined for two-outcome specs only")
        # each action has expected cost  m*p + b  with p = P(outcomes[1])
        lines = []
        for a in self.actions:
            c0 = self.costs[(a, self.outcomes[0])]
            c1 = self.costs[(a, self.outcomes[1])]
            lines.append((a, c1 - c0, c0))
        cuts = sorted({0.0, 1.0} | {self._cross(l1, l2) for i, l1 in enumerate(lines) for l2 in lines[i + 1:]})
        cuts = [c for c in cuts if 0.0 <= c <= 1.0]
        lower: Dict[str, float] = {}
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            mid = (lo + hi) / 2.0
            best = min(lines, key=lambda t: t[1] * mid + t[2])[0]
            lower.setdefault(best, lo)
        # an action that only wins on a boundary still gets its edge recorded
        for a in self.actions:
            lower.setdefault(a, 1.0)
        return lower

    @staticmethod
    def _cross(l1: Tuple[str, float, float], l2: Tuple[str, float, float]) -> float:
        m1, b1 = l1[1], l1[2]
        m2, b2 = l2[1], l2[2]
        if abs(m1 - m2) < 1e-15:
            return 1e9
        return (b2 - b1) / (m1 - m2)

    def oracle_action(self, outcome: str) -> str:
        """The cost-minimising action if the outcome were known in advance.

        This is the reference label for "action accuracy": a decision is correct
        when it matches the action you would have taken with perfect foresight.
        """
        costs = {a: self.costs[(a, outcome)] for a in self.actions}
        best = min(costs.values())
        candidates = [a for a in self.actions if costs[a] <= best + 1e-12]
        if self.abstain is not None and self.abstain in candidates:
            return self.abstain
        return candidates[0]

    def acceptable_actions(self, outcome: str) -> List[str]:
        """Actions counted as correct for ``outcome``.

        Defaults to the single cost-minimising action. Override it for graded
        action sets, where an intermediate action (a review, a step-up) is a
        legitimate outcome even though it is never cost-minimal under certainty.
        """
        if outcome in self.acceptable:
            return list(self.acceptable[outcome])
        return [self.oracle_action(outcome)]

    def is_abstain(self, action: str) -> bool:
        return self.abstain is not None and action == self.abstain

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "actions": self.actions,
            "outcomes": self.outcomes,
            "costs": {"%s|%s" % k: v for k, v in self.costs.items()},
            "abstain": self.abstain,
            "max_abstain_rate": self.max_abstain_rate,
            "acceptable": {k: list(v) for k, v in self.acceptable.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionSpec":
        costs = {}
        for k, v in d["costs"].items():
            a, o = k.split("|", 1)
            costs[(a, o)] = v
        return cls(
            actions=d["actions"],
            costs=costs,
            name=d.get("name", "decision"),
            abstain=d.get("abstain"),
            max_abstain_rate=d.get("max_abstain_rate"),
            outcomes=d.get("outcomes", []),
            acceptable=d.get("acceptable", {}),
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DecisionSpec":
        with open(path) as f:
            return cls.from_dict(json.load(f))
