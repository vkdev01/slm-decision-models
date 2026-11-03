"""Evaluate a policy end to end and produce a committable report."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .metrics import (
    action_distribution,
    brier,
    coverage_at_risk,
    ece,
    expected_cost,
    macro_f1,
    per_class_recall,
)
from .policies import Policy
from .spec import DecisionSpec


@dataclass
class Report:
    policy: str = ""
    n: int = 0
    accuracy: float = 0.0
    macro_f1: Optional[float] = None
    cost_per_case: float = 0.0
    abstain_rate: float = 0.0
    coverage: float = 1.0
    ece: Optional[float] = None
    brier: Optional[float] = None
    coverage_at_5pct_risk: Optional[float] = None
    action_mix: Dict[str, float] = field(default_factory=dict)
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        lines = ["%-22s %s" % ("policy", self.policy), "%-22s %d" % ("n", self.n),
                 "%-22s %.3f" % ("accuracy", self.accuracy),
                 "%-22s %s" % ("macro_f1", "-" if self.macro_f1 is None else "%.3f" % self.macro_f1),
                 "%-22s %.1f" % ("cost_per_case", self.cost_per_case),
                 "%-22s %.3f" % ("abstain_rate", self.abstain_rate),
                 "%-22s %.3f" % ("coverage", self.coverage)]
        if self.ece is not None:
            lines.append("%-22s %.4f" % ("ece", self.ece))
        if self.brier is not None:
            lines.append("%-22s %.4f" % ("brier", self.brier))
        return "\n".join(lines)


def evaluate_policy(spec: DecisionSpec, policy: Policy, y_true: Sequence[int],
                    scores: Optional[Sequence[float]] = None,
                    probs: Optional[Sequence[float]] = None,
                    actions: Optional[Sequence[str]] = None) -> Report:
    """Score a policy on a labelled slice.

    Provide either ``scores`` (a signal the policy consumes), ``probs`` (the
    calibrated probability of the positive outcome) or ``actions`` (already
    decided actions, useful for replaying a production log).
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    if actions is None:
        signal = probs if probs is not None else scores
        if signal is None:
            raise ValueError("provide one of: actions, scores, probs")
        signal = np.asarray(signal, dtype=float).ravel()
        if signal.size != y_true.size:
            raise ValueError("signal and y_true must have the same length")
        actions = policy.act_batch(signal)
    actions = list(actions)
    if len(actions) != y_true.size:
        raise ValueError("actions and y_true must have the same length")

    n = int(y_true.size)
    abstain_rate = float(np.mean([spec.is_abstain(a) for a in actions])) if spec.abstain else 0.0
    decided = [i for i, a in enumerate(actions) if not spec.is_abstain(a)]

    # "correct" means: one of the actions accepted for the outcome that actually
    # happened (by default, the cost-minimising action under perfect foresight).
    allowed = [spec.acceptable_actions(spec.outcomes[int(t)]) for t in y_true]
    correct = [a in s for a, s in zip(actions, allowed)]
    acc = float(np.mean([correct[i] for i in decided])) if decided else 0.0
    # exact-action F1 is only meaningful when each outcome maps to one action
    single = all(len(s) == 1 for s in allowed)
    f1 = (macro_f1([actions[i] for i in decided], [allowed[i][0] for i in decided])
          if (decided and single) else None)

    report = Report(
        policy=spec.name,
        n=n,
        accuracy=float(acc),
        macro_f1=(float(f1) if f1 is not None else None),
        cost_per_case=expected_cost(spec, actions, y_true),
        abstain_rate=abstain_rate,
        coverage=1.0 - abstain_rate,
        action_mix=action_distribution(actions),
        per_class=(per_class_recall([actions[i] for i in decided], [allowed[i][0] for i in decided])
                   if (decided and single) else {}),
    )

    sig = probs if probs is not None else scores
    if sig is not None:
        sig = np.asarray(sig, dtype=float).ravel()
        if sig.size == y_true.size:
            preds = (sig >= 0.5).astype(int)
            report.ece = ece(sig, y_true)
            report.brier = brier(sig, y_true)
            report.coverage_at_5pct_risk = coverage_at_risk(sig, y_true, preds, max_risk=0.05)
    return report


def compare(reports: List[Report]) -> str:
    """Render a markdown table of reports for a PR description or README."""
    cols = ["policy", "accuracy", "macro_f1", "cost_per_case", "abstain_rate", "ece"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in reports:
        d = r.to_dict()
        out.append("| " + " | ".join(
            d[c] if isinstance(d[c], str) else ("%.3f" % d[c] if d[c] is not None else "-")
            for c in cols) + " |")
    return "\n".join(out)
