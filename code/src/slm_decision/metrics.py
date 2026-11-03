"""Decision metrics: calibration, cost, classification quality and selective prediction."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .spec import DecisionSpec


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def ece(p, y, n_bins: int = 15) -> float:
    """Expected calibration error over equal-width probability bins."""
    p, y = _arr(p), _arr(y)
    if p.size == 0:
        return 0.0
    if p.size != y.size:
        raise ValueError("p and y must have the same length")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def brier(p, y) -> float:
    """Brier score: mean squared error of the probability. Lower is better."""
    p, y = _arr(p), _arr(y)
    if p.size != y.size:
        raise ValueError("p and y must have the same length")
    return float(np.mean((p - y) ** 2))


def log_loss(p, y, eps: float = 1e-12) -> float:
    p, y = np.clip(_arr(p), eps, 1 - eps), _arr(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(preds: Sequence, y: Sequence) -> float:
    preds, y = np.asarray(preds), np.asarray(y)
    if preds.size == 0:
        return 0.0
    return float(np.mean(preds == y))


def macro_f1(preds: Sequence, y: Sequence, labels: Sequence | None = None) -> float:
    """Unweighted mean F1 across the classes present in ``y`` (or ``labels``)."""
    preds, y = np.asarray(preds), np.asarray(y)
    if preds.size != y.size:
        raise ValueError("preds and y must have the same length")
    if labels is None:
        labels = sorted(set(y.tolist()) | set(preds.tolist()))
    scores: List[float] = []
    for lab in labels:
        tp = int(np.sum((preds == lab) & (y == lab)))
        fp = int(np.sum((preds == lab) & (y != lab)))
        fn = int(np.sum((preds != lab) & (y == lab)))
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def expected_cost(spec: DecisionSpec, actions: Iterable[str], y: Iterable) -> float:
    """Mean cost per case for a sequence of actions and true outcome labels.

    ``y`` holds either outcome names or integer indices into ``spec.outcomes``.
    """
    actions = list(actions)
    y = list(y)
    if len(actions) != len(y):
        raise ValueError("actions and y must have the same length")
    if not actions:
        return 0.0
    total = 0.0
    for a, truth in zip(actions, y):
        outcome = spec.outcomes[int(truth)] if isinstance(truth, (int, np.integer)) else str(truth)
        total += spec.costs[(str(a), outcome)]
    return float(total / len(actions))


def cost_matrix_loss(spec: DecisionSpec, probs, y) -> float:
    """Expected cost of the argmin action given full outcome distributions.

    ``probs`` has shape (n, n_outcomes); ``y`` holds outcome indices.
    """
    probs = np.atleast_2d(np.asarray(probs, dtype=float))
    y = np.asarray([int(v) for v in y])
    M = np.asarray(spec.cost_matrix(), dtype=float)
    exp = probs @ M.T                       # (n, n_actions)
    chosen = exp.argmin(axis=1)
    return float(np.mean(M[chosen, y]))


def risk_coverage(p, y, preds=None, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Selective-prediction curve: error on the covered slice vs coverage.

    If ``preds`` is None, a 0.5 threshold on ``p`` is used as the decision rule.
    """
    p = _arr(p)
    y = np.asarray(y)                      # may be labels, not only 0/1
    if p.size != y.size:
        raise ValueError("p and y must have the same length")
    if preds is None:
        preds = (p >= 0.5).astype(int)
    preds = np.asarray(preds)
    order = np.argsort(-p)
    errs = np.cumsum((preds[order] != y[order]).astype(int))
    cov = np.arange(1, len(order) + 1) / len(order)
    risk = errs / np.arange(1, len(order) + 1)
    if n_points and len(cov) > n_points:
        keep = np.linspace(0, len(cov) - 1, n_points).astype(int)
        cov, risk = cov[keep], risk[keep]
    return cov, risk


def coverage_at_risk(p, y, preds=None, max_risk: float = 0.05) -> float:
    """Largest coverage at which error on the covered slice stays <= ``max_risk``."""
    cov, risk = risk_coverage(p, y, preds)
    ok = risk <= max_risk
    return float(cov[ok].max()) if ok.any() else 0.0


def per_class_recall(preds: Sequence, y: Sequence) -> Dict[str, Dict[str, float]]:
    preds, y = np.asarray(preds), np.asarray(y)
    out: Dict[str, Dict[str, float]] = {}
    for lab in sorted(set(y.tolist())):
        m = y == lab
        out[str(lab)] = {"recall": float(np.mean(preds[m] == lab)), "support": float(m.sum())}
    return out


def action_distribution(actions: Iterable[str]) -> Dict[str, float]:
    actions = list(actions)
    if not actions:
        return {}
    uniq, counts = np.unique(np.asarray(actions, dtype=object), return_counts=True)
    return {str(u): float(c / len(actions)) for u, c in zip(uniq, counts)}
