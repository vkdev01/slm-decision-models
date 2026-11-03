"""Calibrators: temperature, isotonic and split-conformal.

All of them are numpy-only and operate on a single probability of the positive
class (``P(outcomes[1])`` in the library's convention).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _nll(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class TemperatureCalibrator:
    """Single-parameter calibration: ``p' = sigmoid(logit(p) / T)``.

    Monotone, so it never changes which case is more likely than another - only
    the scale, which is exactly what the cost arithmetic needs.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)
        self.fitted = False

    def fit(self, p: Sequence[float], y: Sequence[int]) -> "TemperatureCalibrator":
        p = np.clip(np.asarray(p, dtype=float).ravel(), 1e-6, 1 - 1e-6)
        y = np.asarray(y, dtype=float).ravel()
        if p.size != y.size:
            raise ValueError("p and y must have the same length")
        z = _logit(p)
        # coarse-to-fine grid search on log T; deterministic and dependency-free
        grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 200))
        losses = [_nll(_sigmoid(z / t), y) for t in grid]
        best = float(grid[int(np.argmin(losses))])
        lo, hi = best / 1.15, best * 1.15
        grid2 = np.linspace(lo, hi, 200)
        losses2 = [_nll(_sigmoid(z / t), y) for t in grid2]
        self.temperature = float(grid2[int(np.argmin(losses2))])
        self.fitted = True
        return self

    def transform(self, p: Sequence[float]) -> np.ndarray:
        p = np.asarray(p, dtype=float).ravel()
        return _sigmoid(_logit(p) / self.temperature)

    def fit_transform(self, p: Sequence[float], y: Sequence[int]) -> np.ndarray:
        return self.fit(p, y).transform(p)

    def to_dict(self) -> dict:
        return {"temperature": self.temperature}


class IsotonicCalibrator:
    """Non-parametric monotone fit via pool-adjacent-violators.

    Needs a few hundred points to be stable; below that, prefer temperature
    scaling, which cannot overfit a small validation set.
    """

    def __init__(self) -> None:
        self.x_: np.ndarray | None = None
        self.y_: np.ndarray | None = None

    def fit(self, p: Sequence[float], y: Sequence[int]) -> "IsotonicCalibrator":
        p_arr = np.asarray(p, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()
        if p_arr.size != y_arr.size:
            raise ValueError("p and y must have the same length")
        order = np.argsort(p_arr, kind="mergesort")
        xs, ys = p_arr[order], y_arr[order]
        # pool adjacent violators
        weights: list[float] = []
        values: list[float] = []
        counts: list[float] = []
        for xi, yi in zip(xs, ys):
            values.append(float(yi))
            weights.append(1.0)
            counts.append(float(xi))
            while len(values) > 1 and values[-2] > values[-1]:
                v2 = values.pop()
                w2 = weights.pop()
                c2 = counts.pop()
                v1 = values.pop()
                w1 = weights.pop()
                c1 = counts.pop()
                values.append((v1 * w1 + v2 * w2) / (w1 + w2))
                weights.append(w1 + w2)
                counts.append((c1 * w1 + c2 * w2) / (w1 + w2))
        self.x_ = np.asarray(counts)
        self.y_ = np.asarray(values)
        return self

    def transform(self, p: Sequence[float]) -> np.ndarray:
        if self.x_ is None or self.y_ is None:
            raise RuntimeError("call fit() first")
        xs, ys = self.x_, self.y_
        p_arr = np.asarray(p, dtype=float).ravel()
        return np.interp(p_arr, xs, ys, left=ys[0], right=ys[-1])


class ConformalCalibrator:
    """Split-conformal prediction sets with a distribution-free coverage guarantee.

    For ``alpha = 0.1`` the returned sets contain the true label at least ~90% of
    the time on exchangeable data - an assumption that holds only if the
    calibration set and the live traffic come from the same distribution.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = float(alpha)
        self.q_ = None
        self.n_ = 0

    def fit(self, p: Sequence[float], y: Sequence[int]) -> "ConformalCalibrator":
        p = np.asarray(p, dtype=float).ravel()
        y = np.asarray(y, dtype=int).ravel()
        if p.size != y.size:
            raise ValueError("p and y must have the same length")
        # nonconformity: 1 - probability assigned to the true class
        scores = np.where(y == 1, 1.0 - p, p)
        n = scores.size
        self.n_ = n
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_ = float(np.quantile(scores, level, method="higher"))
        return self

    def predict_set(self, p: Sequence[float]) -> list[list[int]]:
        if self.q_ is None:
            raise RuntimeError("call fit() first")
        p = np.asarray(p, dtype=float).ravel()
        out = []
        for pi in p:
            s = []
            if (1.0 - pi) <= self.q_:
                s.append(1)
            if pi <= self.q_:
                s.append(0)
            out.append(s)
        return out

    def coverage(self, p: Sequence[float], y: Sequence[int]) -> float:
        sets = self.predict_set(p)
        y = np.asarray(y, dtype=int).ravel()
        return float(np.mean([int(yi) in si for yi, si in zip(y, sets)]))


def calibrate(p: Sequence[float], y: Sequence[int], method: str = "temperature"):
    """Fit and return a calibrator in one call.

    ``method`` is one of ``"temperature"``, ``"isotonic"`` or ``"conformal"``.
    """
    if method == "temperature":
        return TemperatureCalibrator().fit(p, y)
    if method == "isotonic":
        return IsotonicCalibrator().fit(p, y)
    if method == "conformal":
        return ConformalCalibrator().fit(p, y)
    raise ValueError("unknown calibration method: %r" % method)
