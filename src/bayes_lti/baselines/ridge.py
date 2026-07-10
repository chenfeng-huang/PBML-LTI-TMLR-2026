from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np

from .base import Baseline, Task, split_support_query


def _ridge_A_hat(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    n = X.shape[0]
    XXt = X @ X.T
    XXt = 0.5 * (XXt + XXt.T)
    return Y @ X.T @ np.linalg.inv(XXt + float(lam) * np.eye(n))


def _free_rollout_mse(A: np.ndarray, x0: np.ndarray, Y_true: np.ndarray) -> float:
    """Free rollout from x0. Compare predicted x_{t+1} to Y_true[:, t]."""
    n = x0.shape[0]
    q = Y_true.shape[1]
    if q <= 0:
        return float("nan")
    x = x0.copy()
    err = 0.0
    for t in range(q):
        x = A @ x
        y_true = Y_true[:, t]
        err += float(np.sum((x - y_true) ** 2))
    return err / (q * n)


@dataclass
class RidgeBaseline(Baseline):
    lambdas: List[float]
    query_len: int
    tuned_lambda: Optional[float] = None

    @property
    def name(self) -> str:
        return "ridge"

    def fit_global(self, train_tasks: List[Task], support_len: int) -> None:
        grid = [float(l) for l in self.lambdas] if len(self.lambdas) > 0 else [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]

        best_lam = grid[0]
        best = float("inf")
        for lam in grid:
            mses = []
            for t in train_tasks:
                sp = split_support_query(t, support_len=support_len, query_len=self.query_len)
                A_hat = _ridge_A_hat(sp["X_support"], sp["Y_support"], lam=lam)
                mses.append(_free_rollout_mse(A_hat, sp["x0_rollout"], sp["Y_query"]))
            score = float(np.nanmean(mses))
            if score < best:
                best = score
                best_lam = lam

        self.tuned_lambda = float(best_lam)

    def fit_task(self, task: Task, support_len: int) -> np.ndarray:
        lam = float(self.tuned_lambda) if self.tuned_lambda is not None else (float(self.lambdas[0]) if len(self.lambdas) > 0 else 1e-3)
        X = np.asarray(task["X"], dtype=float)
        Y = np.asarray(task["Y"], dtype=float)
        T = X.shape[1]
        s = int(max(1, min(support_len, T)))
        return _ridge_A_hat(X[:, :s], Y[:, :s], lam=lam)

