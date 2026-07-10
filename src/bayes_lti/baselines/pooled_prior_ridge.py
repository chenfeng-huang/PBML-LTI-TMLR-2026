from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .base import Baseline, Task, split_support_query


def _ols_full(X: np.ndarray, Y: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    n = X.shape[0]
    XXt = X @ X.T
    XXt = 0.5 * (XXt + XXt.T)
    try:
        return Y @ X.T @ np.linalg.inv(XXt + jitter * np.eye(n))
    except np.linalg.LinAlgError:
        return Y @ X.T @ np.linalg.pinv(XXt + jitter * np.eye(n))


def _adapt_pooled_prior_ridge(X: np.ndarray, Y: np.ndarray, A_bar: np.ndarray, lam: float) -> np.ndarray:
    n = X.shape[0]
    XXt = X @ X.T
    XXt = 0.5 * (XXt + XXt.T)
    return (Y @ X.T + float(lam) * A_bar) @ np.linalg.inv(XXt + float(lam) * np.eye(n))


def _free_rollout_mse(A: np.ndarray, x0: np.ndarray, Y_true: np.ndarray) -> float:
    n = x0.shape[0]
    q = Y_true.shape[1]
    if q <= 0:
        return float("nan")
    x = x0.copy()
    err = 0.0
    for t in range(q):
        x = A @ x
        err += float(np.sum((x - Y_true[:, t]) ** 2))
    return err / (q * n)


@dataclass
class PooledPriorRidgeBaseline(Baseline):
    lambdas: List[float]
    query_len: int
    A_bar: Optional[np.ndarray] = None
    tuned_lambda: Optional[float] = None

    @property
    def name(self) -> str:
        return "pooled_prior_ridge"

    def fit_global(self, train_tasks: List[Task], support_len: int) -> None:
        # Pooled mean A_bar from training tasks (OLS on the first `support_len` steps).
        # This keeps the prior estimation fair in a few-shot setting.
        A_list = []
        for t in train_tasks:
            X = np.asarray(t["X"], dtype=float)
            Y = np.asarray(t["Y"], dtype=float)
            T = int(X.shape[1])
            s = int(max(1, min(int(support_len), T)))
            A_list.append(_ols_full(X[:, :s], Y[:, :s]))
        self.A_bar = np.mean(np.stack(A_list, axis=0), axis=0) if len(A_list) > 0 else None

        grid = [float(l) for l in self.lambdas] if len(self.lambdas) > 0 else [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
        best_lam = grid[0]
        best = float("inf")
        for lam in grid:
            mses = []
            for t in train_tasks:
                sp = split_support_query(t, support_len=support_len, query_len=self.query_len)
                A_hat = _adapt_pooled_prior_ridge(sp["X_support"], sp["Y_support"], self.A_bar, lam=lam)  # type: ignore[arg-type]
                mses.append(_free_rollout_mse(A_hat, sp["x0_rollout"], sp["Y_query"]))
            score = float(np.nanmean(mses))
            if score < best:
                best = score
                best_lam = lam

        self.tuned_lambda = float(best_lam)

    def fit_task(self, task: Task, support_len: int) -> np.ndarray:
        if self.A_bar is None:
            # Fallback: no training tasks
            self.A_bar = np.zeros((task["X"].shape[0], task["X"].shape[0]), dtype=float)
        lam = float(self.tuned_lambda) if self.tuned_lambda is not None else (float(self.lambdas[0]) if len(self.lambdas) > 0 else 1e-3)
        X = np.asarray(task["X"], dtype=float)
        Y = np.asarray(task["Y"], dtype=float)
        T = X.shape[1]
        s = int(max(1, min(support_len, T)))
        return _adapt_pooled_prior_ridge(X[:, :s], Y[:, :s], self.A_bar, lam=lam)


