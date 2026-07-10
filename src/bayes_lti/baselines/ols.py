from __future__ import annotations

import numpy as np

from .base import Baseline, Task


def _ols_A_hat(X: np.ndarray, Y: np.ndarray, ridge_jitter: float = 1e-10) -> np.ndarray:
    """Closed-form least squares: A = Y X^T (X X^T)^{-1}."""
    n = X.shape[0]
    XXt = X @ X.T
    XXt = 0.5 * (XXt + XXt.T)
    try:
        A = Y @ X.T @ np.linalg.inv(XXt + ridge_jitter * np.eye(n))
    except np.linalg.LinAlgError:
        A = Y @ X.T @ np.linalg.pinv(XXt + ridge_jitter * np.eye(n))
    return A


class OLSBaseline(Baseline):
    @property
    def name(self) -> str:
        return "ols"

    def fit_task(self, task: Task, support_len: int) -> np.ndarray:
        X = np.asarray(task["X"], dtype=float)
        Y = np.asarray(task["Y"], dtype=float)
        T = X.shape[1]
        s = int(max(1, min(support_len, T)))
        return _ols_A_hat(X[:, :s], Y[:, :s])

