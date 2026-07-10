from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .base import Baseline, Task, split_support_query


def _ols_full(X: np.ndarray, Y: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    # Stable pooled OLS: solve X^T A^T ≈ Y^T.
    Xt = np.asarray(X, dtype=float).T
    Yt = np.asarray(Y, dtype=float).T
    A_T, *_ = np.linalg.lstsq(Xt, Yt, rcond=None)
    return A_T.T


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


def _fit_subspace_coeffs(X: np.ndarray, Y: np.ndarray, a0: np.ndarray, U: np.ndarray, lam: float) -> np.ndarray:
    r"""Solve for c in vec(A)=a0+U c using ridge on support set.

    Uses vec(AX) = (X^T \otimes I) vec(A), with column-major vec.
    """
    n, T = X.shape
    k = U.shape[1]

    B = np.kron(X.T, np.eye(n))  # (nT, n^2)
    y = Y.reshape(-1, order="F")

    r = y - (B @ a0)
    G = B @ U

    GTG = G.T @ G
    rhs = G.T @ r
    c = np.linalg.solve(GTG + float(lam) * np.eye(k), rhs)
    return c


@dataclass
class SharedSubspaceBaseline(Baseline):
    k: int
    query_len: int
    lambdas: List[float]

    a0: Optional[np.ndarray] = None
    U: Optional[np.ndarray] = None
    tuned_lambda: Optional[float] = None

    @property
    def name(self) -> str:
        return "shared_subspace"

    def fit_global(self, train_tasks: List[Task], support_len: int) -> None:
        if len(train_tasks) == 0:
            raise ValueError("shared_subspace requires training tasks")

        A_vecs = []
        n = int(np.asarray(train_tasks[0]["X"]).shape[0])
        for t in train_tasks:
            X = np.asarray(t["X"], dtype=float)
            Y = np.asarray(t["Y"], dtype=float)
            A = _ols_full(X, Y)
            A_vecs.append(A.reshape(-1, order="F"))

        A_mat = np.stack(A_vecs, axis=0)  # (M, n^2)
        a0 = A_mat.mean(axis=0)
        Z = A_mat - a0[None, :]

        # PCA via SVD on centered data matrix
        # Z = U_svd S Vt, components are rows of Vt
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        k_eff = int(max(1, min(self.k, Vt.shape[0])))
        U = Vt[:k_eff].T  # (n^2, k)

        self.a0 = a0
        self.U = U

        # Tune ridge on subspace coefficients using training tasks rollout MSE
        grid = [float(l) for l in self.lambdas] if len(self.lambdas) > 0 else [1e-6, 1e-4, 1e-3, 1e-2]
        best_lam = grid[0]
        best = float("inf")
        for lam in grid:
            mses = []
            for t in train_tasks:
                sp = split_support_query(t, support_len=support_len, query_len=self.query_len)
                c = _fit_subspace_coeffs(sp["X_support"], sp["Y_support"], a0=self.a0, U=self.U, lam=lam)
                a = self.a0 + (self.U @ c)
                A_hat = a.reshape((n, n), order="F")
                mses.append(_free_rollout_mse(A_hat, sp["x0_rollout"], sp["Y_query"]))
            score = float(np.nanmean(mses))
            if score < best:
                best = score
                best_lam = lam

        self.tuned_lambda = float(best_lam)

    def fit_task(self, task: Task, support_len: int) -> np.ndarray:
        if self.a0 is None or self.U is None:
            raise RuntimeError("fit_global must be called before fit_task")

        lam = float(self.tuned_lambda) if self.tuned_lambda is not None else (float(self.lambdas[0]) if len(self.lambdas) > 0 else 1e-3)

        X = np.asarray(task["X"], dtype=float)
        Y = np.asarray(task["Y"], dtype=float)
        T = X.shape[1]
        s = int(max(1, min(support_len, T)))

        c = _fit_subspace_coeffs(X[:, :s], Y[:, :s], a0=self.a0, U=self.U, lam=lam)
        a = self.a0 + (self.U @ c)
        n = X.shape[0]
        return a.reshape((n, n), order="F")

