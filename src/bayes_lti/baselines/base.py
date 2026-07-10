from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


Task = Dict[str, np.ndarray]


class Baseline(ABC):
    """Baseline interface for related-work methods."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def fit_global(self, train_tasks: List[Task], support_len: int) -> None:
        """Optional global fit on training tasks.

        Args:
            train_tasks: list of task dicts with keys X,Y,(A_true)
            support_len: support window length used at eval time
        """

    @abstractmethod
    def fit_task(self, task: Task, support_len: int) -> np.ndarray:
        """Fit the method to a single task support set.

        Must only use the first `support_len` columns of task["X"], task["Y"].

        Returns:
            A_hat: (n,n) estimate.
        """
        raise NotImplementedError


def split_support_query(task: Task, support_len: int, query_len: int) -> Dict[str, Any]:
    X = np.asarray(task["X"])
    Y = np.asarray(task["Y"])
    n, T = X.shape

    s = int(max(1, min(support_len, T)))
    # Rollout compares to Y[:, s:s+q], requiring s+q <= T
    q = int(max(0, min(query_len, max(T - s, 0))))

    Xs = X[:, :s]
    Ys = Y[:, :s]

    x0 = X[:, s] if s < T else Y[:, s - 1]
    Yq = Y[:, s : s + q] if q > 0 else Y[:, 0:0]

    return {
        "n": n,
        "T": T,
        "support_len_used": s,
        "query_len_used": q,
        "X_support": Xs,
        "Y_support": Ys,
        "x0_rollout": x0,
        "Y_query": Yq,
    }

