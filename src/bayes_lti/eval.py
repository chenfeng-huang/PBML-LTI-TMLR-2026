from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .baselines.base import split_support_query
from .baselines.registry import make_baseline
from .data import load_dataset
from .model import posterior, load_meta_params


def A_mse(A_hat: np.ndarray, A_true: np.ndarray) -> float:
    # Per-system MSE: sum over matrix entries (||·||_F^2).
    # Aggregation over systems (tasks) is handled by averaging these values.
    return float(np.sum((A_hat - A_true) ** 2))


def traj_rollout_mse(A_hat: np.ndarray, x0: np.ndarray, Y_query: np.ndarray) -> float:
    """Free rollout from x0; compare predicted states to Y_query columns."""
    n = x0.shape[0]
    q = int(Y_query.shape[1])
    if q <= 0:
        return float("nan")
    x = x0.copy()
    err = 0.0
    for t in range(q):
        x = A_hat @ x
        err += float(np.sum((x - Y_query[:, t]) ** 2))
    return err


def _free_rollout_mse(A_hat: np.ndarray, x0: np.ndarray, Y_true: np.ndarray) -> float:
    """Free rollout from x0; compare predicted states to Y_true columns."""
    n = int(x0.shape[0])
    q = int(Y_true.shape[1])
    if q <= 0:
        return float("inf")
    x = x0.copy()
    err = 0.0
    for t in range(q):
        x = A_hat @ x
        err += float(np.sum((x - Y_true[:, t]) ** 2))
    return err / (q * n)


def _default_support_grid(max_s: int) -> List[int]:
    """Candidate prefix lengths for auto-support selection."""
    max_s = int(max_s)
    if max_s <= 0:
        return [1]
    if max_s <= 25:
        return list(range(1, max_s + 1))
    grid = {1, 2, 3, 5, 8, 10, 13, 15, 20, 25, 30, 40, 50, 75, 100, max_s}
    return sorted([g for g in grid if 1 <= g <= max_s])


def _parse_support_grid(grid: Optional[str], max_s: int) -> List[int]:
    if grid is None or str(grid).strip() == "":
        return _default_support_grid(max_s)
    out: List[int] = []
    for tok in str(grid).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            continue
    out = sorted(set([int(x) for x in out if int(x) >= 1]))
    out = [x for x in out if x <= int(max_s)]
    return out if len(out) > 0 else _default_support_grid(max_s)


def _auto_select_support_fit_len(
    *,
    method: str,
    task: Dict[str, np.ndarray],
    support_max: int,
    val_len: int,
    grid: List[int],
    metric: str,
    meta_params,
    baseline,
    device: str,
) -> int:
    """Choose a support prefix length <= support_max via inner validation.

    We fit using the first s columns and validate on the next `val_len` steps
    (still within the support window). This makes longer support windows "safe"
    because we can always reuse the previously best s.
    """
    X = np.asarray(task["X"], dtype=float)
    Y = np.asarray(task["Y"], dtype=float)
    T = int(X.shape[1])
    s_max = int(max(1, min(int(support_max), T)))
    v = int(max(1, min(int(val_len), max(s_max - 1, 0))))
    max_fit = int(max(1, s_max - v))

    candidates = [int(s) for s in grid if 1 <= int(s) <= max_fit]
    if len(candidates) == 0:
        return max_fit

    metric = str(metric).strip().lower()
    if metric not in ("auto", "a_mse", "rollout"):
        metric = "auto"

    best_s = candidates[0]
    best_score = float("inf")

    for s in candidates:
        # Fit A_hat with first s columns.
        if method == "meta":
            Xt = torch.from_numpy(X[:, :s]).to(device=device, dtype=torch.get_default_dtype())
            Yt = torch.from_numpy(Y[:, :s]).to(device=device, dtype=torch.get_default_dtype())
            with torch.no_grad():
                M, _ = posterior(Xt, Yt, meta_params.V, meta_params.W, sigma2=meta_params.sigma2)
            A_hat = M.cpu().numpy()
        else:
            A_hat = baseline.fit_task(task, support_len=int(s))

        # Score candidate.
        use_a_mse = (metric == "a_mse") or (metric == "auto" and "A_true" in task)
        if use_a_mse:
            score = A_mse(A_hat, np.asarray(task["A_true"], dtype=float))
        else:
            # Validate on the next v steps within the support window.
            x0 = X[:, s]
            Yv = Y[:, s : s + v]
            score = _free_rollout_mse(A_hat, x0, Yv)
        # Tie-break: prefer larger s if scores are equal.
        if (score < best_score - 1e-12) or (abs(score - best_score) <= 1e-12 and s > best_s):
            best_score = score
            best_s = int(s)

    return int(best_s)


def evaluate_methods(
    *,
    data_path: str,
    methods: List[str],
    support_len: int,
    query_len: int,
    ridge_lambdas: List[float],
    subspace_k: int,
    auto_support: bool = False,
    auto_support_val_len: int = 5,
    auto_support_grid: Optional[str] = None,
    auto_support_metric: str = "auto",
    ckpt: Optional[str] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    data = load_dataset(data_path)
    train_tasks = list(data.get("train", []))
    test_tasks = list(data.get("test", []))

    if len(test_tasks) == 0:
        raise ValueError(
            "Dataset contains no test tasks. "
            f"(train={len(train_tasks)}, val={len(data.get('val', []))}, test=0). "
            "If you generated a long dataset with --M-test 0, evaluate on the selected dataset (e.g. data/dataset_selected.npz)."
        )

    n = int(np.asarray(test_tasks[0]["X"]).shape[0])
    methods = [m.strip() for m in methods if m.strip()]

    # Meta (PAC-Bayes trained) parameters for 'meta' method.
    meta_params = load_meta_params(ckpt, device=device) if ("meta" in methods) else None

    # Instantiate all non-meta baselines.
    baselines: Dict[str, Any] = {}
    for m in methods:
        if m == "meta":
            continue
        baselines[m] = make_baseline(
            m,
            query_len=int(query_len),
            ridge_lambdas=list(ridge_lambdas),
            subspace_k=int(subspace_k),
            device=str(device),
        )

    # Fit global state for baselines (no test leakage).
    for m, b in baselines.items():
        b.fit_global(train_tasks, support_len=int(support_len))

    # Evaluate per method.
    results: Dict[str, Any] = {}
    for m in methods:
        A_mses: List[float] = []
        traj_mses: List[float] = []
        support_used: List[int] = []
        support_fit_used: List[int] = []
        query_used: List[int] = []

        for t in test_tasks:
            sp = split_support_query(t, support_len=int(support_len), query_len=int(query_len))
            support_used.append(int(sp["support_len_used"]))
            query_used.append(int(sp["query_len_used"]))

            # Optionally choose a safer prefix length for fitting, using an inner
            # validation window that stays inside the support segment.
            s_fit = int(sp["support_len_used"])
            if bool(auto_support):
                grid = _parse_support_grid(auto_support_grid, max_s=int(sp["support_len_used"]))
                if m == "meta":
                    if meta_params is None:
                        raise ValueError("Method 'meta' requires --ckpt.")
                    s_fit = _auto_select_support_fit_len(
                        method=m,
                        task=t,
                        support_max=int(sp["support_len_used"]),
                        val_len=int(auto_support_val_len),
                        grid=grid,
                        metric=str(auto_support_metric),
                        meta_params=meta_params,
                        baseline=None,
                        device=str(device),
                    )
                else:
                    s_fit = _auto_select_support_fit_len(
                        method=m,
                        task=t,
                        support_max=int(sp["support_len_used"]),
                        val_len=int(auto_support_val_len),
                        grid=grid,
                        metric=str(auto_support_metric),
                        meta_params=None,
                        baseline=baselines[m],
                        device=str(device),
                    )
            support_fit_used.append(int(s_fit))

            if m == "meta":
                if meta_params is None:
                    raise ValueError("Method 'meta' requires --ckpt.")
                # Fit using the first s_fit columns (prefix) of the task.
                Xt = torch.from_numpy(np.asarray(t["X"])[:, : int(s_fit)]).to(
                    device=device, dtype=torch.get_default_dtype()
                )
                Yt = torch.from_numpy(np.asarray(t["Y"])[:, : int(s_fit)]).to(
                    device=device, dtype=torch.get_default_dtype()
                )
                with torch.no_grad():
                    M, _ = posterior(Xt, Yt, meta_params.V, meta_params.W, sigma2=meta_params.sigma2)
                A_hat = M.cpu().numpy()
            else:
                A_hat = baselines[m].fit_task(t, support_len=int(s_fit))

            if "A_true" in t:
                A_mses.append(A_mse(A_hat, np.asarray(t["A_true"], dtype=float)))
            traj_mses.append(traj_rollout_mse(A_hat, sp["x0_rollout"], sp["Y_query"]))

        out = {
            "A_mse_mean": float(np.nanmean(A_mses)) if len(A_mses) > 0 else float("nan"),
            "A_mse_std": float(np.nanstd(A_mses)) if len(A_mses) > 0 else float("nan"),
            "traj_mse_mean": float(np.nanmean(traj_mses)),
            "traj_mse_std": float(np.nanstd(traj_mses)),
            "support_len_used_mean": float(np.mean(support_used)) if len(support_used) > 0 else float("nan"),
            "support_fit_len_used_mean": float(np.mean(support_fit_used)) if len(support_fit_used) > 0 else float("nan"),
            "query_len_used_mean": float(np.mean(query_used)) if len(query_used) > 0 else float("nan"),
        }

        # Helpful hyperparam notes.
        b = baselines.get(m)
        if b is not None:
            if hasattr(b, "tuned_lambda"):
                out["tuned_lambda"] = getattr(b, "tuned_lambda")
            if hasattr(b, "k"):
                out["subspace_k_used"] = getattr(b, "k")

        results[m] = out

    return {
        "support_len": int(support_len),
        "query_len": int(query_len),
        "n_state": int(n),
        "num_train_tasks": int(len(train_tasks)),
        "num_test_tasks": int(len(test_tasks)),
        "results": results,
    }


def save_report(path: str, report: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


