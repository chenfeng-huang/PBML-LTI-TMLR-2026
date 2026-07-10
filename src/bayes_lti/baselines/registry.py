from __future__ import annotations

from typing import Dict, List, Optional

from .base import Baseline
from .ols import OLSBaseline
from .ridge import RidgeBaseline
from .pooled_prior_ridge import PooledPriorRidgeBaseline
from .shared_subspace import SharedSubspaceBaseline


def list_baselines() -> List[str]:
    return [
        "ols",
        "ridge",
        "pooled_prior_ridge",
        "shared_subspace",
    ]


def make_baseline(
    name: str,
    *,
    query_len: int,
    ridge_lambdas: List[float],
    subspace_k: int,
    device: str = "cpu",
) -> Baseline:
    name = str(name).strip()

    if name == "ols":
        return OLSBaseline()
    if name == "ridge":
        return RidgeBaseline(lambdas=ridge_lambdas, query_len=int(query_len))
    if name == "pooled_prior_ridge":
        return PooledPriorRidgeBaseline(lambdas=ridge_lambdas, query_len=int(query_len))
    if name == "shared_subspace":
        return SharedSubspaceBaseline(k=int(subspace_k), query_len=int(query_len), lambdas=ridge_lambdas)

    raise ValueError(f"Unknown baseline: {name}")
