"""Related-work baseline methods for bayes_lti.

Each baseline implements a lightweight interface:
- fit_global(train_tasks, support_len): optional meta-fitting using training tasks
- fit_task(task, support_len): produce an A_hat for the given task using only support data
"""

from .base import Baseline
from .ols import OLSBaseline
from .ridge import RidgeBaseline
from .pooled_prior_ridge import PooledPriorRidgeBaseline
from .shared_subspace import SharedSubspaceBaseline
from .registry import make_baseline, list_baselines

__all__ = [
    "Baseline",
    "OLSBaseline",
    "RidgeBaseline",
    "PooledPriorRidgeBaseline",
    "SharedSubspaceBaseline",
    "make_baseline",
    "list_baselines",
]
