"""bayes_lti package initialization.

Sets default dtype to float64 and exposes common utilities.
"""
from __future__ import annotations

import torch

# Default to double precision
torch.set_default_dtype(torch.float64)

__all__ = [
    "torch",
]
