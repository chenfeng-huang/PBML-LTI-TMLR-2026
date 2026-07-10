from __future__ import annotations

from typing import Optional
import torch
import torch.nn.functional as F


def spectral_radius(A: torch.Tensor) -> float:
    """Compute spectral radius (max |eig|) using eigenvalues (CPU if needed).

    Note: Used for diagnostics and stability projection; not used for gradients.
    """
    with torch.no_grad():
        # Move to CPU for potentially more robust complex eigvals
        A_cpu = A.detach().cpu().to(torch.float64)
        vals = torch.linalg.eigvals(A_cpu)
        rho = torch.max(vals.abs()).item()
    return float(rho)


def project_stable(W: torch.Tensor, rho0: float) -> torch.Tensor:
    """Project W to have spectral radius <= rho0 by uniform scaling if needed.

    Args:
        W: square matrix parameter (will be updated in-place under no_grad).
        rho0: desired max spectral radius (<1).
    Returns:
        Projected tensor (same object as W).
    """
    with torch.no_grad():
        rho = spectral_radius(W)
        if rho > rho0:
            scale = rho0 / (rho + 1e-12)
            W.mul_(scale)
    return W


def spd_from_cholesky_params(
    L_unconstrained: torch.Tensor, alpha_unconstrained: torch.Tensor, min_alpha: float = 1e-6
) -> torch.Tensor:
    """Construct SPD matrix V = L L^T + alpha I from unconstrained params.

    Args:
        L_unconstrained: (n,n) raw parameter; lower-triangular part is used.
        alpha_unconstrained: scalar parameter; softplus to ensure positivity.
        min_alpha: minimal added jitter through softplus shift.
    Returns:
        SPD matrix V.
    """
    n = L_unconstrained.shape[0]
    L = torch.tril(L_unconstrained)
    alpha = F.softplus(alpha_unconstrained) + min_alpha
    V = L @ L.mH + alpha * torch.eye(n, dtype=L.dtype, device=L.device)
    return V


def safe_cholesky(A: torch.Tensor, jitter: float = 1e-8, max_tries: int = 5) -> torch.Tensor:
    """Cholesky with adaptive jitter added to the diagonal if needed."""
    assert A.shape[0] == A.shape[1]
    diag_eye = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    for i in range(max_tries):
        try:
            return torch.linalg.cholesky(A)
        except RuntimeError:
            A = A + (jitter * (10.0 ** i)) * diag_eye
    # final attempt (may raise)
    return torch.linalg.cholesky(A)
