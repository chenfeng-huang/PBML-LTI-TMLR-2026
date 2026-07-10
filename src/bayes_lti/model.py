from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .math_utils import spd_from_cholesky_params, safe_cholesky


class MetaParams(nn.Module):
    """Hierarchical meta-parameters (W, V, sigma^2) with SPD parameterization for V.

    Attributes:
        W: free n x n matrix.
        L: raw lower-triangular parameter for V.
        alpha: scalar raw for V jitter.
        s_log: scalar raw for sigma^2.
    """

    def __init__(self, n: int):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(n, n))
        self.L = nn.Parameter(0.01 * torch.randn(n, n))
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.s_log = nn.Parameter(torch.tensor(-2.0))
        self.sigma_min2 = 1e-5

    @property
    def V(self) -> torch.Tensor:
        return spd_from_cholesky_params(self.L, self.alpha, min_alpha=1e-6)

    @property
    def sigma2(self) -> torch.Tensor:
        return F.softplus(self.s_log) + self.sigma_min2


def _woodbury_posterior_Vm(V: torch.Tensor, X: torch.Tensor, sigma2: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute V_m via Woodbury.

    General form with observation noise variance sigma2:
      (V^{-1} + (1/sigma2) X X^T)^{-1} = V - V X (sigma2 I + X^T V X)^{-1} X^T V

    If sigma2 is None, we use sigma2 = 1 to preserve prior behavior.
    """
    # V: (n,n), X: (n,T)
    VX = V @ X
    if sigma2 is None:
        S = torch.eye(X.shape[1], dtype=V.dtype, device=V.device) + X.mH @ VX
    else:
        # S = sigma2 * I + X^T V X
        S = sigma2 * torch.eye(X.shape[1], dtype=V.dtype, device=V.device) + X.mH @ VX
    Ls = safe_cholesky(S)
    # Z = S^{-1} (X^T V)
    Z = torch.cholesky_solve(VX.mH, Ls)  # shape (T, n)
    Vm = V - VX @ Z
    return Vm


def posterior(
    X: torch.Tensor, Y: torch.Tensor, V: torch.Tensor, W: torch.Tensor, sigma2: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Closed-form task posterior parameters (M_m, V_m).

    Args:
        X: (n,T)
        Y: (n,T)
        V: (n,n) SPD
        W: (n,n)
        sigma2: optional observation noise variance (scalar tensor). If None, assumes sigma2 = 1.
    Returns:
        (M_m, V_m)
    """
    Vm = _woodbury_posterior_Vm(V, X, sigma2=sigma2)

    # Compute W V^{-1} without explicit inverse via Cholesky solve
    LV = safe_cholesky(V)
    WVinv_T = torch.cholesky_solve(W.mH, LV)  # (n,n)
    WVinv = WVinv_T.mH

    if sigma2 is None:
        M = (Y @ X.mH + WVinv) @ Vm
    else:
        # Data term weighted by 1/sigma2
        M = ((Y @ X.mH) / sigma2 + WVinv) @ Vm
    return M, Vm


def expected_fit(
    X: torch.Tensor, Y: torch.Tensor, M: torch.Tensor, Vm: torch.Tensor, sigma2: torch.Tensor
) -> torch.Tensor:
    """Expected negative log-likelihood term (drop constants)."""
    n, T = X.shape
    resid = Y - M @ X
    term1 = T * n * torch.log(sigma2)
    term2 = (resid.pow(2).sum()) / sigma2
    XXt = X @ X.mH
    term3 = n * torch.trace(XXt @ Vm)
    return 0.5 * (term1 + term2 + term3)


def kl_matrix_normal(
    M: torch.Tensor,
    Vm: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    sigma2: torch.Tensor,
) -> torch.Tensor:
    """KL between MN(M, Sigma, Vm) and MN(W, Sigma, V) with Sigma = sigma2 * I.

    Uses Kronecker-free formula for the mean term.
    """
    n = V.shape[0]

    # logdet terms via Cholesky
    LV = safe_cholesky(V)
    LVm = safe_cholesky(Vm)
    logdetV = 2.0 * torch.log(torch.diag(LV)).sum()
    logdetVm = 2.0 * torch.log(torch.diag(LVm)).sum()

    # trace(V^{-1} Vm)
    # Solve V X = Vm  => X = V^{-1} Vm
    Xsol = torch.cholesky_solve(Vm, LV)
    tr_term = torch.trace(Xsol)

    # Mean term: (1/sigma2) tr((M-W) V^{-1} (M-W)^T)
    D = M - W
    B = torch.cholesky_solve(D.mH, LV)  # V^{-1} D^T shape (n,n)
    mean_term = torch.trace(D @ B) / sigma2

    kl = 0.5 * (n * (logdetV - logdetVm) - n * n + n * tr_term + mean_term)
    return kl


def stability_penalty(W: torch.Tensor, rho0: float) -> torch.Tensor:
    """Soft stability penalty max(0, rho(W)-rho0)^2.

    Uses spectral norm as a differentiable surrogate upper-bounding the spectral radius.
    """
    # Spectral norm (largest singular value)
    sigma_max = torch.linalg.matrix_norm(W, ord=2)
    excess = torch.relu(sigma_max - rho0)
    return excess * excess


def hyper_reg(W: torch.Tensor, V: torch.Tensor, tauW: float, lambdaV: float) -> torch.Tensor:
    """Hyper-regularization: 0.5/tauW^2 ||W||_F^2 + lambdaV*(0.5||V-I||_F^2 - logdet(V))."""
    n = V.shape[0]
    I = torch.eye(n, dtype=V.dtype, device=V.device)
    LV = safe_cholesky(V)
    logdetV = 2.0 * torch.log(torch.diag(LV)).sum()
    regW = 0.5 * (W.pow(2).sum()) / (tauW * tauW)
    regV = lambdaV * (0.5 * ((V - I).pow(2).sum()) - logdetV)
    return regW + regV


def load_meta_params(ckpt: str, device: str = "cpu") -> MetaParams:
    """Load trained MetaParams from a checkpoint saved by `train.py`."""
    data: Any = torch.load(ckpt, map_location=device)
    n = int(data["n"]) if "n" in data else int(data["state_dict"]["W"].shape[-1])
    params = MetaParams(n).to(device)
    params.load_state_dict(data["state_dict"])  # type: ignore[arg-type]
    params.eval()
    return params


 
