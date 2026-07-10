from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .data import load_dataset
from .math_utils import project_stable, spectral_radius
from .model import MetaParams, posterior, expected_fit, kl_matrix_normal, hyper_reg, stability_penalty


@dataclass
class TrainConfig:
    data: str
    steps: int = 2000
    batch: int = 32
    lr: float = 1e-3
    tauW: float = 5.0
    lambdaV: float = 1e-2
    gamma: float = 1.0
    eta: float = 1.0
    rho0: float = 0.98
    device: str = "cpu"
    seed: int = 1
    out_dir: str = "runs"
    patience: int = 200


def _to_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(x).to(device=device, dtype=torch.get_default_dtype())


def _prepare_tasks(tasks: List[Dict[str, np.ndarray]], device: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    out = []
    for t in tasks:
        out.append((_to_tensor(t["X"], device), _to_tensor(t["Y"], device)))
    return out


def _loss_for_batch(
    params: MetaParams,
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    tauW: float,
    lambdaV: float,
    gamma: float,
    eta: float,
    rho0: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    V = params.V
    W = params.W
    sigma2 = params.sigma2

    fit_terms = []
    kl_terms = []
    for X, Y in batch:
        M, Vm = posterior(X, Y, V, W, sigma2=sigma2)
        T = X.shape[1]
        fit = expected_fit(X, Y, M, Vm, sigma2)
        kl = kl_matrix_normal(M, Vm, W, V, sigma2)
        fit_terms.append(fit)
        kl_terms.append(kl / T)
    fit_mean = torch.stack(fit_terms).mean()
    kl_mean = torch.stack(kl_terms).mean()

    reg = hyper_reg(W, V, tauW=tauW, lambdaV=lambdaV)
    stab = stability_penalty(W, rho0)

    loss = fit_mean + kl_mean + gamma * reg + eta * stab

    with torch.no_grad():
        stats = {
            "loss": float(loss.item()),
            "fit": float(fit_mean.item()),
            "kl": float(kl_mean.item()),
            "normW": float(torch.linalg.norm(W).item()),
            "logdetV": float(torch.logdet(V).item()),
            "sigma2": float(sigma2.item()),
            "rhoW": float(spectral_radius(W)),
        }
    return loss, stats


def train(cfg: TrainConfig) -> Path:
    # Determinism
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load data
    data = load_dataset(cfg.data)
    device = cfg.device
    train_tasks = _prepare_tasks(data["train"], device)
    val_tasks = _prepare_tasks(data["val"], device)

    # Init params
    n = train_tasks[0][0].shape[0]
    params = MetaParams(n).to(device)

    opt = Adam(params.parameters(), lr=cfg.lr)
    scheduler = CosineAnnealingLR(opt, T_max=max(cfg.steps, 1))

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = out_dir / "train_log.csv"

    best_val = float("inf")
    best_path = out_dir / "last.ckpt"
    bad_steps = 0

    with log_csv.open("w", newline="") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "step",
                "loss",
                "fit",
                "kl",
                "normW",
                "logdetV",
                "sigma2",
                "rhoW",
                "val_bound",
            ],
        )
        writer.writeheader()

        pbar = tqdm(range(cfg.steps), desc="train", total=cfg.steps)
        for step in pbar:
            params.train()
            idx = np.random.choice(len(train_tasks), size=min(cfg.batch, len(train_tasks)), replace=False)
            batch = [train_tasks[i] for i in idx]
            opt.zero_grad(set_to_none=True)
            loss, stats = _loss_for_batch(
                params, batch, cfg.tauW, cfg.lambdaV, cfg.gamma, cfg.eta, cfg.rho0
            )
            loss.backward()
            opt.step()
            scheduler.step()

            # Project for stability
            project_stable(params.W, cfg.rho0)

            # Validation bound (skip if no validation tasks)
            if val_tasks:
                with torch.no_grad():
                    params.eval()
                    vidx = np.random.choice(len(val_tasks), size=min(cfg.batch, len(val_tasks)), replace=False)
                    vbatch = [val_tasks[i] for i in vidx]
                    vloss, _ = _loss_for_batch(
                        params, vbatch, cfg.tauW, cfg.lambdaV, cfg.gamma, cfg.eta, cfg.rho0
                    )
                    val_bound = float(vloss.item())
            else:
                val_bound = float("nan")

            tracking_metric = val_bound if val_tasks else stats["loss"]

            row = {"step": step, **stats, "val_bound": val_bound}
            writer.writerow(row)
            fcsv.flush()
            pbar.set_postfix({"loss": stats["loss"], "val": val_bound})

            # Early stopping tracking
            if tracking_metric + 1e-8 < best_val:
                best_val = tracking_metric
                bad_steps = 0
                # Save best/last
                torch.save(
                    {
                        "state_dict": params.state_dict(),
                        "n": n,
                        "cfg": cfg.__dict__,
                        "step": step,
                        "val_bound": val_bound if val_tasks else None,
                    },
                    best_path,
                )
            else:
                bad_steps += 1
                if bad_steps >= cfg.patience:
                    break

    return best_path
