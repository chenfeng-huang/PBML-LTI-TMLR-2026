from __future__ import annotations

from typing import Optional, Sequence
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def save_y_plots(Y_true: np.ndarray, Y_pred: np.ndarray, out_path: Path, drop_first: int = 0) -> None:
    n, T = Y_true.shape
    assert Y_pred.shape == (n, T)
    # Optionally drop an initial transient segment from both series
    d = int(max(0, drop_first))
    if d > 0 and T > d:
        Y_true = Y_true[:, d:]
        Y_pred = Y_pred[:, d:]
        T = Y_true.shape[1]

    fig, axes = plt.subplots(n, 1, figsize=(10, max(2.5 * n, 3)), sharex=True)
    if n == 1:
        axes = [axes]  # type: ignore[list-item]

    time = np.arange(T)
    for i in range(n):
        ax = axes[i]
        ax.plot(time, Y_true[i], label="y_true", color="black", linewidth=1.5)
        ax.plot(time, Y_pred[i], label="y_pred", color="tab:blue", linewidth=1.5, alpha=0.9)
        ax.set_ylabel(f"dim {i}")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def save_A_entry_distribution(A_matrices: Sequence[np.ndarray], out_path: Path, bins: int = 50) -> None:
    if len(A_matrices) == 0:
        raise ValueError("No A matrices provided to plot distribution.")
    vals = np.concatenate([A.ravel() for A in A_matrices], axis=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals, bins=bins, density=True, color="tab:blue", alpha=0.75, edgecolor="white")
    ax.set_xlabel("A entries")
    ax.set_ylabel("density")
    ax.set_title("Distribution of A entries across tasks")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
def save_A_task_mean_distribution(A_matrices: Sequence[np.ndarray], out_path: Path, bins: int = 50) -> None:
    if len(A_matrices) == 0:
        raise ValueError("No A matrices provided to plot mean distribution.")
    means = np.array([float(np.mean(A)) for A in A_matrices], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(means, bins=bins, density=True, color="tab:green", alpha=0.75, edgecolor="white")
    ax.set_xlabel("mean(A) per task")
    ax.set_ylabel("density")
    ax.set_title("Distribution of mean(A) across tasks")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
def save_A_task_mean_distribution_compare(
    A_all: Sequence[np.ndarray],
    A_selected: Sequence[np.ndarray] | None,
    A_selected_test: Sequence[np.ndarray] | None,
    A_edge: Sequence[np.ndarray] | None,
    out_path: Path,
    bins: int = 60,
) -> None:
    if len(A_all) == 0:
        raise ValueError("No A matrices in full dataset to plot.")
    means_all = np.array([float(np.mean(A)) for A in A_all], dtype=float)
    means_sel_trainval = (
        np.array([float(np.mean(A)) for A in (A_selected or [])], dtype=float) if A_selected is not None else np.array([], dtype=float)
    )
    means_sel_test = (
        np.array([float(np.mean(A)) for A in (A_selected_test or [])], dtype=float) if A_selected_test is not None else np.array([], dtype=float)
    )
    means_edge = np.array([float(np.mean(A)) for A in (A_edge or [])], dtype=float) if A_edge is not None else np.array([], dtype=float)
    vmin = float(np.min(means_all))
    vmax = float(np.max(means_all))
    if np.size(means_sel_trainval) > 0:
        vmin = min(vmin, float(np.min(means_sel_trainval)))
        vmax = max(vmax, float(np.max(means_sel_trainval)))
    if np.size(means_sel_test) > 0:
        vmin = min(vmin, float(np.min(means_sel_test)))
        vmax = max(vmax, float(np.max(means_sel_test)))
    if np.size(means_edge) > 0:
        vmin = min(vmin, float(np.min(means_edge)))
        vmax = max(vmax, float(np.max(means_edge)))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        vmin, vmax = -1.0, 1.0
    bin_edges = np.linspace(vmin, vmax, int(bins) + 1)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    # Use dataset_long ("all") as the density base.
    # - all_density integrates to 1
    # - selected_density_base integrates to N_selected / N_all (so it won't look artificially tall)
    counts_all, _ = np.histogram(means_all, bins=bin_edges, density=False)
    widths = np.diff(bin_edges)
    N_all = float(max(means_all.size, 1))
    all_density = counts_all / (N_all * widths)

    ax.stairs(all_density, bin_edges, fill=True, color="tab:blue", alpha=0.35, label="all samples")

    if means_sel_trainval.size > 0:
        counts_sel_tv, _ = np.histogram(means_sel_trainval, bins=bin_edges, density=False)
        # Scale by N_all, so overlays are comparable to "all".
        sel_tv_density_base = counts_sel_tv / (N_all * widths)
        ax.stairs(sel_tv_density_base, bin_edges, fill=True, color="tab:orange", alpha=0.55, label="train/val")

    if means_sel_test.size > 0:
        counts_sel_test, _ = np.histogram(means_sel_test, bins=bin_edges, density=False)
        sel_test_density_base = counts_sel_test / (N_all * widths)
        ax.stairs(sel_test_density_base, bin_edges, fill=True, color="tab:green", alpha=0.55, label="test (common_case)")

    # Edge: show in the SAME histogram style and density units as others.
    if means_edge.size > 0:
        counts_edge, _ = np.histogram(means_edge, bins=bin_edges, density=False)
        edge_density_base = counts_edge / (N_all * widths)
        ax.stairs(edge_density_base, bin_edges, fill=True, color="tab:red", alpha=0.55, label="test (edge_case)")
    ax.set_xlabel("mean(A) per task")
    ax.set_ylabel("density")
    ax.set_title("mean(A) distribution: all vs train/val vs test(common/edge)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
def save_A_entry_distribution_compare(
    A_regular: Sequence[np.ndarray],
    A_edge: Sequence[np.ndarray],
    out_path: Path,
    bins: int = 50,
    labels: tuple[str, str] = ("regular", "extreme values"),
    smooth: bool = False,
    bandwidth: float | None = None,
    line: bool = False,
    hide_bars: bool = False,
) -> None:
    if len(A_regular) == 0 or len(A_edge) == 0:
        raise ValueError("Both regular and edge A sequences must be non-empty.")
    vals_reg = np.concatenate([A.ravel() for A in A_regular], axis=0)
    vals_edge = np.concatenate([A.ravel() for A in A_edge], axis=0)
    vmin = float(min(np.min(vals_reg), np.min(vals_edge)))
    vmax = float(max(np.max(vals_reg), np.max(vals_edge)))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = -1.0, 1.0
    if abs(vmax - vmin) < 1e-12:
        delta = 1e-2 if vmax == 0 else 0.05 * abs(vmax)
        vmin, vmax = vmax - delta, vmax + delta
    bin_edges = np.linspace(vmin, vmax, int(bins) + 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    # Optional histogram outlines for reference
    if not hide_bars:
        ax.hist(vals_reg, bins=bin_edges, density=True, histtype="step", linewidth=1.2, color="tab:blue", alpha=0.45, label=labels[0])
        ax.hist(vals_edge, bins=bin_edges, density=True, histtype="step", linewidth=1.2, color="tab:orange", alpha=0.45, label=labels[1])

    if line:
        # Connect histogram bin centers with straight lines (no kernelization)
        def _polyline(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            counts, edges = np.histogram(samples, bins=bin_edges, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            x_line = np.linspace(vmin, vmax, 512)
            y_line = np.interp(x_line, centers, counts)
            return x_line, y_line
        xr, yr = _polyline(vals_reg)
        xe, ye = _polyline(vals_edge)
        ax.plot(xr, yr, color="tab:blue", linewidth=2.5, label=f"{labels[0]} (line)")
        ax.plot(xe, ye, color="tab:orange", linewidth=2.5, label=f"{labels[1]} (line)")

    if smooth:
        # Simple Gaussian KDE implemented with NumPy (Scott's rule by default)
        def _kde_curve(samples: np.ndarray, x_grid: np.ndarray, bw: float | None) -> np.ndarray:
            n = float(samples.size)
            if n <= 1:
                return np.zeros_like(x_grid)
            std = float(np.std(samples)) if np.size(samples) > 0 else 1.0
            if not np.isfinite(std) or std <= 0:
                std = 1.0
            h = float(bw) if bw is not None and bw > 0 else 1.06 * std * (n ** (-1.0 / 5.0))
            h = max(h, 1e-6)
            # Evaluate density: (1/(n*h*sqrt(2π))) * sum exp(-0.5 * ((x - xi)/h)^2)
            diffs = (x_grid[:, None] - samples[None, :]) / h
            densities = np.exp(-0.5 * (diffs ** 2))
            denom = (samples.size * h * np.sqrt(2.0 * np.pi))
            return np.sum(densities, axis=1) / denom

        x_grid = np.linspace(vmin, vmax, 512)
        y_reg = _kde_curve(vals_reg, x_grid, bandwidth)
        y_edge = _kde_curve(vals_edge, x_grid, bandwidth)
        ax.plot(x_grid, y_reg, color="tab:blue", linewidth=2.5, label=f"{labels[0]} (smoothed)")
        ax.plot(x_grid, y_edge, color="tab:orange", linewidth=2.5, label=f"{labels[1]} (smoothed)")
    ax.set_xlabel("A entries")
    ax.set_ylabel("density")
    ax.set_title(f"Distribution of A entries: {labels[0]} vs {labels[1]}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
def _symmetric_limits(*arrays: np.ndarray) -> float:
    max_abs = 0.0
    for a in arrays:
        if a is None:
            continue
        max_abs = max(max_abs, float(np.max(np.abs(a))))
    return max_abs if max_abs > 0 else 1.0


def save_A_heatmaps(A_hat: np.ndarray, A_true: Optional[np.ndarray], out_path: Path) -> None:
    if A_true is not None:
        cols = 3
    else:
        cols = 1

    fig, axes = plt.subplots(1, cols, figsize=(4 * cols + 2, 4), squeeze=False)
    axes = axes[0]

    # Common limits for A matrices
    lim = _symmetric_limits(A_hat, A_true if A_true is not None else A_hat)

    im0 = axes[0].imshow(A_hat, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[0].set_title("A_hat")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    if A_true is not None:
        im1 = axes[1].imshow(A_true, cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[1].set_title("A_true")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Absolute error heatmap
        err_abs = np.abs(A_hat - A_true)
        # Use a tighter range: smaller of error max and A panels' magnitude limit
        err_max = float(np.max(err_abs)) if np.size(err_abs) > 0 else 1.0
        base_max = float(lim)
        vmax = min(base_max, err_max) if err_max > 0 else 1.0
        # Colored map with white at zero; higher differences are darker red
        im2 = axes[2].imshow(err_abs, cmap="Reds", vmin=0.0, vmax=vmax)
        axes[2].set_title("|A_hat - A_true|")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("column")
        ax.set_ylabel("row")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


