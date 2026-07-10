from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import generate_dataset, save_dataset, load_dataset
from .data import generate_edge_dataset
from .train import TrainConfig, train
from .eval import evaluate_methods, save_report
from .plot import save_A_entry_distribution, save_A_entry_distribution_compare, save_A_task_mean_distribution, save_A_task_mean_distribution_compare


def main() -> None:
    parser = argparse.ArgumentParser(prog="bayes_lti")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # generate
    p_gen = sub.add_parser("generate", help="Generate synthetic dataset")
    p_gen.add_argument("--out", type=str, required=True)
    p_gen.add_argument("--n", type=int, required=True)
    p_gen.add_argument("--M-train", type=int, required=True, dest="M_train")
    p_gen.add_argument("--M-val", type=int, required=True, dest="M_val")
    p_gen.add_argument("--M-test", type=int, required=True, dest="M_test")
    p_gen.add_argument("--T-min", type=int, required=True, dest="T_min")
    p_gen.add_argument("--T-max", type=int, required=True, dest="T_max")
    p_gen.add_argument("--sigma-true", type=float, required=True, dest="sigma_true")
    p_gen.add_argument("--v-true", type=float, required=True, dest="v_true")
    p_gen.add_argument("--rho0-gen", type=float, required=True, dest="rho0_gen")
    p_gen.add_argument("--seed", type=int, default=0)

    # train
    p_train = sub.add_parser("train", help="Train meta-parameters")
    p_train.add_argument("--data", type=str, required=True)
    p_train.add_argument("--steps", type=int, default=2000)
    p_train.add_argument("--batch", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--tauW", type=float, default=5.0)
    p_train.add_argument("--lambdaV", type=float, default=1e-2)
    p_train.add_argument("--gamma", type=float, default=1.0)
    p_train.add_argument("--eta", type=float, default=1.0)
    p_train.add_argument("--rho0", type=float, default=0.98)
    p_train.add_argument("--device", type=str, default="cpu")
    p_train.add_argument("--seed", type=int, default=1)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate few-shot performance")
    p_eval.add_argument("--ckpt", type=str, required=False, help="Checkpoint for method 'meta' (PAC-Bayes trained).")
    p_eval.add_argument("--data", type=str, required=True)
    p_eval.add_argument(
        "--methods",
        type=str,
        default="meta",
        help="Comma-separated methods: meta,ols,ridge,pooled_prior_ridge,shared_subspace",
    )
    p_eval.add_argument("--support-len", type=int, required=True, dest="support_len")
    p_eval.add_argument("--query-len", type=int, required=True, dest="query_len")
    p_eval.add_argument("--ridge-lambdas", type=str, default="1e-6,1e-4,1e-3,1e-2,1e-1", help="Comma-separated ridge grid for ridge-like methods")
    p_eval.add_argument("--subspace-k", type=int, default=5, dest="subspace_k")
    p_eval.add_argument("--auto-support", action="store_true", help="Choose a safer support prefix length <= support-len via inner validation within the support window.")
    p_eval.add_argument("--auto-support-val-len", type=int, default=5, dest="auto_support_val_len", help="Validation rollout length used by --auto-support (within support window).")
    p_eval.add_argument("--auto-support-grid", type=str, default=None, dest="auto_support_grid", help="Optional comma-separated candidate prefix lengths for --auto-support (e.g. '1,2,3,5,10,20').")
    p_eval.add_argument("--auto-support-metric", type=str, default="auto", dest="auto_support_metric", choices=["auto", "A_mse", "rollout"], help="Scoring metric for --auto-support. 'auto' uses A_mse when A_true is present, otherwise rollout.")
    p_eval.add_argument("--device", type=str, default="cpu")
    p_eval.add_argument("--report", type=str, required=True)

    # generate edge-case dataset
    p_gen_edge = sub.add_parser("generate-edge", help="Generate a dataset with one edge-case test task")
    p_gen_edge.add_argument("--out", type=str, required=True)
    p_gen_edge.add_argument("--n", type=int, required=True)
    p_gen_edge.add_argument("--T", type=int, required=True)
    p_gen_edge.add_argument("--sigma-true", type=float, required=True, dest="sigma_true")
    p_gen_edge.add_argument("--v-true", type=float, required=True, dest="v_true")
    p_gen_edge.add_argument("--rho0-gen", type=float, required=True, dest="rho0_gen")
    p_gen_edge.add_argument("--seed", type=int, default=0)
    p_gen_edge.add_argument("--edge-type", type=str, default="near_rho", choices=["near_rho", "large_entry", "large_all", "tails", "spike_all"])
    p_gen_edge.add_argument("--target-rho", type=float, default=None, dest="target_rho")
    p_gen_edge.add_argument("--spike-mult", type=float, default=6.0, dest="spike_multiplier")
    p_gen_edge.add_argument("--spike-i", type=int, default=None)
    p_gen_edge.add_argument("--spike-j", type=int, default=None)
    p_gen_edge.add_argument("--tail-quantile", type=float, default=0.9, dest="tail_quantile", help="Quantile in [0.5, 0.999] used by tails edge-type")

    # A distribution over dataset
    p_adist = sub.add_parser("a-dist", help="Plot distribution of A entries from a dataset")
    p_adist.add_argument("--data", type=str, required=True)
    p_adist.add_argument("--out", type=str, required=True)
    p_adist.add_argument("--bins", type=int, default=50)

    # A distribution compare
    p_adist_cmp = sub.add_parser("a-dist-compare", help="Compare A entry distributions for two datasets")
    p_adist_cmp.add_argument("--data-regular", type=str, required=True, dest="data_regular")
    p_adist_cmp.add_argument("--data-edge", type=str, required=True, dest="data_edge")
    p_adist_cmp.add_argument("--out", type=str, required=True)
    p_adist_cmp.add_argument("--bins", type=int, default=50)
    p_adist_cmp.add_argument("--smooth", action="store_true", help="Overlay smooth KDE curves")
    p_adist_cmp.add_argument("--bw", type=float, default=None, help="KDE bandwidth (default Scott's rule)")
    p_adist_cmp.add_argument("--line", action="store_true", help="Plot piecewise-linear density via connected bin centers")
    p_adist_cmp.add_argument("--hide-bars", action="store_true", help="Hide histogram bar outlines")
    # A mean distribution per-task
    p_amean = sub.add_parser("a-mean-dist", help="Plot distribution of mean(A) per task from a dataset")
    p_amean.add_argument("--data", type=str, required=True)
    p_amean.add_argument("--out", type=str, required=True)
    p_amean.add_argument("--bins", type=int, default=50)
    # Compare mean(A) distributions
    p_amean_cmp = sub.add_parser("a-mean-dist-compare", help="Compare mean(A) distribution (all vs selected/edge)")
    p_amean_cmp.add_argument("--data-all", type=str, required=True, dest="data_all")
    p_amean_cmp.add_argument("--data-selected", type=str, required=True, dest="data_selected")
    p_amean_cmp.add_argument("--data-edge", type=str, default=None, dest="data_edge")
    p_amean_cmp.add_argument("--out", type=str, required=True)
    p_amean_cmp.add_argument("--bins", type=int, default=60)
    # Mean stats for regular and edge datasets
    p_amean_stats = sub.add_parser("a-mean-stats", help="Compute A-entry stats for a specific task in regular and pos/neg extremes in edge")
    p_amean_stats.add_argument("--data-regular", type=str, required=True, dest="data_regular")
    p_amean_stats.add_argument("--data-edge", type=str, required=True, dest="data_edge")
    p_amean_stats.add_argument("--out", type=str, required=True)
    p_amean_stats.add_argument("--regular-split", type=str, default="test", choices=["train", "val", "test"])
    p_amean_stats.add_argument("--regular-index", type=int, default=0)
    p_amean_stats.add_argument("--edge-split", type=str, default="test", choices=["train", "val", "test"], dest="edge_split")
    p_amean_stats.add_argument("--edge-positive-index", type=int, default=1, dest="edge_pos_index")
    p_amean_stats.add_argument("--edge-negative-index", type=int, default=0, dest="edge_neg_index")
    # Select tasks by per-task mean and split into train/val/test; also save edge extremes
    p_select = sub.add_parser("select-tasks", help="Select tasks by mean(A) from a dataset")
    p_select.add_argument("--data-in", type=str, required=True, dest="data_in")
    p_select.add_argument("--out", type=str, required=True, help="Output dataset path for selected inliers")
    p_select.add_argument("--train", type=int, required=True, dest="n_train")
    p_select.add_argument("--val", type=int, required=True, dest="n_val")
    p_select.add_argument("--test", type=int, required=True, dest="n_test")
    p_select.add_argument("--k", type=float, default=1.0, help="Select inliers within mean±k*std (k=1.0 default)")
    p_select.add_argument(
        "--select-strategy",
        type=str,
        default="closest",
        choices=["closest", "spread"],
        help="How to pick tasks from the inlier band. "
        "'closest' picks tasks closest to the global mean (old behavior). "
        "'spread' picks tasks spread across the inlier band to cover a wider mean(A) range.",
    )
    p_select.add_argument(
        "--shuffle-selected",
        action="store_true",
        help="If set (recommended with --select-strategy spread), shuffle the chosen tasks before splitting into train/val/test.",
    )
    p_select.add_argument(
        "--edge-out",
        type=str,
        default=None,
        help=(
            "Optional: save an 'edge' dataset to this path. "
            "It reuses the SAME selected train/val splits, but replaces the test split with extreme-mean(A) tasks. "
            "The number of edge test tasks follows --test."
        ),
    )
    p_select.add_argument("--seed", type=int, default=0)

    # A matrices comparison (task-level)
    p_acomp = sub.add_parser("a-compare", help="Compare A_true for a given task between two datasets")
    p_acomp.add_argument("--data-regular", type=str, required=True, dest="data_regular")
    p_acomp.add_argument("--data-edge", type=str, required=True, dest="data_edge")
    p_acomp.add_argument("--task-index", type=int, default=0, dest="task_index")
    p_acomp.add_argument("--out", type=str, required=True)

    args = parser.parse_args()

    if args.cmd == "generate":
        ds = generate_dataset(
            n=args.n,
            M_train=args.M_train,
            M_val=args.M_val,
            M_test=args.M_test,
            T_min=args.T_min,
            T_max=args.T_max,
            sigma_true=args.sigma_true,
            v_true=args.v_true,
            rho0_gen=args.rho0_gen,
            seed=args.seed,
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        save_dataset(args.out, ds)
        print(f"Saved dataset to {args.out}")

    elif args.cmd == "generate-edge":
        spike_idx = None
        if args.spike_i is not None and args.spike_j is not None:
            spike_idx = (int(args.spike_i), int(args.spike_j))
        ds = generate_edge_dataset(
            n=args.n,
            T=args.T,
            sigma_true=args.sigma_true,
            v_true=args.v_true,
            rho0_gen=args.rho0_gen,
            seed=args.seed,
            edge_type=args.edge_type,
            target_rho=args.target_rho,
            spike_multiplier=args.spike_multiplier,
            spike_index=spike_idx,
            tail_quantile=args.tail_quantile,
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        save_dataset(args.out, ds)
        print(f"Saved edge-case dataset to {args.out}")

    elif args.cmd == "train":
        cfg = TrainConfig(
            data=args.data,
            steps=args.steps,
            batch=args.batch,
            lr=args.lr,
            tauW=args.tauW,
            lambdaV=args.lambdaV,
            gamma=args.gamma,
            eta=args.eta,
            rho0=args.rho0,
            device=args.device,
            seed=args.seed,
        )
        ckpt = train(cfg)
        print(f"Saved checkpoint to {ckpt}")

    elif args.cmd == "eval":
        methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
        ridge_lams = []
        for tok in str(args.ridge_lambdas).split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                ridge_lams.append(float(tok))
            except ValueError:
                pass
        if len(ridge_lams) == 0:
            ridge_lams = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]

        report = evaluate_methods(
            data_path=args.data,
            methods=methods,
            support_len=int(args.support_len),
            query_len=int(args.query_len),
            ridge_lambdas=ridge_lams,
            subspace_k=int(args.subspace_k),
            auto_support=bool(args.auto_support),
            auto_support_val_len=int(args.auto_support_val_len),
            auto_support_grid=args.auto_support_grid,
            auto_support_metric=str(args.auto_support_metric),
            ckpt=args.ckpt,
            device=str(args.device),
        )
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        save_report(args.report, report)
        print(json.dumps(report, indent=2))

    elif args.cmd == "a-dist":
        data = load_dataset(args.data)
        A_list = []
        for split in ["train", "val", "test"]:
            for t in data.get(split, []):
                if "A_true" in t:
                    A_list.append(t["A_true"])
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_A_entry_distribution(A_list, out_path, bins=int(args.bins))
        print(f"Saved A entry distribution to {args.out}")

    elif args.cmd == "a-dist-compare":
        data_reg = load_dataset(args.data_regular)
        data_edge = load_dataset(args.data_edge)
        A_reg = []
        A_edge = []
        for split in ["train", "val", "test"]:
            for t in data_reg.get(split, []):
                if "A_true" in t:
                    A_reg.append(t["A_true"])
            for t in data_edge.get(split, []):
                if "A_true" in t:
                    A_edge.append(t["A_true"])
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_A_entry_distribution_compare(
            A_reg,
            A_edge,
            out_path,
            bins=int(args.bins),
            smooth=bool(args.smooth),
            bandwidth=float(args.bw) if args.bw is not None else None,
            line=bool(args.line),
            hide_bars=bool(args.hide_bars),
        )
        print(f"Saved A entry distribution comparison to {args.out}")
    elif args.cmd == "a-mean-dist":
        data = load_dataset(args.data)
        A_list = []
        for split in ["train", "val", "test"]:
            for t in data.get(split, []):
                if "A_true" in t:
                    A_list.append(t["A_true"])
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_A_task_mean_distribution(A_list, out_path, bins=int(args.bins))
        print(f"Saved mean(A) per-task distribution to {args.out}")
    elif args.cmd == "a-mean-dist-compare":
        data_all = load_dataset(args.data_all)
        data_sel = load_dataset(args.data_selected)
        data_edge = load_dataset(args.data_edge) if args.data_edge is not None else {"train": [], "val": [], "test": []}
        def _collect(ds, splits=("train", "val", "test")):
            arr = []
            for split in list(splits):
                for t in ds.get(split, []):
                    if "A_true" in t:
                        arr.append(t["A_true"])
            return arr
        A_all = _collect(data_all, splits=("train", "val", "test"))
        # Selected dataset now has semantics:
        # - train/val: random tasks used for training/validation
        # - test: common-case tasks selected around mean(A)
        A_sel_trainval = _collect(data_sel, splits=("train", "val"))
        A_sel_test = _collect(data_sel, splits=("test",))
        # Edge dataset shares train/val with selected; only plot the edge TEST tasks.
        A_edge = _collect(data_edge, splits=("test",)) if args.data_edge is not None else []
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_A_task_mean_distribution_compare(A_all, A_sel_trainval, A_sel_test, A_edge, out_path, bins=int(args.bins))
        print(f"Saved mean(A) comparison to {args.out}")
    elif args.cmd == "a-mean-stats":
        reg = load_dataset(args.data_regular)
        edge = load_dataset(args.data_edge)
        # Collect tasks and their per-task mean(A)
        def _collect(ds):
            tasks = []
            means = []
            for split in ["train", "val", "test"]:
                for t in ds.get(split, []):
                    if "A_true" in t:
                        A = np.array(t["A_true"])
                        tasks.append(A)
                        means.append(float(np.mean(A)))
            return tasks, (np.array(means, dtype=float) if len(means) > 0 else np.array([], dtype=float))
        reg_tasks, reg_means = _collect(reg)
        edge_tasks, edge_means = _collect(edge)
        # Helper: stats for a single A matrix (over its entries)
        def _single_A_stats(A: np.ndarray | None) -> dict:
            if A is None or A.size == 0:
                return {"mean": None, "std": None, "min": None, "max": None}
            vals = A.ravel()
            return {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        # Choose the exact requested regular task (split/index)
        A_reg_one = None
        try:
            dsr = load_dataset(args.data_regular)
            split = str(args.regular_split)
            idx = int(max(0, args.regular_index))
            pool = list(dsr.get(split, []))
            if len(pool) == 0:
                # Fallback preference: test -> train -> val
                for alt in (["test", "train", "val"] if split != "test" else ["train", "val", "test"]):
                    pool = list(dsr.get(alt, []))
                    if len(pool) > 0:
                        break
            if len(pool) > 0:
                use_i = min(idx, len(pool) - 1)
                A_reg_one = np.array(pool[use_i]["A_true"])
        except Exception:
            A_reg_one = None
        # For edge: pick explicit indices from requested split (defaults: test: pos=1, neg=0)
        A_edge_pos = None
        A_edge_neg = None
        try:
            dse = load_dataset(args.data_edge)
            esplit = str(args.edge_split)
            pos_i = int(max(0, args.edge_pos_index))
            neg_i = int(max(0, args.edge_neg_index))
            e_pool = list(dse.get(esplit, []))
            if len(e_pool) == 0:
                # Fallback: test -> train -> val
                for alt in (["test", "train", "val"] if esplit != "test" else ["train", "val", "test"]):
                    e_pool = list(dse.get(alt, []))
                    if len(e_pool) > 0:
                        break
            if len(e_pool) > 0:
                if pos_i < len(e_pool):
                    A_edge_pos = np.array(e_pool[pos_i]["A_true"])
                if neg_i < len(e_pool):
                    A_edge_neg = np.array(e_pool[neg_i]["A_true"])
        except Exception:
            A_edge_pos = A_edge_pos
            A_edge_neg = A_edge_neg
        out_obj = {
            "regular_task": _single_A_stats(A_reg_one),
            "edge_positive_task": _single_A_stats(A_edge_pos),
            "edge_negative_task": _single_A_stats(A_edge_neg),
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out_obj, f, indent=2)
        print(json.dumps(out_obj, indent=2))
    elif args.cmd == "select-tasks":
        data = load_dataset(args.data_in)
        # Flatten all tasks
        tasks = []
        for split in ["train", "val", "test"]:
            for t in data.get(split, []):
                tasks.append(t)
        if len(tasks) == 0:
            raise ValueError("Input dataset contains no tasks.")
        # Compute per-task means
        means = np.array([float(np.mean(np.array(t["A_true"]))) for t in tasks], dtype=float)
        mu = float(np.mean(means))
        sig = float(np.std(means)) if means.size > 0 else 0.0
        k = float(args.k)
        lo = mu - k * sig
        hi = mu + k * sig
        rng = np.random.default_rng(int(args.seed))

        # Inliers (used ONLY for the TEST split): within [lo, hi]
        inlier_idx = np.where((means >= lo) & (means <= hi))[0]
        strategy = str(getattr(args, "select_strategy", "closest")).strip().lower()

        def _pick_spread(idxs_sorted_by_mean: np.ndarray, need: int) -> np.ndarray:
            """Pick `need` indices spread across the sorted list (unique picks)."""
            m = int(idxs_sorted_by_mean.size)
            if need <= 0 or m == 0:
                return np.array([], dtype=int)
            if need >= m:
                return idxs_sorted_by_mean.astype(int)
            targets = np.linspace(0, m - 1, num=int(need))
            used = np.zeros(m, dtype=bool)
            out = []
            for t in targets:
                j = int(round(float(t)))
                j = max(0, min(m - 1, j))
                if not used[j]:
                    used[j] = True
                    out.append(int(idxs_sorted_by_mean[j]))
                    continue
                # Search outward for nearest unused slot
                found = False
                for d in range(1, m):
                    lo_j = j - d
                    hi_j = j + d
                    if lo_j >= 0 and not used[lo_j]:
                        used[lo_j] = True
                        out.append(int(idxs_sorted_by_mean[lo_j]))
                        found = True
                        break
                    if hi_j < m and not used[hi_j]:
                        used[hi_j] = True
                        out.append(int(idxs_sorted_by_mean[hi_j]))
                        found = True
                        break
                if not found:
                    break
            return np.array(out, dtype=int)

        n_tr = int(args.n_train)
        n_va = int(args.n_val)
        n_te = int(args.n_test)

        # TEST split: selected around mean (inlier band, strategy-controlled).
        if strategy == "spread":
            # Sort inliers by mean(A) value, then pick evenly across the band.
            inlier_sorted = inlier_idx[np.argsort(means[inlier_idx])]
        else:
            # Default: sort inliers by closeness to mu.
            inlier_sorted = inlier_idx[np.argsort(np.abs(means[inlier_idx] - mu))]

        te_idx = inlier_sorted
        # If insufficient inliers for test, fill with closest-to-mu tasks overall (still "around mean").
        if te_idx.size < n_te:
            all_idx_sorted = np.argsort(np.abs(means - mu))
            seen = set(int(i) for i in te_idx.tolist())
            fill = []
            for i in all_idx_sorted:
                ii = int(i)
                if ii not in seen:
                    fill.append(ii)
                    seen.add(ii)
                if len(seen) >= n_te:
                    break
            te_idx = np.concatenate([te_idx, np.array(fill, dtype=int)], axis=0)

        # If we have more than needed, subsample deterministically by strategy.
        if te_idx.size > n_te:
            if strategy == "spread":
                te_idx = _pick_spread(te_idx, n_te)
            else:
                te_idx = te_idx[:n_te]

        if bool(getattr(args, "shuffle_selected", False)) and te_idx.size > 0:
            te_idx = rng.permutation(te_idx)

        # TRAIN/VAL splits: completely random (uniform) from the remaining tasks (NOT mean-based).
        need_tv = n_tr + n_va
        all_idx = np.arange(len(tasks), dtype=int)
        remaining = np.setdiff1d(all_idx, te_idx.astype(int), assume_unique=False)
        if remaining.size < need_tv:
            raise ValueError(
                f"Not enough remaining tasks for train/val after choosing test ({remaining.size} < {need_tv})."
            )
        tv_idx = rng.choice(remaining, size=int(need_tv), replace=False)
        tr_idx = tv_idx[:n_tr]
        va_idx = tv_idx[n_tr : n_tr + n_va]

        selected = {
            "train": [tasks[int(i)] for i in tr_idx],
            "val": [tasks[int(i)] for i in va_idx],
            "test": [tasks[int(i)] for i in te_idx],
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        save_dataset(args.out, selected)
        # Edge dataset: reuse SAME train/val as selected; replace test with extremes.
        if args.edge_out is not None and str(args.edge_out).strip():
            # Exclude selected train/val so edge test truly differs from selected.
            exclude = set(int(i) for i in np.concatenate([tr_idx, va_idx], axis=0).tolist())

            # Balanced extremes: take half from the low tail and half from the high tail.
            idx_sorted = np.argsort(means)  # ascending
            n_edge_test = int(max(0, args.n_test))
            n_lo = int((n_edge_test + 1) // 2)
            n_hi = int(n_edge_test // 2)

            low = []
            for i in idx_sorted:
                ii = int(i)
                if ii in exclude:
                    continue
                low.append(ii)
                if len(low) >= n_lo:
                    break

            high = []
            for i in idx_sorted[::-1]:
                ii = int(i)
                if ii in exclude or ii in set(low):
                    continue
                high.append(ii)
                if len(high) >= n_hi:
                    break

            edge_test_idx = low + high
            edge_ds = {
                "train": [tasks[int(i)] for i in tr_idx],
                "val": [tasks[int(i)] for i in va_idx],
                "test": [tasks[int(i)] for i in edge_test_idx],
            }
            Path(args.edge_out).parent.mkdir(parents=True, exist_ok=True)
            save_dataset(args.edge_out, edge_ds)
            edge_means = [float(means[i]) for i in edge_test_idx]
            print(
                json.dumps(
                    {
                        "mu": mu,
                        "sigma": sig,
                        "edge_indices": edge_test_idx,
                        "edge_means": edge_means,
                    },
                    indent=2,
                )
            )
        print(
            json.dumps(
                {
                    "mu": mu,
                    "sigma": sig,
                    "k": k,
                    "inlier_band": [lo, hi],
                    "selection_policy": {"train": "uniform_random", "val": "uniform_random", "test": "around_mean"},
                    "selected_counts": {"train": n_tr, "val": n_va, "test": n_te},
                },
                indent=2,
            )
        )

    elif args.cmd == "a-compare":
        data_reg = load_dataset(args.data_regular)
        data_edge = load_dataset(args.data_edge)
        idx = int(max(0, args.task_index))
        # Prefer test split; fallback to train if test empty
        def _get_task(ds, i):
            if len(ds.get("test", [])) > 0:
                return ds["test"][min(i, len(ds["test"]) - 1)]
            elif len(ds.get("train", [])) > 0:
                return ds["train"][min(i, len(ds["train"]) - 1)]
            else:
                raise ValueError("Dataset has no tasks.")
        t_reg = _get_task(data_reg, idx)
        t_edge = _get_task(data_edge, idx)
        A_reg = np.array(t_reg["A_true"])
        A_edge = np.array(t_edge["A_true"])
        def _stats(A: np.ndarray) -> dict:
            return {
                "shape": [int(A.shape[0]), int(A.shape[1])],
                "dtype": str(A.dtype),
                "min": float(np.min(A)) if A.size > 0 else 0.0,
                "max": float(np.max(A)) if A.size > 0 else 0.0,
                "mean": float(np.mean(A)) if A.size > 0 else 0.0,
                "std": float(np.std(A)) if A.size > 0 else 0.0,
            }
        same_shape = bool(A_reg.shape == A_edge.shape)
        same_dtype = bool(A_reg.dtype == A_edge.dtype)
        D = (A_edge - A_reg) if same_shape else np.array([])
        diff_metrics = {}
        if D.size > 0:
            diff_metrics = {
                "l1": float(np.sum(np.abs(D))),
                "frobenius": float(np.linalg.norm(D)),
                "max_abs": float(np.max(np.abs(D))),
                "mean_abs": float(np.mean(np.abs(D))),
                "nonzero_count": int(np.count_nonzero(D)),
            }
        out_obj = {
            "edge": {
                "path": str(args.data_edge),
                "key": "test_0_A_true",
                "error": None,
                "stats": _stats(A_edge),
            },
            "long": {
                "path": str(args.data_regular),
                "key": "test_0_A_true",
                "error": None,
                "stats": _stats(A_reg),
            },
            "comparison": {
                "same_shape": same_shape,
                "same_dtype": same_dtype,
                "allclose": bool(np.allclose(A_edge, A_reg)) if same_shape else False,
                "difference_metrics": diff_metrics,
            },
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out_obj, f, indent=2)
        print(f"Wrote comparison JSON to {args.out}")


if __name__ == "__main__":
    main()
