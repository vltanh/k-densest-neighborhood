import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


sys.path.append(os.path.dirname(__file__))


QUALITY_COLUMNS = {
    "avg_oracle_queries",
    "avg_dir_internal_avg_degree",
    "avg_dir_internal_edge_density",
    "avg_undir_external_expansion",
    "avg_undir_external_conductance",
    "avg_undir_internal_ncut",
}


# Driver table for the hyperparameter columns that participate in resume keys,
# row reconstruction, and sort order. Adding a new hyperparameter means adding
# one entry here; config_key / row_key / sort_key derive from it.
HP_COLUMNS = [
    ("method", str),
    ("k", int),
    ("kappa", int),
    ("depth", int),
    ("time_limit", float),
    ("cg_batch_frac", float),
    ("cg_min_batch", int),
    ("cg_max_batch", int),
    ("node_limit", int),
    ("gap_tol", float),
    ("dinkelbach_iter", int),
]


def parse_int_list(raw):
    return [int(v) for v in raw.split(",") if v.strip() != ""]


def parse_float_list(raw):
    return [float(v) for v in raw.split(",") if v.strip() != ""]


def limit_nodes(nodes, limit, seed):
    if limit is None or limit <= 0 or limit >= len(nodes):
        return list(nodes)
    rng = random.Random(seed)
    return rng.sample(list(nodes), limit)


def _cast(value, caster):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return caster(value)


def config_key(config):
    return tuple(_cast(config.get(col), caster) for col, caster in HP_COLUMNS)


def row_key(row):
    return tuple(
        _cast(row.get(col) if col in row else None, caster)
        for col, caster in HP_COLUMNS
    )


def sort_key(row):
    parts = []
    for col, caster in HP_COLUMNS:
        value = _cast(row.get(col), caster)
        if value is None:
            parts.append((0, "") if caster is str else (0, -1))
        else:
            parts.append((1, value))
    return tuple(parts)


def build_tuning_stages(
    family,
    k_values,
    kappa_values,
    bfs_depth_min,
    bfs_depth_max,
    bp_time_limits,
    bp_cg_batch_frac=None,
    bp_cg_min_batch=None,
    bp_cg_max_batch=None,
    bp_node_limit=None,
    bp_gap_tol=None,
    bp_dinkelbach_iter=None,
):
    avgdeg_configs = []
    bfs_configs = []
    bp_configs = []

    if family in ("all", "avgdeg"):
        avgdeg_configs.append(
            {
                "method": "avgdeg",
                "k": None,
                "kappa": None,
                "depth": None,
                "run_k": None,
                "extra_args": ["--avgdeg"],
                "tmp_name": "avgdeg",
            }
        )

    if family in ("all", "bfs"):
        for depth in range(bfs_depth_min, bfs_depth_max + 1):
            bfs_configs.append(
                {
                    "method": "bfs",
                    "k": None,
                    "kappa": None,
                    "depth": depth,
                    "run_k": None,
                    "extra_args": ["--bfs", "--bfs-depth", str(depth)],
                    "tmp_name": f"bfs_depth{depth}",
                }
            )

    if family in ("all", "bp"):
        cg_extra_args = []
        if bp_cg_batch_frac is not None:
            cg_extra_args += ["--cg-batch-frac", str(bp_cg_batch_frac)]
        if bp_cg_min_batch is not None:
            cg_extra_args += ["--cg-min-batch", str(bp_cg_min_batch)]
        if bp_cg_max_batch is not None:
            cg_extra_args += ["--cg-max-batch", str(bp_cg_max_batch)]

        exact_extra_args = []
        if bp_node_limit is not None:
            exact_extra_args += ["--node-limit", str(bp_node_limit)]
        if bp_gap_tol is not None:
            exact_extra_args += ["--gap-tol", str(bp_gap_tol)]
        if bp_dinkelbach_iter is not None:
            exact_extra_args += ["--dinkelbach-iter", str(bp_dinkelbach_iter)]

        for k in k_values:
            for kappa in kappa_values:
                for bp_time_limit in bp_time_limits:
                    time_label = str(bp_time_limit).replace(".", "p")
                    bp_configs.append(
                        {
                            "method": "bp",
                            "k": k,
                            "kappa": kappa,
                            "depth": None,
                            "time_limit": bp_time_limit,
                            "cg_batch_frac": bp_cg_batch_frac,
                            "cg_min_batch": bp_cg_min_batch,
                            "cg_max_batch": bp_cg_max_batch,
                            "node_limit": bp_node_limit,
                            "gap_tol": bp_gap_tol,
                            "dinkelbach_iter": bp_dinkelbach_iter,
                            "run_k": k,
                            "extra_args": [
                                "--bp",
                                "--kappa",
                                str(kappa),
                                "--time-limit",
                                str(bp_time_limit),
                            ]
                            + cg_extra_args,
                            "tmp_name": f"bp_k{k}_kappa{kappa}_t{time_label}",
                        }
                    )
                    bp_configs[-1]["extra_args"] += exact_extra_args

    return [
        ("avgdeg", avgdeg_configs),
        ("bfs", bfs_configs),
        ("bp", bp_configs),
    ]


def filter_completed_stages(stages, rows):
    completed = {row_key(row) for row in rows}
    return [
        (
            stage_name,
            [config for config in stage_configs if config_key(config) not in completed],
        )
        for stage_name, stage_configs in stages
    ]


def best_rows_by_method(rows, optimize):
    best = {}
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        if method_rows:
            best[method] = max(method_rows, key=lambda r: r[optimize])
    return best


def row_to_jsonable(row):
    clean = {}
    for key, value in row.items():
        if pd.isna(value):
            clean[key] = None
        elif hasattr(value, "item"):
            clean[key] = value.item()
        else:
            clean[key] = value
    return clean


def score_run(
    query_nodes,
    k,
    edge_csv,
    df_nodes,
    bin_path,
    tmp_dir,
    workers,
    weighting,
    extra_args,
    show_progress=True,
    max_in_edges=0,
):
    from solver_utils import evaluate_nodes

    os.makedirs(tmp_dir, exist_ok=True)
    y_true, y_pred, stats = evaluate_nodes(
        query_nodes,
        k,
        edge_csv,
        df_nodes,
        bin_path,
        tmp_dir,
        max_workers=workers,
        weighting=weighting,
        extra_args=extra_args,
        collect_stats=True,
        show_progress=show_progress,
        compute_qualities=True,
        max_in_edges=max_in_edges,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        **stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Tune solver parameters on a validation split.")
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--bin_path", type=str, default="./solver/bin/solver")
    parser.add_argument(
        "--family",
        type=str,
        default="all",
        choices=["all", "bp", "avgdeg", "bfs"],
        help="Which family to tune in this run",
    )
    parser.add_argument("--k_min", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=5)
    parser.add_argument(
        "--k_values",
        type=str,
        default="",
        help="Comma-separated BP k values. Overrides --k_min/--k_max when set.",
    )
    parser.add_argument("--kappa_values", type=str, default="0,1,2")
    parser.add_argument("--bfs_depth_min", type=int, default=1)
    parser.add_argument("--bfs_depth_max", type=int, default=3)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--config_workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max_in_edges", type=int, default=0)
    parser.add_argument("--bp_cg_batch_frac", type=float, default=None)
    parser.add_argument("--bp_cg_min_batch", type=int, default=None)
    parser.add_argument("--bp_cg_max_batch", type=int, default=None)
    parser.add_argument(
        "--bp_node_limit",
        type=int,
        default=None,
        help="B&B node limit for BP solver calls; use -1 to disable.",
    )
    parser.add_argument(
        "--bp_gap_tol",
        type=float,
        default=None,
        help="B&B relative gap tolerance for BP solver calls.",
    )
    parser.add_argument(
        "--bp_dinkelbach_iter",
        type=int,
        default=None,
        help="Maximum Dinkelbach iterations for BP solver calls.",
    )
    parser.add_argument(
        "--limit_nodes",
        type=int,
        default=0,
        help="Evaluate at most this many validation nodes. Use 0 for the full split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore any partial tuning CSV and recompute all configurations.",
    )
    parser.add_argument(
        "--bp_time_limit",
        type=float,
        default=600.0,
        help="Time limit in seconds for each BP solver call; use -1 to disable. Ignored when --bp_time_limit_values is set.",
    )
    parser.add_argument(
        "--bp_time_limit_values",
        type=str,
        default="",
        help="Comma-separated BP time limits in seconds. Overrides --bp_time_limit when set.",
    )
    parser.add_argument(
        "--optimize",
        type=str,
        default="f1",
        choices=["accuracy", "f1", "precision", "recall"],
    )
    parser.add_argument(
        "--weighting",
        type=str,
        default="distance",
        choices=["uniform", "distance"],
    )
    args = parser.parse_args()

    edge_csv = os.path.join("data", args.dataset, "edge.csv")
    node_csv = os.path.join("data", args.dataset, "nodes.csv")
    tmp_dir = os.path.join("exps", "classification", args.dataset, "tmp_tune")
    os.makedirs(tmp_dir, exist_ok=True)

    df_nodes = pd.read_csv(node_csv)
    val_nodes = df_nodes[df_nodes["val"]]["node_id"].tolist()
    if args.k_values.strip():
        k_values = parse_int_list(args.k_values)
        k_values = [k for k in k_values if k >= 3]
    else:
        k_values = list(range(max(3, args.k_min), args.k_max + 1))
    kappa_values = parse_int_list(args.kappa_values)
    if args.bp_time_limit_values.strip():
        bp_time_limits = parse_float_list(args.bp_time_limit_values)
    else:
        bp_time_limits = [args.bp_time_limit]
    val_nodes = limit_nodes(val_nodes, args.limit_nodes, args.seed)

    print(f"Validation nodes: {len(val_nodes)}")
    print(f"Weighting: {args.weighting}")
    print(f"Optimize: {args.optimize}")
    print(f"Family: {args.family}")

    rows = []
    out_path = os.path.join(tmp_dir, f"tuning_results_{args.family}.csv")
    partial_path = os.path.join(tmp_dir, f"tuning_results_{args.family}.partial.csv")

    stages = build_tuning_stages(
        args.family,
        k_values,
        kappa_values,
        args.bfs_depth_min,
        args.bfs_depth_max,
        bp_time_limits,
        args.bp_cg_batch_frac,
        args.bp_cg_min_batch,
        args.bp_cg_max_batch,
        args.bp_node_limit,
        args.bp_gap_tol,
        args.bp_dinkelbach_iter,
    )
    configs = [config for _, stage_configs in stages for config in stage_configs]
    total_configs = len(configs)

    if os.path.exists(partial_path) and not args.force:
        partial_df = pd.read_csv(partial_path)
        if "time_limit" not in partial_df.columns:
            if len(bp_time_limits) == 1:
                partial_df["time_limit"] = pd.NA
                partial_df.loc[
                    partial_df["method"] == "bp", "time_limit"
                ] = bp_time_limits[0]
                print(
                    "Filled missing time_limit in partial BP rows "
                    f"with {bp_time_limits[0]:g}s"
                )
            else:
                print(
                    "Ignoring stale partial results without time_limit while "
                    "multiple BP time limits are configured"
                )
                partial_df = pd.DataFrame()
        configured_cg = any(
            value is not None
            for value in (
                args.bp_cg_batch_frac,
                args.bp_cg_min_batch,
                args.bp_cg_max_batch,
            )
        )
        cg_columns = {"cg_batch_frac", "cg_min_batch", "cg_max_batch"}
        if configured_cg and not cg_columns.issubset(partial_df.columns):
            print("Ignoring stale partial results without CG configuration columns")
            partial_df = pd.DataFrame()
        configured_exact = any(
            value is not None
            for value in (
                args.bp_node_limit,
                args.bp_gap_tol,
                args.bp_dinkelbach_iter,
            )
        )
        exact_columns = {"node_limit", "gap_tol", "dinkelbach_iter"}
        if configured_exact and not exact_columns.issubset(partial_df.columns):
            print("Ignoring stale partial results without BP exactness columns")
            partial_df = pd.DataFrame()
        if QUALITY_COLUMNS.issubset(partial_df.columns):
            rows = partial_df.to_dict("records")
        else:
            missing = sorted(QUALITY_COLUMNS - set(partial_df.columns))
            print(
                f"Ignoring stale partial results without quality columns: {', '.join(missing)}"
            )
            rows = []
        stages = filter_completed_stages(stages, rows)
        configs = [config for _, stage_configs in stages for config in stage_configs]
        print(f"Loaded {len(rows)} completed rows from {partial_path}")
    elif args.force and os.path.exists(partial_path):
        print(f"Ignoring partial results because --force was set: {partial_path}")

    def run_config(config):
        config_tmp_dir = os.path.join(tmp_dir, config["tmp_name"])
        metrics = score_run(
            val_nodes,
            config["run_k"],
            edge_csv,
            df_nodes,
            args.bin_path,
            config_tmp_dir,
            args.workers,
            args.weighting,
            config["extra_args"],
            show_progress=args.config_workers == 1,
            max_in_edges=args.max_in_edges,
        )
        row = {
            "method": config["method"],
            "k": config["k"],
            "kappa": config["kappa"],
            "depth": config["depth"],
            "time_limit": config.get("time_limit"),
            "cg_batch_frac": config.get("cg_batch_frac"),
            "cg_min_batch": config.get("cg_min_batch"),
            "cg_max_batch": config.get("cg_max_batch"),
            "node_limit": config.get("node_limit"),
            "gap_tol": config.get("gap_tol"),
            "dinkelbach_iter": config.get("dinkelbach_iter"),
            **metrics,
        }
        return row

    print(f"Remaining configurations: {len(configs)}")
    print(f"Config workers: {args.config_workers}")
    print(f"Query workers per config: {args.workers}")

    for stage_name, stage_configs in stages:
        if not stage_configs:
            continue
        print(f"\n=== {stage_name} stage ({len(stage_configs)} configs) ===")
        with ThreadPoolExecutor(max_workers=min(args.config_workers, len(stage_configs))) as executor:
            futures = {executor.submit(run_config, config): config for config in stage_configs}
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                label = row["method"]
                if row["method"] == "bp":
                    label += f" k={int(row['k'])} kappa={int(row['kappa'])}"
                    if "time_limit" in row and not pd.isna(row["time_limit"]):
                        label += f" t={float(row['time_limit']):g}s"
                elif row["method"] == "bfs":
                    label += f" depth={int(row['depth'])}"
                print(
                    f"{label} acc={row['accuracy']:.4f} f1={row['f1']:.4f} "
                    f"fallback={row['fallback_rate']:.1f}% "
                    f"queries={row['avg_oracle_queries']:.2f} "
                    f"dir_avg_deg={row['avg_dir_internal_avg_degree']:.4f} "
                    f"dir_density={row['avg_dir_internal_edge_density']:.4f} "
                    f"undir_cond={row['avg_undir_external_conductance']:.4f} "
                    f"undir_ncut={row['avg_undir_internal_ncut']:.4f}"
                )
                pd.DataFrame(rows).to_csv(partial_path, index=False)
                print(f"Progress: {len(rows)}/{total_configs} saved to {partial_path}")

    rows.sort(key=sort_key)

    best_by_method = best_rows_by_method(rows, args.optimize)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    best_path = os.path.join(tmp_dir, f"best_settings_{args.family}.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "family": args.family,
                "optimize": args.optimize,
                "weighting": args.weighting,
                "validation_nodes": len(val_nodes),
                "seed": args.seed,
                "limit_nodes": args.limit_nodes,
                "best": {
                    method: row_to_jsonable(row)
                    for method, row in best_by_method.items()
                },
                "bp_cg_batch_frac": args.bp_cg_batch_frac,
                "bp_cg_min_batch": args.bp_cg_min_batch,
                "bp_cg_max_batch": args.bp_cg_max_batch,
                "bp_node_limit": args.bp_node_limit,
                "bp_gap_tol": args.bp_gap_tol,
                "bp_dinkelbach_iter": args.bp_dinkelbach_iter,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    print("\n=== BEST SETTINGS ===")
    bp_best = best_by_method.get("bp")
    if bp_best is not None:
        time_suffix = ""
        if "time_limit" in bp_best and not pd.isna(bp_best["time_limit"]):
            time_suffix = f", time_limit={bp_best['time_limit']}"
        print(
            f"bp: k={bp_best['k']}, kappa={bp_best['kappa']}{time_suffix} "
            f"({args.optimize}={bp_best[args.optimize]:.4f})"
        )
    bfs_best = best_by_method.get("bfs")
    if bfs_best is not None:
        print(f"bfs: depth={bfs_best['depth']} ({args.optimize}={bfs_best[args.optimize]:.4f})")
    if "avgdeg" in best_by_method:
        print("avgdeg: evaluated")
    print(f"Saved per-run results to {out_path}")
    print(f"Saved best settings to {best_path}")


if __name__ == "__main__":
    main()
