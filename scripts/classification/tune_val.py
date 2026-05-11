"""Validation-set hyperparameter sweep for AvgDeg, BFS, and BP.

Reads split_meta.json for the dataset, sweeps a small grid per method on
eligible.val with forbidden = splits.val | splits.test, persists per-query
records (delegating disk side-effects to evaluate_nodes), and writes:

    exps/classification/<dataset>/tune_val/tune_val_<family>.csv
    exps/classification/<dataset>/tune_val/best_settings_<family>.json
"""

import argparse
import json
import os
import sys
import tempfile

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from solver_utils import (  # noqa: E402
    build_graph_context,
    evaluate_nodes,
    params_hash,
)
from split_utils import assert_split_meta_matches  # noqa: E402


def _parse_int_list(raw: str):
    return [int(x) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str):
    return [float(x) for x in raw.split(",") if x.strip()]


def _avgdeg_grid():
    return [{}]


def _bfs_grid(depths):
    return [{"bfs_depth": d} for d in depths]


def _bp_grid(k_values, kappa_values, time_limits, dinkelbach_iters):
    grid = []
    for k in k_values:
        for kappa in kappa_values:
            for tl in time_limits:
                for di in dinkelbach_iters:
                    grid.append(
                        {
                            "k": k,
                            "kappa": kappa,
                            "time_limit": tl,
                            "dinkelbach_iter": di,
                        }
                    )
    return grid


def _extra_args_for(method, params, gurobi_seed=None):
    if method == "avgdeg":
        return ["--avgdeg"]
    if method == "bfs":
        return ["--bfs", "--bfs-depth", str(params["bfs_depth"])]
    if method == "bp":
        args = ["--bp"]
        if "kappa" in params:
            args += ["--kappa", str(params["kappa"])]
        if params.get("time_limit") is not None:
            args += ["--time-limit", str(params["time_limit"])]
        if params.get("dinkelbach_iter") is not None:
            args += ["--dinkelbach-iter", str(params["dinkelbach_iter"])]
        if gurobi_seed is not None:
            args += ["--gurobi-seed", str(gurobi_seed)]
        return args
    raise ValueError(f"unknown method: {method}")


def _k_for(method, params):
    if method == "bp":
        return int(params["k"])
    return None


def _macro_scores(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--bin-path", type=str, default="./solver/bin/solver")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--exps-dir", type=str, default="exps")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weighting", type=str, default="uniform")
    parser.add_argument("--max-fallback-hops", type=int, default=10)
    parser.add_argument("--keep-solver-dumps", action="store_true")
    parser.add_argument(
        "--family",
        type=str,
        default="all",
        choices=["all", "avgdeg", "bfs", "bp"],
    )
    parser.add_argument("--bp-k", type=str, default="5,10")
    parser.add_argument("--bp-kappa", type=str, default="0,2")
    parser.add_argument("--bp-time-limit", type=str, default="5")
    parser.add_argument("--bp-dinkelbach-iter", type=str, default="-1")
    parser.add_argument("--bfs-depth", type=str, default="1,2")
    args = parser.parse_args()

    df_nodes = pd.read_csv(os.path.join(args.data_dir, args.dataset, "nodes.csv"))
    df_edges = pd.read_csv(os.path.join(args.data_dir, args.dataset, "edge.csv"))
    meta = assert_split_meta_matches(args.dataset, df_nodes, df_edges, args.data_dir)

    from collections import defaultdict
    from split_utils import sha256_node_set

    adj_und = defaultdict(set)
    for s, t in zip(df_edges["source"].astype(int), df_edges["target"].astype(int)):
        if s == t:
            continue
        adj_und[s].add(t)
        adj_und[t].add(s)
    val_ids = df_nodes[df_nodes["val"]]["node_id"].astype(int).tolist()
    eligible_val = [int(q) for q in val_ids if len(adj_und.get(q, ())) >= 2]
    if sha256_node_set(eligible_val) != meta.eligible["val"]["hash"]:
        raise ValueError("eligible.val hash mismatch with split_meta.json")

    forbidden = set(
        df_nodes[df_nodes["val"] | df_nodes["test"]]["node_id"].astype(int).tolist()
    )

    print(
        f"{args.dataset}: eligible.val = {len(eligible_val)} nodes; forbidden = {len(forbidden)} nodes"
    )

    ctx = build_graph_context(
        os.path.join(args.data_dir, args.dataset, "edge.csv"), max_in_edges=0
    )

    family = args.family
    families = ["avgdeg", "bfs", "bp"] if family == "all" else [family]

    out_root = os.path.join(args.exps_dir, "classification", args.dataset, "tune_val")
    os.makedirs(out_root, exist_ok=True)
    run_id = "run"
    records_root = os.path.join(out_root, run_id)

    for fam in families:
        grid_fn = {
            "avgdeg": lambda: _avgdeg_grid(),
            "bfs": lambda: _bfs_grid(_parse_int_list(args.bfs_depth)),
            "bp": lambda: _bp_grid(
                _parse_int_list(args.bp_k),
                _parse_int_list(args.bp_kappa),
                _parse_float_list(args.bp_time_limit),
                _parse_int_list(args.bp_dinkelbach_iter),
            ),
        }[fam]
        grid = grid_fn()
        rows = []
        best_row = None
        best_settings = None
        for params in grid:
            p_hash = params_hash(params)
            extra = _extra_args_for(fam, params, gurobi_seed=args.seed if fam == "bp" else None)
            k = _k_for(fam, params)
            print(
                f"[{args.dataset}] {fam} params={params} hash={p_hash[-12:]} extra={extra}"
            )
            records_dir = os.path.join(records_root, fam, p_hash[-12:])
            os.makedirs(records_dir, exist_ok=True)
            with tempfile.TemporaryDirectory() as td:
                y_true, y_pred, stats = evaluate_nodes(
                    eligible_val,
                    k=k,
                    edge_csv=os.path.join(args.data_dir, args.dataset, "edge.csv"),
                    df_nodes=df_nodes,
                    bin_path=args.bin_path,
                    tmp_dir=td,
                    max_workers=args.max_workers,
                    extra_args=extra,
                    weighting=args.weighting,
                    max_fallback_hops=args.max_fallback_hops,
                    forbidden_nodes=forbidden,
                    graph_context=ctx,
                    records_path=records_dir,
                    dataset_name=args.dataset,
                    seed=args.seed if fam == "bp" else None,
                    method=fam,
                    params=params,
                    split_hash=meta.eligible["val"]["hash"],
                    keep_solver_dumps=args.keep_solver_dumps,
                    query_split="val",
                    collect_stats=True,
                    compute_qualities=True,
                    show_progress=False,
                )
            scores = _macro_scores(y_true, y_pred)
            row = {
                "dataset": args.dataset,
                "method": fam,
                "seed": args.seed if fam == "bp" else None,
                "split_hash": meta.eligible["val"]["hash"],
                "weighting": args.weighting,
                "optimize_target": "f1",
                "params_hash": p_hash,
                "params_json": json.dumps(params, sort_keys=True),
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["f1"],
                "fallback_rate": stats["fallback_rate"],
                "avg_oracle_queries": stats["avg_oracle_queries"],
                "avg_dir_internal_edge_density": stats.get(
                    "avg_dir_internal_edge_density"
                ),
                "avg_undir_internal_ncut": stats.get("avg_undir_internal_ncut"),
                "avg_undir_external_conductance": stats.get(
                    "avg_undir_external_conductance"
                ),
                "records_dir": records_dir,
            }
            rows.append(row)
            if best_row is None or row["f1"] > best_row["f1"]:
                best_row = row
                best_settings = {
                    "dataset": args.dataset,
                    "method": fam,
                    "params": params,
                    "params_hash": p_hash,
                    "extra_args": extra,
                    "k": k,
                    "split_hash": meta.eligible["val"]["hash"],
                    "f1": scores["f1"],
                    "precision": scores["precision"],
                    "recall": scores["recall"],
                    "records_dir": records_dir,
                }
        csv_path = os.path.join(out_root, f"tune_val_{fam}.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        best_path = os.path.join(out_root, f"best_settings_{fam}.json")
        with open(best_path, "w") as f:
            json.dump(best_settings, f, indent=2, sort_keys=True)
        print(f"wrote {csv_path}")
        print(f"wrote {best_path}")


if __name__ == "__main__":
    main()
