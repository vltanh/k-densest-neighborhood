import argparse
import os
import subprocess  # re-exported for tests that patch benchmark_solvers.subprocess.run
import sys
import tempfile
import time
from collections import defaultdict

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _solver_runner import (  # noqa: E402
    base_sim_argv,
    induced_directed_metrics as _shared_induced_metrics,
    overlap_metrics as _shared_overlap_metrics,
    read_predicted_nodes,
)


METHODS = {
    "bp": {
        "label": "BP",
        "flags": ["--bp"],
        "kappa": 0,
        "uses_k": True,
        "uses_baseline_depth": False,
        "uses_bfs_depth": False,
        "problem_scope": "global k-densest",
    },
    "bp_kappa": {
        "label": "BP-Kappa",
        "flags": ["--bp"],
        "kappa": 2,
        "uses_k": True,
        "uses_baseline_depth": False,
        "uses_bfs_depth": False,
        "problem_scope": "global k-densest + kappa-connectivity",
    },
    "avgdeg": {
        "label": "Avgdeg",
        "flags": ["--avgdeg"],
        "kappa": None,
        "uses_k": False,
        "uses_baseline_depth": False,
        "uses_bfs_depth": False,
        "problem_scope": "local BFS average-degree",
    },
    "conn_greedy": {
        "label": "Conn-greedy",
        "flags": ["--conn-greedy"],
        "kappa": None,
        "uses_k": True,
        "uses_baseline_depth": True,
        "uses_bfs_depth": False,
        "problem_scope": "local connected heuristic",
    },
    "conn_avgdeg": {
        "label": "Conn-avgdeg",
        "flags": ["--conn-avgdeg"],
        "kappa": None,
        "uses_k": False,
        "uses_baseline_depth": True,
        "uses_bfs_depth": False,
        "problem_scope": "local connected average-degree",
    },
    "bfs": {
        "label": "BFS",
        "flags": ["--bfs"],
        "kappa": None,
        "uses_k": False,
        "uses_baseline_depth": False,
        "uses_bfs_depth": True,
        "problem_scope": "1-hop BFS neighborhood",
    },
}


def load_ids(path, column):
    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(f"Missing required column '{column}' in {path}")
    return [str(v) for v in df[column].tolist()]


def load_directed_edge_set(edge_csv):
    df = pd.read_csv(edge_csv)
    if len(df.columns) < 2:
        raise ValueError(f"{edge_csv} must have at least two columns")
    src_col, dst_col = df.columns[:2]
    edges = set()
    adj = defaultdict(set)
    for src, dst in zip(df[src_col].astype(str), df[dst_col].astype(str)):
        if src == dst:
            continue
        edges.add((src, dst))
        adj[src].add(dst)
        adj[dst].add(src)
    return edges, adj


read_prediction = read_predicted_nodes
induced_directed_metrics = _shared_induced_metrics


def is_connected_undirected(nodes, adj):
    node_set = set(nodes)
    if len(node_set) <= 1:
        return True
    start = next(iter(node_set))
    seen = {start}
    stack = [start]
    while stack:
        curr = stack.pop()
        for nb in adj.get(curr, ()):
            if nb in node_set and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(node_set)


overlap_metrics = _shared_overlap_metrics


def run_solver(
    bin_path,
    edge_csv,
    query_node,
    k,
    method,
    time_limit,
    node_limit,
    baseline_depth,
    bfs_depth,
    output_dir,
):
    method_spec = METHODS[method]
    out_csv = os.path.join(output_dir, f"{method}_q{query_node}_k{k}.csv")
    cmd = base_sim_argv(bin_path, edge_csv, query_node, out_csv) + [
        "--time-limit", str(time_limit),
        "--node-limit", str(node_limit),
    ]
    if method_spec["uses_k"]:
        cmd += ["--k", str(k)]
    if method_spec["kappa"] is not None:
        cmd += ["--kappa", str(method_spec["kappa"])]
    if method_spec["uses_bfs_depth"]:
        cmd += ["--bfs-depth", str(bfs_depth)]
    if method_spec["uses_baseline_depth"] and baseline_depth is not None:
        cmd += ["--baseline-depth", str(baseline_depth)]
    cmd += method_spec["flags"]

    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall_time = time.perf_counter() - started

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    status = "ok" if proc.returncode == 0 else f"exit_{proc.returncode}"

    pred_nodes = read_prediction(out_csv) if proc.returncode == 0 else []
    if os.path.exists(out_csv):
        os.remove(out_csv)

    return {
        "status": status,
        "wall_time_sec": wall_time,
        "stdout": stdout,
        "stderr": stderr,
        "pred_nodes": pred_nodes,
    }


def summarize(df):
    metric_cols = [
        "wall_time_sec",
        "pred_size",
        "internal_edges",
        "avg_degree_density",
        "edge_density",
        "precision",
        "recall",
        "f1",
        "jaccard",
        "contains_query",
    ]
    grouped = df.groupby(["method_label", "problem_scope"], dropna=False)
    rows = []
    for (label, scope), chunk in grouped:
        row = {"method_label": label, "problem_scope": scope, "runs": len(chunk)}
        for col in metric_cols:
            row[f"{col}_mean"] = chunk[col].mean()
            row[f"{col}_std"] = chunk[col].std(ddof=0) if len(chunk) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Benchmark solver variants on planted-community datasets.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to a dataset directory containing edge.csv and gt_comm.csv")
    parser.add_argument("--bin_path", type=str, default="./solver/bin/solver", help="Path to the compiled solver binary")
    parser.add_argument("--k", type=int, required=True, help="Target community size")
    parser.add_argument("--time_limit", type=float, default=60.0, help="Per-run solver time limit")
    parser.add_argument("--node_limit", type=int, default=100000, help="Per-run branch-and-bound node limit")
    parser.add_argument(
        "--baseline_depth",
        type=int,
        default=-1,
        help="Local exploration depth for baseline solvers; use -1 for the full reachable graph",
    )
    parser.add_argument(
        "--bfs_depth",
        type=int,
        default=1,
        help="BFS depth for the BFS baseline",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bp", "bp_kappa", "avgdeg", "bfs"],
        choices=sorted(METHODS.keys()),
        help="Solver variants to benchmark",
    )
    parser.add_argument("--query_limit", type=int, default=0, help="Optional cap on the number of query nodes from gt_comm.csv")
    parser.add_argument("--output_csv", type=str, default="", help="Optional path to write per-run results as CSV")
    args = parser.parse_args()

    edge_csv = os.path.join(args.dataset_dir, "edge.csv")
    gt_csv = os.path.join(args.dataset_dir, "gt_comm.csv")
    if not os.path.exists(edge_csv):
        raise FileNotFoundError(f"Missing edge list: {edge_csv}")
    if not os.path.exists(gt_csv):
        raise FileNotFoundError(f"Missing ground-truth community: {gt_csv}")
    if not os.path.exists(args.bin_path):
        raise FileNotFoundError(f"Missing solver binary: {args.bin_path}")

    query_nodes = load_ids(gt_csv, "node_id")
    if args.query_limit and args.query_limit > 0:
        query_nodes = query_nodes[: args.query_limit]
    gt_nodes = query_nodes

    edges, adj = load_directed_edge_set(edge_csv)

    rows = []
    with tempfile.TemporaryDirectory(prefix="kdensest_bench_") as tmp_dir:
        for method in args.methods:
            spec = METHODS[method]
            for q in query_nodes:
                result = run_solver(
                    args.bin_path,
                    edge_csv,
                    q,
                    args.k,
                    method,
                    args.time_limit,
                    args.node_limit,
                    args.baseline_depth,
                    args.bfs_depth,
                    tmp_dir,
                )
                pred_nodes = result["pred_nodes"]
                overlap = overlap_metrics(gt_nodes, pred_nodes)
                density = induced_directed_metrics(pred_nodes, edges)
                rows.append(
                    {
                        "query_node": q,
                        "method": method,
                        "method_label": spec["label"],
                        "problem_scope": spec["problem_scope"],
                        "status": result["status"],
                        "wall_time_sec": result["wall_time_sec"],
                        "pred_size": len(pred_nodes),
                        "contains_query": q in pred_nodes,
                        **density,
                        **overlap,
                    }
                )

    df = pd.DataFrame(rows)
    summary = summarize(df)

    print("=" * 72)
    print("BENCHMARK SUMMARY")
    print("=" * 72)
    print(summary.sort_values(["method_label"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 72)
    print("PER-RUN RESULTS")
    print("=" * 72)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
