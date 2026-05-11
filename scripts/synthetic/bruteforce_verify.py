import argparse
import itertools
import os
import subprocess
import tempfile

import pandas as pd


DEMO_EDGES = [
    ("0", "1"),
    ("1", "2"),
    ("2", "0"),
    ("1", "3"),
    ("3", "4"),
    ("4", "1"),
]


def load_edges(edge_csv):
    df = pd.read_csv(edge_csv)
    if len(df.columns) < 2:
        raise ValueError(f"{edge_csv} must have at least two columns")
    src_col, dst_col = df.columns[:2]
    edges = []
    nodes = set()
    for src, dst in zip(df[src_col].astype(str), df[dst_col].astype(str)):
        if src == dst:
            continue
        edges.append((src, dst))
        nodes.add(src)
        nodes.add(dst)
    return sorted(nodes), edges


def write_demo_csv(path):
    df = pd.DataFrame(DEMO_EDGES, columns=["source", "target"])
    df.to_csv(path, index=False)


def internal_edge_count(edges, subset):
    subset = set(subset)
    return sum(1 for u, v in edges if u in subset and v in subset)


def brute_force_avgdeg(nodes, edges, query):
    best_subset = None
    best_score = float("-inf")
    for r in range(1, len(nodes) + 1):
        for subset in itertools.combinations(nodes, r):
            if query not in subset:
                continue
            internal = internal_edge_count(edges, subset)
            score = internal / len(subset)
            if score > best_score:
                best_score = score
                best_subset = tuple(sorted(subset))
    return best_subset, best_score


def brute_force_bp(nodes, edges, query, k):
    best_subset = None
    best_score = float("-inf")
    for r in range(max(1, k), len(nodes) + 1):
        for subset in itertools.combinations(nodes, r):
            if query not in subset:
                continue
            internal = internal_edge_count(edges, subset)
            score = internal / (len(subset) * (len(subset) - 1)) if len(subset) > 1 else 0.0
            if score > best_score:
                best_score = score
                best_subset = tuple(sorted(subset))
    return best_subset, best_score


def run_solver(bin_path, edge_csv, query, method, args):
    out_csv = tempfile.mktemp(suffix=".csv")
    cmd = [
        bin_path,
        "--mode",
        "sim",
        "--input",
        edge_csv,
        "--query",
        str(query),
        "--output",
        out_csv,
    ]
    if method == "avgdeg":
        cmd.append("--avgdeg")
    if method == "bp":
        cmd.extend(["--bp", "--k", str(args.k), "--kappa", str(args.kappa)])
    if args.time_limit is not None:
        cmd.extend(["--time-limit", str(args.time_limit)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        pred_df = pd.read_csv(out_csv)
        pred_nodes = tuple(sorted(pred_df["node_id"].astype(str).tolist()))
    except Exception:
        pred_nodes = tuple()
    finally:
        if os.path.exists(out_csv):
            os.remove(out_csv)
    return result.returncode, result.stdout, result.stderr, pred_nodes


def score_subset(edges, subset):
    subset = set(subset)
    internal = internal_edge_count(edges, subset)
    n = len(subset)
    avgdeg = internal / n if n else 0.0
    bp_density = internal / (n * (n - 1)) if n > 1 else 0.0
    return avgdeg, bp_density, internal


def main():
    parser = argparse.ArgumentParser(description="Brute-force verification on a tiny graph.")
    parser.add_argument("--edge-csv", type=str, default=None, help="Optional edge list CSV; if omitted, a built-in demo graph is used.")
    parser.add_argument("--query", type=str, default="1")
    parser.add_argument("--mode", type=str, choices=["avgdeg", "bp", "both"], default="both")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--kappa", type=int, default=0)
    parser.add_argument("--bin-path", type=str, default="./solver/bin/solver")
    parser.add_argument("--time-limit", type=float, default=-1.0)
    args = parser.parse_args()

    if args.edge_csv:
        nodes, edges = load_edges(args.edge_csv)
        edge_csv = args.edge_csv
        tmp_handle = None
    else:
        tmp_handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        tmp_handle.close()
        write_demo_csv(tmp_handle.name)
        nodes, edges = load_edges(tmp_handle.name)
        edge_csv = tmp_handle.name

    if args.query not in nodes:
        raise ValueError(f"Query node {args.query} is not present in the graph.")

    print(f"Graph nodes: {len(nodes)}")
    print(f"Graph directed edges: {len(edges)}")
    print(f"Query: {args.query}")

    if args.mode in ("avgdeg", "both"):
        exact_subset, exact_score = brute_force_avgdeg(nodes, edges, args.query)
        rc, stdout, stderr, solver_subset = run_solver(args.bin_path, edge_csv, args.query, "avgdeg", args)
        solver_avgdeg, _, solver_internal = score_subset(edges, solver_subset)
        print("\n[avgdeg]")
        print(f"exact subset:  {exact_subset}")
        print(f"exact score:   {exact_score:.6f}")
        print(f"solver subset: {solver_subset}")
        print(f"solver score:  {solver_avgdeg:.6f}")
        print(f"solver internal edges: {solver_internal}")
        print(f"solver exit:   {rc}")

    if args.mode in ("bp", "both"):
        exact_subset, exact_score = brute_force_bp(nodes, edges, args.query, args.k)
        rc, stdout, stderr, solver_subset = run_solver(args.bin_path, edge_csv, args.query, "bp", args)
        solver_avgdeg, solver_bp_density, solver_internal = score_subset(edges, solver_subset)
        print("\n[bp]")
        print(f"k:             {args.k}")
        print(f"exact subset:  {exact_subset}")
        print(f"exact density: {exact_score:.6f}")
        print(f"solver subset: {solver_subset}")
        print(f"solver density:{solver_bp_density:.6f}")
        print(f"solver internal edges: {solver_internal}")
        print(f"solver exit:   {rc}")

    if tmp_handle is not None and os.path.exists(tmp_handle.name):
        os.remove(tmp_handle.name)


if __name__ == "__main__":
    main()
