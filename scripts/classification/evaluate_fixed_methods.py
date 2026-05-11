import os
import sys
import argparse
import json

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver_utils import evaluate_nodes
from tune_methods import limit_nodes
from _solver_runner import build_bp_extra_args


def _maybe_int(value):
    return None if value is None or pd.isna(value) else int(value)


def _maybe_float(value):
    return None if value is None or pd.isna(value) else float(value)


def _bp_extra_args(
    kappa,
    time_limit,
    cg_batch_frac,
    cg_min_batch,
    cg_max_batch,
    node_limit,
    gap_tol,
    dinkelbach_iter,
):
    return build_bp_extra_args(
        kappa=kappa,
        time_limit=time_limit,
        cg_batch_frac=cg_batch_frac,
        cg_min_batch=cg_min_batch,
        cg_max_batch=cg_max_batch,
        node_limit=node_limit,
        gap_tol=gap_tol,
        dinkelbach_iter=dinkelbach_iter,
    )


def methods_from_settings(settings_path):
    with open(settings_path, "r", encoding="utf-8") as f:
        settings = json.load(f)

    best = settings["best"]
    methods = []

    if "bfs" in best:
        row = best["bfs"]
        depth = _maybe_int(row.get("depth")) or 1
        methods.append(
            {
                "method": "bfs",
                "k": None,
                "kappa": None,
                "depth": depth,
                "time_limit": None,
                "cg_batch_frac": None,
                "cg_min_batch": None,
                "cg_max_batch": None,
                "node_limit": None,
                "gap_tol": None,
                "dinkelbach_iter": None,
                "extra_args": ["--bfs", "--bfs-depth", str(depth)],
                "tmp": f"bfs_depth{depth}",
            }
        )

    if "bp" in best:
        row = best["bp"]
        k = _maybe_int(row.get("k"))
        kappa = _maybe_int(row.get("kappa")) or 0
        time_limit = _maybe_float(row.get("time_limit"))
        if time_limit is None:
            time_limit = -1.0
        cg_batch_frac = _maybe_float(row.get("cg_batch_frac"))
        cg_min_batch = _maybe_int(row.get("cg_min_batch"))
        cg_max_batch = _maybe_int(row.get("cg_max_batch"))
        node_limit = _maybe_int(row.get("node_limit"))
        gap_tol = _maybe_float(row.get("gap_tol"))
        dinkelbach_iter = _maybe_int(row.get("dinkelbach_iter"))
        methods.append(
            {
                "method": "bp",
                "k": k,
                "kappa": kappa,
                "depth": None,
                "time_limit": time_limit,
                "cg_batch_frac": cg_batch_frac,
                "cg_min_batch": cg_min_batch,
                "cg_max_batch": cg_max_batch,
                "node_limit": node_limit,
                "gap_tol": gap_tol,
                "dinkelbach_iter": dinkelbach_iter,
                "extra_args": _bp_extra_args(
                    kappa,
                    time_limit,
                    cg_batch_frac,
                    cg_min_batch,
                    cg_max_batch,
                    node_limit,
                    gap_tol,
                    dinkelbach_iter,
                ),
                "tmp": f"bp_k{k}_kappa{kappa}",
            }
        )

    if "avgdeg" in best:
        methods.append(
            {
                "method": "avgdeg",
                "k": None,
                "kappa": None,
                "depth": None,
                "time_limit": None,
                "cg_batch_frac": None,
                "cg_min_batch": None,
                "cg_max_batch": None,
                "node_limit": None,
                "gap_tol": None,
                "dinkelbach_iter": None,
                "extra_args": ["--avgdeg"],
                "tmp": "avgdeg",
            }
        )

    return methods


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fixed classification methods on a split."
    )
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"]
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        choices=["all", "bfs_depth1_wrong"],
        help="Optional subset of the split to evaluate.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--limit_nodes",
        type=int,
        default=0,
        help="Evaluate at most this many nodes from the selected split. Use 0 for the full split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_in_edges", type=int, default=0)
    parser.add_argument(
        "--settings_json",
        type=str,
        default="",
        help="Optional best_settings JSON produced by tune_methods.py.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="",
        help="Optional output directory name under exps/classification/<dataset>/.",
    )
    parser.add_argument("--bp_k", type=int, default=5)
    parser.add_argument("--bp_kappa", type=int, default=2)
    parser.add_argument("--bp_time_limit", type=float, default=-1.0)
    parser.add_argument("--bp_cg_batch_frac", type=float, default=None)
    parser.add_argument("--bp_cg_min_batch", type=int, default=None)
    parser.add_argument("--bp_cg_max_batch", type=int, default=None)
    parser.add_argument("--bp_node_limit", type=int, default=None)
    parser.add_argument("--bp_gap_tol", type=float, default=None)
    parser.add_argument("--bp_dinkelbach_iter", type=int, default=None)
    parser.add_argument("--bfs_depth", type=int, default=1)
    args = parser.parse_args()

    dataset = args.dataset
    edge_csv = os.path.join("data", dataset, "edge.csv")
    node_csv = os.path.join("data", dataset, "nodes.csv")
    bin_path = "./solver/bin/solver"
    if args.output_name:
        out_name = args.output_name
    elif args.settings_json:
        out_name = (
            f"{args.split}_best"
            if args.subset == "all"
            else f"{args.split}_best_{args.subset}"
        )
    else:
        out_name = (
            f"{args.split}_fixed"
            if args.subset == "all"
            else f"{args.split}_fixed_{args.subset}"
        )
    out_dir = os.path.join("exps", "classification", dataset, out_name)
    os.makedirs(out_dir, exist_ok=True)

    df_nodes = pd.read_csv(node_csv)
    target_nodes = df_nodes[df_nodes[args.split]]["node_id"].tolist()
    target_nodes = limit_nodes(target_nodes, args.limit_nodes, args.seed)

    if args.settings_json:
        methods = methods_from_settings(args.settings_json)
    else:
        methods = [
            {
                "method": "bfs",
                "k": None,
                "kappa": None,
                "depth": args.bfs_depth,
                "time_limit": None,
                "cg_batch_frac": None,
                "cg_min_batch": None,
                "cg_max_batch": None,
                "node_limit": None,
                "gap_tol": None,
                "dinkelbach_iter": None,
                "extra_args": ["--bfs", "--bfs-depth", str(args.bfs_depth)],
                "tmp": f"bfs_depth{args.bfs_depth}",
            },
            {
                "method": "bp",
                "k": args.bp_k,
                "kappa": args.bp_kappa,
                "depth": None,
                "time_limit": args.bp_time_limit,
                "cg_batch_frac": args.bp_cg_batch_frac,
                "cg_min_batch": args.bp_cg_min_batch,
                "cg_max_batch": args.bp_cg_max_batch,
                "node_limit": args.bp_node_limit,
                "gap_tol": args.bp_gap_tol,
                "dinkelbach_iter": args.bp_dinkelbach_iter,
                "extra_args": _bp_extra_args(
                    args.bp_kappa,
                    args.bp_time_limit,
                    args.bp_cg_batch_frac,
                    args.bp_cg_min_batch,
                    args.bp_cg_max_batch,
                    args.bp_node_limit,
                    args.bp_gap_tol,
                    args.bp_dinkelbach_iter,
                ),
                "tmp": f"bp_k{args.bp_k}_kappa{args.bp_kappa}",
            },
            {
                "method": "avgdeg",
                "k": None,
                "kappa": None,
                "depth": None,
                "time_limit": None,
                "cg_batch_frac": None,
                "cg_min_batch": None,
                "cg_max_batch": None,
                "node_limit": None,
                "gap_tol": None,
                "dinkelbach_iter": None,
                "extra_args": ["--avgdeg"],
                "tmp": "avgdeg",
            },
        ]

    if args.subset == "bfs_depth1_wrong":
        bfs_method = methods[0]
        print(
            f"\n=== SELECTING {args.split} nodes where BFS depth=1 is wrong ===",
            flush=True,
        )
        tmp_dir = os.path.join(out_dir, "_subset_bfs_depth1")
        os.makedirs(tmp_dir, exist_ok=True)
        evaluated_nodes, y_true, y_pred, _ = evaluate_nodes(
            target_nodes,
            bfs_method["k"],
            edge_csv,
            df_nodes,
            bin_path,
            tmp_dir,
            max_workers=args.workers,
            weighting="distance",
            extra_args=bfs_method["extra_args"],
            collect_stats=True,
            show_progress=True,
            compute_qualities=False,
            max_in_edges=args.max_in_edges,
            return_query_nodes=True,
        )
        target_nodes = [
            node
            for node, true_label, pred_label in zip(
                evaluated_nodes, y_true, y_pred
            )
            if true_label != pred_label
        ]
        pd.DataFrame({"node_id": target_nodes}).to_csv(
            os.path.join(out_dir, "subset_nodes.csv"), index=False
        )
        print(f"Selected {len(target_nodes)} nodes", flush=True)

    rows = []
    partial_path = os.path.join(out_dir, "fixed_methods.partial.csv")
    out_path = os.path.join(out_dir, "fixed_methods.csv")

    for method in methods:
        print(
            f"\n=== {args.split.upper()} {method['method']} "
            f"k={method['k']} kappa={method['kappa']} depth={method['depth']} ===",
            flush=True,
        )
        tmp_dir = os.path.join(out_dir, method["tmp"])
        os.makedirs(tmp_dir, exist_ok=True)
        y_true, y_pred, stats = evaluate_nodes(
            target_nodes,
            method["k"],
            edge_csv,
            df_nodes,
            bin_path,
            tmp_dir,
            max_workers=args.workers,
            weighting="distance",
            extra_args=method["extra_args"],
            collect_stats=True,
            show_progress=True,
            compute_qualities=True,
            max_in_edges=args.max_in_edges,
        )

        row = {
            "method": method["method"],
            "k": method["k"],
            "kappa": method["kappa"],
            "depth": method["depth"],
            "time_limit": method["time_limit"],
            "cg_batch_frac": method["cg_batch_frac"],
            "cg_min_batch": method["cg_min_batch"],
            "cg_max_batch": method["cg_max_batch"],
            "node_limit": method["node_limit"],
            "gap_tol": method["gap_tol"],
            "dinkelbach_iter": method["dinkelbach_iter"],
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            **stats,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(partial_path, index=False)
        print(row, flush=True)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved {out_path}", flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
