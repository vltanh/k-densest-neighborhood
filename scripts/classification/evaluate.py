import os
import pandas as pd
import argparse
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from solver_utils import evaluate_nodes
from tune_methods import limit_nodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate K-Densest on a specific split."
    )
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument(
        "--split", type=str, required=True, choices=["train", "val", "test"]
    )
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--bin_path", type=str, default="./solver/bin/solver")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--max_in_edges", type=int, default=0)
    parser.add_argument(
        "--limit_nodes",
        type=int,
        default=0,
        help="Evaluate at most this many nodes from the selected split. Use 0 for the full split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--weighting",
        type=str,
        default="uniform",
        choices=["uniform", "distance"],
        help="Voting weight strategy",
    )
    args = parser.parse_args()

    edge_csv = os.path.join("data", args.dataset, "edge.csv")
    node_csv = os.path.join("data", args.dataset, "nodes.csv")
    tmp_dir = os.path.join("exps", "classification", args.dataset, "tmp_outputs")
    os.makedirs(tmp_dir, exist_ok=True)

    df_nodes = pd.read_csv(node_csv)
    target_nodes = df_nodes[df_nodes[args.split]]["node_id"].tolist()
    target_nodes = limit_nodes(target_nodes, args.limit_nodes, args.seed)

    print(f"--- Evaluating '{args.split}' split for {args.dataset} (k={args.k}) ---")

    y_true, y_pred = evaluate_nodes(
        target_nodes,
        args.k,
        edge_csv,
        df_nodes,
        args.bin_path,
        tmp_dir,
        args.workers,
        weighting=args.weighting,
        max_in_edges=args.max_in_edges,
    )

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n==================================================")
    print(f"{args.split.upper()} METRICS (k={args.k})")
    print("==================================================")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f} (Macro)")
    print(f"Recall    : {rec:.4f} (Macro)")
    print(f"F1 Score  : {f1:.4f} (Macro)")
    print("--------------------------------------------------")
    print("PER-CLASS REPORT:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print("==================================================")
