import os
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from solver_utils import evaluate_nodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for k on validation set."
    )
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--bin_path", type=str, default="./solver/bin/solver")
    parser.add_argument("--k_min", type=int, default=5)
    parser.add_argument("--k_max", type=int, default=25)
    parser.add_argument("--k_step", type=int, default=5)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument(
        "--optimize",
        type=str,
        default="accuracy",
        choices=["accuracy", "f1", "precision", "recall"],
        help="The target metric to optimize when selecting the best k",
    )
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
    tmp_dir = os.path.join("data", args.dataset, "tmp_outputs")
    os.makedirs(tmp_dir, exist_ok=True)

    df_nodes = pd.read_csv(node_csv)
    val_nodes = df_nodes[df_nodes["val"]]["node_id"].tolist()

    print(
        f"--- Sweeping k for {args.dataset} Validation Set ({len(val_nodes)} nodes) ---"
    )
    print(f"--- Optimizing for: {args.optimize.upper()} ---")

    best_k = args.k_min
    best_score = 0.0

    for k in range(args.k_min, args.k_max + 1, args.k_step):
        y_true, y_pred = evaluate_nodes(
            val_nodes,
            k,
            edge_csv,
            df_nodes,
            args.bin_path,
            tmp_dir,
            args.workers,
            weighting=args.weighting,
        )

        # Calculate all metrics (using macro average for multiclass)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        print(
            f"[k={k:<2}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}"
        )

        # Select the target metric to track
        metrics_dict = {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}
        current_target_score = metrics_dict[args.optimize]

        if current_target_score > best_score:
            best_score = current_target_score
            best_k = k

    print("==================================================")
    print(f"Optimal k found : {best_k}")
    print(f"Best {args.optimize.capitalize()} : {best_score:.4f}")
    print("==================================================")
