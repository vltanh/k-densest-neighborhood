import os
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.append(os.path.dirname(__file__))
from solver_utils import evaluate_nodes


def main():
    dataset = "Cora"
    base_dir = os.path.join(
        "exps", "classification", dataset, "test_fixed_bfs_depth1_wrong"
    )
    subset_path = os.path.join(base_dir, "subset_nodes.csv")
    node_csv = os.path.join("data", dataset, "nodes.csv")
    edge_csv = os.path.join("data", dataset, "edge.csv")
    bin_path = "./solver/bin/solver"

    df_nodes = pd.read_csv(node_csv)
    query_nodes = pd.read_csv(subset_path)["node_id"].astype(int).tolist()

    rows = []
    out_path = os.path.join(base_dir, "bfs_depth_sweep.csv")
    for depth in [1, 2, 3]:
        print(f"\n=== hard subset bfs depth={depth} ===", flush=True)
        tmp_dir = os.path.join(base_dir, f"bfs_depth{depth}_sweep")
        os.makedirs(tmp_dir, exist_ok=True)
        y_true, y_pred, stats = evaluate_nodes(
            query_nodes,
            None,
            edge_csv,
            df_nodes,
            bin_path,
            tmp_dir,
            max_workers=4,
            weighting="distance",
            extra_args=["--bfs", "--bfs-depth", str(depth)],
            collect_stats=True,
            show_progress=True,
            compute_qualities=True,
            max_in_edges=0,
        )
        row = {
            "method": "bfs",
            "depth": depth,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            **stats,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(row, flush=True)

    print(f"\nSaved {out_path}", flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
