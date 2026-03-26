import os
import pandas as pd
import networkx as nx
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def classify_bfs(q_node, G, train_mask, labels, global_majority, max_hops):
    """Worker function to classify a single node using Concentric BFS."""
    try:
        paths = nx.single_source_shortest_path_length(G, q_node, cutoff=max_hops)
        reachable_train = {
            n: d for n, d in paths.items() if train_mask[n] and n != q_node
        }

        if reachable_train:
            min_dist = min(reachable_train.values())
            nearest_train_nodes = [
                n for n, d in reachable_train.items() if d == min_dist
            ]

            fallback_labels = [labels[n] for n in nearest_train_nodes]
            return Counter(fallback_labels).most_common(1)[0][0]

    except Exception:
        pass

    return global_majority


def evaluate_baseline(query_nodes, edge_csv, df_nodes, max_hops=10, max_workers=8):
    """Runs the BFS baseline on a batch of query nodes."""
    train_mask = df_nodes["train"].values
    labels = df_nodes["label"].values

    train_labels = labels[train_mask]
    global_majority = Counter(train_labels).most_common(1)[0][0]

    df_edges = pd.read_csv(edge_csv)
    G = nx.from_pandas_edgelist(
        df_edges, source="source", target="target", create_using=nx.Graph()
    )

    y_true = []
    y_pred = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                classify_bfs, q, G, train_mask, labels, global_majority, max_hops
            ): q
            for q in query_nodes
        }

        for future in tqdm(
            as_completed(futures), total=len(query_nodes), desc="Running BFS Baseline"
        ):
            q_node = futures[future]
            pred_label = future.result()

            y_true.append(labels[q_node])
            y_pred.append(pred_label)

    return y_true, y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concentric BFS Baseline Classifier")
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument(
        "--split", type=str, required=True, choices=["train", "val", "test"]
    )
    parser.add_argument("--max_hops", type=int, default=10)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    edge_csv = os.path.join("data", args.dataset, "edge.csv")
    node_csv = os.path.join("data", args.dataset, "nodes.csv")

    if not os.path.exists(edge_csv) or not os.path.exists(node_csv):
        print(f"Error: Data for {args.dataset} not found. Run prepare_data.py first.")
        exit(1)

    df_nodes = pd.read_csv(node_csv)
    target_nodes = df_nodes[df_nodes[args.split]]["node_id"].tolist()

    print(f"--- Running BFS Baseline on '{args.split}' split for {args.dataset} ---")

    y_true, y_pred = evaluate_baseline(
        target_nodes, edge_csv, df_nodes, args.max_hops, args.workers
    )

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n==================================================")
    print(f"BFS BASELINE METRICS ({args.split.upper()})")
    print("==================================================")
    print(f"Max Search Hops: {args.max_hops}")
    print("--------------------------------------------------")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f} (Macro)")
    print(f"Recall    : {rec:.4f} (Macro)")
    print(f"F1 Score  : {f1:.4f} (Macro)")
    print("--------------------------------------------------")
    print("PER-CLASS REPORT:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print("==================================================")
