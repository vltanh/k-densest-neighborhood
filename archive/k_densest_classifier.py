import os
import subprocess
import pandas as pd
import numpy as np
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# PyTorch Geometric imports
try:
    from torch_geometric.datasets import Planetoid
except ImportError:
    print(
        "Error: PyTorch Geometric not found. Please install via: pip install torch_geometric"
    )
    exit(1)


def prepare_dataset(dataset_name, data_dir="data"):
    """Loads the Planetoid dataset and exports edges and node splits to disk."""
    print(f"\n--- 1. Loading {dataset_name} Dataset ---")
    dataset = Planetoid(root=data_dir, name=dataset_name, split="public")
    data = dataset[0]

    ds_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(ds_dir, exist_ok=True)

    edge_csv = os.path.join(ds_dir, "edge.csv")
    node_csv = os.path.join(ds_dir, "nodes.csv")

    # 1. Export Edges (PyG edge_index contains both directions for undirected graphs, which is perfect)
    if not os.path.exists(edge_csv):
        edges = data.edge_index.numpy().T
        df_edges = pd.DataFrame(edges, columns=["source", "target"])
        df_edges.to_csv(edge_csv, index=False)
        print(f"Exported {len(df_edges)} edges to {edge_csv}")

    # 2. Export Node Metadata
    if not os.path.exists(node_csv):
        df_nodes = pd.DataFrame(
            {
                "node_id": range(data.num_nodes),
                "label": data.y.numpy(),
                "train": data.train_mask.numpy(),
                "val": data.val_mask.numpy(),
                "test": data.test_mask.numpy(),
            }
        )
        df_nodes.to_csv(node_csv, index=False)
        print(f"Exported {len(df_nodes)} nodes to {node_csv}")

    return edge_csv, node_csv, data


def run_solver(q_node, k, edge_csv, bin_path, tmp_dir):
    """Executes the C++ Branch-and-Price solver for a single query node."""
    out_csv = os.path.join(tmp_dir, f"out_q{q_node}_k{k}.csv")
    cmd = [bin_path, edge_csv, str(q_node), str(k), out_csv]

    try:
        # Hide standard output to prevent console flooding; capture errors
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Solver failed for node {q_node}: {e.stderr}")
        return q_node, []

    # Parse the output file
    if os.path.exists(out_csv):
        try:
            pred_df = pd.read_csv(out_csv)
            neighborhood = pred_df["node_id"].astype(int).tolist()
        except Exception:
            neighborhood = []
        # Cleanup temp file to save disk space
        os.remove(out_csv)
        return q_node, neighborhood

    return q_node, []


def evaluate_split(
    split_nodes, k, edge_csv, df_nodes, bin_path, tmp_dir, max_workers=8
):
    """Evaluates the K-Densest classification on a specific set of nodes."""
    # Pre-extract numpy arrays for fast O(1) lookups during majority voting
    train_mask = df_nodes["train"].values
    labels = df_nodes["label"].values

    # Global fallback: Most frequent class in the entire training set
    train_labels = labels[train_mask]
    global_majority = Counter(train_labels).most_common(1)[0][0]

    correct = 0
    total = len(split_nodes)

    print(f"Batched queries for k={k} ({total} nodes) using {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_solver, q, k, edge_csv, bin_path, tmp_dir): q
            for q in split_nodes
        }

        for future in tqdm(
            as_completed(futures), total=total, desc=f"Evaluating k={k}"
        ):
            q_node, neighborhood = future.result()

            # Step 1: Filter the neighborhood to strictly contain Train nodes
            train_neighbors = [n for n in neighborhood if train_mask[n]]

            # Step 2: Majority Vote
            if not train_neighbors:
                # Fallback if the neighborhood is entirely unlabeled or disconnected
                pred_label = global_majority
            else:
                neighbor_labels = [labels[n] for n in train_neighbors]
                # Tie-breaking natively handled by Counter (returns first encountered)
                pred_label = Counter(neighbor_labels).most_common(1)[0][0]

            # Step 3: Score
            if pred_label == labels[q_node]:
                correct += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="K-Densest Neighborhood Node Classification"
    )
    parser.add_argument(
        "--dataset", type=str, default="Cora", choices=["Cora", "CiteSeer", "PubMed"]
    )
    parser.add_argument(
        "--bin_path", type=str, default="./bin/solver", help="Path to C++ solver binary"
    )
    parser.add_argument("--k_min", type=int, default=5, help="Minimum k to sweep")
    parser.add_argument("--k_max", type=int, default=25, help="Maximum k to sweep")
    parser.add_argument("--k_step", type=int, default=5, help="Step size for k sweep")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Threads for parallel execution",
    )
    args = parser.parse_args()

    if not os.path.exists(args.bin_path):
        print(f"Error: C++ binary not found at {args.bin_path}. Please compile first.")
        exit(1)

    # 1. Prepare Data
    edge_csv, node_csv, data = prepare_dataset(args.dataset)
    df_nodes = pd.read_csv(node_csv)

    val_nodes = df_nodes[df_nodes["val"]]["node_id"].tolist()
    test_nodes = df_nodes[df_nodes["test"]]["node_id"].tolist()

    tmp_dir = os.path.join("data", args.dataset, "tmp_outputs")
    os.makedirs(tmp_dir, exist_ok=True)

    # 2. Hyperparameter Tuning on Validation Split
    print("\n--- 2. Validation Sweep (Tuning k) ---")
    best_k = args.k_min
    best_val_acc = 0.0

    for k in range(args.k_min, args.k_max + 1, args.k_step):
        acc = evaluate_split(
            val_nodes, k, edge_csv, df_nodes, args.bin_path, tmp_dir, args.workers
        )
        print(f"Validation Accuracy (k={k}): {acc * 100:.2f}%")

        if acc > best_val_acc:
            best_val_acc = acc
            best_k = k

    print(
        f"\n=> Best k determined by validation split: {best_k} (Acc: {best_val_acc * 100:.2f}%)"
    )

    # 3. Final Evaluation on Test Split
    print("\n--- 3. Final Test Evaluation ---")
    test_acc = evaluate_split(
        test_nodes, best_k, edge_csv, df_nodes, args.bin_path, tmp_dir, args.workers
    )

    print("==================================================")
    print("FINAL CLASSIFICATION RESULTS")
    print("==================================================")
    print(f"Dataset      : {args.dataset}")
    print(f"Optimal k    : {best_k}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print("==================================================")
