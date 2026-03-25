import os
import pandas as pd
import numpy as np
import argparse

try:
    from torch_geometric.datasets import CitationFull
except ImportError:
    print("Error: PyTorch Geometric not found.")
    exit(1)


def prepare_citation_full(dataset_name, data_dir="data"):
    print(f"--- Loading {dataset_name} (CitationFull) ---")

    # 1. Establish the strict working directory first
    ds_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(ds_dir, exist_ok=True)

    # 2. Force PyG to download and process strictly inside this directory
    dataset = CitationFull(root=ds_dir, name=dataset_name, to_undirected=False)
    data = dataset[0]

    edge_csv = os.path.join(ds_dir, "edge.csv")
    node_csv = os.path.join(ds_dir, "nodes.csv")

    # 3. Export Edges
    edges = data.edge_index.numpy().T
    df_edges = pd.DataFrame(edges, columns=["source", "target"])
    df_edges.to_csv(edge_csv, index=False)
    print(f"Exported {len(df_edges)} directed edges to {edge_csv}")

    # 4. Identify the Temporal/Inductive Split
    # "Old" papers are targets (they receive citations)
    cited_nodes = set(df_edges["target"].unique())

    # "New" papers are sources that cite others but have NEVER been cited yet
    all_citing_nodes = set(df_edges["source"].unique())
    pure_sources = list(all_citing_nodes - cited_nodes)

    print(f"Total Nodes: {data.num_nodes}")
    print(f"Found {len(cited_nodes)} foundational papers (assigned to Train).")
    print(f"Found {len(pure_sources)} brand new papers (assigned to Val/Test).")

    # 5. Create the Validation and Test splits from the new papers
    np.random.seed(42)
    np.random.shuffle(pure_sources)

    split_idx = int(len(pure_sources) * 0.5)
    val_nodes = set(pure_sources[:split_idx])
    test_nodes = set(pure_sources[split_idx:])

    # 6. Build the Nodes DataFrame with perfectly aligned masks
    df_nodes = pd.DataFrame({"node_id": range(data.num_nodes), "label": data.y.numpy()})

    # Map the boolean masks
    df_nodes["train"] = df_nodes["node_id"].isin(cited_nodes)
    df_nodes["val"] = df_nodes["node_id"].isin(val_nodes)
    df_nodes["test"] = df_nodes["node_id"].isin(test_nodes)

    df_nodes.to_csv(node_csv, index=False)
    print(
        f"Exported masks to {node_csv}! You can now run tune.py and evaluate.py directly."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora_ML",
        choices=["Cora", "Cora_ML", "CiteSeer", "DBLP", "PubMed"],
    )
    args = parser.parse_args()

    prepare_citation_full(args.dataset)
