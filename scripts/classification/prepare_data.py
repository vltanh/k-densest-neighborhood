import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from split_utils import (  # noqa: E402
    build_out_adjacency,
    compute_hard_subset,
    sha256_edge_list,
    write_split_meta,
)

try:
    from torch_geometric.datasets import CitationFull
except ImportError:
    print("Error: PyTorch Geometric not found.")
    exit(1)


def _library_versions() -> dict:
    versions = {}
    try:
        import torch_geometric

        versions["torch_geometric"] = torch_geometric.__version__
    except Exception:
        pass
    try:
        import numpy

        versions["numpy"] = numpy.__version__
    except Exception:
        pass
    try:
        import pandas as _pd

        versions["pandas"] = _pd.__version__
    except Exception:
        pass
    return versions


def prepare_citation_full(dataset_name, seed: int = 42, data_dir: str = "data"):
    print(f"--- Loading {dataset_name} (CitationFull) ---")

    ds_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(ds_dir, exist_ok=True)

    dataset = CitationFull(root=ds_dir, name=dataset_name, to_undirected=False)
    data = dataset[0]

    edge_csv = os.path.join(ds_dir, "edge.csv")
    node_csv = os.path.join(ds_dir, "nodes.csv")

    edges = data.edge_index.numpy().T
    df_edges = pd.DataFrame(edges, columns=["source", "target"])
    df_edges.to_csv(edge_csv, index=False)
    print(f"Exported {len(df_edges)} directed edges to {edge_csv}")

    cited_nodes = set(df_edges["target"].astype(int).unique())
    all_citing_nodes = set(df_edges["source"].astype(int).unique())
    pure_sources = sorted(all_citing_nodes - cited_nodes)

    out_adj = build_out_adjacency(df_edges)
    pool = sorted(q for q in pure_sources if len(out_adj.get(q, ())) >= 2)

    print(f"Total Nodes: {data.num_nodes}")
    print(f"Pure-source candidates: {len(pure_sources)}")
    print(f"Pool after out-degree >= 2 filter: {len(pool)}")

    rng = np.random.default_rng(seed)
    pool = list(pool)
    rng.shuffle(pool)
    split_idx = int(len(pool) * 0.5)
    val_nodes = sorted(pool[:split_idx])
    test_nodes = sorted(pool[split_idx:])

    df_nodes = pd.DataFrame(
        {"node_id": range(data.num_nodes), "label": data.y.numpy().astype(int)}
    )
    df_nodes["val"] = df_nodes["node_id"].isin(val_nodes)
    df_nodes["test"] = df_nodes["node_id"].isin(test_nodes)
    df_nodes["train"] = ~(df_nodes["val"] | df_nodes["test"])

    df_nodes.to_csv(node_csv, index=False)
    print(f"Exported masks to {node_csv}")

    train_ids = df_nodes[df_nodes["train"]]["node_id"].astype(int).tolist()
    val_ids = df_nodes[df_nodes["val"]]["node_id"].astype(int).tolist()
    test_ids = df_nodes[df_nodes["test"]]["node_id"].astype(int).tolist()
    labels = df_nodes["label"].values
    train_mask = df_nodes["train"].values

    hard_subset = compute_hard_subset(val_ids, out_adj, train_mask, labels)

    edges_iter = zip(
        df_edges["source"].astype(int).tolist(),
        df_edges["target"].astype(int).tolist(),
    )
    edges_hash = sha256_edge_list(edges_iter)

    payload = write_split_meta(
        dataset_name=dataset_name,
        seed=seed,
        num_nodes=int(data.num_nodes),
        num_edges=int(len(df_edges)),
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        hard_subset_ids=hard_subset,
        edges_hash=edges_hash,
        library_versions=_library_versions(),
        data_dir=data_dir,
    )
    print(
        f"split_meta.json: train={payload['splits']['train']['size']}, "
        f"val={payload['splits']['val']['size']}, "
        f"test={payload['splits']['test']['size']}, "
        f"hard_subset={payload['hard_subset']['size']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora_ML",
        choices=["Cora", "Cora_ML", "CiteSeer", "DBLP"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    prepare_citation_full(args.dataset, seed=args.seed, data_dir=args.data_dir)
