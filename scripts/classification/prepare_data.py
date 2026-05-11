import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from split_utils import sha256_edge_list, write_split_meta  # noqa: E402

try:
    from torch_geometric.datasets import CitationFull
except ImportError:
    print("Error: PyTorch Geometric not found.")
    exit(1)


def _argmax_label(counter: Counter):
    if not counter:
        return None
    return min(((-count, label) for label, count in counter.items()))[1]


def _undirected_adjacency(df_edges: pd.DataFrame):
    adj = defaultdict(set)
    for s, t in zip(df_edges["source"].astype(int), df_edges["target"].astype(int)):
        if s == t:
            continue
        adj[s].add(t)
        adj[t].add(s)
    return adj


def _bfs_depth1_predict(q_node, adj_und, train_mask, labels, global_majority):
    neighbors = adj_und.get(q_node, ())
    train_neighbors = [n for n in neighbors if n != q_node and train_mask[n]]
    if not train_neighbors:
        return global_majority
    return _argmax_label(Counter(labels[n] for n in train_neighbors))


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

    print(f"Total Nodes: {data.num_nodes}")
    print(f"Cited papers (Train): {len(cited_nodes)}")
    print(f"New papers (Val/Test): {len(pure_sources)}")

    rng = np.random.default_rng(seed)
    pure_sources = list(pure_sources)
    rng.shuffle(pure_sources)
    split_idx = int(len(pure_sources) * 0.5)
    val_nodes = sorted(pure_sources[:split_idx])
    test_nodes = sorted(pure_sources[split_idx:])

    df_nodes = pd.DataFrame(
        {"node_id": range(data.num_nodes), "label": data.y.numpy().astype(int)}
    )
    df_nodes["train"] = df_nodes["node_id"].isin(cited_nodes)
    df_nodes["val"] = df_nodes["node_id"].isin(val_nodes)
    df_nodes["test"] = df_nodes["node_id"].isin(test_nodes)

    df_nodes.to_csv(node_csv, index=False)
    print(f"Exported masks to {node_csv}")

    train_ids = df_nodes[df_nodes["train"]]["node_id"].astype(int).tolist()
    val_ids = df_nodes[df_nodes["val"]]["node_id"].astype(int).tolist()
    test_ids = df_nodes[df_nodes["test"]]["node_id"].astype(int).tolist()
    labels = df_nodes["label"].values
    train_mask = df_nodes["train"].values

    adj_und = _undirected_adjacency(df_edges)
    eligible_val = [q for q in val_ids if len(adj_und.get(q, ())) >= 2]
    eligible_test = [q for q in test_ids if len(adj_und.get(q, ())) >= 2]

    global_majority = _argmax_label(Counter(labels[train_mask]))

    hard_subset = [
        q for q in eligible_val
        if _bfs_depth1_predict(q, adj_und, train_mask, labels, global_majority) != labels[q]
    ]

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
        eligible_val_ids=eligible_val,
        eligible_test_ids=eligible_test,
        hard_subset_ids=hard_subset,
        edges_hash=edges_hash,
        library_versions=_library_versions(),
        data_dir=data_dir,
    )
    print(
        f"split_meta.json: train={payload['splits']['train']['size']}, "
        f"val={payload['splits']['val']['size']} (eligible {payload['eligible']['val']['size']}), "
        f"test={payload['splits']['test']['size']} (eligible {payload['eligible']['test']['size']}), "
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
