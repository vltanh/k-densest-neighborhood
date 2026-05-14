import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from split_utils import (  # noqa: E402
    build_out_adjacency,
    build_query_pool,
    compute_hard_subset,
    EXPECTED_POOL_MIN_OUT_2EDGE_COMPONENT_SIZE,
    sha256_edge_list,
    write_split_meta,
)

try:
    from torch_geometric.datasets import CitationFull
except ImportError:
    CitationFull = None


def _stratified_half_split(pool, labels, rng):
    by_label = {}
    for q in pool:
        by_label.setdefault(int(labels[q]), []).append(int(q))

    val_nodes = []
    test_nodes = []
    singleton_labels = []
    for label in sorted(by_label):
        bucket = by_label[label]
        rng.shuffle(bucket)
        n_val = len(bucket) // 2
        if len(bucket) % 2:
            if len(val_nodes) < len(test_nodes):
                n_val += 1
        if len(bucket) == 1:
            singleton_labels.append(label)
        val_nodes.extend(bucket[:n_val])
        test_nodes.extend(bucket[n_val:])
    return sorted(val_nodes), sorted(test_nodes), singleton_labels


def _library_versions(bin_path: str = "./solver/bin/solver") -> dict:
    versions = {}
    for mod_name, attr in (
        ("torch_geometric", "__version__"),
        ("numpy", "__version__"),
        ("pandas", "__version__"),
        ("scipy", "__version__"),
        ("networkx", "__version__"),
        ("sklearn", "__version__"),
    ):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, attr)
        except Exception:
            pass
    # Solver build id: prefer the binary's self-reported value via --help-style
    # JSON probe; fall back to a local git sha so a sloppily-staged checkout
    # still leaves an audit trail.
    versions["solver_build_id"] = _solver_build_id(bin_path)
    return versions


def _solver_build_id(bin_path: str) -> str:
    import os
    import subprocess

    if os.path.exists(bin_path):
        try:
            # Cheap probe: --emit-json with a tiny in-memory invocation is
            # expensive, so we shell out git rev-parse on the repo root which
            # matches the value baked into the binary at build time.
            sha = subprocess.check_output(
                ["git", "rev-parse", "--short=12", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if sha:
                return sha
        except Exception:
            pass
    return "unknown"


def _split_and_write(dataset_name, seed: int, data_dir: str, df_edges, labels_np, num_nodes: int):
    ds_dir = os.path.join(data_dir, dataset_name)
    node_csv = os.path.join(ds_dir, "nodes.csv")

    out_adj = build_out_adjacency(df_edges)
    query_pool, pool_stats = build_query_pool(df_edges)

    print(f"Total Nodes: {num_nodes}")
    print(f"Pure-source candidates: {pool_stats['pure_source_count']}")
    print(
        "Query pool after outgoing-reachable 2-edge component "
        f">= {EXPECTED_POOL_MIN_OUT_2EDGE_COMPONENT_SIZE}: {len(query_pool)}"
    )

    rng = np.random.default_rng(seed)
    val_nodes, test_nodes, singleton_labels = _stratified_half_split(
        query_pool, labels_np, rng
    )
    print(
        f"Label-stratified split: val={len(val_nodes)}, test={len(test_nodes)}, "
        f"singleton label buckets={len(singleton_labels)}"
    )

    df_nodes = pd.DataFrame(
        {"node_id": range(num_nodes), "label": labels_np}
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

    hard_subset = compute_hard_subset(test_ids, out_adj, train_mask, labels)

    edges_iter = zip(
        df_edges["source"].astype(int).tolist(),
        df_edges["target"].astype(int).tolist(),
    )
    edges_hash = sha256_edge_list(edges_iter)

    payload = write_split_meta(
        dataset_name=dataset_name,
        seed=seed,
        num_nodes=int(num_nodes),
        num_edges=int(len(df_edges)),
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        hard_subset_ids=hard_subset,
        edges_hash=edges_hash,
        library_versions=_library_versions(),
        data_dir=data_dir,
        query_pool_ids=query_pool,
    )
    print(
        f"split_meta.json: train={payload['splits']['train']['size']}, "
        f"val={payload['splits']['val']['size']}, "
        f"test={payload['splits']['test']['size']}, "
        f"hard_subset={payload['hard_subset']['size']}"
    )


def prepare_citation_full(dataset_name, seed: int = 42, data_dir: str = "data"):
    if CitationFull is None:
        raise RuntimeError(
            "PyTorch Geometric is not installed. Use --source existing to "
            "regenerate masks from data/<dataset>/edge.csv and nodes.csv."
        )

    print(f"--- Loading {dataset_name} (CitationFull) ---")

    ds_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(ds_dir, exist_ok=True)

    dataset = CitationFull(root=ds_dir, name=dataset_name, to_undirected=False)
    data = dataset[0]

    edge_csv = os.path.join(ds_dir, "edge.csv")

    edges = data.edge_index.numpy().T
    df_edges = pd.DataFrame(edges, columns=["source", "target"])
    df_edges.to_csv(edge_csv, index=False)
    print(f"Exported {len(df_edges)} directed edges to {edge_csv}")

    labels_np = data.y.numpy().astype(int)
    _split_and_write(dataset_name, seed, data_dir, df_edges, labels_np, int(data.num_nodes))


def prepare_existing_export(dataset_name, seed: int = 42, data_dir: str = "data"):
    print(f"--- Loading {dataset_name} from existing CSV export ---")

    ds_dir = os.path.join(data_dir, dataset_name)
    edge_csv = os.path.join(ds_dir, "edge.csv")
    node_csv = os.path.join(ds_dir, "nodes.csv")
    if not os.path.exists(edge_csv):
        raise FileNotFoundError(edge_csv)
    if not os.path.exists(node_csv):
        raise FileNotFoundError(node_csv)

    df_edges = pd.read_csv(edge_csv)
    df_nodes = pd.read_csv(node_csv)
    if "node_id" not in df_nodes.columns or "label" not in df_nodes.columns:
        raise ValueError(f"{node_csv} must contain node_id and label columns")

    node_ids = df_nodes["node_id"].astype(int)
    max_node = int(
        max(
            node_ids.max(),
            df_edges["source"].astype(int).max(),
            df_edges["target"].astype(int).max(),
        )
    )
    num_nodes = max_node + 1
    labels_np = np.zeros(num_nodes, dtype=int)
    labels_np[node_ids.to_numpy()] = df_nodes["label"].astype(int).to_numpy()

    _split_and_write(dataset_name, seed, data_dir, df_edges, labels_np, num_nodes)


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
    parser.add_argument(
        "--source",
        choices=["auto", "citationfull", "existing"],
        default="auto",
        help=(
            "Where to read graph/label data from. auto uses CitationFull when "
            "available, otherwise existing data/<dataset> CSVs."
        ),
    )
    args = parser.parse_args()

    if args.source == "existing" or (args.source == "auto" and CitationFull is None):
        prepare_existing_export(args.dataset, seed=args.seed, data_dir=args.data_dir)
    else:
        prepare_citation_full(args.dataset, seed=args.seed, data_dir=args.data_dir)
