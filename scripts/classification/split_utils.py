"""Split-meta helpers shared by prepare_data, evaluate_nodes, and the
classification aggregators. Builds hash-pinned manifests so downstream phases
can verify they are operating on the same split they were tuned against.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class SplitMeta:
    schema_version: str
    dataset_name: str
    seed: int
    num_nodes: int
    num_edges: int
    splits: dict
    eligible: dict
    hard_subset: Optional[dict]
    edges_hash: str
    library_versions: dict
    created_at: str


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_node_set(node_ids: Iterable[int]) -> str:
    ids = sorted(int(n) for n in node_ids)
    payload = b"\n".join(str(i).encode("ascii") for i in ids)
    return _sha256_bytes(payload)


def sha256_edge_list(edges: Iterable[Tuple[int, int]]) -> str:
    canonical = sorted((int(u), int(v)) for u, v in edges)
    payload = b"\n".join(f"{u},{v}".encode("ascii") for u, v in canonical)
    return _sha256_bytes(payload)


def _split_meta_path(dataset: str, data_dir: str) -> str:
    return os.path.join(data_dir, dataset, "split_meta.json")


def load_split_meta(dataset: str, data_dir: str = "data") -> SplitMeta:
    path = _split_meta_path(dataset, data_dir)
    with open(path) as f:
        payload = json.load(f)
    return SplitMeta(
        schema_version=payload["schema_version"],
        dataset_name=payload["dataset_name"],
        seed=payload["seed"],
        num_nodes=payload["num_nodes"],
        num_edges=payload["num_edges"],
        splits=payload["splits"],
        eligible=payload["eligible"],
        hard_subset=payload.get("hard_subset"),
        edges_hash=payload["edges_hash"],
        library_versions=payload.get("library_versions", {}),
        created_at=payload["created_at"],
    )


def write_split_meta(
    dataset_name: str,
    seed: int,
    num_nodes: int,
    num_edges: int,
    train_ids: List[int],
    val_ids: List[int],
    test_ids: List[int],
    eligible_val_ids: List[int],
    eligible_test_ids: List[int],
    hard_subset_ids: Optional[List[int]],
    edges_hash: str,
    library_versions: dict,
    data_dir: str = "data",
) -> dict:
    payload = {
        "schema_version": "1.0",
        "dataset_name": dataset_name,
        "seed": seed,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "splits": {
            "train": {"size": len(train_ids), "hash": sha256_node_set(train_ids)},
            "val": {"size": len(val_ids), "hash": sha256_node_set(val_ids)},
            "test": {"size": len(test_ids), "hash": sha256_node_set(test_ids)},
        },
        "eligible": {
            "criterion": "undirected_1hop_neighbors_geq_2",
            "val": {
                "size": len(eligible_val_ids),
                "hash": sha256_node_set(eligible_val_ids),
                "n_filtered": len(val_ids) - len(eligible_val_ids),
            },
            "test": {
                "size": len(eligible_test_ids),
                "hash": sha256_node_set(eligible_test_ids),
                "n_filtered": len(test_ids) - len(eligible_test_ids),
            },
        },
        "hard_subset": None
        if hard_subset_ids is None
        else {
            "criterion": "bfs_depth1_label_vote_wrong",
            "split": "val",
            "size": len(hard_subset_ids),
            "hash": sha256_node_set(hard_subset_ids),
        },
        "edges_hash": edges_hash,
        "library_versions": library_versions,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = _split_meta_path(dataset_name, data_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return payload


def assert_split_meta_matches(
    dataset: str,
    df_nodes,
    df_edges,
    data_dir: str = "data",
) -> SplitMeta:
    meta = load_split_meta(dataset, data_dir)

    train_ids = df_nodes[df_nodes["train"]]["node_id"].astype(int).tolist()
    val_ids = df_nodes[df_nodes["val"]]["node_id"].astype(int).tolist()
    test_ids = df_nodes[df_nodes["test"]]["node_id"].astype(int).tolist()

    if sha256_node_set(train_ids) != meta.splits["train"]["hash"]:
        raise ValueError(f"{dataset}: train split hash mismatch with split_meta.json")
    if sha256_node_set(val_ids) != meta.splits["val"]["hash"]:
        raise ValueError(f"{dataset}: val split hash mismatch with split_meta.json")
    if sha256_node_set(test_ids) != meta.splits["test"]["hash"]:
        raise ValueError(f"{dataset}: test split hash mismatch with split_meta.json")

    edges_iter = zip(
        df_edges["source"].astype(int).tolist(),
        df_edges["target"].astype(int).tolist(),
    )
    if sha256_edge_list(edges_iter) != meta.edges_hash:
        raise ValueError(f"{dataset}: edges_hash mismatch with split_meta.json")

    return meta
