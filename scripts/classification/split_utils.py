"""Split-meta helpers shared by prepare_data, evaluate_nodes, and the
classification aggregators. Builds hash-pinned manifests so downstream phases
can verify they are operating on the same split they were tuned against.
"""

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Set, Tuple


EXPECTED_SCHEMA_VERSION = "3.0"
EXPECTED_POOL_MIN_OUT_2EDGE_COMPONENT_SIZE = 5
EXPECTED_POOL_CRITERION = "pure_source_with_out_reachable_2edge_component_geq_5"
EXPECTED_SPLIT_STRATEGY = "label_stratified_50_50_out_2edge_component_eligible"


@dataclass(frozen=True)
class SplitMeta:
    schema_version: str
    dataset_name: str
    seed: int
    num_nodes: int
    num_edges: int
    splits: dict
    hard_subset: Optional[dict]
    edges_hash: str
    library_versions: dict
    created_at: str
    pool_criterion: Optional[str] = None
    pool_min_out_reachable_size: Optional[int] = None
    pool_min_out_2edge_component_size: Optional[int] = None
    split_strategy: Optional[str] = None
    query_pool: Optional[dict] = None


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
        hard_subset=payload.get("hard_subset"),
        edges_hash=payload["edges_hash"],
        library_versions=payload.get("library_versions", {}),
        created_at=payload["created_at"],
        pool_criterion=payload.get("pool_criterion"),
        pool_min_out_reachable_size=payload.get("pool_min_out_reachable_size"),
        pool_min_out_2edge_component_size=payload.get("pool_min_out_2edge_component_size"),
        split_strategy=payload.get("split_strategy"),
        query_pool=payload.get("query_pool"),
    )


def write_split_meta(
    dataset_name: str,
    seed: int,
    num_nodes: int,
    num_edges: int,
    train_ids: List[int],
    val_ids: List[int],
    test_ids: List[int],
    hard_subset_ids: Optional[List[int]],
    edges_hash: str,
    library_versions: dict,
    data_dir: str = "data",
    query_pool_ids: Optional[List[int]] = None,
) -> dict:
    payload = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "seed": seed,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "pool_criterion": EXPECTED_POOL_CRITERION,
        "pool_min_out_2edge_component_size": EXPECTED_POOL_MIN_OUT_2EDGE_COMPONENT_SIZE,
        "split_strategy": EXPECTED_SPLIT_STRATEGY,
        "splits": {
            "train": {"size": len(train_ids), "hash": sha256_node_set(train_ids)},
            "val": {"size": len(val_ids), "hash": sha256_node_set(val_ids)},
            "test": {"size": len(test_ids), "hash": sha256_node_set(test_ids)},
        },
        "query_pool": None
        if query_pool_ids is None
        else {
            "criterion": EXPECTED_POOL_CRITERION,
            "size": len(query_pool_ids),
            "hash": sha256_node_set(query_pool_ids),
        },
        "hard_subset": None
        if hard_subset_ids is None
        else {
            "criterion": "bfs_depth1_label_vote_wrong_outgoing",
            "split": "test",
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


def build_out_adjacency(df_edges) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = defaultdict(set)
    for s, t in zip(df_edges["source"].astype(int), df_edges["target"].astype(int)):
        if s == t:
            continue
        adj[s].add(t)
    return adj


def build_undirected_adjacency(df_edges) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = defaultdict(set)
    for s, t in zip(df_edges["source"].astype(int), df_edges["target"].astype(int)):
        if s == t:
            continue
        adj[s].add(t)
        adj[t].add(s)
    return adj


def out_reachable_nodes(start: int, out_adj: Dict[int, Set[int]]) -> Set[int]:
    seen = {int(start)}
    stack = [int(start)]
    while stack:
        node = stack.pop()
        for nb in out_adj.get(node, ()):
            if nb in seen:
                continue
            seen.add(nb)
            stack.append(nb)
    return seen


def out_reachable_2edge_component_size(
    q_node: int,
    out_adj: Dict[int, Set[int]],
    undirected_adj: Dict[int, Set[int]],
) -> int:
    """Size of q's bridge-free component in its outgoing-reachable support.

    The solver sweep uses ``max_in_edges=0``, so query feasibility should be
    judged in the undirected support induced by nodes reachable from q via
    outgoing edges. Removing bridges leaves the 2-edge-connected components.
    """
    q_node = int(q_node)
    reachable = out_reachable_nodes(q_node, out_adj)
    if len(reachable) <= 1:
        return len(reachable)

    timer = 0
    tin: Dict[int, int] = {}
    low: Dict[int, int] = {}
    bridges: Set[Tuple[int, int]] = set()

    def edge_key(u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u < v else (v, u)

    def dfs(u: int, parent: Optional[int]) -> None:
        nonlocal timer
        tin[u] = timer
        low[u] = timer
        timer += 1
        for v in undirected_adj.get(u, ()):
            if v not in reachable:
                continue
            if v == parent:
                continue
            if v in tin:
                low[u] = min(low[u], tin[v])
                continue
            dfs(v, u)
            low[u] = min(low[u], low[v])
            if low[v] > tin[u]:
                bridges.add(edge_key(u, v))

    dfs(q_node, None)

    component = {q_node}
    stack = [q_node]
    while stack:
        u = stack.pop()
        for v in undirected_adj.get(u, ()):
            if v not in reachable or v in component:
                continue
            if edge_key(u, v) in bridges:
                continue
            component.add(v)
            stack.append(v)
    return len(component)


def build_query_pool(df_edges) -> Tuple[List[int], dict]:
    """Return query ids satisfying the consolidated split-pool criterion."""
    cited_nodes = set(df_edges["target"].astype(int).unique())
    all_citing_nodes = set(df_edges["source"].astype(int).unique())
    pure_sources = sorted(all_citing_nodes - cited_nodes)

    out_adj = build_out_adjacency(df_edges)
    undirected_adj = build_undirected_adjacency(df_edges)
    component_sizes = {
        int(q): out_reachable_2edge_component_size(q, out_adj, undirected_adj)
        for q in pure_sources
    }
    query_pool = sorted(
        q
        for q, size in component_sizes.items()
        if size >= EXPECTED_POOL_MIN_OUT_2EDGE_COMPONENT_SIZE
    )
    stats = {
        "pure_source_count": len(pure_sources),
        "eligible_count": len(query_pool),
        "component_size_counts": dict(Counter(component_sizes.values())),
    }
    return query_pool, stats


def query_has_undirected_triangle(q_node: int, undirected_adj: Dict[int, Set[int]]) -> bool:
    """Return true when q participates in at least one triangle in the
    undirected support graph.

    For k=3, kappa=2 this is exactly the local feasibility condition for a
    3-node query-containing answer under the solver's undirected support
    interpretation of edge connectivity.
    """
    neighbours = sorted(undirected_adj.get(int(q_node), ()))
    for i, u in enumerate(neighbours):
        u_neighbours = undirected_adj.get(u, set())
        for v in neighbours[i + 1 :]:
            if v in u_neighbours:
                return True
    return False


def argmax_label_value(counter: Counter):
    """Deterministic argmax over label counts; ties broken by ascending label."""
    if not counter:
        return None
    best = None
    for label, count in counter.items():
        key = (-count, label)
        if best is None or key < best[0]:
            best = (key, label)
    return best[1]


def bfs_depth1_label_vote(q_node, out_adj, train_mask, labels, global_majority):
    """Majority label over outgoing 1-hop train neighbours of q, with
    deterministic argmax and a global-majority fallback when no train neighbour
    is reachable.
    """
    neighbours = out_adj.get(q_node, ())
    train_neighbours = [n for n in neighbours if n != q_node and train_mask[n]]
    if not train_neighbours:
        return global_majority
    return argmax_label_value(Counter(labels[n] for n in train_neighbours))


def compute_hard_subset(query_ids, out_adj, train_mask, labels) -> List[int]:
    global_majority = argmax_label_value(Counter(labels[train_mask]))
    return [
        int(q)
        for q in query_ids
        if bfs_depth1_label_vote(q, out_adj, train_mask, labels, global_majority) != labels[q]
    ]


def assert_split_meta_matches(
    dataset: str,
    df_nodes,
    df_edges,
    data_dir: str = "data",
) -> SplitMeta:
    meta = load_split_meta(dataset, data_dir)

    if meta.schema_version != EXPECTED_SCHEMA_VERSION:
        raise ValueError(
            f"{dataset}: split_meta.json schema_version {meta.schema_version!r} "
            f"does not match expected {EXPECTED_SCHEMA_VERSION!r}. Regenerate via prepare_data.py."
        )
    if meta.pool_criterion != EXPECTED_POOL_CRITERION:
        raise ValueError(
            f"{dataset}: split_meta.json pool_criterion {meta.pool_criterion!r} "
            f"does not match expected {EXPECTED_POOL_CRITERION!r}. Regenerate via prepare_data.py."
        )
    if (
        meta.pool_min_out_2edge_component_size
        != EXPECTED_POOL_MIN_OUT_2EDGE_COMPONENT_SIZE
    ):
        raise ValueError(
            f"{dataset}: split_meta.json pool_min_out_2edge_component_size "
            f"{meta.pool_min_out_2edge_component_size!r} does not match expected "
            f"{EXPECTED_POOL_MIN_OUT_2EDGE_COMPONENT_SIZE!r}. Regenerate via prepare_data.py."
        )
    if meta.split_strategy != EXPECTED_SPLIT_STRATEGY:
        raise ValueError(
            f"{dataset}: split_meta.json split_strategy {meta.split_strategy!r} "
            f"does not match expected {EXPECTED_SPLIT_STRATEGY!r}. Regenerate via prepare_data.py."
        )

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
