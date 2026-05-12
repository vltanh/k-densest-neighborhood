import hashlib
import json
import math
import os
import sys
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd
from pymincut.pygraph import PyGraph
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _solver_runner import (  # noqa: E402
    count_internal_directed_edges,
    invoke_solver,
)


@dataclass
class GraphContext:
    """Per-dataset graph caches shared across (method, params, seed) cells."""

    G: Any
    G_dir: Any
    out_neighbors: Dict[int, Set[int]]
    mincut_neighbors: Dict[int, Set[int]]
    fallback_graph: Any


def build_graph_context(edge_csv: str, max_in_edges: int = 0) -> GraphContext:
    df_edges = pd.read_csv(edge_csv)
    G = nx.from_pandas_edgelist(
        df_edges, source="source", target="target", create_using=nx.Graph()
    )
    G_dir = nx.from_pandas_edgelist(
        df_edges, source="source", target="target", create_using=nx.DiGraph()
    )
    out_neighbors = {node: set(G_dir.successors(node)) for node in G_dir.nodes()}
    mincut_neighbors = {node: set(G.neighbors(node)) for node in G.nodes()}
    fallback_graph = G_dir if max_in_edges == 0 else G
    return GraphContext(
        G=G,
        G_dir=G_dir,
        out_neighbors=out_neighbors,
        mincut_neighbors=mincut_neighbors,
        fallback_graph=fallback_graph,
    )


def params_hash(params: dict) -> str:
    payload = json.dumps(params or {}, sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_forbidden(forbidden_nodes: Optional[Iterable[int]]) -> str:
    ids = sorted(int(n) for n in (forbidden_nodes or ()))
    payload = b"\n".join(str(i).encode("ascii") for i in ids)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def effective_params(
    params: Optional[dict],
    *,
    weighting: str,
    max_fallback_hops: int,
    forbidden_nodes: Optional[Iterable[int]] = None,
    code_hash: Optional[str] = None,
) -> Tuple[dict, str]:
    """Bake the bits that change the classifier's output into the params dict
    before hashing, so cached records emitted under one configuration are not
    reused under another. Returns (effective_params, params_hash).

    Fields included in ``_eval``:
      - weighting (uniform / distance)
      - max_fallback_hops
      - forbidden_hash: sha256 of the sorted forbidden node id list
      - code_hash (optional): caller-stamped solver build id or git sha
    """
    base = dict(params or {})
    eval_block = {
        "weighting": weighting,
        "max_fallback_hops": int(max_fallback_hops),
        "forbidden_hash": _sha256_forbidden(forbidden_nodes),
    }
    if code_hash is not None:
        eval_block["code_hash"] = str(code_hash)
    base["_eval"] = eval_block
    return base, params_hash(base)


def method_extra_args(method: str, params: Optional[dict] = None, gurobi_seed: Optional[int] = None) -> List[str]:
    """Single source of truth for the solver argv per (method, params).

    BP cells optionally carry kappa, time_limit, dinkelbach_iter, node_limit,
    gap_tol, and cg_* values. Unset values stay at solver defaults (-1).
    """
    params = params or {}
    if method == "avgdeg":
        return ["--avgdeg"]
    if method == "bfs":
        depth = params.get("bfs_depth", 1)
        return ["--bfs", "--bfs-depth", str(depth)]
    if method == "bp":
        args: List[str] = ["--bp"]
        if "kappa" in params:
            args += ["--kappa", str(params["kappa"])]
        if params.get("time_limit") is not None:
            args += ["--time-limit", str(params["time_limit"])]
        if params.get("dinkelbach_iter") is not None:
            args += ["--dinkelbach-iter", str(params["dinkelbach_iter"])]
        if params.get("node_limit") is not None:
            args += ["--node-limit", str(params["node_limit"])]
        if params.get("gap_tol") is not None:
            args += ["--gap-tol", str(params["gap_tol"])]
        for cg_key, flag in (
            ("cg_batch_frac", "--cg-batch-frac"),
            ("cg_min_batch", "--cg-min-batch"),
            ("cg_max_batch", "--cg-max-batch"),
        ):
            if params.get(cg_key) is not None:
                args += [flag, str(params[cg_key])]
        if gurobi_seed is not None:
            args += ["--gurobi-seed", str(gurobi_seed)]
        return args
    raise ValueError(f"unknown method: {method}")


def _record_key(record: dict) -> Tuple:
    method = record.get("method")
    keys = [
        record.get("query_node"),
        method,
        record.get("params_hash"),
        record.get("dataset"),
        record.get("split_hash"),
    ]
    if method == "bp":
        keys.append(record.get("seed"))
    return tuple(keys)


def load_ndjson_records(path: str) -> List[dict]:
    """Read an NDJSON file of records; skip malformed lines silently."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# Backwards-compatible alias for any local caller still using the private name.
_records_from_ndjson = load_ndjson_records


def argmax_label(counter: Counter):
    """Deterministic argmax over label counts. Ties broken by ascending label."""
    if not counter:
        return None
    best = None
    for label, count in counter.items():
        key = (-count, label)
        if best is None or key < best[0]:
            best = (key, label)
    return best[1]


def _bfs_with_forbidden(
    graph,
    source,
    cutoff: int,
    forbidden: Optional[Set[int]],
):
    """BFS that never visits forbidden nodes as destinations or stepping stones.

    The source is always visited (distance 0) so its non-forbidden neighbours
    can be expanded; the caller filters the source out of the voting pool via
    ``n != q_node``.
    """
    if forbidden is None:
        forbidden = set()
    distances = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        depth = distances[node]
        if depth >= cutoff:
            continue
        try:
            neighbours = graph.neighbors(node)
        except nx.NetworkXError:
            continue
        for nb in neighbours:
            if nb in distances or nb in forbidden:
                continue
            distances[nb] = depth + 1
            queue.append(nb)
    return distances


def run_solver(
    q_node,
    k,
    edge_csv,
    bin_path,
    tmp_dir,
    extra_args=None,
    max_in_edges=0,
):
    """Executes the C++ solver for a single query node.

    Returns (q_node, neighborhood_int, oracle_queries). tmp_dir is retained for
    backwards-compatible call sites; the solver no longer writes a CSV output.
    """
    cmd_extra = list(extra_args) if extra_args else []
    cmd_extra += ["--max-in-edges", str(max_in_edges)]
    if k is not None:
        cmd_extra += ["--k", str(k)]

    result = invoke_solver(
        bin_path=bin_path,
        edge_csv=edge_csv,
        query=q_node,
        extra_args=cmd_extra,
        as_int_nodes=True,
    )
    if result["returncode"] != 0:
        return q_node, [], result["oracle_queries"]
    return q_node, result["pred_nodes"], result["oracle_queries"]


def _safe_mean(values):
    finite = [v for v in values if v is not None and not math.isnan(v)]
    return sum(finite) / len(finite) if finite else math.nan


def compute_mS_cS(neighbors, com):
    m_count = 0
    c_count = 0
    for node in com:
        for neighbor in neighbors.get(node, []):
            if neighbor in com:
                m_count += 1
            else:
                c_count += 1
    return m_count // 2, c_count


def compute_mincut(neighbors, com):
    cluster_edges = set()
    for node in com:
        for neighbor in neighbors.get(node, []):
            if neighbor in com:
                cluster_edges.add((node, neighbor))
    sub_graph = PyGraph(list(com), list(cluster_edges))
    return sub_graph.mincut("noi", "bqueue", False)


_LAMBDA2_DENSE_THRESHOLD = 64


def _algebraic_connectivity_lambda2(mincut_neighbors, node_set):
    """Second-smallest eigenvalue of the normalized Laplacian of the undirected
    induced subgraph. scipy.sparse.linalg.eigsh handles graphs of any practical
    size; dense numpy.linalg.eigvalsh is used as a fallback for very small
    subgraphs where ARPACK requires k < n - 1."""
    n = len(node_set)
    if n < 2:
        return 0.0
    try:
        import numpy as np
        from scipy import sparse
        from scipy.sparse.linalg import eigsh
    except ImportError:
        return math.nan

    nodes = sorted(node_set)
    idx = {v: i for i, v in enumerate(nodes)}
    rows: List[int] = []
    cols: List[int] = []
    for u in nodes:
        for v in mincut_neighbors.get(u, ()):
            if v in idx and v != u:
                rows.append(idx[u])
                cols.append(idx[v])
    if not rows:
        return 0.0
    data = np.ones(len(rows), dtype=float)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    A = (A + A.T) * 0.5
    deg = np.asarray(A.sum(axis=1)).ravel()
    if (deg <= 0).any():
        return 0.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    D_inv_sqrt = sparse.diags(d_inv_sqrt)
    L_norm = sparse.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    L_norm = (L_norm + L_norm.T) * 0.5

    if n <= _LAMBDA2_DENSE_THRESHOLD:
        try:
            vals = np.linalg.eigvalsh(L_norm.toarray())
            return float(vals[1])
        except Exception:
            return math.nan

    try:
        vals = eigsh(L_norm, k=2, which="SM", return_eigenvectors=False)
        vals = sorted(float(v) for v in vals)
        return vals[1]
    except Exception:
        try:
            vals = np.linalg.eigvalsh(L_norm.toarray())
            return float(vals[1])
        except Exception:
            return math.nan


def _size_bucket(n: int) -> str:
    if n <= 5:
        return "small"
    if n <= 20:
        return "medium"
    return "large"


def compute_per_class_breakdown(
    nodes: Iterable[int],
    out_neighbors: Dict[int, Set[int]],
    mincut_neighbors: Dict[int, Set[int]],
    labels,
    train_mask,
    query_node: int,
) -> dict:
    """Per-class diagnostics for a returned subgraph. Reports:

      within_class_internal_edges_ratio
          Of the undirected internal edges of S, the fraction whose two
          endpoints share a label.
      train_label_entropy
          Shannon entropy (base 2) of the label distribution over train
          neighbours of S, in nats-of-log2; 0 when one class dominates,
          log2(num_classes) when uniform.
      true_class_vote_share
          Fraction of train neighbours of S whose label equals labels[query_node].
      n_train_neighbours
          Number of train neighbours of S used to build the vote.
      size_bucket
          Discretised size class: small (|S| <= 5), medium (6..20), large (>20).
    """
    node_set = set(int(n) for n in nodes)
    n = len(node_set)
    if n == 0 or query_node is None:
        return {
            "within_class_internal_edges_ratio": math.nan,
            "train_label_entropy": math.nan,
            "true_class_vote_share": math.nan,
            "n_train_neighbours": 0,
            "size_bucket": _size_bucket(n),
        }

    same = 0
    total = 0
    seen_pairs: Set[Tuple[int, int]] = set()
    for u in node_set:
        for v in mincut_neighbors.get(u, ()):
            if v == u or v not in node_set:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in seen_pairs:
                continue
            seen_pairs.add((a, b))
            total += 1
            try:
                if labels[a] == labels[b]:
                    same += 1
            except IndexError:
                pass
    within_ratio = (same / total) if total else math.nan

    train_neighbours = [
        v for v in node_set if v != query_node and v < len(train_mask) and train_mask[v]
    ]
    train_labels = [int(labels[v]) for v in train_neighbours]
    if train_labels:
        counts = Counter(train_labels)
        total_t = sum(counts.values())
        entropy = 0.0
        for c in counts.values():
            p = c / total_t
            entropy -= p * math.log2(p)
        query_label = int(labels[query_node]) if query_node < len(labels) else None
        vote_share = (
            counts.get(query_label, 0) / total_t if query_label is not None else math.nan
        )
    else:
        entropy = math.nan
        vote_share = math.nan

    return {
        "within_class_internal_edges_ratio": within_ratio,
        "train_label_entropy": entropy,
        "true_class_vote_share": vote_share,
        "n_train_neighbours": len(train_neighbours),
        "size_bucket": _size_bucket(n),
    }


def compute_subgraph_quality(nodes, out_neighbors, mincut_neighbors):
    """Compute post-solve quality metrics for the retrieved subgraph."""
    node_set = set(nodes)
    n = len(node_set)
    if n == 0:
        return {
            "dir_internal_avg_degree": 0.0,
            "dir_internal_edge_density": 0.0,
            "undir_external_expansion": 0.0,
            "undir_external_conductance": 0.0,
            "undir_internal_ncut": math.nan,
            "mixing_param": math.nan,
            "algebraic_connectivity_lambda2": 0.0,
            "size": 0,
        }

    undir_internal_edges, undir_boundary_edges = compute_mS_cS(mincut_neighbors, node_set)
    undir_external_volume = 2 * undir_internal_edges + undir_boundary_edges
    full_undir_volume = sum(len(v) for v in mincut_neighbors.values())
    outside_undir_volume = full_undir_volume - undir_external_volume

    internal_edges = count_internal_directed_edges(node_set, out_neighbors)

    dir_internal_avg_degree = internal_edges / n
    dir_internal_edge_density = internal_edges / (n * (n - 1)) if n > 1 else 0.0
    undir_external_expansion = undir_boundary_edges / n
    conductance_denominator = min(undir_external_volume, outside_undir_volume)
    undir_external_conductance = (
        undir_boundary_edges / conductance_denominator
        if conductance_denominator > 0
        else 0.0
    )

    mixing_denominator = undir_boundary_edges + 2 * undir_internal_edges
    mixing_param = (
        undir_boundary_edges / mixing_denominator
        if mixing_denominator > 0
        else math.nan
    )

    undir_internal_ncut = math.nan
    if n >= 2:
        if undir_internal_edges == 0:
            undir_internal_ncut = 0.0
        else:
            try:
                part_a, part_b, cut_value = compute_mincut(mincut_neighbors, node_set)
                vol_a = sum(
                    1
                    for u in part_a
                    for v in mincut_neighbors.get(u, [])
                    if v in node_set
                )
                vol_b = sum(
                    1
                    for u in part_b
                    for v in mincut_neighbors.get(u, [])
                    if v in node_set
                )
                if vol_a > 0 and vol_b > 0:
                    undir_internal_ncut = (
                        cut_value * (vol_a + vol_b) / (vol_a * vol_b)
                    )
                else:
                    undir_internal_ncut = 0.0
            except Exception:
                undir_internal_ncut = math.nan

    lambda2 = _algebraic_connectivity_lambda2(mincut_neighbors, node_set)

    return {
        "dir_internal_avg_degree": dir_internal_avg_degree,
        "dir_internal_edge_density": dir_internal_edge_density,
        "undir_external_expansion": undir_external_expansion,
        "undir_external_conductance": undir_external_conductance,
        "undir_internal_ncut": undir_internal_ncut,
        "mixing_param": mixing_param,
        "algebraic_connectivity_lambda2": lambda2,
        "size": n,
    }


def run_one_query(
    q_node: int,
    method: str,
    k: Optional[int],
    edge_csv: str,
    bin_path: str,
    extra_args: Optional[Iterable[str]] = None,
    max_in_edges: int = 0,
    json_output_path: Optional[str] = None,
) -> dict:
    """Single-query solver invocation. Returns the parsed solver dict."""
    cmd_extra: List[str] = list(extra_args) if extra_args else []
    cmd_extra += ["--max-in-edges", str(max_in_edges)]
    if k is not None and "--k" not in cmd_extra:
        cmd_extra += ["--k", str(k)]
    return invoke_solver(
        bin_path=bin_path,
        edge_csv=edge_csv,
        query=q_node,
        extra_args=cmd_extra,
        as_int_nodes=True,
        capture_wall_time=True,
        json_output_path=json_output_path,
    )


def _classify_query(
    q_node: int,
    neighborhood: List[int],
    train_mask,
    labels,
    forbidden_set: Set[int],
    weighting: str,
    G,
    fallback_graph,
    max_fallback_hops: int,
    global_majority,
) -> Tuple[Any, bool]:
    train_neighbors = [
        n
        for n in neighborhood
        if train_mask[n] and n != q_node and n not in forbidden_set
    ]
    if not train_neighbors:
        try:
            paths = _bfs_with_forbidden(
                fallback_graph, q_node, max_fallback_hops, forbidden_set
            )
            reachable_train = {
                n: d for n, d in paths.items() if train_mask[n] and n != q_node
            }
            if reachable_train:
                min_dist = min(reachable_train.values())
                nearest = [n for n, d in reachable_train.items() if d == min_dist]
                pred_label = argmax_label(Counter(labels[n] for n in nearest))
            else:
                pred_label = global_majority
        except Exception:
            pred_label = global_majority
        return pred_label, True

    if weighting == "distance":
        class_scores: Counter = Counter()
        for n in train_neighbors:
            try:
                d = nx.shortest_path_length(G, source=q_node, target=n)
                w = 1.0 / d
            except nx.NetworkXNoPath:
                w = 0.0
            class_scores[labels[n]] += w
        return argmax_label(class_scores), False

    return argmax_label(Counter(labels[n] for n in train_neighbors)), False


def evaluate_nodes(
    query_nodes,
    k,
    edge_csv,
    df_nodes,
    bin_path,
    tmp_dir,
    max_workers=8,
    weighting="uniform",
    max_fallback_hops=10,
    extra_args=None,
    collect_stats=False,
    show_progress=True,
    compute_qualities=False,
    max_in_edges=0,
    return_query_nodes=False,
    forbidden_nodes: Optional[Iterable[int]] = None,
    graph_context: Optional[GraphContext] = None,
    records_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    seed: Optional[int] = None,
    method: Optional[str] = None,
    params: Optional[dict] = None,
    split_hash: Optional[str] = None,
    keep_solver_dumps: bool = False,
    query_split: Optional[str] = None,
):
    """Runs k-densest classification and returns aligned y_true / y_pred arrays.

    When records_path is set, per-query lean records are appended to
    records.ndjson under that directory; heavy solver dumps are written under
    solver_dumps/ when keep_solver_dumps is True. Existing matching records on
    disk short-circuit re-solving for deterministic methods (and for BP when
    the seed matches).
    """
    train_mask = df_nodes["train"].values
    labels = df_nodes["label"].values

    forbidden_set: Set[int] = set(int(n) for n in (forbidden_nodes or ()))
    train_labels = labels[train_mask]
    global_majority = argmax_label(Counter(train_labels))

    ctx = graph_context or build_graph_context(edge_csv, max_in_edges)
    G = ctx.G
    fallback_graph = ctx.fallback_graph
    out_neighbors = ctx.out_neighbors
    mincut_neighbors = ctx.mincut_neighbors

    p_hash = params_hash(params or {})
    method_name = method or ("bp" if k is not None else "unknown")

    cached_records: Dict[Tuple, dict] = {}
    records_file_path: Optional[str] = None
    solver_dumps_dir: Optional[str] = None
    if records_path is not None:
        os.makedirs(records_path, exist_ok=True)
        records_file_path = os.path.join(records_path, "records.ndjson")
        if keep_solver_dumps:
            solver_dumps_dir = os.path.join(records_path, "solver_dumps")
            os.makedirs(solver_dumps_dir, exist_ok=True)
        for rec in _records_from_ndjson(records_file_path):
            cached_records[_record_key(rec)] = rec

    record_lock = Lock()

    def _make_lookup_key(q_node: int) -> Tuple:
        method_for_seed = method_name
        seed_part = seed if method_for_seed == "bp" else None
        return (
            int(q_node),
            method_name,
            p_hash,
            dataset_name,
            split_hash,
            *([seed_part] if method_for_seed == "bp" else []),
        )

    def _resolve(q_node: int):
        key = _make_lookup_key(q_node)
        cached = cached_records.get(key)
        if cached is not None:
            return cached, True
        solver_dump_path: Optional[str] = None
        if solver_dumps_dir is not None:
            seed_tag = "noseed" if (seed is None or method_name != "bp") else f"s{seed}"
            dump_name = f"{q_node}_{method_name}_{p_hash[-12:]}_{seed_tag}.json"
            solver_dump_path = os.path.join(solver_dumps_dir, dump_name)

        result = run_one_query(
            q_node=q_node,
            method=method_name,
            k=k,
            edge_csv=edge_csv,
            bin_path=bin_path,
            extra_args=extra_args,
            max_in_edges=max_in_edges,
            json_output_path=solver_dump_path,
        )
        neighborhood = result["pred_nodes"] if result["returncode"] == 0 else []
        oracle_queries = result["oracle_queries"]
        wall_time = result.get("wall_time")
        solver_payload = result.get("solver_json") or {}
        qualities_local = None
        if compute_qualities:
            qualities_local = compute_subgraph_quality(
                neighborhood, out_neighbors, mincut_neighbors
            )
            qualities_local.update(
                compute_per_class_breakdown(
                    neighborhood,
                    out_neighbors,
                    mincut_neighbors,
                    labels,
                    train_mask,
                    int(q_node),
                )
            )
        record = {
            "dataset": dataset_name,
            "method": method_name,
            "params": params or {},
            "params_hash": p_hash,
            "seed": seed if method_name == "bp" else None,
            "split_hash": split_hash,
            "query_node": int(q_node),
            "query_split": query_split,
            "query_label": int(labels[q_node]) if q_node < len(labels) else None,
            "size": len(neighborhood),
            "neighborhood": [int(n) for n in neighborhood],
            "oracle_queries": (
                None if isinstance(oracle_queries, float) and math.isnan(oracle_queries)
                else int(oracle_queries)
            ),
            "wall_time_s": wall_time,
            "qualities": qualities_local,
            "solver_dump_path": (
                os.path.relpath(solver_dump_path, records_path)
                if solver_dump_path is not None
                else None
            ),
            "returncode": result["returncode"],
            "solver_build_id": solver_payload.get("solver_build_id"),
            "kappa_verified": result.get("kappa_verified"),
            "kappa_verify_failed": result.get("kappa_verify_failed"),
            "hard_cap_hit": result.get("hard_cap_hit"),
            "soft_time_limit_s": (solver_payload.get("config") or {}).get("time_limit"),
            "hard_time_limit_s": (solver_payload.get("config") or {}).get("hard_time_limit"),
            "solver_wall_time_s": solver_payload.get("wall_time_s"),
            "stats": result.get("stats"),
        }
        return record, False

    y_true: List = []
    y_pred: List = []
    evaluated_query_nodes: List = []
    records_out: List[dict] = []
    pred_sizes: List[int] = []
    oracle_query_counts: List = []
    quality_values: Dict[str, List] = {
        "dir_internal_avg_degree": [],
        "dir_internal_edge_density": [],
        "undir_external_expansion": [],
        "undir_external_conductance": [],
        "undir_internal_ncut": [],
    }
    fallback_count = 0
    total_queries = len(query_nodes)
    eval_label = f"k={k}" if k is not None else "no-k"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_resolve, q): q for q in query_nodes}
        for future in tqdm(
            as_completed(futures),
            total=total_queries,
            desc=f"Evaluating {eval_label}",
            disable=not show_progress,
        ):
            record, from_cache = future.result()
            q_node = record["query_node"]
            neighborhood = record.get("neighborhood")
            pred_sizes.append(record.get("size") or (len(neighborhood) if neighborhood else 0))
            oracle_query_counts.append(record.get("oracle_queries"))

            if compute_qualities:
                qualities_local = record.get("qualities")
                if qualities_local is None and neighborhood is not None:
                    qualities_local = compute_subgraph_quality(
                        neighborhood, out_neighbors, mincut_neighbors
                    )
                    qualities_local.update(
                        compute_per_class_breakdown(
                            neighborhood,
                            out_neighbors,
                            mincut_neighbors,
                            labels,
                            train_mask,
                            int(q_node),
                        )
                    )
                    record["qualities"] = qualities_local
                if qualities_local is None:
                    qualities_local = {}
                for key in quality_values:
                    quality_values[key].append(qualities_local.get(key, math.nan))

            if from_cache and record.get("predicted_label") is not None:
                pred_label = record["predicted_label"]
                fb = bool(record.get("fallback_used"))
            else:
                pred_label, fb = _classify_query(
                    q_node,
                    neighborhood or [],
                    train_mask,
                    labels,
                    forbidden_set,
                    weighting,
                    G,
                    fallback_graph,
                    max_fallback_hops,
                    global_majority,
                )
                record["predicted_label"] = (
                    int(pred_label) if pred_label is not None else None
                )
                record["fallback_used"] = fb
            if fb:
                fallback_count += 1

            if records_file_path is not None and not from_cache:
                lean = {k_: v for k_, v in record.items() if k_ != "neighborhood"}
                with record_lock:
                    with open(records_file_path, "a") as f:
                        f.write(json.dumps(lean) + "\n")

            y_true.append(int(labels[q_node]))
            y_pred.append(pred_label)
            evaluated_query_nodes.append(q_node)
            records_out.append(record)

    # Sort results by query_node so callers (e.g. paired bootstrap) get the
    # same row order regardless of as_completed arrival ordering, making the
    # i-th sample correspond to the same query across methods and seeds.
    if evaluated_query_nodes:
        order = sorted(range(len(evaluated_query_nodes)), key=lambda i: evaluated_query_nodes[i])
        y_true = [y_true[i] for i in order]
        y_pred = [y_pred[i] for i in order]
        evaluated_query_nodes = [evaluated_query_nodes[i] for i in order]
        records_out = [records_out[i] for i in order]

    fallback_rate = (fallback_count / total_queries) * 100 if total_queries > 0 else 0
    print(
        f"\n[Diagnostic] {eval_label} | Fallback Triggered: {fallback_count}/{total_queries} times ({fallback_rate:.1f}%)"
    )

    if collect_stats:
        stats = {
            "fallback_count": fallback_count,
            "fallback_rate": fallback_rate,
            "avg_pred_size": (sum(pred_sizes) / len(pred_sizes)) if pred_sizes else 0.0,
            "avg_oracle_queries": _safe_mean(oracle_query_counts),
            "query_count": total_queries,
            "records": records_out,
        }
        if compute_qualities:
            stats.update(
                {
                    "avg_dir_internal_avg_degree": _safe_mean(
                        quality_values["dir_internal_avg_degree"]
                    ),
                    "avg_dir_internal_edge_density": _safe_mean(
                        quality_values["dir_internal_edge_density"]
                    ),
                    "avg_undir_external_expansion": _safe_mean(
                        quality_values["undir_external_expansion"]
                    ),
                    "avg_undir_external_conductance": _safe_mean(
                        quality_values["undir_external_conductance"]
                    ),
                    "avg_undir_internal_ncut": _safe_mean(
                        quality_values["undir_internal_ncut"]
                    ),
                }
            )
        if return_query_nodes:
            return evaluated_query_nodes, y_true, y_pred, stats
        return y_true, y_pred, stats

    if return_query_nodes:
        return evaluated_query_nodes, y_true, y_pred
    return y_true, y_pred
