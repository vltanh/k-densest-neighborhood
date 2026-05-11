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


def _records_from_ndjson(path: str) -> List[dict]:
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
    """BFS that never visits forbidden nodes (neither destinations nor stepping stones)."""
    if forbidden is None:
        forbidden = set()
    if source in forbidden:
        return {}
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

    undir_internal_ncut = math.nan
    if n >= 2:
        if undir_internal_edges == 0:
            # Disconnected induced subgraph: empty cut, no internal volume on
            # one side. Conventional choice is 0 (perfectly cuttable).
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

    return {
        "dir_internal_avg_degree": dir_internal_avg_degree,
        "dir_internal_edge_density": dir_internal_edge_density,
        "undir_external_expansion": undir_external_expansion,
        "undir_external_conductance": undir_external_conductance,
        "undir_internal_ncut": undir_internal_ncut,
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
        qualities_local = None
        if compute_qualities:
            qualities_local = compute_subgraph_quality(
                neighborhood, out_neighbors, mincut_neighbors
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
            neighborhood = record.get("neighborhood") or []
            pred_sizes.append(len(neighborhood))
            oracle_query_counts.append(record["oracle_queries"])

            if compute_qualities:
                qualities_local = record.get("qualities")
                if qualities_local is None:
                    qualities_local = compute_subgraph_quality(
                        neighborhood, out_neighbors, mincut_neighbors
                    )
                    record["qualities"] = qualities_local
                for key in quality_values:
                    quality_values[key].append(qualities_local.get(key, math.nan))

            pred_label, fb = _classify_query(
                q_node,
                neighborhood,
                train_mask,
                labels,
                forbidden_set,
                weighting,
                G,
                fallback_graph,
                max_fallback_hops,
                global_majority,
            )
            if fb:
                fallback_count += 1
            record["predicted_label"] = (
                int(pred_label) if pred_label is not None else None
            )
            record["fallback_used"] = fb

            if records_file_path is not None and not from_cache:
                lean = {k_: v for k_, v in record.items() if k_ != "neighborhood"}
                with record_lock:
                    with open(records_file_path, "a") as f:
                        f.write(json.dumps(lean) + "\n")

            y_true.append(int(labels[q_node]))
            y_pred.append(pred_label)
            evaluated_query_nodes.append(q_node)
            records_out.append(record)

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
