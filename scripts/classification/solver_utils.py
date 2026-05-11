import math
import os
import sys
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Optional, Set

import networkx as nx
import pandas as pd
from pymincut.pygraph import PyGraph
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _solver_runner import (  # noqa: E402
    count_internal_directed_edges,
    invoke_solver,
)


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
):
    """Runs k-densest classification and returns perfectly aligned y_true and y_pred arrays."""
    train_mask = df_nodes["train"].values
    labels = df_nodes["label"].values

    forbidden_set: Set[int] = set(int(n) for n in (forbidden_nodes or ()))

    train_labels = labels[train_mask]
    global_majority = argmax_label(Counter(train_labels))

    # Build the NetworkX graph once for distance weighting, BFS fallbacks, and qualities.
    df_edges = pd.read_csv(edge_csv)
    G = nx.from_pandas_edgelist(
        df_edges, source="source", target="target", create_using=nx.Graph()
    )
    G_dir = nx.from_pandas_edgelist(
        df_edges, source="source", target="target", create_using=nx.DiGraph()
    )
    out_neighbors = {node: set(G_dir.successors(node)) for node in G_dir.nodes()}
    if max_in_edges == 0:
        fallback_graph = G_dir
    else:
        fallback_graph = G
    mincut_neighbors = {node: set(G.neighbors(node)) for node in G.nodes()}

    y_true = []
    y_pred = []
    evaluated_query_nodes = []
    pred_sizes = []
    oracle_query_counts = []
    quality_values = {
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
        futures = {
            executor.submit(
                run_solver,
                q,
                k,
                edge_csv,
                bin_path,
                tmp_dir,
                extra_args,
                max_in_edges,
            ): q
            for q in query_nodes
        }

        # as_completed yields out of order, but y_true and y_pred stay pairwise matched
        for future in tqdm(
            as_completed(futures),
            total=total_queries,
            desc=f"Evaluating {eval_label}",
            disable=not show_progress,
        ):
            q_node, neighborhood, oracle_queries = future.result()
            pred_sizes.append(len(neighborhood))
            oracle_query_counts.append(oracle_queries)

            if compute_qualities:
                qualities = compute_subgraph_quality(
                    neighborhood, out_neighbors, mincut_neighbors
                )
                for key in quality_values:
                    quality_values[key].append(qualities[key])

            # Filter neighborhood to strictly contain Train nodes (excluding query and forbidden nodes)
            train_neighbors = [
                n
                for n in neighborhood
                if train_mask[n] and n != q_node and n not in forbidden_set
            ]

            if not train_neighbors:
                # Concentric-BFS fallback: solver returned no training-labelled
                # neighbours, so vote with the nearest labelled ring instead.
                fallback_count += 1
                try:
                    paths = _bfs_with_forbidden(
                        fallback_graph,
                        q_node,
                        max_fallback_hops,
                        forbidden_set,
                    )
                    reachable_train = {
                        n: d for n, d in paths.items() if train_mask[n] and n != q_node
                    }

                    if reachable_train:
                        min_dist = min(reachable_train.values())
                        nearest_train_nodes = [
                            n for n, d in reachable_train.items() if d == min_dist
                        ]
                        pred_label = argmax_label(
                            Counter(labels[n] for n in nearest_train_nodes)
                        )
                    else:
                        pred_label = global_majority
                except Exception:
                    pred_label = global_majority

            else:
                if weighting == "distance":
                    class_scores = Counter()
                    for n in train_neighbors:
                        try:
                            d = nx.shortest_path_length(G, source=q_node, target=n)
                            w = 1.0 / d
                        except nx.NetworkXNoPath:
                            w = 0.0

                        class_scores[labels[n]] += w

                    pred_label = argmax_label(class_scores)

                else:  # Uniform weighting
                    pred_label = argmax_label(Counter(labels[n] for n in train_neighbors))

            y_true.append(labels[q_node])
            y_pred.append(pred_label)
            evaluated_query_nodes.append(q_node)

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
