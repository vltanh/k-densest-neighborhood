"""Shared C++ solver invocation + graph metric helpers used by both
classification and synthetic benchmark scripts. Single source of truth for the
sim-mode argv and the post-run CSV parse so the four ad-hoc wrappers stay
gated identically to backend/server.py:_variant_argv and solver/src/main.cpp.
"""

import math
import os
import re
import subprocess
import tempfile
import time
from typing import Optional, Sequence, Tuple

import pandas as pd


def base_sim_argv(bin_path: str, edge_csv: str, query: str, output_csv: str) -> list:
    return [
        bin_path,
        "--mode", "sim",
        "--input", edge_csv,
        "--query", str(query),
        "--output", output_csv,
    ]


_QUERIES_PATTERN = re.compile(r"API Queries Made\s*:\s*(\d+)")


def parse_oracle_queries(output: str) -> float:
    match = _QUERIES_PATTERN.search(output or "")
    return int(match.group(1)) if match else math.nan


def read_predicted_nodes(path: str, as_int: bool = False) -> list:
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if "node_id" not in df.columns:
        return []
    if as_int:
        return df["node_id"].astype(int).tolist()
    return df["node_id"].astype(str).tolist()


def invoke_solver(
    bin_path: str,
    edge_csv: str,
    query: str,
    output_csv: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
    capture_wall_time: bool = False,
    as_int_nodes: bool = False,
    check: bool = False,
) -> dict:
    """Spawns the C++ solver in sim mode, returns the parsed result.

    Result keys: returncode, stdout, stderr, pred_nodes, oracle_queries,
    wall_time (if capture_wall_time). When output_csv is None a NamedTemporaryFile
    is used and cleaned up before returning.
    """
    owns_tmp = output_csv is None
    if owns_tmp:
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        tmp.close()
        output_csv = tmp.name

    cmd = base_sim_argv(bin_path, edge_csv, query, output_csv)
    if extra_args:
        cmd += list(extra_args)

    started = time.perf_counter() if capture_wall_time else None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=check)
        rc = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.CalledProcessError as exc:
        rc = exc.returncode
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""

    wall = (time.perf_counter() - started) if started is not None else None

    pred = read_predicted_nodes(output_csv, as_int=as_int_nodes) if rc == 0 else []

    if owns_tmp and os.path.exists(output_csv):
        os.remove(output_csv)

    result = {
        "returncode": rc,
        "stdout": stdout,
        "stderr": stderr,
        "pred_nodes": pred,
        "oracle_queries": parse_oracle_queries(stdout),
    }
    if wall is not None:
        result["wall_time"] = wall
    return result


def count_internal_directed_edges(nodes, out_neighbors) -> int:
    node_set = set(nodes)
    return sum(
        1
        for u in node_set
        for v in out_neighbors.get(u, ())
        if v != u and v in node_set
    )


def count_internal_edges_from_edge_list(nodes, edges) -> int:
    node_set = set(nodes)
    return sum(1 for u, v in edges if u in node_set and v in node_set)


def directed_density(internal_edges: int, n: int) -> float:
    return internal_edges / (n * (n - 1)) if n > 1 else 0.0


def avg_degree_density(internal_edges: int, n: int) -> float:
    return internal_edges / n if n > 0 else 0.0


def induced_directed_metrics(nodes, edges) -> dict:
    node_set = set(nodes)
    if not node_set:
        return {"internal_edges": 0, "avg_degree_density": 0.0, "edge_density": 0.0}
    internal = count_internal_edges_from_edge_list(node_set, edges)
    n = len(node_set)
    return {
        "internal_edges": internal,
        "avg_degree_density": avg_degree_density(internal, n),
        "edge_density": directed_density(internal, n),
    }


def overlap_metrics(gt_nodes, pred_nodes) -> dict:
    gt = set(gt_nodes)
    pred = set(pred_nodes)
    intersection = gt & pred
    union = gt | pred
    tp = len(intersection)
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt) if gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    jaccard = tp / len(union) if union else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
    }


def nodes_for_split(df_nodes: pd.DataFrame, split: str, as_int: bool = True) -> list:
    nodes = df_nodes[df_nodes[split]]["node_id"].tolist()
    return [int(n) for n in nodes] if as_int else nodes


def build_bp_extra_args(
    kappa: Optional[int] = None,
    time_limit: Optional[float] = None,
    cg_batch_frac: Optional[float] = None,
    cg_min_batch: Optional[int] = None,
    cg_max_batch: Optional[int] = None,
    node_limit: Optional[int] = None,
    gap_tol: Optional[float] = None,
    dinkelbach_iter: Optional[int] = None,
) -> list:
    extra = ["--bp"]
    if kappa is not None:
        extra += ["--kappa", str(kappa)]
    if time_limit is not None:
        extra += ["--time-limit", str(time_limit)]
    if cg_batch_frac is not None:
        extra += ["--cg-batch-frac", str(cg_batch_frac)]
    if cg_min_batch is not None:
        extra += ["--cg-min-batch", str(cg_min_batch)]
    if cg_max_batch is not None:
        extra += ["--cg-max-batch", str(cg_max_batch)]
    if node_limit is not None:
        extra += ["--node-limit", str(node_limit)]
    if gap_tol is not None:
        extra += ["--gap-tol", str(gap_tol)]
    if dinkelbach_iter is not None:
        extra += ["--dinkelbach-iter", str(dinkelbach_iter)]
    return extra


def _add_scripts_root_to_path() -> str:
    """Helper for subdir scripts to import this module via sys.path."""
    return os.path.dirname(os.path.abspath(__file__))
