"""Brute-force solver-optimality experiment plus the original tiny-graph
verification utility.

Subcommands:

    bf_generate    : sample G(n, p) graphs and write edge.csv + meta.json.
    bf_optima      : enumerate every subset containing the query node and
                     record the avg-degree and edge-density optima.
    bf_solver_runs : invoke the C++ solver across (method, k) and compare to
                     the brute-force optima.
    verify         : retains the original tiny-graph demo (kept for parity).

A reusable enumerator brute_force_optima is exported at module top-level for
direct use by other scripts and the test harness.
"""

import argparse
import hashlib
import itertools
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _bf_code_hash() -> str:
    """sha256 of this file's source. Stamped on every optima row so a CSV from
    a stale checkout fails loudly downstream rather than silently mismatching
    solver_runs records emitted from a different version."""
    try:
        with open(__file__, "rb") as f:
            return "sha256:" + hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return "unknown"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _solver_runner import (  # noqa: E402
    count_internal_edges_from_edge_list,
    invoke_solver,
)


DEMO_EDGES = [
    ("0", "1"),
    ("1", "2"),
    ("2", "0"),
    ("1", "3"),
    ("3", "4"),
    ("4", "1"),
]


# ----------------------------------------------------------------------------
# Tiny-graph demo (retained)
# ----------------------------------------------------------------------------


def load_edges(edge_csv):
    df = pd.read_csv(edge_csv)
    if len(df.columns) < 2:
        raise ValueError(f"{edge_csv} must have at least two columns")
    src_col, dst_col = df.columns[:2]
    edges = []
    nodes = set()
    for src, dst in zip(df[src_col].astype(str), df[dst_col].astype(str)):
        if src == dst:
            continue
        edges.append((src, dst))
        nodes.add(src)
        nodes.add(dst)
    return sorted(nodes), edges


def write_demo_csv(path):
    df = pd.DataFrame(DEMO_EDGES, columns=["source", "target"])
    df.to_csv(path, index=False)


def internal_edge_count(edges, subset):
    return count_internal_edges_from_edge_list(subset, edges)


def brute_force_avgdeg(nodes, edges, query):
    best_subset = None
    best_score = float("-inf")
    for r in range(1, len(nodes) + 1):
        for subset in itertools.combinations(nodes, r):
            if query not in subset:
                continue
            internal = internal_edge_count(edges, subset)
            score = internal / len(subset)
            if score > best_score:
                best_score = score
                best_subset = tuple(sorted(subset))
    return best_subset, best_score


def brute_force_bp(nodes, edges, query, k):
    best_subset = None
    best_score = float("-inf")
    for r in range(max(1, k), len(nodes) + 1):
        for subset in itertools.combinations(nodes, r):
            if query not in subset:
                continue
            internal = internal_edge_count(edges, subset)
            score = internal / (len(subset) * (len(subset) - 1)) if len(subset) > 1 else 0.0
            if score > best_score:
                best_score = score
                best_subset = tuple(sorted(subset))
    return best_subset, best_score


def run_solver(bin_path, edge_csv, query, method, args):
    extra = []
    if method == "avgdeg":
        extra.append("--avgdeg")
    if method == "bp":
        extra += ["--bp", "--k", str(args.k), "--kappa", str(args.kappa)]
    if args.time_limit is not None:
        extra += ["--time-limit", str(args.time_limit)]

    result = invoke_solver(bin_path, edge_csv, query, extra_args=extra)
    pred_nodes = tuple(sorted(result["pred_nodes"]))
    return result["returncode"], result["stdout"], result["stderr"], pred_nodes


def score_subset(edges, subset):
    subset = set(subset)
    internal = internal_edge_count(edges, subset)
    n = len(subset)
    avgdeg = internal / n if n else 0.0
    bp_density = internal / (n * (n - 1)) if n > 1 else 0.0
    return avgdeg, bp_density, internal


def _run_demo(args):
    if args.edge_csv:
        nodes, edges = load_edges(args.edge_csv)
        edge_csv = args.edge_csv
        tmp_handle = None
    else:
        tmp_handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        tmp_handle.close()
        write_demo_csv(tmp_handle.name)
        nodes, edges = load_edges(tmp_handle.name)
        edge_csv = tmp_handle.name

    if args.query not in nodes:
        raise ValueError(f"Query node {args.query} is not present in the graph.")

    print(f"Graph nodes: {len(nodes)}")
    print(f"Graph directed edges: {len(edges)}")
    print(f"Query: {args.query}")

    if args.mode in ("avgdeg", "both"):
        exact_subset, exact_score = brute_force_avgdeg(nodes, edges, args.query)
        rc, stdout, stderr, solver_subset = run_solver(
            args.bin_path, edge_csv, args.query, "avgdeg", args
        )
        solver_avgdeg, _, solver_internal = score_subset(edges, solver_subset)
        print("\n[avgdeg]")
        print(f"exact subset:  {exact_subset}")
        print(f"exact score:   {exact_score:.6f}")
        print(f"solver subset: {solver_subset}")
        print(f"solver score:  {solver_avgdeg:.6f}")
        print(f"solver internal edges: {solver_internal}")
        print(f"solver exit:   {rc}")

    if args.mode in ("bp", "both"):
        exact_subset, exact_score = brute_force_bp(nodes, edges, args.query, args.k)
        rc, stdout, stderr, solver_subset = run_solver(
            args.bin_path, edge_csv, args.query, "bp", args
        )
        solver_avgdeg, solver_bp_density, solver_internal = score_subset(
            edges, solver_subset
        )
        print("\n[bp]")
        print(f"k:             {args.k}")
        print(f"exact subset:  {exact_subset}")
        print(f"exact density: {exact_score:.6f}")
        print(f"solver subset: {solver_subset}")
        print(f"solver density:{solver_bp_density:.6f}")
        print(f"solver internal edges: {solver_internal}")
        print(f"solver exit:   {rc}")

    if tmp_handle is not None and os.path.exists(tmp_handle.name):
        os.remove(tmp_handle.name)


# ----------------------------------------------------------------------------
# Phase BF: brute-force experiment over G(n, p)
# ----------------------------------------------------------------------------


def _bitmask_adjacency(n: int, edges_dir: Iterable[Tuple[int, int]]):
    """Build symmetric directed adjacency as bitmasks. Returns adj_out (list of
    ints) and adj_und (list of ints, same as adj_out when edges are symmetric).
    """
    adj_out = [0] * n
    for u, v in edges_dir:
        if u == v:
            continue
        adj_out[u] |= 1 << v
    return adj_out


def _connected_in_subset(adj_und: List[int], s: int, start: int) -> bool:
    visited = 1 << start
    frontier = visited
    while frontier:
        new_frontier = 0
        ff = frontier
        while ff:
            lsb = ff & -ff
            v = lsb.bit_length() - 1
            new_frontier |= adj_und[v] & s & ~visited
            ff &= ff - 1
        visited |= new_frontier
        frontier = new_frontier
    return visited == s


def _edge_connectivity_in_subset(adj_und: List[int], s_mask: int, q: int) -> int:
    """Edge connectivity of the undirected subgraph induced by s_mask, anchored
    at q. Uses unit-capacity BFS max-flow from q to every other vertex in S;
    returns the minimum, which equals lambda(S) by Menger's theorem.
    """
    nodes: List[int] = []
    tmp = s_mask
    while tmp:
        lsb = tmp & -tmp
        nodes.append(lsb.bit_length() - 1)
        tmp &= tmp - 1
    nn = len(nodes)
    if nn < 2:
        return 0
    idx = {v: i for i, v in enumerate(nodes)}
    adj_compact: List[int] = [0] * nn
    for i, u in enumerate(nodes):
        a = adj_und[u] & s_mask
        while a:
            lsb = a & -a
            v = lsb.bit_length() - 1
            adj_compact[i] |= 1 << idx[v]
            a &= a - 1
    s_local = idx[q]
    min_flow = nn
    for t_local in range(nn):
        if t_local == s_local:
            continue
        cap_to = list(adj_compact)
        flow = 0
        while True:
            parent_arr = [-1] * nn
            parent_arr[s_local] = s_local
            queue: List[int] = [s_local]
            head = 0
            while head < len(queue) and parent_arr[t_local] == -1:
                u = queue[head]
                head += 1
                outs = cap_to[u]
                while outs:
                    lsb = outs & -outs
                    v = lsb.bit_length() - 1
                    outs &= outs - 1
                    if parent_arr[v] == -1:
                        parent_arr[v] = u
                        queue.append(v)
            if parent_arr[t_local] == -1:
                break
            v = t_local
            while v != s_local:
                u = parent_arr[v]
                cap_to[u] &= ~(1 << v)
                cap_to[v] |= 1 << u
                v = u
            flow += 1
        if flow < min_flow:
            min_flow = flow
        if min_flow == 0:
            break
    return min_flow


def brute_force_optima(
    adj_out: List[int],
    n: int,
    q: int,
    k_set: Iterable[int],
    kappa_set: Iterable[int] = (0, 1, 2, 3, 4),
):
    """Enumerate every subset that contains q and is at least 2 in size.
    Returns argmax records for the avg-degree and edge-density objectives.

    For avg-degree, both an unconstrained variant and a connected-only
    variant (undirected projection) are returned. For BP, kappa=0 applies no
    connectivity filter; kappa>=1 requires rooted edge-connectivity at least
    kappa in the undirected support.
    """
    bit_q = 1 << q
    k_list = sorted(set(int(k) for k in k_set))
    kappa_list = sorted(set(int(kp) for kp in kappa_set))
    if 0 not in kappa_list:
        kappa_list = [0] + kappa_list
    # Symmetric directed edges -> adj_und = adj_out.
    adj_und = adj_out

    best_avg = (-1.0, -1, -1, 0, None)  # (score, size, m, mask, actual_kappa)
    best_avg_conn = (-1.0, -1, -1, 0, None)
    # best_bp_kappa[k][kappa] = (score, size, m, mask, actual_kappa).
    # kappa=0 means no edge-connectivity threshold; kappa>=1 means
    # lambda_q(S) >= kappa in the undirected support.
    best_bp_kappa = {k: {kp: (-1.0, -1, -1, 0, None) for kp in kappa_list} for k in k_list}

    full = 1 << n
    for s in range(full):
        if not (s & bit_q):
            continue
        sz = s.bit_count()
        if sz < 2:
            continue
        m = 0
        ss = s
        while ss:
            lsb = ss & -ss
            v = lsb.bit_length() - 1
            m += (adj_out[v] & s).bit_count()
            ss &= ss - 1
        avg = m / sz
        if avg > best_avg[0]:
            best_avg = (avg, sz, m, s, None)

        could_improve_avg_conn = avg > best_avg_conn[0]
        could_improve_bp_any_kappa = False
        for k in k_list:
            if sz >= k:
                ed = m / (sz * (sz - 1))
                for kp in kappa_list:
                    if ed > best_bp_kappa[k][kp][0]:
                        could_improve_bp_any_kappa = True
                        break
                if could_improve_bp_any_kappa:
                    break
        if not (could_improve_avg_conn or could_improve_bp_any_kappa):
            continue
        kappa_S: Optional[int] = None
        for k in k_list:
            if sz >= k:
                ed = m / (sz * (sz - 1))
                if ed > best_bp_kappa[k][0][0]:
                    best_bp_kappa[k][0] = (ed, sz, m, s, None)
        connected = _connected_in_subset(adj_und, s, q)
        if not connected:
            continue
        if could_improve_avg_conn:
            best_avg_conn = (avg, sz, m, s, None)
        need_kappa_check = False
        for k in k_list:
            if sz < k:
                continue
            ed = m / (sz * (sz - 1))
            for kp in kappa_list:
                if kp == 0:
                    continue
                if ed > best_bp_kappa[k][kp][0]:
                    need_kappa_check = True
                    break
            if need_kappa_check:
                break
        if not need_kappa_check:
            continue
        if kappa_S is None:
            kappa_S = _edge_connectivity_in_subset(adj_und, s, q)
        for k in k_list:
            if sz < k:
                continue
            ed = m / (sz * (sz - 1))
            for kp in kappa_list:
                if kp == 0:
                    continue
                if kappa_S >= kp and ed > best_bp_kappa[k][kp][0]:
                    best_bp_kappa[k][kp] = (ed, sz, m, s, kappa_S)

    def _mask_to_nodes(mask: int) -> List[int]:
        nodes = []
        while mask:
            lsb = mask & -mask
            nodes.append(lsb.bit_length() - 1)
            mask &= mask - 1
        return nodes

    out = {
        "avgdeg": {
            "score": best_avg[0],
            "size": best_avg[1],
            "internal_edges": best_avg[2],
            "nodes": _mask_to_nodes(best_avg[3]),
            "actual_kappa": best_avg[4],
        },
        "avgdeg_connected": {
            "score": best_avg_conn[0],
            "size": best_avg_conn[1],
            "internal_edges": best_avg_conn[2],
            "nodes": _mask_to_nodes(best_avg_conn[3]),
            "actual_kappa": best_avg_conn[4],
        },
        "bp_kappa": {},
    }
    for k in k_list:
        out["bp_kappa"][k] = {}
        for kp in kappa_list:
            out["bp_kappa"][k][kp] = {
                "score": best_bp_kappa[k][kp][0],
                "size": best_bp_kappa[k][kp][1],
                "internal_edges": best_bp_kappa[k][kp][2],
                "nodes": _mask_to_nodes(best_bp_kappa[k][kp][3]),
                "actual_kappa": best_bp_kappa[k][kp][4],
            }
    return out


def _graph_dir(root: Path, n: int, p: float, seed: int) -> Path:
    p_tag = f"p{int(round(p * 100)):03d}"
    return root / "synthetic" / "bf" / f"n{n}" / p_tag / f"s{seed}"


def _generate_one(n: int, p: float, seed: int, root: Path):
    # Symmetric directed graph: draw each unordered pair {u, v} with probability
    # p, then write both directed arcs (u, v) and (v, u). The directed solver
    # input therefore has reciprocal arcs everywhere; p is the undirected-pair
    # rate, not the per-arc rate.
    rng = np.random.default_rng(seed)
    edges_und = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges_und.append((u, v))
    edges_dir = []
    for u, v in edges_und:
        edges_dir.append((u, v))
        edges_dir.append((v, u))

    degree = [0] * n
    for u, v in edges_dir:
        degree[u] += 1
    if max(degree) == 0:
        query = 0
    else:
        query = max(range(n), key=lambda i: (degree[i], i))

    out_dir = _graph_dir(root, n, p, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(edges_dir, columns=["source", "target"]).to_csv(
        out_dir / "edge.csv", index=False
    )
    with open(out_dir / "meta.json", "w") as f:
        json.dump(
            {
                "n": n,
                "p": p,
                "seed": seed,
                "query_node": int(query),
                "edge_count_directed": len(edges_dir),
                "edge_count_undirected": len(edges_und),
            },
            f,
            indent=2,
            sort_keys=True,
        )
    return out_dir


def _compute_optima_one(meta_path: Path, k_set: List[int], kappa_set: List[int]):
    with open(meta_path) as f:
        meta = json.load(f)
    n = meta["n"]
    q = meta["query_node"]
    df_edges = pd.read_csv(meta_path.with_name("edge.csv"))
    edges_dir = list(zip(df_edges["source"].astype(int), df_edges["target"].astype(int)))
    adj_out = _bitmask_adjacency(n, edges_dir)
    started = time.perf_counter()
    optima = brute_force_optima(adj_out, n, q, k_set, kappa_set=kappa_set)
    elapsed = time.perf_counter() - started
    rows = []
    code_hash = _bf_code_hash()
    common = {
        "n": n,
        "p": meta["p"],
        "seed": meta["seed"],
        "query_node": q,
        "code_hash": code_hash,
    }
    rows.append(
        {
            **common,
            "optimum_kind": "avgdeg",
            "k": None,
            "kappa": None,
            "opt_value": optima["avgdeg"]["score"],
            "opt_size": optima["avgdeg"]["size"],
            "actual_kappa": optima["avgdeg"]["actual_kappa"],
            "opt_nodes_json": json.dumps(optima["avgdeg"]["nodes"]),
            "enumerate_time_s": elapsed,
        }
    )
    rows.append(
        {
            **common,
            "optimum_kind": "avgdeg_connected",
            "k": None,
            "kappa": None,
            "opt_value": optima["avgdeg_connected"]["score"],
            "opt_size": optima["avgdeg_connected"]["size"],
            "actual_kappa": optima["avgdeg_connected"]["actual_kappa"],
            "opt_nodes_json": json.dumps(optima["avgdeg_connected"]["nodes"]),
            "enumerate_time_s": elapsed,
        }
    )
    for k in k_set:
        kappa_cells = optima["bp_kappa"].get(k, {})
        for kp in sorted(kappa_cells.keys()):
            rows.append(
                {
                    **common,
                    "optimum_kind": "edge_density_kappa",
                    "k": k,
                    "kappa": kp,
                    "opt_value": kappa_cells[kp]["score"],
                    "opt_size": kappa_cells[kp]["size"],
                    "actual_kappa": kappa_cells[kp]["actual_kappa"],
                    "opt_nodes_json": json.dumps(kappa_cells[kp]["nodes"]),
                    "enumerate_time_s": elapsed,
                }
            )
    return rows


def _run_solver_cell(meta_path: Path, method: str, k: Optional[int], kappa: Optional[int], args):
    with open(meta_path) as f:
        meta = json.load(f)
    edge_csv = str(meta_path.with_name("edge.csv"))
    query = str(meta["query_node"])
    extra: List[str] = []
    if method == "avgdeg":
        extra += ["--avgdeg"]
    elif method == "bp":
        extra += ["--bp", "--k", str(k), "--kappa", str(kappa), "--gurobi-seed", str(args.gurobi_seed)]
    if args.time_limit is not None:
        extra += ["--time-limit", str(args.time_limit)]
    extra += ["--compute-qualities"]

    if method == "bp":
        method_dir = f"bp_k{k}_kappa{kappa}"
    else:
        method_dir = method
    dump_path = meta_path.parent / "solver_dumps" / f"{method_dir}.json"
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = invoke_solver(
        args.bin_path,
        edge_csv,
        query,
        extra_args=extra,
        as_int_nodes=True,
        json_output_path=str(dump_path),
    )
    wall = time.perf_counter() - started
    return meta, str(meta_path), method, k, kappa, result, wall


def _solver_runs_one(args_tuple):
    return _run_solver_cell(*args_tuple)


def _extract_opt_row(row: dict):
    actual = row.get("actual_kappa")
    return (
        row["opt_value"],
        row["opt_size"],
        json.loads(row["opt_nodes_json"]),
        None if pd.isna(actual) else int(actual),
    )


def _opt_value_from_optima(opt_rows: List[dict], method: str, k: Optional[int], kappa: Optional[int]):
    """Return ((primary_value, primary_size, primary_nodes, primary_actual_kappa),
    (secondary_value, secondary_size, secondary_nodes, secondary_actual_kappa) | None).

    Primary follows the solver's actual feasibility region:
      - avgdeg: unconstrained avg-degree optimum (|S| >= 2, contains q, possibly disconnected).
      - bp + any kappa: edge_density_kappa[k][kappa]. kappa = 0 means contains
        q with |S| >= k and no connectivity filter.
    Secondary is the connected-only counterpart kept for cross-check on avgdeg
    only; for BP the primary and the only meaningful baseline coincide.
    """
    target_kappa = 0 if (kappa is None or pd.isna(kappa)) else int(kappa)
    primary = None
    secondary = None
    for row in opt_rows:
        kind = row["optimum_kind"]
        if method == "avgdeg":
            if kind == "avgdeg":
                primary = _extract_opt_row(row)
            elif kind == "avgdeg_connected":
                secondary = _extract_opt_row(row)
        elif method == "bp":
            if k is None:
                continue
            if (
                kind == "edge_density_kappa"
                and int(row["k"]) == int(k)
                and int(row["kappa"]) == target_kappa
            ):
                primary = _extract_opt_row(row)
    if primary is None:
        raise KeyError(f"optimum not found for method={method}, k={k}, kappa={kappa}")
    return primary, secondary


def _do_generate(args):
    root = Path(args.data_dir)
    p_values = [float(x) for x in args.p_values.split(",")]
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    for p in p_values:
        for seed in seeds:
            _generate_one(args.n, p, seed, root)
    print(f"wrote {len(p_values) * len(seeds)} graphs under {root}/synthetic/bf/n{args.n}")


def _filter_metas_by_seed(metas: List[Path], seeds_arg: Optional[str]) -> List[Path]:
    if not seeds_arg:
        return metas
    wanted = set(int(x) for x in seeds_arg.split(",") if x.strip())
    out: List[Path] = []
    for m in metas:
        try:
            with open(m) as f:
                seed = int(json.load(f).get("seed"))
        except Exception:
            continue
        if seed in wanted:
            out.append(m)
    return out


def _do_optima(args):
    root = Path(args.data_dir)
    k_set = [int(x) for x in args.k_values.split(",")]
    kappa_set = [int(x) for x in args.kappa_values.split(",")]
    metas = sorted((root / "synthetic" / "bf" / f"n{args.n}").rglob("meta.json"))
    metas = _filter_metas_by_seed(metas, getattr(args, "seeds", None))
    if not metas:
        raise FileNotFoundError(f"no meta.json found under {root}/synthetic/bf/n{args.n}")
    out_root = Path(args.exps_dir) / "synthetic" / "bf"
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = out_root / "optima.csv"
    rows: List[dict] = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as exe:
        futures = {exe.submit(_compute_optima_one, m, k_set, kappa_set): m for m in metas}
        for i, fut in enumerate(as_completed(futures), 1):
            rows.extend(fut.result())
            if i % 5 == 0 or i == len(metas):
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                print(f"optima: {i}/{len(metas)} graphs done", flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


def _do_solver_runs(args):
    root = Path(args.data_dir)
    metas = sorted((root / "synthetic" / "bf" / f"n{args.n}").rglob("meta.json"))
    metas = _filter_metas_by_seed(metas, getattr(args, "seeds", None))
    if not metas:
        raise FileNotFoundError(f"no meta.json under {root}/synthetic/bf/n{args.n}")
    k_set = [int(x) for x in args.k_values.split(",")]
    kappa_set = [int(x) for x in args.kappa_values.split(",")]
    optima_csv = Path(args.exps_dir) / "synthetic" / "bf" / "optima.csv"
    if not optima_csv.exists():
        raise FileNotFoundError(f"run bf_optima first; missing {optima_csv}")
    optima_df = pd.read_csv(optima_csv)

    out_root = Path(args.exps_dir) / "synthetic" / "bf"
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = out_root / "solver_runs.csv"

    optima_code_hash: Optional[str] = None
    if "code_hash" in optima_df.columns and not optima_df.empty:
        first = optima_df["code_hash"].dropna().head(1)
        if len(first):
            optima_code_hash = str(first.iloc[0])

    # Index optima per (n, p, seed, k, kappa) so feasibility lookups are O(1).
    bp_opt_index: Dict[Tuple[int, float, int, int, int], dict] = {}
    if "optimum_kind" in optima_df.columns:
        kappa_df = optima_df[optima_df["optimum_kind"] == "edge_density_kappa"]
        for row in kappa_df.itertuples(index=False):
            key = (int(row.n), float(row.p), int(row.seed), int(row.k), int(row.kappa))
            bp_opt_index[key] = {
                "opt_value": float(row.opt_value),
                "opt_size": (int(row.opt_size) if pd.notna(row.opt_size) else None),
            }

    cells: List[Tuple] = []
    skipped_rows: List[dict] = []
    for m in metas:
        cells.append((m, "avgdeg", None, None, args))
        with open(m) as f:
            mj = json.load(f)
        graph_key = (int(mj["n"]), float(mj["p"]), int(mj["seed"]))
        for k in k_set:
            for kappa in kappa_set:
                key = graph_key + (int(k), int(kappa))
                bf_opt = bp_opt_index.get(key)
                # Sentinel score < 0 means no subset containing q with |S| >= k
                # has edge-connectivity >= kappa. Running the solver here only
                # exhausts B&B trying to prove infeasibility, so the cell is
                # recorded as skipped instead of dispatched.
                if bf_opt is not None and bf_opt["opt_value"] < 0:
                    skipped_rows.append(
                        {
                            "n": mj["n"],
                            "p": mj["p"],
                            "seed": mj["seed"],
                            "method": "bp",
                            "k": k,
                            "kappa": kappa,
                            "opt_value_brute": bf_opt["opt_value"],
                            "opt_value_solver": float("nan"),
                            "opt_match": False,
                            "opt_match_within_tol": False,
                            "opt_match_size_only": False,
                            "opt_value_brute_secondary": float("nan"),
                            "opt_size_brute_secondary": None,
                            "opt_match_secondary": False,
                            "opt_match_within_tol_secondary": False,
                            "brute_actual_kappa": None,
                            "brute_actual_kappa_secondary": None,
                            "solver_actual_kappa": None,
                            "kappa_verified": None,
                            "kappa_verify_failed": None,
                            "hard_cap_hit": None,
                            "solver_size": 0,
                            "brute_size": bf_opt["opt_size"],
                            "wall_time_s": 0.0,
                            "total_bb_nodes": None,
                            "returncode": None,
                            "solver_build_id": None,
                            "optima_code_hash": optima_code_hash,
                            "status": "skipped_bf_infeasible",
                        }
                    )
                    continue
                cells.append((m, "bp", k, kappa, args))

    print(
        f"solver_runs: {len(cells)} cells to run, {len(skipped_rows)} cells skipped (BF infeasible)",
        flush=True,
    )

    rows: List[dict] = list(skipped_rows)
    tol = float(args.match_tol)
    # Cache (n, p, seed) -> adj_out bitmask so post-checks don't reread edge.csv per cell.
    adj_cache: Dict[Tuple[int, float, int], List[int]] = {}

    def _adj_for_graph(meta, meta_path_str):
        key = (int(meta["n"]), float(meta["p"]), int(meta["seed"]))
        cached = adj_cache.get(key)
        if cached is not None:
            return cached
        df_e = pd.read_csv(str(Path(meta_path_str).with_name("edge.csv")))
        edges_dir_local = list(
            zip(df_e["source"].astype(int), df_e["target"].astype(int))
        )
        adj_local = _bitmask_adjacency(int(meta["n"]), edges_dir_local)
        adj_cache[key] = adj_local
        return adj_local

    with ProcessPoolExecutor(max_workers=args.max_workers) as exe:
        futures = {exe.submit(_solver_runs_one, cell): cell for cell in cells}
        for i, fut in enumerate(as_completed(futures), 1):
            meta, meta_path_str, method, k, kappa, result, wall = fut.result()
            payload = result.get("solver_json") or {}
            stats = payload.get("stats") or {}
            qualities = payload.get("qualities") or {}
            opt_rows = optima_df[
                (optima_df["n"] == meta["n"])
                & (optima_df["p"] == meta["p"])
                & (optima_df["seed"] == meta["seed"])
            ].to_dict("records")
            primary, secondary = _opt_value_from_optima(opt_rows, method, k, kappa)
            opt_value, opt_size, opt_nodes, brute_actual_kappa = primary
            if secondary is not None:
                opt_value_secondary, opt_size_secondary, opt_nodes_secondary, brute_actual_kappa_secondary = secondary
            else:
                opt_value_secondary = float("nan")
                opt_size_secondary = None
                opt_nodes_secondary = []
                brute_actual_kappa_secondary = None
            solver_size = int(payload.get("size") or len(result["pred_nodes"]))
            if method == "avgdeg":
                solver_value = qualities.get("avg_degree_density", float("nan"))
            else:
                solver_value = qualities.get("edge_density", float("nan"))
            try:
                within_tol = abs(float(solver_value) - float(opt_value)) <= tol
            except Exception:
                within_tol = False
            try:
                within_tol_secondary = (
                    abs(float(solver_value) - float(opt_value_secondary)) <= tol
                    if secondary is not None
                    else False
                )
            except Exception:
                within_tol_secondary = False
            solver_nodes_set = set(int(n) for n in result["pred_nodes"])
            opt_set = set(int(n) for n in opt_nodes)
            opt_set_secondary = set(int(n) for n in (opt_nodes_secondary or []))
            opt_match = solver_nodes_set == opt_set
            opt_match_secondary = (
                solver_nodes_set == opt_set_secondary if secondary is not None else False
            )
            opt_match_size_only = solver_size == opt_size
            kappa_verified = payload.get("kappa_verified")
            kappa_verify_failed = payload.get("kappa_verify_failed")
            hard_cap_hit = payload.get("hard_cap_hit")
            solver_build_id = payload.get("solver_build_id")
            # Compute the actual edge-connectivity of the solver's returned set.
            solver_actual_kappa: Optional[int] = None
            if result["returncode"] == 0 and solver_nodes_set:
                adj_local = _adj_for_graph(meta, meta_path_str)
                s_mask_local = 0
                for nid in solver_nodes_set:
                    s_mask_local |= 1 << int(nid)
                q_local = int(meta["query_node"])
                if (s_mask_local >> q_local) & 1:
                    solver_actual_kappa = _edge_connectivity_in_subset(
                        adj_local, s_mask_local, q_local
                    )
            rows.append(
                {
                    "n": meta["n"],
                    "p": meta["p"],
                    "seed": meta["seed"],
                    "method": method,
                    "k": k,
                    "kappa": kappa,
                    "opt_value_brute": opt_value,
                    "opt_value_solver": solver_value,
                    "opt_match": opt_match,
                    "opt_match_within_tol": within_tol,
                    "opt_match_size_only": opt_match_size_only,
                    "opt_value_brute_secondary": opt_value_secondary,
                    "opt_size_brute_secondary": opt_size_secondary,
                    "opt_match_secondary": opt_match_secondary,
                    "opt_match_within_tol_secondary": within_tol_secondary,
                    "brute_actual_kappa": brute_actual_kappa,
                    "brute_actual_kappa_secondary": brute_actual_kappa_secondary,
                    "solver_actual_kappa": solver_actual_kappa,
                    "kappa_verified": kappa_verified,
                    "kappa_verify_failed": kappa_verify_failed,
                    "hard_cap_hit": hard_cap_hit,
                    "solver_size": solver_size,
                    "brute_size": opt_size,
                    "wall_time_s": wall,
                    "total_bb_nodes": stats.get("total_bb_nodes"),
                    "returncode": result["returncode"],
                    "solver_build_id": solver_build_id,
                    "optima_code_hash": optima_code_hash,
                    "status": "ran",
                }
            )
            if i % 50 == 0 or i == len(cells):
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                print(f"solver_runs: {i}/{len(cells)} done")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


def _do_verify(args):
    _run_demo(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("bf_generate", help="sample G(n, p) graphs")
    gen.add_argument("--n", type=int, default=25)
    gen.add_argument("--p-values", type=str, default="0.25,0.50,0.75")
    gen.add_argument("--seed-start", type=int, default=0)
    gen.add_argument("--num-seeds", type=int, default=50)
    gen.add_argument("--data-dir", type=str, default="data")
    gen.set_defaults(func=_do_generate)

    opt = sub.add_parser("bf_optima", help="brute-force optima per graph")
    opt.add_argument("--n", type=int, default=25)
    opt.add_argument("--k-values", type=str, default="3,4,5")
    opt.add_argument("--kappa-values", type=str, default="0,1,2")
    opt.add_argument("--data-dir", type=str, default="data")
    opt.add_argument("--exps-dir", type=str, default="exps")
    opt.add_argument("--max-workers", type=int, default=4)
    opt.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated graph seeds (e.g. 0,1,2,3,4). Default: every meta.json found.",
    )
    opt.set_defaults(func=_do_optima)

    runs = sub.add_parser("bf_solver_runs", help="run solver across (method, k, kappa)")
    runs.add_argument("--n", type=int, default=25)
    runs.add_argument("--k-values", type=str, default="3,4,5")
    runs.add_argument("--kappa-values", type=str, default="0,1,2")
    runs.add_argument("--data-dir", type=str, default="data")
    runs.add_argument("--exps-dir", type=str, default="exps")
    runs.add_argument("--bin-path", type=str, default="./solver/bin/solver")
    runs.add_argument("--gurobi-seed", type=int, default=42)
    runs.add_argument("--time-limit", type=float, default=-1.0)
    runs.add_argument("--max-workers", type=int, default=4)
    runs.add_argument("--match-tol", type=float, default=1e-9)
    runs.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated graph seeds (e.g. 0,1,2,3,4). Default: every meta.json found.",
    )
    runs.set_defaults(func=_do_solver_runs)

    verify = sub.add_parser("verify", help="legacy tiny-graph demo")
    verify.add_argument("--edge-csv", type=str, default=None)
    verify.add_argument("--query", type=str, default="1")
    verify.add_argument("--mode", type=str, choices=["avgdeg", "bp", "both"], default="both")
    verify.add_argument("--k", type=int, default=3)
    verify.add_argument("--kappa", type=int, default=0)
    verify.add_argument("--bin-path", type=str, default="./solver/bin/solver")
    verify.add_argument("--time-limit", type=float, default=-1.0)
    verify.set_defaults(func=_do_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
