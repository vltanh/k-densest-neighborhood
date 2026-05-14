"""Cost-perspective aggregator. Reads records.ndjson and reports solver
computational cost per (dataset, method, params_hash, query_split). Cost is
the third experimental perspective alongside cluster quality (intrinsic
geometry) and classification (extrinsic label utility); it lives on the same
records produced by sweep_cluster_quality.py and is therefore label-agnostic
and runs on the val + test union.

Reported metrics, all aggregated per cell:

    wall_time_s
    oracle_queries
    total_bb_nodes
    total_lp_solves
    total_columns_added
    total_cuts_added
    t_lp_solve
    t_pricing
    size_solver
    optimality_gap

Each metric column emits mean, std, median, q1, q3, and n_finite. AvgDeg and
BFS rows usually carry empty BB or LP counts; those columns surface as NaN
with n_finite=0.
"""

import argparse
import json
import math
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from record_io import load_ndjson_records  # noqa: E402


COST_NUMERIC_FIELDS = [
    "wall_time_s",
    "oracle_queries",
    "size_solver",
    "total_bb_nodes",
    "total_lp_solves",
    "total_columns_added",
    "total_cuts_added",
    "t_lp_solve",
    "t_pricing",
    "t_separation",
    "t_sync",
    "t_total",
    "optimality_gap",
    "bb_incumbent_obj",
    "bb_best_bound",
    "open_bb_nodes",
]


def _validate_split_hashes(records, data_dir: str, allow_mismatch: bool):
    if allow_mismatch:
        return
    meta_cache = {}
    mismatches = []
    for record in records:
        dataset = record.get("dataset")
        qsplit = record.get("query_split")
        if not dataset or qsplit not in {"val", "test"}:
            continue
        if dataset not in meta_cache:
            meta_path = os.path.join(data_dir, dataset, "split_meta.json")
            if not os.path.exists(meta_path):
                meta_cache[dataset] = {}
            else:
                with open(meta_path) as f:
                    meta_cache[dataset] = json.load(f).get("splits") or {}
        expected_hash = (meta_cache.get(dataset, {}).get(qsplit) or {}).get("hash")
        if expected_hash and record.get("split_hash") != expected_hash:
            mismatches.append(
                (
                    dataset,
                    record.get("method"),
                    str(record.get("params_hash"))[-12:],
                    qsplit,
                    record.get("query_node"),
                    record.get("split_hash"),
                    expected_hash,
                )
            )
    if mismatches:
        details = "; ".join(
            f"{dataset}/{method}/{ph}/{qsplit}/q={query}: {seen} != {want}"
            for dataset, method, ph, qsplit, query, seen, want in mismatches[:5]
        )
        if len(mismatches) > 5:
            details += f"; ... {len(mismatches) - 5} more"
        raise RuntimeError(
            "split_hash mismatch detected. "
            f"{details}. Re-run records for the current split_meta.json "
            "or pass --allow-split-hash-mismatch for legacy artifacts."
        )


def _expected_split_sizes(dataset: str, data_dir: str) -> dict:
    meta_path = os.path.join(data_dir, dataset, "split_meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        splits = (json.load(f).get("splits") or {})
    return {
        split: int(info["size"])
        for split, info in splits.items()
        if split in {"val", "test"} and "size" in info
    }


def _validate_complete(per_query: pd.DataFrame, data_dir: str, allow_partial: bool):
    if allow_partial or per_query.empty:
        return
    expected_cache = {}
    groups = {}
    base_keys = set()
    for row in per_query.itertuples(index=False):
        dataset = getattr(row, "dataset", None)
        qsplit = getattr(row, "query_split", None)
        if not dataset or qsplit not in {"val", "test"}:
            continue
        if dataset not in expected_cache:
            expected_cache[dataset] = _expected_split_sizes(dataset, data_dir)
        if qsplit not in expected_cache[dataset]:
            continue
        seed = getattr(row, "seed", None)
        if pd.isna(seed):
            seed = None
        base_key = (
            dataset,
            getattr(row, "method", None),
            getattr(row, "params_hash", None),
            seed,
        )
        base_keys.add(base_key)
        key = (*base_key, qsplit)
        groups.setdefault(key, set()).add(int(getattr(row, "query_node")))

    short = []
    for dataset, method, params_hash, seed in base_keys:
        for qsplit in ("val", "test"):
            want = expected_cache.get(dataset, {}).get(qsplit)
            if want is None:
                continue
            count = len(groups.get((dataset, method, params_hash, seed, qsplit), set()))
            if count != want:
                short.append((dataset, method, str(params_hash)[-12:], seed, qsplit, count, want))

    if short:
        details = "; ".join(
            f"{dataset}/{method}/{ph}/seed={seed}/{qsplit}: {count}/{want}"
            for dataset, method, ph, seed, qsplit, count, want in short[:10]
        )
        if len(short) > 10:
            details += f"; ... {len(short) - 10} more"
        raise RuntimeError(
            "partial cost records detected. "
            f"{details}. Re-run missing cells or pass --allow-partial."
        )


def _row_for(record):
    stats = record.get("stats") or {}
    row = {
        "dataset": record.get("dataset"),
        "method": record.get("method"),
        "params_hash": record.get("params_hash"),
        "params": json.dumps(record.get("params") or {}, sort_keys=True),
        "seed": record.get("seed"),
        "query_node": record.get("query_node"),
        "query_split": record.get("query_split"),
        "wall_time_s": record.get("wall_time_s"),
        "oracle_queries": record.get("oracle_queries"),
        "size_solver": record.get("size"),
        "total_bb_nodes": stats.get("total_bb_nodes"),
        "total_lp_solves": stats.get("total_lp_solves"),
        "total_columns_added": stats.get("total_columns_added"),
        "total_cuts_added": stats.get("total_cuts_added"),
        "t_lp_solve": stats.get("t_lp_solve"),
        "t_pricing": stats.get("t_pricing"),
        "t_separation": stats.get("t_separation"),
        "t_sync": stats.get("t_sync"),
        "t_total": stats.get("t_total"),
        "optimality_gap": record.get("optimality_gap", stats.get("optimality_gap")),
        "bb_incumbent_obj": record.get("bb_incumbent_obj", stats.get("bb_incumbent_obj")),
        "bb_best_bound": record.get("bb_best_bound", stats.get("bb_best_bound")),
        "open_bb_nodes": stats.get("open_bb_nodes"),
        "gap_status": record.get("gap_status", stats.get("gap_status")),
        "solver_build_id": record.get("solver_build_id"),
        "hard_cap_hit": record.get("hard_cap_hit"),
        "incumbent_trajectory_json": json.dumps(record.get("incumbent_trajectory") or []),
    }
    return row


def _summarise(per_query: pd.DataFrame, combine_splits: bool = False) -> pd.DataFrame:
    if combine_splits:
        group_cols = ["dataset", "method", "params_hash"]
    else:
        group_cols = ["dataset", "method", "params_hash", "query_split"]

    rows = []
    for keys, chunk in per_query.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["params"] = chunk["params"].iloc[0]
        row["n_queries"] = len(chunk)
        if "hard_cap_hit" in chunk.columns:
            hits = chunk["hard_cap_hit"].fillna(False).astype(bool)
            row["n_hard_cap_hit"] = int(hits.sum())
            row["hard_cap_hit_rate"] = float(hits.mean()) if len(chunk) else 0.0
        else:
            row["n_hard_cap_hit"] = 0
            row["hard_cap_hit_rate"] = 0.0
        for key in COST_NUMERIC_FIELDS:
            if key not in chunk.columns:
                row[f"{key}_mean"] = math.nan
                row[f"{key}_std"] = math.nan
                row[f"{key}_median"] = math.nan
                row[f"{key}_q1"] = math.nan
                row[f"{key}_q3"] = math.nan
                row[f"{key}_n_finite"] = 0
                continue
            vals = pd.to_numeric(chunk[key], errors="coerce").to_numpy(dtype=float)
            finite = vals[~np.isnan(vals)]
            if finite.size == 0:
                row[f"{key}_mean"] = math.nan
                row[f"{key}_std"] = math.nan
                row[f"{key}_median"] = math.nan
                row[f"{key}_q1"] = math.nan
                row[f"{key}_q3"] = math.nan
                row[f"{key}_n_finite"] = 0
            else:
                row[f"{key}_mean"] = float(np.mean(finite))
                row[f"{key}_std"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
                row[f"{key}_median"] = float(np.median(finite))
                row[f"{key}_q1"] = float(np.percentile(finite, 25))
                row[f"{key}_q3"] = float(np.percentile(finite, 75))
                row[f"{key}_n_finite"] = int(finite.size)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=str,
        required=True,
        help="Path to records.ndjson (or a directory of them).",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument(
        "--combine-splits",
        action="store_true",
        help="Drop query_split from the group key; aggregate val + test as one pool.",
    )
    parser.add_argument(
        "--allow-split-hash-mismatch",
        action="store_true",
        help="Allow legacy records whose split_hash differs from current split_meta.json.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow incomplete cells and aggregate only the records present.",
    )
    args = parser.parse_args()

    records_path = args.records
    if os.path.isdir(records_path):
        gathered = []
        for root, _dirs, files in os.walk(records_path):
            for fn in files:
                if fn == "records.ndjson":
                    gathered.extend(load_ndjson_records(os.path.join(root, fn)))
        records = gathered
    else:
        if not os.path.exists(records_path):
            raise FileNotFoundError(records_path)
        records = load_ndjson_records(records_path)

    if args.dataset:
        records = [r for r in records if r.get("dataset") == args.dataset]
    if args.method:
        records = [r for r in records if r.get("method") == args.method]
    if not records:
        print(f"No records matched at {records_path}")
        return
    _validate_split_hashes(records, args.data_dir, args.allow_split_hash_mismatch)

    per_query = pd.DataFrame([_row_for(r) for r in records])
    _validate_complete(per_query, args.data_dir, args.allow_partial)
    summary = _summarise(per_query, combine_splits=args.combine_splits)

    os.makedirs(args.output, exist_ok=True)
    per_query_path = os.path.join(args.output, "cost_per_query.csv")
    summary_path = os.path.join(args.output, "cost_summary.csv")
    per_query.to_csv(per_query_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"wrote {per_query_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
