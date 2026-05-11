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

Each metric column emits median, q1, q3, n_finite, plus a top-line mean for
quick comparison across methods. AvgDeg and BFS rows usually carry empty BB
or LP counts; those columns surface as NaN with n_finite=0.
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
from solver_utils import load_ndjson_records  # noqa: E402


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
]


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
        "solver_build_id": record.get("solver_build_id"),
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
        for key in COST_NUMERIC_FIELDS:
            if key not in chunk.columns:
                row[f"{key}_median"] = math.nan
                row[f"{key}_q1"] = math.nan
                row[f"{key}_q3"] = math.nan
                row[f"{key}_mean"] = math.nan
                row[f"{key}_n_finite"] = 0
                continue
            vals = pd.to_numeric(chunk[key], errors="coerce").to_numpy(dtype=float)
            finite = vals[~np.isnan(vals)]
            if finite.size == 0:
                row[f"{key}_median"] = math.nan
                row[f"{key}_q1"] = math.nan
                row[f"{key}_q3"] = math.nan
                row[f"{key}_mean"] = math.nan
                row[f"{key}_n_finite"] = 0
            else:
                row[f"{key}_median"] = float(np.median(finite))
                row[f"{key}_q1"] = float(np.percentile(finite, 25))
                row[f"{key}_q3"] = float(np.percentile(finite, 75))
                row[f"{key}_mean"] = float(np.mean(finite))
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
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument(
        "--combine-splits",
        action="store_true",
        help="Drop query_split from the group key; aggregate val + test as one pool.",
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

    per_query = pd.DataFrame([_row_for(r) for r in records])
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
