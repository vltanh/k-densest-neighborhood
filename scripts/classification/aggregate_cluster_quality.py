"""Cluster-quality aggregator. Reads records.ndjson emitted by evaluate_nodes
and prints two CSVs per (dataset, method, params_hash) group:

    cluster_quality_per_query.csv  -- one row per query node.
    cluster_quality_summary.csv    -- median, IQR (Q1, Q3), n_queries.

The aggregator never re-solves anything; it consumes existing records.
"""

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd


METRIC_KEYS = [
    "size",
    "dir_internal_avg_degree",
    "dir_internal_edge_density",
    "undir_external_expansion",
    "undir_external_conductance",
    "undir_internal_ncut",
    "mixing_param",
    "algebraic_connectivity_lambda2",
]


def _read_records(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _row_for(record):
    qualities = record.get("qualities") or {}
    row = {
        "dataset": record.get("dataset"),
        "method": record.get("method"),
        "params_hash": record.get("params_hash"),
        "params": json.dumps(record.get("params") or {}, sort_keys=True),
        "seed": record.get("seed"),
        "split_hash": record.get("split_hash"),
        "query_node": record.get("query_node"),
        "query_split": record.get("query_split"),
        "fallback_used": record.get("fallback_used"),
        "size_solver": record.get("size"),
    }
    for key in METRIC_KEYS:
        row[key] = qualities.get(key, math.nan)
    return row


def _summarise(per_query: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "method", "params_hash"]
    summary_rows = []
    for keys, chunk in per_query.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["params"] = chunk["params"].iloc[0]
        row["n_queries"] = len(chunk)
        for key in METRIC_KEYS:
            vals = chunk[key].astype(float).to_numpy()
            finite = vals[~np.isnan(vals)]
            if finite.size == 0:
                row[f"{key}_median"] = math.nan
                row[f"{key}_q1"] = math.nan
                row[f"{key}_q3"] = math.nan
            else:
                row[f"{key}_median"] = float(np.median(finite))
                row[f"{key}_q1"] = float(np.percentile(finite, 25))
                row[f"{key}_q3"] = float(np.percentile(finite, 75))
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=str,
        required=True,
        help="Path to records.ndjson (or a directory containing it)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for the two CSVs",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset filter",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Optional method filter (bp / avgdeg / bfs)",
    )
    args = parser.parse_args()

    records_path = args.records
    if os.path.isdir(records_path):
        records_path = os.path.join(records_path, "records.ndjson")
    if not os.path.exists(records_path):
        raise FileNotFoundError(records_path)

    records = _read_records(records_path)
    if args.dataset:
        records = [r for r in records if r.get("dataset") == args.dataset]
    if args.method:
        records = [r for r in records if r.get("method") == args.method]

    if not records:
        print(f"No records matched at {records_path}")
        return

    per_query = pd.DataFrame([_row_for(r) for r in records])
    summary = _summarise(per_query)

    os.makedirs(args.output, exist_ok=True)
    per_query_path = os.path.join(args.output, "cluster_quality_per_query.csv")
    summary_path = os.path.join(args.output, "cluster_quality_summary.csv")
    per_query.to_csv(per_query_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"wrote {per_query_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
