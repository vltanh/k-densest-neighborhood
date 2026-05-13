"""Cluster-quality aggregator. Reads records.ndjson emitted by evaluate_nodes
and emits per-(dataset, method, params_hash, split) summaries plus, optionally,
per-query deltas vs a baseline method with Wilcoxon signed-rank p-values.

Grouping is split-aware (val, test, or the union both via --split=union) and
seed-aware: BP records carry a Gurobi seed and are first collapsed by median
across seeds for each (query, params, split) so a seed-replicated cell does not
get over-counted in the cross-query median.

Headline metrics:

    size
    dir_internal_avg_degree
    dir_internal_edge_density
    undir_external_expansion
    undir_internal_ncut
    mixing_param                       (boundary mixing; preferred over conductance)
    algebraic_connectivity_lambda2     (scipy eigsh, no node-count cap)
    edge_connectivity

undir_external_conductance is intentionally not part of the headline list: on
small induced subgraphs it collapses to mixing_param numerically. The
per-query CSV still carries the conductance column for traceability.
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


HEADLINE_METRICS = [
    "size",
    "dir_internal_avg_degree",
    "dir_internal_edge_density",
    "undir_external_expansion",
    "undir_internal_ncut",
    "mixing_param",
    "algebraic_connectivity_lambda2",
    "edge_connectivity",
    "within_class_internal_edges_ratio",
    "train_label_entropy",
    "true_class_vote_share",
]

# Carried through to the per-query CSV but excluded from the summary headline.
TRACE_METRICS = ["undir_external_conductance", "n_train_neighbours"]
# Categorical, surfaced separately as a stratification axis rather than a number.
CATEGORICAL_METRICS = ["size_bucket"]


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
        "hard_cap_hit": bool(record.get("hard_cap_hit") or False),
    }
    for key in HEADLINE_METRICS + TRACE_METRICS:
        row[key] = qualities.get(key, math.nan)
    for key in CATEGORICAL_METRICS:
        row[key] = qualities.get(key)
    return row


def _filter_by_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == "union":
        return df
    return df[df["query_split"] == split].copy()


def _collapse_seeds(per_query: pd.DataFrame) -> pd.DataFrame:
    """For BP records, replace the (query, params, split) seed cluster by its
    median. AvgDeg and BFS records are deterministic so seed is None and the
    groupby is a no-op."""
    metric_cols = HEADLINE_METRICS + TRACE_METRICS
    keys = ["dataset", "method", "params_hash", "params", "query_split", "query_node"]
    agg = per_query.groupby(keys, dropna=False, as_index=False)[metric_cols].median(numeric_only=True)
    # Re-attach split_hash / fallback_used / size_solver / size_bucket from the
    # first row of each group; these are seed-invariant for deterministic
    # methods and a representative for BP.
    extras_agg = {"split_hash": "first", "fallback_used": "any", "size_solver": "median", "hard_cap_hit": "any"}
    for cat in CATEGORICAL_METRICS:
        if cat in per_query.columns:
            extras_agg[cat] = "first"
    extras = per_query.groupby(keys, dropna=False, as_index=False).agg(extras_agg)
    return agg.merge(extras, on=keys, how="left")


def _summarise(
    collapsed: pd.DataFrame,
    group_extra: Optional[list] = None,
    combine_splits: bool = False,
) -> pd.DataFrame:
    if combine_splits:
        group_cols = ["dataset", "method", "params_hash"]
    else:
        group_cols = ["dataset", "method", "params_hash", "query_split"]
    if group_extra:
        group_cols = group_cols + list(group_extra)
    rows = []
    for keys, chunk in collapsed.groupby(group_cols, dropna=False):
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
        for key in HEADLINE_METRICS:
            vals = chunk[key].astype(float).to_numpy()
            finite = vals[~np.isnan(vals)]
            if finite.size == 0:
                row[f"{key}_median"] = math.nan
                row[f"{key}_q1"] = math.nan
                row[f"{key}_q3"] = math.nan
                row[f"{key}_n_finite"] = 0
            else:
                row[f"{key}_median"] = float(np.median(finite))
                row[f"{key}_q1"] = float(np.percentile(finite, 25))
                row[f"{key}_q3"] = float(np.percentile(finite, 75))
                row[f"{key}_n_finite"] = int(finite.size)
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_deltas(collapsed: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Per-query metric deltas vs a baseline method. Records are paired on
    (dataset, params_hash, query_split, query_node); a query is included only
    when both methods produced a finite value for the given metric."""
    from scipy.stats import wilcoxon

    other = collapsed[collapsed["method"] != baseline].copy()
    base = collapsed[collapsed["method"] == baseline].copy()
    if base.empty:
        print(f"--paired-vs {baseline}: no baseline records found")
        return pd.DataFrame()

    out_rows = []
    for (dataset, params_hash, qsplit), chunk_o in other.groupby(
        ["dataset", "params_hash", "query_split"], dropna=False
    ):
        # The baseline does not vary on the experimental method's params_hash,
        # so we pair on (dataset, query_split, query_node) only. When there
        # are multiple baseline params_hashes, take the best by f1 not handled
        # here; the aggregator pairs against the FIRST baseline row per query.
        base_for_dataset = base[
            (base["dataset"] == dataset) & (base["query_split"] == qsplit)
        ]
        if base_for_dataset.empty:
            continue
        b = base_for_dataset.drop_duplicates("query_node", keep="first").set_index("query_node")
        o = chunk_o.drop_duplicates("query_node", keep="first").set_index("query_node")
        common = sorted(set(o.index) & set(b.index))
        if not common:
            continue
        method_name = chunk_o["method"].iloc[0]
        for metric in HEADLINE_METRICS:
            diffs = []
            for q in common:
                a_val = float(o.loc[q, metric])
                b_val = float(b.loc[q, metric])
                if math.isnan(a_val) or math.isnan(b_val):
                    continue
                diffs.append(a_val - b_val)
            if len(diffs) < 2:
                p_val = math.nan
            else:
                non_zero = [d for d in diffs if d != 0]
                if not non_zero:
                    p_val = 1.0
                else:
                    try:
                        _, p_val = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
                        p_val = float(p_val)
                    except ValueError:
                        p_val = math.nan
            out_rows.append(
                {
                    "dataset": dataset,
                    "method": method_name,
                    "baseline": baseline,
                    "params_hash": params_hash,
                    "query_split": qsplit,
                    "metric": metric,
                    "n_paired": len(diffs),
                    "delta_median": float(np.median(diffs)) if diffs else math.nan,
                    "delta_q1": float(np.percentile(diffs, 25)) if diffs else math.nan,
                    "delta_q3": float(np.percentile(diffs, 75)) if diffs else math.nan,
                    "wilcoxon_p": p_val,
                }
            )
    return pd.DataFrame(out_rows)


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
        help="Output directory for the aggregated CSVs",
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument(
        "--split",
        type=str,
        default="union",
        choices=["val", "test", "union"],
        help="Filter records by query_split before aggregating (default: union of val+test).",
    )
    parser.add_argument(
        "--combine-splits",
        action="store_true",
        help=(
            "Group by (dataset, method, params_hash) only, treating val and "
            "test records as one pool. Use for the headline cluster-quality "
            "frame which is intrinsic and label-agnostic across both splits."
        ),
    )
    parser.add_argument(
        "--params-hash",
        type=str,
        default=None,
        help="Filter records to one params_hash; combine with --combine-splits to emit one row per method.",
    )
    parser.add_argument(
        "--paired-vs",
        type=str,
        default=None,
        help="Emit per-query deltas vs the named baseline method (avgdeg / bfs / bp).",
    )
    args = parser.parse_args()

    records_path = args.records
    if os.path.isdir(records_path):
        # Allow passing a tune_val or evaluate_test directory: walk it for
        # any records.ndjson.
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
    if args.params_hash:
        records = [r for r in records if r.get("params_hash") == args.params_hash]

    if not records:
        print(f"No records matched at {records_path}")
        return

    per_query = pd.DataFrame([_row_for(r) for r in records])
    per_query = _filter_by_split(per_query, args.split)
    collapsed = _collapse_seeds(per_query)
    summary = _summarise(collapsed, combine_splits=args.combine_splits)

    os.makedirs(args.output, exist_ok=True)
    per_query_path = os.path.join(args.output, "cluster_quality_per_query.csv")
    collapsed_path = os.path.join(args.output, "cluster_quality_collapsed.csv")
    summary_path = os.path.join(args.output, "cluster_quality_summary.csv")
    per_query.to_csv(per_query_path, index=False)
    collapsed.to_csv(collapsed_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"wrote {per_query_path}")
    print(f"wrote {collapsed_path}")
    print(f"wrote {summary_path}")

    # Size-stratified summary so density gains can be read separately from
    # size effects in the headline frames.
    if "size_bucket" in collapsed.columns:
        size_summary = _summarise(
            collapsed, group_extra=["size_bucket"], combine_splits=args.combine_splits
        )
        size_path = os.path.join(args.output, "cluster_quality_summary_by_size.csv")
        size_summary.to_csv(size_path, index=False)
        print(f"wrote {size_path}")

    if args.paired_vs:
        deltas = _paired_deltas(collapsed, args.paired_vs)
        if not deltas.empty:
            deltas_path = os.path.join(args.output, f"cluster_quality_paired_vs_{args.paired_vs}.csv")
            deltas.to_csv(deltas_path, index=False)
            print(f"wrote {deltas_path}")


if __name__ == "__main__":
    main()
