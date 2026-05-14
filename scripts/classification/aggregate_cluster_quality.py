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
    kappa_s                            (alias for achieved edge_connectivity)

Derived rates:

    size_lt_k_rate                     fraction of query results with |S| < requested k
    kappa_s_lt_1_rate                  fraction with achieved edge connectivity < 1
    kappa_s_lt_2_rate                  fraction with achieved edge connectivity < 2

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
from record_io import load_records_from_path  # noqa: E402


HEADLINE_METRICS = [
    "size",
    "dir_internal_avg_degree",
    "dir_internal_edge_density",
    "undir_external_expansion",
    "undir_internal_ncut",
    "mixing_param",
    "algebraic_connectivity_lambda2",
    "edge_connectivity",
    "kappa_s",
    "within_class_internal_edges_ratio",
    "train_label_entropy",
    "true_class_vote_share",
]

DERIVED_RATE_COLUMNS = [
    "size_lt_k",
    "kappa_s_lt_1",
    "kappa_s_lt_2",
]

# Carried through to the per-query CSV but excluded from the summary headline.
TRACE_METRICS = ["undir_external_conductance", "n_train_neighbours"]
# Categorical, surfaced separately as a stratification axis rather than a number.
CATEGORICAL_METRICS = ["size_bucket"]


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


def _validate_complete(per_query: pd.DataFrame, data_dir: str, allow_partial: bool, requested_split: str):
    if allow_partial or per_query.empty:
        return
    wanted_splits = ["val", "test"] if requested_split == "union" else [requested_split]
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
        for qsplit in wanted_splits:
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
            "partial cluster-quality records detected. "
            f"{details}. Re-run missing cells or pass --allow-partial."
        )


def _requested_k(params: dict):
    try:
        value = (params or {}).get("k")
        if value is None:
            return math.nan
        return int(value)
    except (TypeError, ValueError):
        return math.nan


def _finite_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _add_derived_quality_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "kappa_s" not in out.columns and "edge_connectivity" in out.columns:
        out["kappa_s"] = out["edge_connectivity"]
    if "requested_k" not in out.columns:
        out["requested_k"] = math.nan

    size_vals = pd.to_numeric(out.get("size"), errors="coerce")
    k_vals = pd.to_numeric(out.get("requested_k"), errors="coerce")
    kappa_vals = pd.to_numeric(out.get("kappa_s"), errors="coerce")

    out["size_lt_k"] = np.where(size_vals.notna() & k_vals.notna(), size_vals < k_vals, np.nan)
    out["kappa_s_lt_1"] = np.where(kappa_vals.notna(), kappa_vals < 1, np.nan)
    out["kappa_s_lt_2"] = np.where(kappa_vals.notna(), kappa_vals < 2, np.nan)
    return out


def _row_for(record):
    qualities = record.get("qualities") or {}
    params = record.get("params") or {}
    edge_connectivity = qualities.get("edge_connectivity", math.nan)
    row = {
        "dataset": record.get("dataset"),
        "method": record.get("method"),
        "params_hash": record.get("params_hash"),
        "params": json.dumps(params, sort_keys=True),
        "requested_k": _requested_k(params),
        "seed": record.get("seed"),
        "split_hash": record.get("split_hash"),
        "query_node": record.get("query_node"),
        "query_split": record.get("query_split"),
        "fallback_used": record.get("fallback_used"),
        "size_solver": record.get("size"),
        "hard_cap_hit": bool(record.get("hard_cap_hit") or False),
        "incumbent_trajectory_json": json.dumps(record.get("incumbent_trajectory") or []),
    }
    for key in HEADLINE_METRICS + TRACE_METRICS:
        if key == "kappa_s":
            row[key] = edge_connectivity
        else:
            row[key] = qualities.get(key, math.nan)
    for key in CATEGORICAL_METRICS:
        row[key] = qualities.get(key)
    return _add_derived_quality_columns(pd.DataFrame([row])).iloc[0].to_dict()


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
    extras_agg = {
        "split_hash": "first",
        "requested_k": "first",
        "fallback_used": "any",
        "size_solver": "median",
        "hard_cap_hit": "any",
    }
    for cat in CATEGORICAL_METRICS:
        if cat in per_query.columns:
            extras_agg[cat] = "first"
    extras = per_query.groupby(keys, dropna=False, as_index=False).agg(extras_agg)
    return _add_derived_quality_columns(agg.merge(extras, on=keys, how="left"))


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
        if "requested_k" in chunk.columns:
            k_vals = pd.to_numeric(chunk["requested_k"], errors="coerce").dropna().unique()
            row["requested_k"] = int(k_vals[0]) if len(k_vals) else math.nan
        row["n_queries"] = len(chunk)
        if "hard_cap_hit" in chunk.columns:
            hits = chunk["hard_cap_hit"].fillna(False).astype(bool)
            row["n_hard_cap_hit"] = int(hits.sum())
            row["hard_cap_hit_rate"] = float(hits.mean()) if len(chunk) else 0.0
        else:
            row["n_hard_cap_hit"] = 0
            row["hard_cap_hit_rate"] = 0.0
        for col in DERIVED_RATE_COLUMNS:
            if col not in chunk.columns:
                row[f"n_{col}"] = 0
                row[f"{col}_rate"] = math.nan
                row[f"{col}_pct"] = math.nan
                continue
            vals = chunk[col].dropna().astype(bool)
            row[f"n_{col}"] = int(vals.sum()) if len(vals) else 0
            row[f"{col}_rate"] = float(vals.mean()) if len(vals) else math.nan
            row[f"{col}_pct"] = float(100.0 * vals.mean()) if len(vals) else math.nan
        for key in HEADLINE_METRICS:
            vals = chunk[key].astype(float).to_numpy()
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
    parser.add_argument("--data-dir", type=str, default="data")
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
    records = load_records_from_path(records_path)

    if args.dataset:
        records = [r for r in records if r.get("dataset") == args.dataset]
    if args.method:
        records = [r for r in records if r.get("method") == args.method]
    if args.params_hash:
        records = [r for r in records if r.get("params_hash") == args.params_hash]

    if not records:
        print(f"No records matched at {records_path}")
        return
    _validate_split_hashes(records, args.data_dir, args.allow_split_hash_mismatch)

    per_query = pd.DataFrame([_row_for(r) for r in records])
    per_query = _filter_by_split(per_query, args.split)
    _validate_complete(per_query, args.data_dir, args.allow_partial, args.split)
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
