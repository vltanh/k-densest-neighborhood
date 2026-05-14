"""Wide CSV aggregator for classification/community-search records.

This script is intentionally table-agnostic.  It reads the records.ndjson files
emitted by sweep_cluster_quality.py / evaluate_nodes and writes broad CSVs that
preserve enough information to choose paper tables later:

    records_flat.csv
        One row per query record with params, qualities, stats, and core
        top-level fields flattened into columns.

    cell_summary_by_seed.csv
        One row per (dataset, method, params_hash, split, seed), with macro
        classification scores and distribution summaries for every numeric
        metric found in the flat records.

    cell_summary_union_by_seed.csv
        Same as above, but val+test are pooled.  This is the natural input for
        intrinsic quality and cost tables.

    classification_grid_by_seed.csv
        Compact classification-only view for val, test, and the canonical
        BFS-depth-1-wrong hard subset when it can be identified.

    classification_best_by_val_f1.csv
        Validation-best params per (dataset, method, seed), with the matching
        val/test/hard-subset scores carried through for quick inspection.
"""

import argparse
import json
import math
import os
import re
import sys
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from record_io import load_ndjson_records  # noqa: E402
from split_utils import build_out_adjacency, compute_hard_subset, sha256_node_set  # noqa: E402


IDENTIFIER_COLS = {
    "query_node",
    "query_label",
    "predicted_label",
}

FLAG_COLS = [
    "fallback_used",
    "hard_cap_hit",
    "kappa_verified",
    "kappa_verify_failed",
    "prediction_correct",
    "size_lt_k",
    "kappa_s_lt_1",
    "kappa_s_lt_2",
]

CATEGORICAL_COUNT_COLS = [
    "gap_status",
    "quality_size_bucket",
    "quality_schema_version",
    "solver_build_id",
]


def _load_records(path: str) -> List[dict]:
    if os.path.isdir(path):
        records: List[dict] = []
        for root, _dirs, files in os.walk(path):
            for fn in files:
                if fn == "records.ndjson":
                    records.extend(load_ndjson_records(os.path.join(root, fn)))
        return records
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return load_ndjson_records(path)


def _safe_col(raw: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(raw)).strip("_")


def _jsonish(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True)


def _finite_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _flatten_nested(prefix: str, value, out: dict):
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten_nested(f"{prefix}_{_safe_col(key)}", child, out)
        return
    out[prefix] = _jsonish(value)


def _flatten_params(params: dict) -> dict:
    out = {}
    for key, value in (params or {}).items():
        key_s = _safe_col(key)
        prefix = "eval" if key == "_eval" else f"param_{key_s}"
        _flatten_nested(prefix, value, out)
    return out


def _flatten_record(record: dict) -> dict:
    params = record.get("params") or {}
    qualities = record.get("qualities") or {}
    row = {
        "dataset": record.get("dataset"),
        "method": record.get("method"),
        "params_hash": record.get("params_hash"),
        "params_hash_short": str(record.get("params_hash") or "")[-12:],
        "params_json": json.dumps(params, sort_keys=True),
        "seed": record.get("seed"),
        "split_hash": record.get("split_hash"),
        "query_split": record.get("query_split"),
        "query_node": record.get("query_node"),
        "query_label": record.get("query_label"),
        "predicted_label": record.get("predicted_label"),
        "returncode": record.get("returncode"),
        "size": record.get("size"),
        "oracle_queries": record.get("oracle_queries"),
        "wall_time_s": record.get("wall_time_s"),
        "solver_wall_time_s": record.get("solver_wall_time_s"),
        "soft_time_limit_s": record.get("soft_time_limit_s"),
        "hard_time_limit_s": record.get("hard_time_limit_s"),
        "solver_build_id": record.get("solver_build_id"),
        "solver_dump_path": record.get("solver_dump_path"),
        "quality_schema_version": record.get("quality_schema_version"),
        "kappa_verified": record.get("kappa_verified"),
        "kappa_verify_failed": record.get("kappa_verify_failed"),
        "hard_cap_hit": record.get("hard_cap_hit"),
        "optimality_gap": record.get("optimality_gap"),
        "bb_incumbent_obj": record.get("bb_incumbent_obj"),
        "bb_best_bound": record.get("bb_best_bound"),
        "gap_status": record.get("gap_status"),
        "fallback_used": record.get("fallback_used"),
        "retrieved_nodes_json": json.dumps(record.get("retrieved_nodes") or record.get("neighborhood") or []),
        "incumbent_trajectory_json": json.dumps(record.get("incumbent_trajectory") or []),
    }
    requested_k = _finite_float(params.get("k"))
    size = _finite_float(qualities.get("size", record.get("size")))
    kappa_s = _finite_float(qualities.get("edge_connectivity"))
    row["requested_k"] = requested_k
    row["quality_kappa_s"] = kappa_s
    row["size_lt_k"] = (size < requested_k) if math.isfinite(size) and math.isfinite(requested_k) else None
    row["kappa_s_lt_1"] = (kappa_s < 1) if math.isfinite(kappa_s) else None
    row["kappa_s_lt_2"] = (kappa_s < 2) if math.isfinite(kappa_s) else None
    if row["query_label"] is not None and row["predicted_label"] is not None:
        row["prediction_correct"] = int(row["query_label"]) == int(row["predicted_label"])
    else:
        row["prediction_correct"] = None
    row.update(_flatten_params(params))
    for key, value in qualities.items():
        row[f"quality_{_safe_col(key)}"] = _jsonish(value)
    for key, value in (record.get("stats") or {}).items():
        row[f"stat_{_safe_col(key)}"] = _jsonish(value)
    return row


def _validate_split_hashes(records: Iterable[dict], data_dir: str, allow_mismatch: bool):
    if allow_mismatch:
        return
    meta_cache: Dict[str, dict] = {}
    mismatches = []
    for record in records:
        dataset = record.get("dataset")
        qsplit = record.get("query_split")
        if not dataset or qsplit not in {"val", "test"}:
            continue
        if dataset not in meta_cache:
            meta_path = os.path.join(data_dir, dataset, "split_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta_cache[dataset] = json.load(f).get("splits") or {}
            else:
                meta_cache[dataset] = {}
        expected = (meta_cache.get(dataset, {}).get(qsplit) or {}).get("hash")
        if expected and record.get("split_hash") != expected:
            mismatches.append(
                (
                    dataset,
                    record.get("method"),
                    str(record.get("params_hash"))[-12:],
                    record.get("seed"),
                    qsplit,
                    record.get("query_node"),
                    record.get("split_hash"),
                    expected,
                )
            )
    if not mismatches:
        return
    details = "; ".join(
        f"{dataset}/{method}/{ph}/seed={seed}/{qsplit}/q={query}: {seen} != {want}"
        for dataset, method, ph, seed, qsplit, query, seen, want in mismatches[:8]
    )
    if len(mismatches) > 8:
        details += f"; ... {len(mismatches) - 8} more"
    raise RuntimeError(
        "split_hash mismatch detected. "
        f"{details}. Re-run records for the current split_meta.json or pass "
        "--allow-split-hash-mismatch for legacy artifacts."
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


def _validate_complete(df: pd.DataFrame, data_dir: str, allow_partial: bool):
    if allow_partial or df.empty:
        return
    expected_cache: Dict[str, dict] = {}
    groups: Dict[tuple, set] = {}
    base_keys = set()
    for row in df.itertuples(index=False):
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
        groups.setdefault((*base_key, qsplit), set()).add(int(getattr(row, "query_node")))

    short = []
    for dataset, method, params_hash, seed in base_keys:
        for qsplit in ("val", "test"):
            want = expected_cache.get(dataset, {}).get(qsplit)
            if want is None:
                continue
            count = len(groups.get((dataset, method, params_hash, seed, qsplit), set()))
            if count != want:
                short.append((dataset, method, str(params_hash)[-12:], seed, qsplit, count, want))
    if not short:
        return
    details = "; ".join(
        f"{dataset}/{method}/{ph}/seed={seed}/{qsplit}: {count}/{want}"
        for dataset, method, ph, seed, qsplit, count, want in short[:12]
    )
    if len(short) > 12:
        details += f"; ... {len(short) - 12} more"
    raise RuntimeError(
        "partial records detected. "
        f"{details}. Re-run missing cells or pass --allow-partial."
    )


def _metric_labels(y_true, y_pred):
    return sorted(set(int(y) for y in y_true) | set(int(y) for y in y_pred))


def _bootstrap_scores(y_true, y_pred, bootstrap: int, seed: int = 0):
    if bootstrap <= 0 or len(y_true) == 0:
        return {}
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = _metric_labels(y_true, y_pred)
    samples = {"precision": [], "recall": [], "f1": [], "accuracy": []}
    n = len(y_true)
    for _ in range(bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        samples["precision"].append(precision_score(yt, yp, average="macro", zero_division=0, labels=labels))
        samples["recall"].append(recall_score(yt, yp, average="macro", zero_division=0, labels=labels))
        samples["f1"].append(f1_score(yt, yp, average="macro", zero_division=0, labels=labels))
        samples["accuracy"].append(accuracy_score(yt, yp))
    out = {}
    for metric, vals in samples.items():
        arr = np.asarray(vals, dtype=float)
        lo = float(np.percentile(arr, 2.5))
        hi = float(np.percentile(arr, 97.5))
        out[f"{metric}_ci_lo"] = lo
        out[f"{metric}_ci_hi"] = hi
        out[f"{metric}_boot_se"] = (hi - lo) / 4.0
    return out


def _classification_scores(chunk: pd.DataFrame, bootstrap: int) -> dict:
    valid = chunk.dropna(subset=["query_label", "predicted_label"])
    if valid.empty:
        return {
            "classification_n": 0,
            "precision_macro": math.nan,
            "recall_macro": math.nan,
            "f1_macro": math.nan,
            "accuracy": math.nan,
        }
    y_true = valid["query_label"].astype(int).to_numpy()
    y_pred = valid["predicted_label"].astype(int).to_numpy()
    labels = _metric_labels(y_true, y_pred)
    out = {
        "classification_n": int(len(valid)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    out.update(_bootstrap_scores(y_true, y_pred, bootstrap=bootstrap))
    return out


def _numeric_metric_columns(df: pd.DataFrame) -> List[str]:
    prefixes = (
        "quality_",
        "stat_",
    )
    explicit = {
        "size",
        "oracle_queries",
        "wall_time_s",
        "solver_wall_time_s",
        "soft_time_limit_s",
        "hard_time_limit_s",
        "optimality_gap",
        "bb_incumbent_obj",
        "bb_best_bound",
        "returncode",
    }
    skip = IDENTIFIER_COLS | set(FLAG_COLS)
    cols = []
    for col in df.columns:
        if col in skip or col.startswith("param_") or col.startswith("eval_"):
            continue
        if col in explicit or col.startswith(prefixes):
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().any():
                cols.append(col)
    return cols


def _config_columns(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.startswith("param_") or c.startswith("eval_")])


def _summarise_chunks(df: pd.DataFrame, group_cols: List[str], bootstrap: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    numeric_cols = _numeric_metric_columns(df)
    config_cols = _config_columns(df)
    rows = []
    for keys, chunk in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["params_json"] = chunk["params_json"].iloc[0] if "params_json" in chunk else "{}"
        row["n_records"] = int(len(chunk))
        row["n_queries"] = int(chunk["query_node"].nunique()) if "query_node" in chunk else int(len(chunk))
        row["split_hashes_json"] = json.dumps(sorted(str(x) for x in chunk["split_hash"].dropna().unique())) if "split_hash" in chunk else "[]"
        solver_build_ids = sorted(str(x) for x in chunk["solver_build_id"].dropna().unique()) if "solver_build_id" in chunk else []
        row["n_solver_build_ids"] = int(len(solver_build_ids))
        row["solver_build_ids_json"] = json.dumps(solver_build_ids)
        for col in config_cols:
            vals = chunk[col].dropna()
            row[col] = vals.iloc[0] if not vals.empty else np.nan
        for col in CATEGORICAL_COUNT_COLS:
            if col in chunk.columns:
                counts = chunk[col].dropna().astype(str).value_counts().sort_index().to_dict()
                row[f"{col}_counts_json"] = json.dumps(counts, sort_keys=True)
        for flag in FLAG_COLS:
            if flag in chunk.columns:
                vals = chunk[flag].dropna().astype(bool)
                row[f"{flag}_count"] = int(vals.sum()) if len(vals) else 0
                row[f"{flag}_rate"] = float(vals.mean()) if len(vals) else math.nan
        row.update(_classification_scores(chunk, bootstrap=bootstrap))
        for col in numeric_cols:
            vals = pd.to_numeric(chunk[col], errors="coerce").to_numpy(dtype=float)
            finite = vals[~np.isnan(vals)]
            prefix = _safe_col(col)
            if finite.size == 0:
                for suffix in ("mean", "std", "median", "q1", "q3", "min", "max"):
                    row[f"{prefix}_{suffix}"] = math.nan
                row[f"{prefix}_n_finite"] = 0
                continue
            row[f"{prefix}_mean"] = float(np.mean(finite))
            row[f"{prefix}_std"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            row[f"{prefix}_median"] = float(np.median(finite))
            row[f"{prefix}_q1"] = float(np.percentile(finite, 25))
            row[f"{prefix}_q3"] = float(np.percentile(finite, 75))
            row[f"{prefix}_min"] = float(np.min(finite))
            row[f"{prefix}_max"] = float(np.max(finite))
            row[f"{prefix}_n_finite"] = int(finite.size)
        rows.append(row)
    return pd.DataFrame(rows)


def _hard_subset_ids(dataset: str, data_dir: str) -> List[int]:
    meta_path = os.path.join(data_dir, dataset, "split_meta.json")
    nodes_csv = os.path.join(data_dir, dataset, "nodes.csv")
    edges_csv = os.path.join(data_dir, dataset, "edge.csv")
    if not all(os.path.exists(p) for p in (meta_path, nodes_csv, edges_csv)):
        return []
    df_nodes = pd.read_csv(nodes_csv)
    df_edges = pd.read_csv(edges_csv)
    test_ids = [int(q) for q in df_nodes[df_nodes["test"]]["node_id"].astype(int).tolist()]
    labels = df_nodes["label"].values
    train_mask = df_nodes["train"].values
    hard = compute_hard_subset(test_ids, build_out_adjacency(df_edges), train_mask, labels)
    with open(meta_path) as f:
        expected = (json.load(f).get("hard_subset") or {}).get("hash")
    if expected is not None and sha256_node_set(hard) != expected:
        raise RuntimeError(f"hard_subset hash mismatch for {dataset}: split_meta.json drift")
    return sorted(int(q) for q in hard)


def _classification_grid(df: pd.DataFrame, data_dir: str, bootstrap: int) -> pd.DataFrame:
    rows = []
    datasets = sorted(str(d) for d in df["dataset"].dropna().unique()) if "dataset" in df else []
    subset_frames = []
    for split in ("val", "test"):
        split_df = df[df["query_split"] == split].copy()
        if not split_df.empty:
            split_df["subset"] = split
            subset_frames.append(split_df)
    for dataset in datasets:
        hard_ids = _hard_subset_ids(dataset, data_dir)
        if not hard_ids:
            continue
        hard_df = df[
            (df["dataset"] == dataset)
            & (df["query_split"] == "test")
            & (df["query_node"].astype(int).isin(set(hard_ids)))
        ].copy()
        if not hard_df.empty:
            hard_df["subset"] = "bfs_depth1_wrong"
            subset_frames.append(hard_df)
    if not subset_frames:
        return pd.DataFrame()
    all_subsets = pd.concat(subset_frames, ignore_index=True)
    group_cols = ["dataset", "subset", "method", "params_hash", "seed"]
    summary = _summarise_chunks(all_subsets, group_cols, bootstrap=bootstrap)
    keep_prefixes = ("param_", "eval_")
    keep = [
        c
        for c in summary.columns
        if c in {
            "dataset",
            "subset",
            "method",
            "params_hash",
            "seed",
            "params_json",
            "n_records",
            "n_queries",
            "classification_n",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "accuracy",
            "precision_ci_lo",
            "precision_ci_hi",
            "precision_boot_se",
            "recall_ci_lo",
            "recall_ci_hi",
            "recall_boot_se",
            "f1_ci_lo",
            "f1_ci_hi",
            "f1_boot_se",
            "accuracy_ci_lo",
            "accuracy_ci_hi",
            "accuracy_boot_se",
        }
        or c.startswith(keep_prefixes)
    ]
    return summary[keep]


def _best_by_val_f1(classification_grid: pd.DataFrame) -> pd.DataFrame:
    if classification_grid.empty:
        return pd.DataFrame()
    val = classification_grid[classification_grid["subset"] == "val"].copy()
    if val.empty:
        return pd.DataFrame()
    rows = []
    key_cols = ["dataset", "method", "seed"]
    for keys, chunk in val.groupby(key_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        best = chunk.sort_values(
            ["f1_macro", "precision_macro", "recall_macro"],
            ascending=[False, False, False],
        ).iloc[0]
        row = {col: val_ for col, val_ in zip(key_cols, keys)}
        row["best_params_hash"] = best["params_hash"]
        row["best_params_json"] = best["params_json"]
        for col in classification_grid.columns:
            if col.startswith("param_") or col.startswith("eval_"):
                row[col] = best.get(col)
        for subset in ("val", "test", "bfs_depth1_wrong"):
            match = classification_grid[
                (classification_grid["dataset"] == row["dataset"])
                & (classification_grid["method"] == row["method"])
                & (classification_grid["seed"].fillna("__nan__") == pd.Series([row["seed"]]).fillna("__nan__").iloc[0])
                & (classification_grid["params_hash"] == row["best_params_hash"])
                & (classification_grid["subset"] == subset)
            ]
            if match.empty:
                continue
            m = match.iloc[0]
            for metric in ("precision_macro", "recall_macro", "f1_macro", "accuracy", "n_queries"):
                row[f"{subset}_{metric}"] = m.get(metric)
            for metric in ("precision", "recall", "f1", "accuracy"):
                row[f"{subset}_{metric}_ci_lo"] = m.get(f"{metric}_ci_lo")
                row[f"{subset}_{metric}_ci_hi"] = m.get(f"{metric}_ci_hi")
                row[f"{subset}_{metric}_boot_se"] = m.get(f"{metric}_boot_se")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=str, required=True, help="records.ndjson or a directory tree containing them")
    parser.add_argument("--output", type=str, required=True, help="Output directory for aggregate CSVs")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--allow-split-hash-mismatch", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    records = _load_records(args.records)
    if args.dataset:
        records = [r for r in records if r.get("dataset") == args.dataset]
    if not records:
        print(f"No records matched {args.records}")
        return

    _validate_split_hashes(records, args.data_dir, args.allow_split_hash_mismatch)
    flat = pd.DataFrame([_flatten_record(r) for r in records])
    _validate_complete(flat, args.data_dir, args.allow_partial)

    os.makedirs(args.output, exist_ok=True)

    flat_path = os.path.join(args.output, "records_flat.csv")
    flat.to_csv(flat_path, index=False)
    print(f"wrote {flat_path}")

    by_seed = _summarise_chunks(
        flat,
        ["dataset", "method", "params_hash", "query_split", "seed"],
        bootstrap=args.bootstrap,
    )
    by_seed_path = os.path.join(args.output, "cell_summary_by_seed.csv")
    by_seed.to_csv(by_seed_path, index=False)
    print(f"wrote {by_seed_path}")

    union = _summarise_chunks(
        flat,
        ["dataset", "method", "params_hash", "seed"],
        bootstrap=args.bootstrap,
    )
    union_path = os.path.join(args.output, "cell_summary_union_by_seed.csv")
    union.to_csv(union_path, index=False)
    print(f"wrote {union_path}")

    classification = _classification_grid(flat, args.data_dir, bootstrap=args.bootstrap)
    if not classification.empty:
        cls_path = os.path.join(args.output, "classification_grid_by_seed.csv")
        classification.to_csv(cls_path, index=False)
        print(f"wrote {cls_path}")
        best = _best_by_val_f1(classification)
        if not best.empty:
            best_path = os.path.join(args.output, "classification_best_by_val_f1.csv")
            best.to_csv(best_path, index=False)
            print(f"wrote {best_path}")


if __name__ == "__main__":
    main()
