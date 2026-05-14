"""Per-cell paired-bootstrap ties for classification metrics.

Reads cluster_quality records.ndjson (BFS d=1 grow-to-k, AvgDeg grow-to-k, BP)
and emits a CSV with per-cell macro P/R/F1, their bootstrap std, the section
best per (k, metric), the overall best across k, and per-cell flags marking
paired-bootstrap ties with the section/overall best (95% CI of the metric
difference contains zero).

Uses vectorized numpy to evaluate B bootstrap samples in batch for all cells
on shared resample indices, so paired differences are derived directly from
the same B draws.

Usage:

    python scripts/classification/paired_bootstrap_ties.py \\
        --records exps/classification/<dataset>/cluster_quality \\
        --output  exps/classification/<dataset>/agg/classification/ties_val.csv \\
        --dataset <dataset> \\
        --split   val
"""

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from record_io import load_ndjson_records  # noqa: E402


def _cell_key(record: dict) -> Optional[Tuple]:
    method = record.get("method")
    params = record.get("params") or {}
    if method == "bp":
        k = params.get("k")
        kappa = params.get("kappa")
        if k is None or kappa is None:
            return None
        return ("bp", int(k), int(kappa))
    if method == "bfs":
        depth = params.get("bfs_depth")
        k = params.get("k")
        if depth is None or k is None:
            return None
        return ("bfs", int(depth), int(k))
    if method == "avgdeg":
        k = params.get("k")
        if k is None:
            return None
        return ("avgdeg", int(k))
    return None


def _cell_k(cell: Tuple) -> int:
    if cell[0] == "bp":
        return cell[1]
    if cell[0] == "bfs":
        return cell[2]
    return cell[1]


def _cell_label(cell: Tuple) -> str:
    if cell[0] == "bp":
        return f"BP κ={cell[2]}"
    if cell[0] == "bfs":
        return f"BFS d={cell[1]}"
    return "AvgDeg"


def _collect(records: Iterable[dict], split: str,
             hard_ids: Optional[set] = None) -> Dict[Tuple, Dict[int, Tuple[int, int]]]:
    cells: Dict[Tuple, Dict[int, Tuple[int, int]]] = {}
    for record in records:
        if record.get("query_split") != split:
            continue
        seed = record.get("seed")
        if seed is not None and seed != 42:
            continue
        if record.get("query_label") is None:
            continue
        cell = _cell_key(record)
        if cell is None:
            continue
        query_node = int(record["query_node"])
        if hard_ids is not None and query_node not in hard_ids:
            continue
        truth = int(record["query_label"])
        pred = int(record["predicted_label"]) if record.get("predicted_label") is not None else -1
        cells.setdefault(cell, {})[query_node] = (truth, pred)
    return cells


def _macro_metrics_batch(yt: np.ndarray, yp: np.ndarray,
                         idx_batches: np.ndarray, classes: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised macro precision/recall/F1 across a batch of bootstrap
    resample index arrays.

    yt, yp: shape (n,). idx_batches: shape (B, n). classes: shape (C,).
    Returns three (B,) arrays for precision, recall, f1, macro-averaged.
    """
    P = np.zeros(idx_batches.shape[0], dtype=np.float64)
    R = np.zeros_like(P)
    F = np.zeros_like(P)
    for c in classes:
        is_t = (yt == c).astype(np.int32)
        is_p = (yp == c).astype(np.int32)
        tp_ind = is_t * is_p
        fp_ind = (1 - is_t) * is_p
        fn_ind = is_t * (1 - is_p)
        tp = tp_ind[idx_batches].sum(axis=1)
        fp = fp_ind[idx_batches].sum(axis=1)
        fn = fn_ind[idx_batches].sum(axis=1)
        denom_p = tp + fp
        denom_r = tp + fn
        denom_f = 2 * tp + fp + fn
        with np.errstate(invalid="ignore", divide="ignore"):
            P += np.where(denom_p > 0, tp / denom_p, 0.0)
            R += np.where(denom_r > 0, tp / denom_r, 0.0)
            F += np.where(denom_f > 0, (2 * tp) / denom_f, 0.0)
    C = len(classes)
    return P / C, R / C, F / C


def _macro_metrics(yt: np.ndarray, yp: np.ndarray,
                   classes: np.ndarray) -> Tuple[float, float, float]:
    P = R = F = 0.0
    for c in classes:
        is_t = (yt == c).astype(np.int32)
        is_p = (yp == c).astype(np.int32)
        tp = int((is_t * is_p).sum())
        fp = int(((1 - is_t) * is_p).sum())
        fn = int((is_t * (1 - is_p)).sum())
        P += (tp / (tp + fp)) if (tp + fp) else 0.0
        R += (tp / (tp + fn)) if (tp + fn) else 0.0
        F += ((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    C = len(classes)
    return P / C, R / C, F / C


def _resolve_hard_subset(dataset: str, data_dir: str) -> Optional[set]:
    from split_utils import build_out_adjacency, compute_hard_subset, sha256_node_set
    nodes_csv = os.path.join(data_dir, dataset, "nodes.csv")
    edge_csv = os.path.join(data_dir, dataset, "edge.csv")
    meta_path = os.path.join(data_dir, dataset, "split_meta.json")
    if not (os.path.exists(nodes_csv) and os.path.exists(edge_csv)):
        return None
    df_nodes = pd.read_csv(nodes_csv)
    df_edges = pd.read_csv(edge_csv)
    test_ids = df_nodes[df_nodes["test"]]["node_id"].astype(int).tolist()
    labels = df_nodes["label"].values
    train_mask = df_nodes["train"].values
    out_adj = build_out_adjacency(df_edges)
    hard = compute_hard_subset(test_ids, out_adj, train_mask, labels)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        expected = (meta.get("hard_subset") or {}).get("hash")
        if expected is not None and sha256_node_set(hard) != expected:
            raise RuntimeError(
                f"hard_subset hash mismatch for {dataset}: split_meta.json drift"
            )
    return set(int(q) for q in hard)


def _expected_query_count(dataset: str, data_dir: str, split: str,
                          hard_ids: Optional[set]) -> Optional[int]:
    if hard_ids is not None:
        return len(hard_ids)
    meta_path = os.path.join(data_dir, dataset, "split_meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    split_info = (meta.get("splits") or {}).get(split)
    if not split_info:
        return None
    return int(split_info["size"])


def _validate_split_hashes(records: Iterable[dict], dataset: str, data_dir: str,
                           split: str, allow_mismatch: bool):
    if allow_mismatch:
        return
    meta_path = os.path.join(data_dir, dataset, "split_meta.json")
    if not os.path.exists(meta_path):
        return
    with open(meta_path) as f:
        splits = (json.load(f).get("splits") or {})
    expected_hash = (splits.get(split) or {}).get("hash")
    if not expected_hash:
        return
    mismatches = []
    for record in records:
        if record.get("query_split") != split:
            continue
        if record.get("split_hash") != expected_hash:
            mismatches.append(
                (
                    record.get("method"),
                    str(record.get("params_hash"))[-12:],
                    record.get("seed"),
                    record.get("query_node"),
                    record.get("split_hash"),
                    expected_hash,
                )
            )
    if mismatches:
        details = "; ".join(
            f"{method}/{ph}/seed={seed}/q={query}: {seen} != {want}"
            for method, ph, seed, query, seen, want in mismatches[:5]
        )
        if len(mismatches) > 5:
            details += f"; ... {len(mismatches) - 5} more"
        raise RuntimeError(
            "split_hash mismatch detected. "
            f"{details}. Re-run records for the current split_meta.json "
            "or pass --allow-split-hash-mismatch for legacy artifacts."
        )


def _load_records(records_path: str, dataset: Optional[str]) -> List[dict]:
    if os.path.isdir(records_path):
        records: List[dict] = []
        for root, _dirs, files in os.walk(records_path):
            for fn in files:
                if fn == "records.ndjson":
                    records.extend(load_ndjson_records(os.path.join(root, fn)))
    else:
        if not os.path.exists(records_path):
            raise FileNotFoundError(records_path)
        records = load_ndjson_records(records_path)
    if dataset is not None:
        records = [r for r in records if r.get("dataset") == dataset]
    return records


def compute_table(records: List[dict], split: str, dataset: str, data_dir: str,
                  hard: bool, bootstrap: int, rng_seed: int = 42,
                  allow_partial: bool = False) -> pd.DataFrame:
    hard_ids: Optional[set] = None
    if hard:
        hard_ids = _resolve_hard_subset(dataset, data_dir)
    cells = _collect(records, split=split, hard_ids=hard_ids)
    if not cells:
        return pd.DataFrame()
    common = sorted(set.intersection(*[set(d) for d in cells.values()]))
    if not common:
        return pd.DataFrame()
    n_queries = len(common)
    expected_n = _expected_query_count(dataset, data_dir, split, hard_ids)
    if expected_n is not None and not allow_partial:
        short_cells = {
            f"{_cell_label(cell)} k={_cell_k(cell)}": len(rows)
            for cell, rows in cells.items()
            if len(rows) != expected_n
        }
        if n_queries != expected_n or short_cells:
            details = ", ".join(
                f"{label}={count}/{expected_n}" for label, count in sorted(short_cells.items())
            )
            if not details:
                details = f"intersection={n_queries}/{expected_n}"
            raise RuntimeError(
                "partial classification table input: "
                f"split={split} subset={'hard' if hard else split}; {details}. "
                "Re-run the missing cells or pass --allow-partial."
            )

    aligned: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}
    yt_ref: Optional[np.ndarray] = None
    for cell, rows in cells.items():
        yt = np.array([rows[q][0] for q in common], dtype=np.int64)
        yp = np.array([rows[q][1] for q in common], dtype=np.int64)
        aligned[cell] = (yt, yp)
        if yt_ref is None:
            yt_ref = yt
        elif not np.array_equal(yt_ref, yt):
            raise RuntimeError("true labels differ across cells for the same query node")

    classes = np.unique(np.concatenate([yt_ref] + [yp for _, yp in aligned.values()]))

    rng = np.random.default_rng(rng_seed)
    idx_batches = rng.integers(0, n_queries, size=(bootstrap, n_queries))

    samples: Dict[Tuple, Dict[str, np.ndarray]] = {}
    observed: Dict[Tuple, Dict[str, float]] = {}
    for cell, (yt, yp) in aligned.items():
        p_arr, r_arr, f_arr = _macro_metrics_batch(yt, yp, idx_batches, classes)
        samples[cell] = {"precision": p_arr, "recall": r_arr, "f1": f_arr}
        Pobs, Robs, Fobs = _macro_metrics(yt, yp, classes)
        observed[cell] = {"precision": Pobs, "recall": Robs, "f1": Fobs}

    rows: List[dict] = []
    for cell in aligned:
        rows.append({
            "dataset": dataset,
            "split": split,
            "subset": "hard" if hard else split,
            "method": cell[0],
            "cell_label": _cell_label(cell),
            "k": _cell_k(cell),
            "kappa": cell[2] if cell[0] == "bp" else None,
            "bfs_depth": cell[1] if cell[0] == "bfs" else None,
            "n_queries": n_queries,
            "precision": observed[cell]["precision"],
            "precision_std": float(samples[cell]["precision"].std(ddof=1)) if bootstrap > 1 else 0.0,
            "recall": observed[cell]["recall"],
            "recall_std": float(samples[cell]["recall"].std(ddof=1)) if bootstrap > 1 else 0.0,
            "f1": observed[cell]["f1"],
            "f1_std": float(samples[cell]["f1"].std(ddof=1)) if bootstrap > 1 else 0.0,
        })
    df = pd.DataFrame(rows)

    for metric in ("precision", "recall", "f1"):
        section_best: Dict[int, Tuple] = {}
        for k_val in sorted(df["k"].unique()):
            section_cells = [c for c in aligned if _cell_k(c) == k_val]
            section_best[k_val] = max(section_cells, key=lambda c: observed[c][metric])
        overall_best = max(aligned.keys(), key=lambda c: observed[c][metric])

        section_best_flags = []
        overall_best_flags = []
        section_tied_flags = []
        overall_tied_flags = []
        diff_lo_section = []
        diff_hi_section = []
        diff_lo_overall = []
        diff_hi_overall = []

        for _, r in df.iterrows():
            cell = _row_cell(r)
            k_val = int(r["k"])
            best_sec = section_best[k_val]
            best_all = overall_best
            section_best_flags.append(cell == best_sec)
            overall_best_flags.append(cell == best_all)

            if cell == best_sec:
                section_tied_flags.append(True)
                diff_lo_section.append(0.0)
                diff_hi_section.append(0.0)
            else:
                diff = samples[best_sec][metric] - samples[cell][metric]
                lo, hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
                section_tied_flags.append(lo <= 0 <= hi)
                diff_lo_section.append(lo)
                diff_hi_section.append(hi)

            if cell == best_all:
                overall_tied_flags.append(True)
                diff_lo_overall.append(0.0)
                diff_hi_overall.append(0.0)
            else:
                diff = samples[best_all][metric] - samples[cell][metric]
                lo, hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
                overall_tied_flags.append(lo <= 0 <= hi)
                diff_lo_overall.append(lo)
                diff_hi_overall.append(hi)

        df[f"{metric}__section_best"] = section_best_flags
        df[f"{metric}__overall_best"] = overall_best_flags
        df[f"{metric}__section_tied"] = section_tied_flags
        df[f"{metric}__overall_tied"] = overall_tied_flags
        df[f"{metric}__diff_lo_section"] = diff_lo_section
        df[f"{metric}__diff_hi_section"] = diff_hi_section
        df[f"{metric}__diff_lo_overall"] = diff_lo_overall
        df[f"{metric}__diff_hi_overall"] = diff_hi_overall

    return df.sort_values(["k", "method", "kappa", "bfs_depth"], na_position="first").reset_index(drop=True)


def _row_cell(row: pd.Series) -> Tuple:
    if row["method"] == "bp":
        return ("bp", int(row["k"]), int(row["kappa"]))
    if row["method"] == "bfs":
        return ("bfs", int(row["bfs_depth"]), int(row["k"]))
    return ("avgdeg", int(row["k"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--hard", action="store_true",
                        help="Restrict test queries to the BFS-hard subset (split=test only).")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow incomplete cells and aggregate only their query intersection.",
    )
    parser.add_argument(
        "--allow-split-hash-mismatch",
        action="store_true",
        help="Allow legacy records whose split_hash differs from current split_meta.json.",
    )
    args = parser.parse_args()

    if args.hard and args.split != "test":
        raise SystemExit("--hard requires --split test")

    records = _load_records(args.records, args.dataset)
    if not records:
        raise SystemExit("no records matched")
    _validate_split_hashes(
        records,
        args.dataset,
        args.data_dir,
        args.split,
        args.allow_split_hash_mismatch,
    )

    df = compute_table(records, split=args.split, dataset=args.dataset,
                       data_dir=args.data_dir, hard=args.hard,
                       bootstrap=args.bootstrap, rng_seed=args.rng_seed,
                       allow_partial=args.allow_partial)
    if df.empty:
        raise SystemExit("no cells produced")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output}  rows={len(df)} n_queries={int(df['n_queries'].iloc[0])}")
    summary_cols = ["cell_label", "k", "n_queries",
                    "precision", "precision_std",
                    "recall", "recall_std",
                    "f1", "f1_std",
                    "f1__section_best", "f1__section_tied",
                    "f1__overall_best", "f1__overall_tied"]
    print(df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
