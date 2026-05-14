"""Classification aggregator. Reads records.ndjson (emitted by the unified
sweep) and derives the classification perspective without re-running the
solver:

  1. For each method, pick the params_hash that maximises macro F1 on
     query_split == 'val' records.
  2. At the best params_hash, compute per-seed macro precision/recall/F1 on
     query_split == 'test' records, with paired-by-query bootstrap CI shared
     across methods.
  3. Optionally restrict the test pool to the hard subset (test records where
     the BFS depth-1 vote was wrong) via --subset bfs_depth1_wrong.

Deterministic methods (avgdeg, bfs) report a single-seed paired CI; BP uses
a stratified-by-seed bootstrap so the across-seed variance is preserved.
Wilson interval substitutes when bootstrap variance is zero.

Output:

    classification_best_settings_<family>.json   -- chosen params + scores
    classification_per_seed_<subset>.csv         -- one row per (method, seed)
    classification_aggregate_<subset>.csv        -- one row per method
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from record_io import load_ndjson_records  # noqa: E402


SEEDS = [42, 43, 44, 45, 46]
DETERMINISTIC = {"avgdeg", "bfs"}


def _ordered_seed_keys(fam: str, by_seed: Dict[Optional[int], List[dict]]) -> List[Optional[int]]:
    """Return the actual seed keys present for a method in stable order.

    Deterministic methods store records under seed=None and should contribute a
    single row, not one synthetic row per BP seed slot.
    """
    if fam in DETERMINISTIC:
        if None in by_seed:
            return [None]
        return sorted(by_seed, key=lambda s: (-1 if s is None else int(s)))[:1]

    preferred = [s for s in SEEDS if s in by_seed]
    extras = sorted(s for s in by_seed if s is not None and s not in SEEDS)
    return preferred + extras


def _primary_seed_key(fam: str, by_seed: Dict[Optional[int], List[dict]]) -> Optional[int]:
    keys = _ordered_seed_keys(fam, by_seed)
    if not keys:
        return None
    return keys[0]


def _metric_labels(y_true, y_pred):
    return sorted(set(y_true) | set(y_pred))


def _macro_scores(y_true, y_pred, labels=None):
    kwargs = {"average": "macro", "zero_division": 0}
    if labels is not None:
        kwargs["labels"] = labels
    return {
        "precision": precision_score(y_true, y_pred, **kwargs),
        "recall": recall_score(y_true, y_pred, **kwargs),
        "f1": f1_score(y_true, y_pred, **kwargs),
    }


def _wilson_interval(p_hat: float, n: int, z: float = 1.959963984540054):
    if n <= 0:
        return (math.nan, math.nan)
    p_hat = max(0.0, min(1.0, float(p_hat)))
    denom = 1.0 + (z * z) / n
    centre = (p_hat + (z * z) / (2 * n)) / denom
    half = (z * math.sqrt((p_hat * (1 - p_hat) / n) + (z * z) / (4 * n * n))) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def _ci_from_samples(precisions, recalls, f1s, n_total):
    out = {}
    for name, samples in (("precision", precisions), ("recall", recalls), ("f1", f1s)):
        arr = np.asarray(samples, dtype=float)
        lo = float(np.percentile(arr, 2.5))
        hi = float(np.percentile(arr, 97.5))
        if lo == hi and n_total > 0:
            lo, hi = _wilson_interval(float(arr[0]), n_total)
        out[name] = (lo, hi)
    return out


def _paired_indices(n: int, B: int, rng_seed: int):
    rng = np.random.default_rng(rng_seed)
    return [rng.integers(0, n, size=n) for _ in range(B)]


def _stratified_indices(per_seed_n: int, n_seeds: int, B: int, rng_seed: int):
    rng = np.random.default_rng(rng_seed)
    out = []
    for _ in range(B):
        chunks = []
        for s in range(n_seeds):
            chunks.append(rng.integers(s * per_seed_n, (s + 1) * per_seed_n, size=per_seed_n))
        out.append(np.concatenate(chunks))
    return out


def _bootstrap_ci(y_true, y_pred, indices_per_replicate, labels=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return {"precision": (math.nan, math.nan), "recall": (math.nan, math.nan), "f1": (math.nan, math.nan)}
    precisions, recalls, f1s = [], [], []
    for idx in indices_per_replicate:
        yt = y_true[idx]
        yp = y_pred[idx]
        precisions.append(precision_score(yt, yp, average="macro", zero_division=0, labels=labels))
        recalls.append(recall_score(yt, yp, average="macro", zero_division=0, labels=labels))
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0, labels=labels))
    return _ci_from_samples(precisions, recalls, f1s, n)


def _records_by_method_seed(records, query_split: str) -> Dict[str, Dict[str, Dict[Optional[int], List[dict]]]]:
    """Group records by method -> params_hash -> seed -> list (ordered by query_node)."""
    grouped: Dict[str, Dict[str, Dict[Optional[int], List[dict]]]] = {}
    for r in records:
        if r.get("query_split") != query_split:
            continue
        m = r.get("method")
        ph = r.get("params_hash")
        seed = r.get("seed")
        grouped.setdefault(m, {}).setdefault(ph, {}).setdefault(seed, []).append(r)
    # Sort each list by query_node so paired CI lines up.
    for m in grouped:
        for ph in grouped[m]:
            for seed in grouped[m][ph]:
                grouped[m][ph][seed].sort(key=lambda r: int(r.get("query_node") or 0))
    return grouped


def _pick_best_params_by_val_f1(records) -> Dict[str, str]:
    """For each family return the params_hash whose val records maximise macro F1.
    Deterministic methods use seed=None records; BP uses seed=SEEDS[0] (42)."""
    val_grouped = _records_by_method_seed(records, "val")
    best_for: Dict[str, str] = {}
    for fam, by_hash in val_grouped.items():
        best_hash = None
        best_f1 = -1.0
        for ph, by_seed in by_hash.items():
            seed_key = _primary_seed_key(fam, by_seed)
            recs = by_seed.get(seed_key) if seed_key in by_seed else None
            if not recs:
                continue
            y_true = [int(r.get("query_label")) for r in recs if r.get("query_label") is not None]
            y_pred = [int(r.get("predicted_label")) if r.get("predicted_label") is not None else -1 for r in recs if r.get("query_label") is not None]
            if not y_true:
                continue
            f1 = f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
                labels=_metric_labels(y_true, y_pred),
            )
            if f1 > best_f1:
                best_f1 = f1
                best_hash = ph
        if best_hash is not None:
            best_for[fam] = best_hash
    return best_for


def _hard_subset(records, dataset=None, data_dir="data") -> List[int]:
    """Canonical hard subset: test nodes whose plain depth-1 BFS majority vote
    over 1-hop train neighbours (no grow-to-k) disagrees with the true label.
    When dataset is given, recompute from data/<dataset>/{nodes.csv, edge.csv}
    and verify against split_meta.json's hash. Otherwise derive from BFS d=1
    records (non-canonical fallback)."""
    if dataset is not None:
        meta_path = os.path.join(data_dir, dataset, "split_meta.json")
        nodes_csv = os.path.join(data_dir, dataset, "nodes.csv")
        edges_csv = os.path.join(data_dir, dataset, "edge.csv")
        if all(os.path.exists(p) for p in (meta_path, nodes_csv, edges_csv)):
            from split_utils import build_out_adjacency, compute_hard_subset, sha256_node_set
            df_nodes = pd.read_csv(nodes_csv)
            df_edges = pd.read_csv(edges_csv)
            test_ids = [int(q) for q in df_nodes[df_nodes["test"]]["node_id"].astype(int).tolist()]
            labels = df_nodes["label"].values
            train_mask = df_nodes["train"].values
            out_adj = build_out_adjacency(df_edges)
            hard = compute_hard_subset(test_ids, out_adj, train_mask, labels)
            with open(meta_path) as f:
                meta = json.load(f)
            expected = (meta.get("hard_subset") or {}).get("hash")
            if expected is not None and sha256_node_set(hard) != expected:
                raise RuntimeError(
                    f"hard_subset hash mismatch for {dataset}: split_meta.json drift"
                )
            return sorted(int(n) for n in hard)
    candidates = [r for r in records if r.get("method") == "bfs" and r.get("query_split") == "test"]
    if not candidates:
        return []
    pool = []
    for r in candidates:
        params = r.get("params") or {}
        if int(params.get("bfs_depth", 0)) == 1:
            pool.append(r)
    if not pool:
        return []
    wrong = [int(r["query_node"]) for r in pool if r.get("query_label") is not None and r.get("predicted_label") != r.get("query_label")]
    return sorted(set(wrong))


def _filter_test_records_to_subset(records, hard_ids):
    if not hard_ids:
        return records
    hard_set = set(int(n) for n in hard_ids)
    return [r for r in records if r.get("query_split") != "test" or int(r.get("query_node")) in hard_set]


def _expected_split_info(dataset: Optional[str], data_dir: str, subset: str, records) -> Dict[str, dict]:
    if dataset is None:
        return {}
    meta_path = os.path.join(data_dir, dataset, "split_meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        meta = json.load(f)
    splits = meta.get("splits") or {}
    out = {}
    if "val" in splits:
        out["val"] = {
            "size": int(splits["val"]["size"]),
            "hash": splits["val"].get("hash"),
        }
    if subset == "bfs_depth1_wrong":
        out["test"] = {
            "size": len(_hard_subset(records, dataset=dataset, data_dir=data_dir)),
            "hash": splits.get("test", {}).get("hash"),
        }
    elif "test" in splits:
        out["test"] = {
            "size": int(splits["test"]["size"]),
            "hash": splits["test"].get("hash"),
        }
    return out


def _validate_complete(
    records,
    dataset: Optional[str],
    data_dir: str,
    subset: str,
    allow_partial: bool,
    allow_split_hash_mismatch: bool,
):
    expected = _expected_split_info(dataset, data_dir, subset, records)
    if expected and not allow_split_hash_mismatch:
        mismatches = []
        for r in records:
            qsplit = r.get("query_split")
            expected_hash = (expected.get(qsplit) or {}).get("hash")
            if not expected_hash:
                continue
            seen_hash = r.get("split_hash")
            if seen_hash != expected_hash:
                mismatches.append(
                    (
                        r.get("method"),
                        str(r.get("params_hash"))[-12:],
                        r.get("seed"),
                        qsplit,
                        r.get("query_node"),
                        seen_hash,
                        expected_hash,
                    )
                )
        if mismatches:
            details = "; ".join(
                f"{method}/{ph}/seed={seed}/{qsplit}/q={query}: "
                f"{seen} != {want}"
                for method, ph, seed, qsplit, query, seen, want in mismatches[:5]
            )
            if len(mismatches) > 5:
                details += f"; ... {len(mismatches) - 5} more"
            raise RuntimeError(
                "split_hash mismatch detected. "
                f"{details}. Re-run records for the current split_meta.json "
                "or pass --allow-split-hash-mismatch for legacy artifacts."
            )

    if allow_partial:
        return
    if not expected:
        return
    groups: Dict[tuple, set] = {}
    base_keys = set()
    for r in records:
        qsplit = r.get("query_split")
        method_key = (r.get("method"), r.get("params_hash"), r.get("seed"))
        if qsplit in expected:
            base_keys.add(method_key)
        if qsplit not in expected:
            continue
        key = (*method_key, qsplit)
        groups.setdefault(key, set()).add(int(r["query_node"]))
    short = [
        (
            method,
            str(params_hash)[-12:],
            seed,
            qsplit,
            len(groups.get((method, params_hash, seed, qsplit), set())),
            want,
        )
        for (method, params_hash, seed) in base_keys
        for qsplit, info in expected.items()
        for want in [info["size"]]
        if len(groups.get((method, params_hash, seed, qsplit), set())) != want
    ]
    if short:
        details = "; ".join(
            f"{method}/{ph}/seed={seed}/{qsplit}: {count}/{want}"
            for method, ph, seed, qsplit, count, want in short[:10]
        )
        if len(short) > 10:
            details += f"; ... {len(short) - 10} more"
        raise RuntimeError(
            "partial classification records detected. "
            f"{details}. Re-run missing cells or pass --allow-partial."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--subset",
        type=str,
        default="test",
        choices=["test", "bfs_depth1_wrong"],
    )
    parser.add_argument("--bootstrap", type=int, default=500)
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

    records_path = args.records
    if os.path.isdir(records_path):
        gathered = []
        for root, _dirs, files in os.walk(records_path):
            for fn in files:
                if fn == "records.ndjson":
                    gathered.extend(load_ndjson_records(os.path.join(root, fn)))
        records = gathered
    else:
        records = load_ndjson_records(records_path)

    if args.dataset:
        records = [r for r in records if r.get("dataset") == args.dataset]
    if not records:
        print(f"No records under {records_path}")
        return

    if args.subset == "bfs_depth1_wrong":
        hard_ids = _hard_subset(records, dataset=args.dataset, data_dir=args.data_dir)
        if not hard_ids:
            print("No hard subset queries identified (no BFS depth-1 records with wrong label).")
            return
        records = _filter_test_records_to_subset(records, hard_ids)

    _validate_complete(
        records,
        args.dataset,
        args.data_dir,
        args.subset,
        args.allow_partial,
        args.allow_split_hash_mismatch,
    )

    best = _pick_best_params_by_val_f1(records)
    if not best:
        print("No val records found to pick best params from.")
        return

    test_grouped = _records_by_method_seed(records, "test")
    # Establish the intersection of test queries across validation-best methods
    # so paired bootstrap indices map to the same query for every method.
    common_test_qn = None
    for fam in best:
        ph = best[fam]
        by_seed = test_grouped.get(fam, {}).get(ph, {})
        seed_key = _primary_seed_key(fam, by_seed)
        recs = by_seed.get(seed_key) if seed_key in by_seed else None
        if recs is None:
            continue
        qns = {int(r["query_node"]) for r in recs}
        common_test_qn = qns if common_test_qn is None else (common_test_qn & qns)
    if common_test_qn is None:
        print("No test records found for any best params hash.")
        return
    common_test_qn = sorted(common_test_qn)
    if not common_test_qn:
        print("No common test queries found across validation-best methods.")
        return
    paired_idx = _paired_indices(len(common_test_qn), args.bootstrap, rng_seed=0)

    per_seed_rows = []
    aggregate_rows = []
    for fam, ph in best.items():
        by_seed = test_grouped.get(fam, {}).get(ph, {})
        if not by_seed:
            print(f"[{fam}] best params_hash {ph[-12:]} has no test records.")
            continue
        method_preds: Dict[Optional[int], List[int]] = {}
        method_truth = None
        seed_keys = _ordered_seed_keys(fam, by_seed)
        for seed in seed_keys:
            recs = by_seed.get(seed)
            if recs is None:
                continue
            recs_by_qn = {int(r["query_node"]): r for r in recs}
            ordered = [recs_by_qn[qn] for qn in common_test_qn if qn in recs_by_qn]
            y_true = [int(r["query_label"]) for r in ordered]
            y_pred = [int(r["predicted_label"]) if r["predicted_label"] is not None else -1 for r in ordered]
            labels_for_seed = _metric_labels(y_true, y_pred)
            method_truth = y_true
            method_preds[seed] = y_pred
            scores = _macro_scores(y_true, y_pred, labels=labels_for_seed)
            ci = _bootstrap_ci(y_true, y_pred, paired_idx, labels=labels_for_seed)
            per_seed_rows.append(
                {
                    "dataset": ordered[0].get("dataset") if ordered else args.dataset,
                    "subset": args.subset,
                    "method": fam,
                    "seed": seed,
                    "params_hash": ph,
                    "params_json": json.dumps(ordered[0].get("params") or {}, sort_keys=True) if ordered else "{}",
                    "precision": scores["precision"],
                    "recall": scores["recall"],
                    "f1": scores["f1"],
                    "precision_ci_lo": ci["precision"][0],
                    "precision_ci_hi": ci["precision"][1],
                    "recall_ci_lo": ci["recall"][0],
                    "recall_ci_hi": ci["recall"][1],
                    "f1_ci_lo": ci["f1"][0],
                    "f1_ci_hi": ci["f1"][1],
                    "n_queries": len(y_true),
                }
            )
        if method_truth is None:
            continue
        precisions, recalls, f1s = [], [], []
        for seed in seed_keys:
            yp = method_preds.get(seed)
            if yp is None:
                continue
            s = _macro_scores(method_truth, yp, labels=_metric_labels(method_truth, yp))
            precisions.append(s["precision"])
            recalls.append(s["recall"])
            f1s.append(s["f1"])
        present_seeds = [s for s in seed_keys if s in method_preds]
        if fam in DETERMINISTIC or len(present_seeds) <= 1:
            ci_kind = "single_seed_paired"
            single_yp = method_preds.get(present_seeds[0]) if present_seeds else next(iter(method_preds.values()))
            pooled_ci = _bootstrap_ci(
                method_truth,
                single_yp,
                paired_idx,
                labels=_metric_labels(method_truth, single_yp),
            )
        else:
            ci_kind = "stratified_seed_pooled"
            n_per_seed = len(method_truth)
            strat_idx = _stratified_indices(n_per_seed, len(present_seeds), args.bootstrap, rng_seed=0)
            yt_stack = np.concatenate([np.asarray(method_truth) for _ in present_seeds])
            yp_stack = np.concatenate([np.asarray(method_preds[s]) for s in present_seeds])
            pooled_ci = _bootstrap_ci(
                yt_stack,
                yp_stack,
                strat_idx,
                labels=_metric_labels(yt_stack.tolist(), yp_stack.tolist()),
            )
        aggregate_rows.append(
            {
                "dataset": args.dataset,
                "subset": args.subset,
                "method": fam,
                "params_hash": ph,
                "ci_kind": ci_kind,
                "n_seeds": len(precisions),
                "n_queries": len(method_truth),
                "precision_mean": float(np.mean(precisions)) if precisions else math.nan,
                "recall_mean": float(np.mean(recalls)) if recalls else math.nan,
                "f1_mean": float(np.mean(f1s)) if f1s else math.nan,
                "precision_pooled_ci_lo": pooled_ci["precision"][0],
                "precision_pooled_ci_hi": pooled_ci["precision"][1],
                "recall_pooled_ci_lo": pooled_ci["recall"][0],
                "recall_pooled_ci_hi": pooled_ci["recall"][1],
                "f1_pooled_ci_lo": pooled_ci["f1"][0],
                "f1_pooled_ci_hi": pooled_ci["f1"][1],
            }
        )
        out_best = {
            "dataset": args.dataset,
            "method": fam,
            "params_hash": ph,
            "subset": args.subset,
        }
        best_settings_path = os.path.join(args.output, f"classification_best_settings_{fam}_{args.subset}.json")
        os.makedirs(args.output, exist_ok=True)
        with open(best_settings_path, "w") as f:
            json.dump(out_best, f, indent=2, sort_keys=True)
        print(f"wrote {best_settings_path}")

    os.makedirs(args.output, exist_ok=True)
    per_seed_path = os.path.join(args.output, f"classification_per_seed_{args.subset}.csv")
    aggregate_path = os.path.join(args.output, f"classification_aggregate_{args.subset}.csv")
    pd.DataFrame(per_seed_rows).to_csv(per_seed_path, index=False)
    pd.DataFrame(aggregate_rows).to_csv(aggregate_path, index=False)
    print(f"wrote {per_seed_path}")
    print(f"wrote {aggregate_path}")


if __name__ == "__main__":
    main()
