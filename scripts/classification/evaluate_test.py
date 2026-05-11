"""Test-set evaluation with paired bootstrap CI for AvgDeg, BFS, and BP.

Reads best_settings_<family>.json from tune_val.py, runs the solver on
eligible.test (or hard_subset if --subset bfs_depth1_wrong), bootstraps a
95 percent CI per metric, and writes per-seed plus aggregate CSVs under

    exps/classification/<dataset>/evaluate_test/aggregate.csv
    exps/classification/<dataset>/evaluate_test/per_seed.csv
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils import resample

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from solver_utils import (  # noqa: E402
    build_graph_context,
    effective_params,
    evaluate_nodes,
    method_extra_args,
)
from split_utils import (  # noqa: E402
    assert_split_meta_matches,
    build_out_adjacency,
    compute_hard_subset,
    sha256_node_set,
)


SEEDS = [42, 43, 44, 45, 46]
DETERMINISTIC = {"avgdeg", "bfs"}


def _macro_scores(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def _bootstrap_ci(y_true, y_pred, B: int = 500, rng_seed: int = 0, indices_per_replicate=None):
    """Standard bootstrap CI. When indices_per_replicate is supplied (list of
    arrays length B), use those indices for the b-th replicate so multiple
    methods reuse the same paired-by-query resamples."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return {"precision": (np.nan, np.nan), "recall": (np.nan, np.nan), "f1": (np.nan, np.nan)}
    precisions, recalls, f1s = [], [], []
    indices = np.arange(n)
    for b in range(B):
        if indices_per_replicate is not None:
            idx = indices_per_replicate[b]
        else:
            idx = resample(indices, replace=True, n_samples=n, random_state=rng_seed + b)
        yt = y_true[idx]
        yp = y_pred[idx]
        precisions.append(precision_score(yt, yp, average="macro", zero_division=0))
        recalls.append(recall_score(yt, yp, average="macro", zero_division=0))
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
    return _ci_from_samples(precisions, recalls, f1s, n)


def _ci_from_samples(precisions, recalls, f1s, n_total):
    """Convert bootstrap sample lists to a 95 percent CI dict. When the metric
    is constant across replicates (typical on the hard subset when one method
    is locked at zero F1), substitute the Wilson interval on n_total Bernoulli
    trials with success rate equal to the point estimate."""
    out = {}
    for name, samples in (("precision", precisions), ("recall", recalls), ("f1", f1s)):
        arr = np.asarray(samples)
        lo = float(np.percentile(arr, 2.5))
        hi = float(np.percentile(arr, 97.5))
        if lo == hi and n_total > 0:
            lo, hi = _wilson_interval(float(arr[0]), n_total)
        out[name] = (lo, hi)
    return out


def _wilson_interval(p_hat: float, n: int, z: float = 1.959963984540054) -> tuple:
    """95 percent Wilson score interval for a proportion p_hat over n trials."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p_hat = max(0.0, min(1.0, float(p_hat)))
    denom = 1.0 + (z * z) / n
    centre = (p_hat + (z * z) / (2 * n)) / denom
    half = (z * np.sqrt((p_hat * (1 - p_hat) / n) + (z * z) / (4 * n * n))) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def _paired_indices(n: int, B: int, rng_seed: int):
    """B arrays of indices in [0, n), used as the common resample for every
    method so the i-th sample maps to the same query across methods."""
    rng = np.random.default_rng(rng_seed)
    return [rng.integers(0, n, size=n) for _ in range(B)]


def _stratified_indices(per_seed_n: int, n_seeds: int, B: int, rng_seed: int):
    """Stratified resample for BP pooled CI. For each replicate, draw
    per_seed_n indices per seed independently (so the variance from seed
    randomness is preserved rather than artificially shrunk by sqrt(n_seeds))."""
    rng = np.random.default_rng(rng_seed)
    replicates = []
    for _ in range(B):
        per_seed = []
        for s in range(n_seeds):
            per_seed.append(rng.integers(s * per_seed_n, (s + 1) * per_seed_n, size=per_seed_n))
        replicates.append(np.concatenate(per_seed))
    return replicates


def _load_best(out_root: str, family: str):
    path = os.path.join(out_root, f"best_settings_{family}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        return json.load(f)




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--bin-path", type=str, default="./solver/bin/solver")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--exps-dir", type=str, default="exps")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument(
        "--families",
        type=str,
        default="avgdeg,bfs,bp",
        help="comma-separated families to evaluate",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="test",
        choices=["test", "bfs_depth1_wrong"],
    )
    parser.add_argument("--weighting", type=str, default="uniform")
    parser.add_argument("--max-fallback-hops", type=int, default=10)
    parser.add_argument("--keep-solver-dumps", action="store_true")
    args = parser.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]

    df_nodes = pd.read_csv(os.path.join(args.data_dir, args.dataset, "nodes.csv"))
    df_edges = pd.read_csv(os.path.join(args.data_dir, args.dataset, "edge.csv"))
    meta = assert_split_meta_matches(args.dataset, df_nodes, df_edges, args.data_dir)

    if args.subset == "test":
        target_nodes = [int(q) for q in df_nodes[df_nodes["test"]]["node_id"].astype(int).tolist()]
        split_hash = meta.splits["test"]["hash"]
        query_split = "test"
    else:
        if meta.hard_subset is None:
            raise ValueError("split_meta.json carries no hard_subset; rerun prepare_data")
        out_adj = build_out_adjacency(df_edges)
        val_ids = df_nodes[df_nodes["val"]]["node_id"].astype(int).tolist()
        hard = compute_hard_subset(
            val_ids,
            out_adj,
            df_nodes["train"].values,
            df_nodes["label"].values,
        )
        if sha256_node_set(hard) != meta.hard_subset["hash"]:
            raise ValueError("hard_subset hash mismatch with split_meta.json")
        target_nodes = hard
        split_hash = meta.hard_subset["hash"]
        query_split = "val"

    forbidden = set(
        df_nodes[df_nodes["val"] | df_nodes["test"]]["node_id"].astype(int).tolist()
    )

    print(
        f"{args.dataset}: subset={args.subset}, |target|={len(target_nodes)}, |forbidden|={len(forbidden)}"
    )

    ctx = build_graph_context(
        os.path.join(args.data_dir, args.dataset, "edge.csv"), max_in_edges=0
    )

    tune_root = os.path.join(args.exps_dir, "classification", args.dataset, "tune_val")
    out_root = os.path.join(args.exps_dir, "classification", args.dataset, "evaluate_test")
    os.makedirs(out_root, exist_ok=True)

    per_seed_rows = []
    aggregate_rows = []

    # Paired-by-query bootstrap: build one shared set of resample indices over
    # the query axis and reuse it for every (method, seed) so the b-th replicate
    # samples the same queries everywhere. evaluate_nodes sorts results by
    # query_node, so position i maps to the same query across methods.
    paired_idx = _paired_indices(len(target_nodes), args.bootstrap, rng_seed=0)

    method_state = {}
    for family in families:
        best = _load_best(tune_root, family)
        k = best.get("k")
        params = best.get("params") or {}
        p_hash = best["params_hash"]
        # Recompute the effective params hash so cached records bound to the
        # eval signature in tune_val are reused if and only if eval bits match.
        eff_params, eff_hash = effective_params(
            params,
            weighting=args.weighting,
            max_fallback_hops=args.max_fallback_hops,
            forbidden_nodes=forbidden,
        )
        if eff_hash != p_hash:
            print(
                f"[{args.dataset}] {family}: params_hash drift {p_hash[-12:]} -> {eff_hash[-12:]}; "
                "tune_val and evaluate_test eval signatures differ. Using effective hash."
            )
        method_preds = {}
        method_truth = None
        seed_aux = {}
        for seed in SEEDS:
            if family in DETERMINISTIC and seed != SEEDS[0] and method_truth is not None:
                method_preds[seed] = method_preds[SEEDS[0]]
                seed_aux[seed] = seed_aux[SEEDS[0]]
                continue
            extra = method_extra_args(family, params, gurobi_seed=seed if family == "bp" else None)
            records_dir = os.path.join(
                out_root, family, args.subset, eff_hash[-12:], f"seed_{seed}"
            )
            os.makedirs(records_dir, exist_ok=True)
            with tempfile.TemporaryDirectory() as td:
                _, y_true, y_pred, stats = evaluate_nodes(
                    target_nodes,
                    k=k,
                    edge_csv=os.path.join(args.data_dir, args.dataset, "edge.csv"),
                    df_nodes=df_nodes,
                    bin_path=args.bin_path,
                    tmp_dir=td,
                    max_workers=args.max_workers,
                    extra_args=extra,
                    weighting=args.weighting,
                    max_fallback_hops=args.max_fallback_hops,
                    forbidden_nodes=forbidden,
                    graph_context=ctx,
                    records_path=records_dir,
                    dataset_name=args.dataset,
                    seed=seed if family == "bp" else None,
                    method=family,
                    params=eff_params,
                    split_hash=split_hash,
                    keep_solver_dumps=args.keep_solver_dumps,
                    query_split=query_split,
                    collect_stats=True,
                    compute_qualities=True,
                    show_progress=False,
                    return_query_nodes=True,
                )
            method_truth = y_true
            method_preds[seed] = y_pred
            seed_aux[seed] = {
                "fallback_rate": stats["fallback_rate"],
                "avg_oracle_queries": stats.get("avg_oracle_queries"),
            }
            scores = _macro_scores(y_true, y_pred)
            ci = _bootstrap_ci(
                y_true, y_pred, B=args.bootstrap, rng_seed=seed,
                indices_per_replicate=paired_idx,
            )
            per_seed_rows.append(
                {
                    "dataset": args.dataset,
                    "subset": args.subset,
                    "method": family,
                    "seed": seed,
                    "params_hash": eff_hash,
                    "params_json": json.dumps(params, sort_keys=True),
                    "effective_params_json": json.dumps(eff_params, sort_keys=True),
                    "split_hash": split_hash,
                    "precision": scores["precision"],
                    "recall": scores["recall"],
                    "f1": scores["f1"],
                    "precision_ci_lo": ci["precision"][0],
                    "precision_ci_hi": ci["precision"][1],
                    "recall_ci_lo": ci["recall"][0],
                    "recall_ci_hi": ci["recall"][1],
                    "f1_ci_lo": ci["f1"][0],
                    "f1_ci_hi": ci["f1"][1],
                    "fallback_rate": stats["fallback_rate"],
                    "fallback_rate_flag": stats["fallback_rate"] > 10.0,
                    "avg_oracle_queries": stats.get("avg_oracle_queries"),
                }
            )
        method_state[family] = {
            "method_truth": method_truth,
            "method_preds": method_preds,
            "seed_aux": seed_aux,
            "p_hash": eff_hash,
            "params": params,
            "eff_params": eff_params,
        }

        precisions, recalls, f1s = [], [], []
        for seed in SEEDS:
            scores = _macro_scores(method_truth, method_preds[seed])
            precisions.append(scores["precision"])
            recalls.append(scores["recall"])
            f1s.append(scores["f1"])

        if family in DETERMINISTIC:
            # Deterministic methods produce identical predictions across seeds;
            # pooling 5 copies shrinks the CI by sqrt(5) without adding signal.
            # Report the single-seed CI computed above for SEEDS[0].
            single_ci = _bootstrap_ci(
                method_truth, method_preds[SEEDS[0]],
                B=args.bootstrap, rng_seed=SEEDS[0],
                indices_per_replicate=paired_idx,
            )
            pooled_ci = single_ci
            ci_kind = "single_seed_paired"
        else:
            # BP: stratified bootstrap over the seed axis preserves the across-
            # seed variance instead of treating 5 seeds as 5N independent draws.
            n_per_seed = len(method_truth)
            strat_idx = _stratified_indices(n_per_seed, len(SEEDS), args.bootstrap, rng_seed=0)
            yt_stack = np.concatenate([np.asarray(method_truth) for _ in SEEDS])
            yp_stack = np.concatenate([np.asarray(method_preds[s]) for s in SEEDS])
            pooled_ci = _bootstrap_ci(
                yt_stack, yp_stack,
                B=args.bootstrap, rng_seed=0,
                indices_per_replicate=strat_idx,
            )
            ci_kind = "stratified_seed_pooled"
        mean_fallback = float(np.mean([seed_aux[s]["fallback_rate"] for s in SEEDS]))
        aggregate_rows.append(
            {
                "dataset": args.dataset,
                "subset": args.subset,
                "method": family,
                "params_hash": eff_hash,
                "params_json": json.dumps(params, sort_keys=True),
                "split_hash": split_hash,
                "ci_kind": ci_kind,
                "n_seeds": len(SEEDS),
                "n_queries": len(method_truth),
                "precision_mean": float(np.mean(precisions)),
                "recall_mean": float(np.mean(recalls)),
                "f1_mean": float(np.mean(f1s)),
                "precision_std": float(np.std(precisions, ddof=0)),
                "recall_std": float(np.std(recalls, ddof=0)),
                "f1_std": float(np.std(f1s, ddof=0)),
                "precision_pooled_ci_lo": pooled_ci["precision"][0],
                "precision_pooled_ci_hi": pooled_ci["precision"][1],
                "recall_pooled_ci_lo": pooled_ci["recall"][0],
                "recall_pooled_ci_hi": pooled_ci["recall"][1],
                "f1_pooled_ci_lo": pooled_ci["f1"][0],
                "f1_pooled_ci_hi": pooled_ci["f1"][1],
                "fallback_rate_mean": mean_fallback,
                "fallback_rate_flag": mean_fallback > 10.0,
            }
        )

    per_seed_path = os.path.join(out_root, f"per_seed_{args.subset}.csv")
    aggregate_path = os.path.join(out_root, f"aggregate_{args.subset}.csv")
    pd.DataFrame(per_seed_rows).to_csv(per_seed_path, index=False)
    pd.DataFrame(aggregate_rows).to_csv(aggregate_path, index=False)
    print(f"wrote {per_seed_path}")
    print(f"wrote {aggregate_path}")


if __name__ == "__main__":
    main()
