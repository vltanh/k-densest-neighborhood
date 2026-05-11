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
    evaluate_nodes,
    method_extra_args,
    params_hash,
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


def _bootstrap_ci(y_true, y_pred, B: int = 500, rng_seed: int = 0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return {"precision": (np.nan, np.nan), "recall": (np.nan, np.nan), "f1": (np.nan, np.nan)}
    precisions, recalls, f1s = [], [], []
    indices = np.arange(n)
    for b in range(B):
        idx = resample(indices, replace=True, n_samples=n, random_state=rng_seed + b)
        yt = y_true[idx]
        yp = y_pred[idx]
        precisions.append(precision_score(yt, yp, average="macro", zero_division=0))
        recalls.append(recall_score(yt, yp, average="macro", zero_division=0))
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
    return {
        "precision": (float(np.percentile(precisions, 2.5)), float(np.percentile(precisions, 97.5))),
        "recall": (float(np.percentile(recalls, 2.5)), float(np.percentile(recalls, 97.5))),
        "f1": (float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))),
    }


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

    for family in families:
        best = _load_best(tune_root, family)
        k = best.get("k")
        params = best.get("params") or {}
        p_hash = best["params_hash"]
        method_preds = {}
        method_truth = None
        for seed in SEEDS:
            if family in DETERMINISTIC and seed != SEEDS[0] and method_truth is not None:
                method_preds[seed] = method_preds[SEEDS[0]]
                continue
            extra = method_extra_args(family, params, gurobi_seed=seed if family == "bp" else None)
            records_dir = os.path.join(
                out_root, family, args.subset, p_hash[-12:], f"seed_{seed}"
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
                    params=params,
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
            scores = _macro_scores(y_true, y_pred)
            ci = _bootstrap_ci(y_true, y_pred, B=args.bootstrap, rng_seed=seed)
            per_seed_rows.append(
                {
                    "dataset": args.dataset,
                    "subset": args.subset,
                    "method": family,
                    "seed": seed,
                    "params_hash": p_hash,
                    "params_json": json.dumps(params, sort_keys=True),
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
                    "avg_oracle_queries": stats.get("avg_oracle_queries"),
                }
            )

        precisions, recalls, f1s = [], [], []
        for seed in SEEDS:
            scores = _macro_scores(method_truth, method_preds[seed])
            precisions.append(scores["precision"])
            recalls.append(scores["recall"])
            f1s.append(scores["f1"])
        yt_stack = np.concatenate([np.asarray(method_truth) for _ in SEEDS])
        yp_stack = np.concatenate([np.asarray(method_preds[s]) for s in SEEDS])
        pooled_ci = _bootstrap_ci(yt_stack, yp_stack, B=args.bootstrap, rng_seed=0)
        aggregate_rows.append(
            {
                "dataset": args.dataset,
                "subset": args.subset,
                "method": family,
                "params_hash": p_hash,
                "params_json": json.dumps(params, sort_keys=True),
                "split_hash": split_hash,
                "n_seeds": len(SEEDS),
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
