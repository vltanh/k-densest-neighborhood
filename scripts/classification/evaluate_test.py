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
    params_hash,
)
from split_utils import assert_split_meta_matches, sha256_node_set  # noqa: E402


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


def _extra_args_for_seed(extra_args, gurobi_seed):
    # Strip any pre-existing --gurobi-seed and append the current one.
    out = []
    i = 0
    while i < len(extra_args):
        if extra_args[i] == "--gurobi-seed":
            i += 2
            continue
        out.append(extra_args[i])
        i += 1
    if "--bp" in out and gurobi_seed is not None:
        out += ["--gurobi-seed", str(gurobi_seed)]
    return out


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
        default="eligible_test",
        choices=["eligible_test", "bfs_depth1_wrong"],
    )
    parser.add_argument("--weighting", type=str, default="uniform")
    parser.add_argument("--max-fallback-hops", type=int, default=10)
    parser.add_argument("--keep-solver-dumps", action="store_true")
    args = parser.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]

    df_nodes = pd.read_csv(os.path.join(args.data_dir, args.dataset, "nodes.csv"))
    df_edges = pd.read_csv(os.path.join(args.data_dir, args.dataset, "edge.csv"))
    meta = assert_split_meta_matches(args.dataset, df_nodes, df_edges, args.data_dir)

    from collections import defaultdict

    adj_und = defaultdict(set)
    for s, t in zip(df_edges["source"].astype(int), df_edges["target"].astype(int)):
        if s == t:
            continue
        adj_und[s].add(t)
        adj_und[t].add(s)

    if args.subset == "eligible_test":
        test_ids = df_nodes[df_nodes["test"]]["node_id"].astype(int).tolist()
        target_nodes = [int(q) for q in test_ids if len(adj_und.get(q, ())) >= 2]
        expected_hash = meta.eligible["test"]["hash"]
        if sha256_node_set(target_nodes) != expected_hash:
            raise ValueError("eligible.test hash mismatch with split_meta.json")
        split_hash = expected_hash
        query_split = "test"
    else:
        if meta.hard_subset is None:
            raise ValueError("split_meta.json carries no hard_subset; rerun prepare_data")
        # Recompute hard_subset from split_meta + eligible.val to verify.
        from collections import Counter
        from solver_utils import argmax_label

        train_mask = df_nodes["train"].values
        labels = df_nodes["label"].values
        global_majority = argmax_label(Counter(labels[train_mask]))
        val_ids = df_nodes[df_nodes["val"]]["node_id"].astype(int).tolist()
        eligible_val = [int(q) for q in val_ids if len(adj_und.get(q, ())) >= 2]

        def _bfs1(q):
            train_neighbours = [n for n in adj_und.get(q, ()) if n != q and train_mask[n]]
            if not train_neighbours:
                return global_majority
            return argmax_label(Counter(labels[n] for n in train_neighbours))

        hard = [int(q) for q in eligible_val if _bfs1(q) != labels[q]]
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
        extra_args_base = best["extra_args"]
        k = best.get("k")
        params = best.get("params") or {}
        p_hash = best["params_hash"]
        seeds = SEEDS if family not in DETERMINISTIC else SEEDS  # always five rows
        method_preds = {}  # seed -> y_pred
        method_truth = None
        for seed in seeds:
            if family in DETERMINISTIC and seed != seeds[0] and method_truth is not None:
                # Reuse the deterministic prediction across seeds.
                method_preds[seed] = method_preds[seeds[0]]
                continue
            extra = _extra_args_for_seed(extra_args_base, seed if family == "bp" else None)
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

        # Aggregate across seeds.
        precisions, recalls, f1s = [], [], []
        for seed in seeds:
            yt = method_truth
            yp = method_preds[seed]
            scores = _macro_scores(yt, yp)
            precisions.append(scores["precision"])
            recalls.append(scores["recall"])
            f1s.append(scores["f1"])
        # Pooled CI: bootstrap over the union of (yt, yp) seed-stack.
        yt_stack = np.concatenate([np.asarray(method_truth) for _ in seeds])
        yp_stack = np.concatenate([np.asarray(method_preds[s]) for s in seeds])
        pooled_ci = _bootstrap_ci(yt_stack, yp_stack, B=args.bootstrap, rng_seed=0)
        aggregate_rows.append(
            {
                "dataset": args.dataset,
                "subset": args.subset,
                "method": family,
                "params_hash": p_hash,
                "params_json": json.dumps(params, sort_keys=True),
                "split_hash": split_hash,
                "n_seeds": len(seeds),
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
