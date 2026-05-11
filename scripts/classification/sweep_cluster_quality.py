"""Full-grid cluster-quality sweep on the val + test union.

Mirrors the tune_val grid (bp-k, bp-kappa, bfs-depth) and runs the solver on
val_ids + test_ids for every cell, emitting per-query records into a shared
directory. Aggregation downstream:

    python scripts/classification/aggregate_cluster_quality.py \
        --records exps/classification/<dataset>/cluster_quality/ \
        --output  exps/classification/<dataset>/cluster_quality/agg/ \
        --combine-splits

produces one cluster-quality row per (method, params_hash) over the union,
which lets the slides + paper show how cluster quality moves with k and kappa
independently of the classification objective.

The cluster-quality perspective is intrinsic and label-agnostic: only the
geometry of the returned subgraph matters, so val and test are treated as one
query pool. Classification (tune-val / eval-test) is handled separately by
tune_val.py + evaluate_test.py and is not re-run here.
"""

import argparse
import json
import os
import sys
import tempfile

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from solver_utils import (  # noqa: E402
    build_graph_context,
    effective_params,
    evaluate_nodes,
    method_extra_args,
)
from split_utils import assert_split_meta_matches  # noqa: E402


def _parse_int_list(raw: str):
    return [int(x) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str):
    return [float(x) for x in raw.split(",") if x.strip()]


def _avgdeg_grid():
    return [{}]


def _bfs_grid(depths):
    return [{"bfs_depth": d} for d in depths]


def _bp_grid(k_values, kappa_values, time_limits, dinkelbach_iters):
    grid = []
    for k in k_values:
        for kappa in kappa_values:
            for tl in time_limits:
                for di in dinkelbach_iters:
                    grid.append(
                        {
                            "k": k,
                            "kappa": kappa,
                            "time_limit": tl,
                            "dinkelbach_iter": di,
                        }
                    )
    return grid


def _k_for(method, params):
    if method == "bp":
        return int(params["k"])
    return None


def _run_on_split(
    *,
    query_nodes,
    split_label,
    split_hash,
    family,
    params,
    eff_params,
    extra,
    k,
    args,
    df_nodes,
    ctx,
    records_dir,
    forbidden,
    seed,
):
    """Run evaluate_nodes for one (family, params, seed) cell on one split.
    seed is None for deterministic methods, int for BP. evaluate_nodes dedupes
    cached records on _record_key so a repeat run is cheap if records already
    exist (deterministic methods share a single record across seeds; BP records
    are per-seed)."""
    with tempfile.TemporaryDirectory() as td:
        evaluate_nodes(
            query_nodes,
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
            seed=seed,
            method=family,
            params=eff_params,
            split_hash=split_hash,
            keep_solver_dumps=args.keep_solver_dumps,
            query_split=split_label,
            collect_stats=False,
            compute_qualities=True,
            show_progress=False,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--bin-path", type=str, default="./solver/bin/solver")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--exps-dir", type=str, default="exps")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44,45,46",
        help="Comma-separated Gurobi seeds for BP cells; deterministic methods ignore them and run once.",
    )
    parser.add_argument("--weighting", type=str, default="uniform")
    parser.add_argument("--max-fallback-hops", type=int, default=10)
    parser.add_argument("--keep-solver-dumps", action="store_true")
    parser.add_argument(
        "--family",
        type=str,
        default="all",
        choices=["all", "avgdeg", "bfs", "bp"],
    )
    parser.add_argument("--bp-k", type=str, default="3,4,5")
    parser.add_argument("--bp-kappa", type=str, default="0,1,2")
    parser.add_argument("--bp-time-limit", type=str, default="-1")
    parser.add_argument("--bp-dinkelbach-iter", type=str, default="-1")
    parser.add_argument("--bfs-depth", type=str, default="1,2")
    args = parser.parse_args()

    df_nodes = pd.read_csv(os.path.join(args.data_dir, args.dataset, "nodes.csv"))
    df_edges = pd.read_csv(os.path.join(args.data_dir, args.dataset, "edge.csv"))
    meta = assert_split_meta_matches(args.dataset, df_nodes, df_edges, args.data_dir)

    val_ids = [int(q) for q in df_nodes[df_nodes["val"]]["node_id"].astype(int).tolist()]
    test_ids = [int(q) for q in df_nodes[df_nodes["test"]]["node_id"].astype(int).tolist()]
    forbidden = set(val_ids) | set(test_ids)
    print(
        f"{args.dataset}: val={len(val_ids)} + test={len(test_ids)} -> union={len(val_ids) + len(test_ids)};"
        f" forbidden={len(forbidden)}"
    )

    ctx = build_graph_context(
        os.path.join(args.data_dir, args.dataset, "edge.csv"), max_in_edges=0
    )

    out_root = os.path.join(args.exps_dir, "classification", args.dataset, "cluster_quality")
    os.makedirs(out_root, exist_ok=True)

    bp_seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    families = ["avgdeg", "bfs", "bp"] if args.family == "all" else [args.family]
    manifest_rows = []
    for fam in families:
        grid_fn = {
            "avgdeg": lambda: _avgdeg_grid(),
            "bfs": lambda: _bfs_grid(_parse_int_list(args.bfs_depth)),
            "bp": lambda: _bp_grid(
                _parse_int_list(args.bp_k),
                _parse_int_list(args.bp_kappa),
                _parse_float_list(args.bp_time_limit),
                _parse_int_list(args.bp_dinkelbach_iter),
            ),
        }[fam]
        grid = grid_fn()
        for params in grid:
            eff_params, p_hash = effective_params(
                params,
                weighting=args.weighting,
                max_fallback_hops=args.max_fallback_hops,
                forbidden_nodes=forbidden,
            )
            k = _k_for(fam, params)
            records_dir = os.path.join(out_root, fam, p_hash[-12:])
            os.makedirs(records_dir, exist_ok=True)
            print(
                f"[{args.dataset}] cluster-quality sweep {fam} params={params} hash={p_hash[-12:]}"
            )
            seeds_for_this_fam = bp_seeds if fam == "bp" else [None]
            for seed in seeds_for_this_fam:
                extra = method_extra_args(fam, params, gurobi_seed=seed if fam == "bp" else None)
                _run_on_split(
                    query_nodes=val_ids,
                    split_label="val",
                    split_hash=meta.splits["val"]["hash"],
                    family=fam,
                    params=params,
                    eff_params=eff_params,
                    extra=extra,
                    k=k,
                    args=args,
                    df_nodes=df_nodes,
                    ctx=ctx,
                    records_dir=records_dir,
                    forbidden=forbidden,
                    seed=seed,
                )
                _run_on_split(
                    query_nodes=test_ids,
                    split_label="test",
                    split_hash=meta.splits["test"]["hash"],
                    family=fam,
                    params=params,
                    eff_params=eff_params,
                    extra=extra,
                    k=k,
                    args=args,
                    df_nodes=df_nodes,
                    ctx=ctx,
                    records_dir=records_dir,
                    forbidden=forbidden,
                    seed=seed,
                )
            manifest_rows.append(
                {
                    "dataset": args.dataset,
                    "method": fam,
                    "params_hash": p_hash,
                    "params_json": json.dumps(params, sort_keys=True),
                    "records_dir": records_dir,
                    "seeds": ",".join(str(s) for s in seeds_for_this_fam),
                }
            )

    manifest_path = os.path.join(out_root, "manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"wrote {manifest_path}")
    print(
        "Next: python scripts/classification/aggregate_cluster_quality.py "
        f"--records {out_root} --output {out_root}/agg/ --combine-splits"
    )


if __name__ == "__main__":
    main()
