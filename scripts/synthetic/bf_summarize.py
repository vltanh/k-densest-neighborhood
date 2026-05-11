"""Summarise exps/synthetic/bf/solver_runs.csv into a per-cell match table.

Output columns:

    method, k, p, n_runs, n_match_within_tol, match_rate,
    median_wall_time_s, median_total_bb_nodes
"""

import argparse
import os

import pandas as pd


def _summarise(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["method", "k", "p"]
    rows = []
    for keys, chunk in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        n_runs = len(chunk)
        n_match = int(chunk["opt_match_within_tol"].sum())
        row["n_runs"] = n_runs
        row["n_match_within_tol"] = n_match
        row["match_rate"] = n_match / n_runs if n_runs else 0.0
        row["median_wall_time_s"] = float(chunk["wall_time_s"].median())
        if "total_bb_nodes" in chunk.columns:
            row["median_total_bb_nodes"] = float(
                chunk["total_bb_nodes"].dropna().median()
            ) if chunk["total_bb_nodes"].notna().any() else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solver-runs",
        type=str,
        default="exps/synthetic/bf/solver_runs.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exps/synthetic/bf/summary.csv",
    )
    args = parser.parse_args()

    if not os.path.exists(args.solver_runs):
        raise FileNotFoundError(args.solver_runs)
    df = pd.read_csv(args.solver_runs)
    summary = _summarise(df)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"wrote {args.output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
