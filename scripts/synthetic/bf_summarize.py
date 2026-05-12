"""Summarise exps/synthetic/bf/solver_runs.csv into a per-cell match table.

The summary gates "match" counts on actual feasibility of the solver's returned
set: returncode == 0; |S| >= k for BP cells; solver_actual_kappa >= requested
kappa for kappa >= 1 cells. Non-feasible cells are surfaced as separate rows
(``status_breakdown``) instead of being silently counted as a non-match.

Output columns:

    method, k, kappa, p, n_runs, n_feasible, n_match_primary,
    match_rate_primary, n_match_secondary, match_rate_secondary,
    n_solver_error, n_infeasible_size, n_infeasible_kappa,
    median_wall_time_s, median_total_bb_nodes
"""

import argparse
import os

import pandas as pd


def _row_status(row) -> str:
    """Per-row feasibility status. 'feasible' means the row is eligible to be
    counted in the match-rate denominator."""
    status = row.get("status")
    if isinstance(status, str) and status.startswith("skipped"):
        return status
    rc = row.get("returncode")
    if pd.notna(rc) and int(rc) != 0:
        return "solver_error"
    method = row["method"]
    k = row.get("k")
    if method == "bp" and pd.notna(k):
        if pd.isna(row.get("solver_size")) or int(row["solver_size"]) < int(k):
            return "infeasible_size"
    kappa = row.get("kappa")
    if pd.notna(kappa) and int(kappa) >= 1:
        actual = row.get("solver_actual_kappa")
        if pd.isna(actual) or int(actual) < int(kappa):
            return "infeasible_kappa"
    return "feasible"


def _summarise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_status"] = df.apply(_row_status, axis=1)

    group_cols = ["method", "k", "kappa", "p"] if "kappa" in df.columns else ["method", "k", "p"]
    rows = []
    for keys, chunk in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        n_runs = len(chunk)
        feasible_mask = chunk["_status"] == "feasible"
        feasible_chunk = chunk[feasible_mask]
        n_feasible = int(feasible_mask.sum())
        n_solver_error = int((chunk["_status"] == "solver_error").sum())
        n_infeasible_size = int((chunk["_status"] == "infeasible_size").sum())
        n_infeasible_kappa = int((chunk["_status"] == "infeasible_kappa").sum())
        n_skipped_bf_infeasible = int((chunk["_status"] == "skipped_bf_infeasible").sum())

        n_match_primary = int(feasible_chunk["opt_match_within_tol"].sum()) if n_feasible else 0
        match_rate_primary = n_match_primary / n_feasible if n_feasible else 0.0

        if "opt_match_within_tol_secondary" in chunk.columns:
            n_match_secondary = int(feasible_chunk["opt_match_within_tol_secondary"].sum()) if n_feasible else 0
            match_rate_secondary = n_match_secondary / n_feasible if n_feasible else 0.0
        else:
            n_match_secondary = None
            match_rate_secondary = None

        row.update(
            {
                "n_runs": n_runs,
                "n_feasible": n_feasible,
                "n_match_primary": n_match_primary,
                "match_rate_primary": match_rate_primary,
                "n_match_secondary": n_match_secondary,
                "match_rate_secondary": match_rate_secondary,
                "n_solver_error": n_solver_error,
                "n_infeasible_size": n_infeasible_size,
                "n_infeasible_kappa": n_infeasible_kappa,
                "n_skipped_bf_infeasible": n_skipped_bf_infeasible,
                "median_wall_time_s": float(feasible_chunk["wall_time_s"].median())
                if n_feasible
                else float("nan"),
            }
        )
        if "total_bb_nodes" in feasible_chunk.columns and feasible_chunk["total_bb_nodes"].notna().any():
            row["median_total_bb_nodes"] = float(
                feasible_chunk["total_bb_nodes"].dropna().median()
            )
        else:
            row["median_total_bb_nodes"] = None
        if "hard_cap_hit" in chunk.columns:
            hits = chunk["hard_cap_hit"].fillna(False).astype(bool)
            row["n_hard_cap_hit"] = int(hits.sum())
            row["hard_cap_hit_rate"] = float(hits.mean()) if n_runs else 0.0
        else:
            row["n_hard_cap_hit"] = 0
            row["hard_cap_hit_rate"] = 0.0
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
