"""Render paper-style LaTeX tables from aggregate_experiment.py CSV outputs.

This is deliberately a thin presentation layer.  Regenerate the CSVs first:

    python scripts/classification/aggregate_experiment.py \
        --records exps/classification/Cora_ML/cluster_quality \
        --output exps/classification/Cora_ML/agg/wide \
        --dataset Cora_ML --allow-partial

Then render table snippets:

    python scripts/classification/aggregate_to_latex.py \
        --aggregate-dir exps/classification/Cora_ML/agg/wide \
        --output docs/paper/generated_tables \
        --dataset Cora_ML

Cells whose row is missing or whose query count is below the current
split_meta.json expectation are rendered as ``(pending)``.  The script writes
one .tex file per table plus all_tables.tex.
"""

import argparse
import json
import math
import os
from typing import Optional

import pandas as pd


METHOD_ORDER = [
    ("bfs", None),
    ("avgdeg", None),
    ("bp", 0),
    ("bp", 1),
    ("bp", 2),
]


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _split_sizes(dataset: str, data_dir: str) -> dict:
    meta_path = os.path.join(data_dir, dataset, "split_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    splits = meta.get("splits") or {}
    return {
        "val": int(splits.get("val", {}).get("size", 0)),
        "test": int(splits.get("test", {}).get("size", 0)),
        "union": int(splits.get("val", {}).get("size", 0)) + int(splits.get("test", {}).get("size", 0)),
        "bfs_depth1_wrong": int((meta.get("hard_subset") or {}).get("size", 0)),
    }


def _is_missing(value) -> bool:
    try:
        return pd.isna(value)
    except TypeError:
        return value is None


def _latex_num(value, digits: int) -> str:
    if _is_missing(value):
        return "(pending)"
    return f"{float(value):.{digits}f}"


def _mean_std(row: Optional[pd.Series], prefix: str, digits: int = 2) -> str:
    if row is None:
        return "(pending)"
    mean_col = f"{prefix}_mean"
    std_col = f"{prefix}_std"
    if mean_col not in row or std_col not in row or _is_missing(row[mean_col]) or _is_missing(row[std_col]):
        return "--"
    return f"${float(row[mean_col]):.{digits}f}\\pm{float(row[std_col]):.{digits}f}$"


def _score(row: Optional[pd.Series], metric: str, prefix: str = "", digits: int = 3) -> str:
    if row is None:
        return "(pending)"
    value_col = f"{prefix}{metric}_macro" if metric != "accuracy" else f"{prefix}accuracy"
    se_col = f"{prefix}{metric}_boot_se"
    if value_col not in row or se_col not in row or _is_missing(row[value_col]) or _is_missing(row[se_col]):
        return "(pending)"
    return f"${float(row[value_col]):.{digits}f}\\pm{float(row[se_col]):.{digits}f}$"


def _rate(row: Optional[pd.Series], col: str, digits: int = 3) -> str:
    if row is None or col not in row or _is_missing(row[col]):
        return "(pending)"
    return f"${float(row[col]):.{digits}f}$"


def _method_label(method: str, kappa) -> str:
    if method == "bfs":
        return r"BFS $d=1$"
    if method == "avgdeg":
        return "AvgDeg"
    if method == "bp":
        return rf"BP $\kappa={int(kappa)}$"
    return method


def _dataset_label(dataset: str) -> str:
    return str(dataset).replace("_", "-")


def _subset_label(subset: str) -> str:
    if subset == "bfs_depth1_wrong":
        return "BFS-depth-1-wrong"
    return str(subset).replace("_", "-")


def _params_label(row: pd.Series) -> str:
    method = row["method"]
    k = int(row["param_k"]) if "param_k" in row and not _is_missing(row["param_k"]) else None
    if method == "bfs":
        depth = int(row["param_bfs_depth"]) if "param_bfs_depth" in row and not _is_missing(row["param_bfs_depth"]) else 1
        return rf"$d={depth}, k={k}$"
    if method == "avgdeg":
        return rf"$k={k}$"
    if method == "bp":
        kappa = int(row["param_kappa"]) if "param_kappa" in row and not _is_missing(row["param_kappa"]) else 0
        return rf"$k={k}, \kappa={kappa}$"
    return "--"


def _pending_caption_note(lines: list[str]) -> str:
    return " Pending cells are incomplete in the aggregate CSV." if any("(pending)" in line for line in lines) else ""


def _select_cell(df: pd.DataFrame, dataset: str, method: str, k: int, kappa, expected_n: int) -> Optional[pd.Series]:
    if df.empty:
        return None
    mask = (df["dataset"] == dataset) & (df["method"] == method) & (pd.to_numeric(df["param_k"], errors="coerce") == k)
    if method == "bp":
        mask = mask & (pd.to_numeric(df["param_kappa"], errors="coerce") == int(kappa))
    chunk = df[mask]
    if chunk.empty:
        return None
    row = chunk.iloc[0]
    if "n_queries" in row and int(row["n_queries"]) < expected_n:
        return None
    return row


def _select_classification(
    df: pd.DataFrame,
    dataset: str,
    subset: str,
    method: str,
    k: int,
    kappa,
    expected_n: int,
) -> Optional[pd.Series]:
    if df.empty:
        return None
    mask = (
        (df["dataset"] == dataset)
        & (df["subset"] == subset)
        & (df["method"] == method)
        & (pd.to_numeric(df["param_k"], errors="coerce") == k)
    )
    if method == "bp":
        mask = mask & (pd.to_numeric(df["param_kappa"], errors="coerce") == int(kappa))
    chunk = df[mask]
    if chunk.empty:
        return None
    row = chunk.iloc[0]
    if "n_queries" in row and int(row["n_queries"]) < expected_n:
        return None
    return row


def _table_env(
    label: str,
    caption: str,
    colspec: str,
    header: str,
    body_lines: list[str],
    size: str = r"\scriptsize",
    resize: bool = False,
) -> str:
    body = "\n".join(body_lines)
    tabular = [
        f"  \\begin{{tabular}}{{{colspec}}}",
        r"    \toprule",
        f"    {header} \\\\",
        r"    \midrule",
        body,
        r"    \bottomrule",
        r"  \end{tabular}",
    ]
    if resize:
        tabular = [
            r"  \resizebox{\textwidth}{!}{%",
            *tabular,
            r"  }",
        ]
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"  \centering",
            f"  {size}",
            f"  \\caption{{{caption}}}",
            f"  \\label{{{label}}}",
            *tabular,
            r"\end{table}",
            "",
        ]
    )


def render_quality(union: pd.DataFrame, dataset: str, expected_union: int) -> str:
    lines = []
    for k in (3, 4, 5):
        if lines:
            lines.append(r"    \midrule")
        lines.append(rf"    \multicolumn{{8}}{{l}}{{\textit{{$k={k}$}}}} \\")
        for method, kappa in METHOD_ORDER:
            row = _select_cell(union, dataset, method, k, kappa, expected_union)
            lines.append(
                "    "
                + " & ".join(
                    [
                        _method_label(method, kappa),
                        _mean_std(row, "size"),
                        _mean_std(row, "quality_kappa_s"),
                        _rate(row, "kappa_s_lt_1_rate"),
                        _rate(row, "kappa_s_lt_2_rate"),
                        _mean_std(row, "quality_algebraic_connectivity_lambda2"),
                        _mean_std(row, "quality_dir_internal_avg_degree"),
                        _mean_std(row, "quality_dir_internal_edge_density"),
                    ]
                )
                + r" \\"
            )
    return _table_env(
        "tab:quality",
        rf"Cluster quality on {_dataset_label(dataset)} validation+test queries (${expected_union}$ queries), mean $\pm$ standard deviation.{_pending_caption_note(lines)}",
        "lccccccc",
        r"Method & $|S|$ & $\kappa_S$ & $\Pr[\kappa_S<1]$ & $\Pr[\kappa_S<2]$ & $\lambda_2$ & avg.deg & dens.",
        lines,
        resize=True,
    )


def render_cost(union: pd.DataFrame, dataset: str, expected_union: int) -> str:
    lines = []
    for k in (3, 4, 5):
        if lines:
            lines.append(r"    \midrule")
        lines.append(rf"    \multicolumn{{9}}{{l}}{{\textit{{$k={k}$}}}} \\")
        for method, kappa in METHOD_ORDER:
            row = _select_cell(union, dataset, method, k, kappa, expected_union)
            lines.append(
                "    "
                + " & ".join(
                    [
                        _method_label(method, kappa),
                        _mean_std(row, "wall_time_s"),
                        _mean_std(row, "oracle_queries", digits=0),
                        _mean_std(row, "stat_total_bb_nodes", digits=0),
                        _mean_std(row, "stat_total_lp_solves", digits=0),
                        _mean_std(row, "stat_total_cuts_added", digits=0),
                        _mean_std(row, "stat_t_lp_solve"),
                        _mean_std(row, "optimality_gap"),
                        _rate(row, "hard_cap_hit_rate"),
                    ]
                )
                + r" \\"
            )
    return _table_env(
        "tab:cost",
        rf"Cost on {_dataset_label(dataset)} validation+test queries (${expected_union}$ queries), mean $\pm$ standard deviation.{_pending_caption_note(lines)}",
        "lcccccccc",
        r"Method & wall (s) & queries & BB nodes & LP solves & cuts & $t_{\mathrm{LP}}$ (s) & gap & hc",
        lines,
        resize=True,
    )


def render_validation(grid: pd.DataFrame, dataset: str, expected_val: int) -> str:
    lines = []
    for k in (3, 4, 5):
        if lines:
            lines.append(r"    \midrule")
        lines.append(rf"    \multicolumn{{4}}{{l}}{{\textit{{$k={k}$}}}} \\")
        for method, kappa in METHOD_ORDER:
            row = _select_classification(grid, dataset, "val", method, k, kappa, expected_val)
            lines.append(
                "    "
                + " & ".join([_method_label(method, kappa), _score(row, "precision"), _score(row, "recall"), _score(row, "f1")])
                + r" \\"
            )
    return _table_env(
        "tab:validation",
        rf"Validation performance on {_dataset_label(dataset)} (${expected_val}$ queries), mean $\pm$ bootstrap standard error.{_pending_caption_note(lines)}",
        "llll",
        "Method & Precision & Recall & F1",
        lines,
    )


def _best_rows(grid: pd.DataFrame, dataset: str, expected_val: int) -> list[pd.Series]:
    rows = []
    for method in ("bfs", "avgdeg", "bp"):
        chunk = grid[(grid["dataset"] == dataset) & (grid["subset"] == "val") & (grid["method"] == method)].copy()
        if chunk.empty:
            continue
        chunk = chunk[pd.to_numeric(chunk["n_queries"], errors="coerce") >= expected_val]
        if chunk.empty:
            continue
        rows.append(chunk.sort_values(["f1_macro", "precision_macro", "recall_macro"], ascending=[False, False, False]).iloc[0])
    return rows


def render_best_eval(grid: pd.DataFrame, dataset: str, subset: str, expected_val: int, expected_subset: int, label: str) -> str:
    lines = []
    for best in _best_rows(grid, dataset, expected_val):
        method = best["method"]
        k = int(best["param_k"])
        kappa = int(best["param_kappa"]) if method == "bp" and not _is_missing(best["param_kappa"]) else None
        row = _select_classification(grid, dataset, subset, method, k, kappa, expected_subset)
        lines.append(
            "    "
            + " & ".join(
                [
                    "BFS" if method == "bfs" else "AvgDeg" if method == "avgdeg" else "BP",
                    _params_label(best),
                    _score(row, "precision"),
                    _score(row, "recall"),
                    _score(row, "f1"),
                ]
            )
            + r" \\"
        )
    caption = (
        rf"Validation-best configurations evaluated on {_dataset_label(dataset)} {_subset_label(subset)} queries (${expected_subset}$ queries), "
        rf"mean $\pm$ bootstrap standard error.{_pending_caption_note(lines)}"
    )
    return _table_env(
        label,
        caption,
        "lllll",
        "Method & validation-best parameters & Precision & Recall & F1",
        lines,
        size=r"\scriptsize",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", required=True, help="Directory written by aggregate_experiment.py")
    parser.add_argument("--output", required=True, help="Directory for generated .tex snippets")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    sizes = _split_sizes(args.dataset, args.data_dir)
    union = _read_csv(os.path.join(args.aggregate_dir, "cell_summary_union_by_seed.csv"))
    grid = _read_csv(os.path.join(args.aggregate_dir, "classification_grid_by_seed.csv"))

    tables = {
        "quality_table.tex": render_quality(union, args.dataset, sizes["union"]),
        "cost_table.tex": render_cost(union, args.dataset, sizes["union"]),
        "validation_table.tex": render_validation(grid, args.dataset, sizes["val"]),
        "test_table.tex": render_best_eval(grid, args.dataset, "test", sizes["val"], sizes["test"], "tab:test"),
        "hard_table.tex": render_best_eval(
            grid,
            args.dataset,
            "bfs_depth1_wrong",
            sizes["val"],
            sizes["bfs_depth1_wrong"],
            "tab:hard",
        ),
    }

    os.makedirs(args.output, exist_ok=True)
    all_parts = []
    for filename, content in tables.items():
        path = os.path.join(args.output, filename)
        with open(path, "w") as f:
            f.write(content)
        all_parts.append(content)
        print(f"wrote {path}")
    all_path = os.path.join(args.output, "all_tables.tex")
    with open(all_path, "w") as f:
        f.write("\n".join(all_parts))
    print(f"wrote {all_path}")


if __name__ == "__main__":
    main()
