import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _solver_runner import overlap_metrics  # noqa: E402


def evaluate(gt_path, pred_path):
    if not os.path.exists(gt_path):
        print(f"Error: Ground truth file '{gt_path}' not found.")
        return
    if not os.path.exists(pred_path):
        print(f"Error: Prediction file '{pred_path}' not found.")
        return

    gt_set = set(pd.read_csv(gt_path)["node_id"].astype(str))
    pred_set = set(pd.read_csv(pred_path)["node_id"].astype(str))
    m = overlap_metrics(gt_set, pred_set)

    print("==================================================")
    print("EVALUATION METRICS")
    print("==================================================")
    print(f"Ground Truth Size : {len(gt_set)}")
    print(f"Prediction Size   : {len(pred_set)}")
    print("--------------------------------------------------")
    print(f"True Positives    : {m['tp']}")
    print(f"False Positives   : {m['fp']}")
    print(f"False Negatives   : {m['fn']}")
    print("--------------------------------------------------")
    print(f"Precision         : {m['precision']:.4f}")
    print(f"Recall            : {m['recall']:.4f}")
    print(f"F1 Score          : {m['f1']:.4f}")
    print(f"Jaccard Similarity: {m['jaccard']:.4f}")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Densest Subgraph Predictions")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth CSV (gt_comm.csv)")
    parser.add_argument("--pred", type=str, required=True, help="Path to predicted CSV (pred_comm.csv)")
    args = parser.parse_args()

    evaluate(args.gt, args.pred)
