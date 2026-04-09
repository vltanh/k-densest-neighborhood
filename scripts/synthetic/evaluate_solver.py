import pandas as pd
import argparse
import os


def evaluate(gt_path, pred_path):
    if not os.path.exists(gt_path):
        print(f"Error: Ground truth file '{gt_path}' not found.")
        return
    if not os.path.exists(pred_path):
        print(f"Error: Prediction file '{pred_path}' not found.")
        return

    # Read CSVs and cast to string to ensure safe matching
    gt_df = pd.read_csv(gt_path)
    pred_df = pd.read_csv(pred_path)

    gt_set = set(gt_df["node_id"].astype(str))
    pred_set = set(pred_df["node_id"].astype(str))

    # Compute Set Intersections
    intersection = gt_set.intersection(pred_set)
    union = gt_set.union(pred_set)

    tp = len(intersection)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    # Compute Metrics safely
    precision = tp / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = tp / len(gt_set) if len(gt_set) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    jaccard = tp / len(union) if len(union) > 0 else 0.0

    print("==================================================")
    print("EVALUATION METRICS")
    print("==================================================")
    print(f"Ground Truth Size : {len(gt_set)}")
    print(f"Prediction Size   : {len(pred_set)}")
    print("--------------------------------------------------")
    print(f"True Positives    : {tp}")
    print(f"False Positives   : {fp}")
    print(f"False Negatives   : {fn}")
    print("--------------------------------------------------")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1 Score          : {f1:.4f}")
    print(f"Jaccard Similarity: {jaccard:.4f}")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Densest Subgraph Predictions"
    )
    parser.add_argument(
        "--gt", type=str, required=True, help="Path to ground truth CSV (gt_comm.csv)"
    )
    parser.add_argument(
        "--pred", type=str, required=True, help="Path to predicted CSV (pred_comm.csv)"
    )
    args = parser.parse_args()

    evaluate(args.gt, args.pred)
