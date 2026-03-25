import os
import subprocess
import pandas as pd
import networkx as nx
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def run_solver(q_node, k, edge_csv, bin_path, tmp_dir):
    """Executes the C++ Branch-and-Price solver for a single query node."""
    out_csv = os.path.join(tmp_dir, f"out_q{q_node}_k{k}.csv")
    cmd = [bin_path, edge_csv, str(q_node), str(k), out_csv]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        return q_node, []

    neighborhood = []
    if os.path.exists(out_csv):
        try:
            pred_df = pd.read_csv(out_csv)
            neighborhood = pred_df["node_id"].astype(int).tolist()
        except Exception:
            pass
        os.remove(out_csv)

    return q_node, neighborhood


def evaluate_nodes(
    query_nodes,
    k,
    edge_csv,
    df_nodes,
    bin_path,
    tmp_dir,
    max_workers=8,
    weighting="uniform",
    max_fallback_hops=10,
):
    """Runs k-densest classification and returns perfectly aligned y_true and y_pred arrays."""
    train_mask = df_nodes["train"].values
    labels = df_nodes["label"].values

    # Global fallback: Most frequent class in the entire training set
    train_labels = labels[train_mask]
    global_majority = Counter(train_labels).most_common(1)[0][0]

    # Build the NetworkX graph once for distance weighting and BFS fallbacks
    df_edges = pd.read_csv(edge_csv)
    G = nx.from_pandas_edgelist(
        df_edges, source="source", target="target", create_using=nx.Graph()
    )

    y_true = []
    y_pred = []

    # --- NEW: Initialize the starvation counter ---
    fallback_count = 0
    total_queries = len(query_nodes)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_solver, q, k, edge_csv, bin_path, tmp_dir): q
            for q in query_nodes
        }

        # as_completed yields out of order, but y_true and y_pred stay pairwise matched
        for future in tqdm(
            as_completed(futures), total=total_queries, desc=f"Evaluating k={k}"
        ):
            q_node, neighborhood = future.result()

            # Filter neighborhood to strictly contain Train nodes (EXCLUDING the query node itself)
            train_neighbors = [n for n in neighborhood if train_mask[n] and n != q_node]

            if not train_neighbors:
                # --- NEW: Increment the starvation counter ---
                fallback_count += 1

                # ==========================================================
                # LABEL STARVATION DETECTED: Trigger Concentric BFS Fallback
                # ==========================================================
                try:
                    paths = nx.single_source_shortest_path_length(
                        G, q_node, cutoff=max_fallback_hops
                    )
                    reachable_train = {
                        n: d for n, d in paths.items() if train_mask[n] and n != q_node
                    }

                    if reachable_train:
                        # Find the radius of the closest labeled ring
                        min_dist = min(reachable_train.values())
                        nearest_train_nodes = [
                            n for n, d in reachable_train.items() if d == min_dist
                        ]

                        fallback_labels = [labels[n] for n in nearest_train_nodes]
                        pred_label = Counter(fallback_labels).most_common(1)[0][0]
                    else:
                        # Disconnected component or no labels within cutoff
                        pred_label = global_majority
                except Exception:
                    pred_label = global_majority

            else:
                # ==========================================================
                # STANDARD VOTING (K-Densest Subgraph)
                # ==========================================================
                if weighting == "distance":
                    class_scores = Counter()
                    for n in train_neighbors:
                        try:
                            d = nx.shortest_path_length(G, source=q_node, target=n)
                            # d is guaranteed >= 1 because we excluded q_node
                            w = 1.0 / d
                        except nx.NetworkXNoPath:
                            w = 0.0

                        class_scores[labels[n]] += w

                    pred_label = class_scores.most_common(1)[0][0]

                else:  # Uniform weighting
                    neighbor_labels = [labels[n] for n in train_neighbors]
                    pred_label = Counter(neighbor_labels).most_common(1)[0][0]

            y_true.append(labels[q_node])
            y_pred.append(pred_label)

    # --- NEW: Print the telemetry summary before returning ---
    fallback_rate = (fallback_count / total_queries) * 100 if total_queries > 0 else 0
    print(
        f"\n[Diagnostic] k={k} | Fallback Triggered: {fallback_count}/{total_queries} times ({fallback_rate:.1f}%)"
    )

    return y_true, y_pred
