import json
import networkx as nx
import pandas as pd
import random
import time
import os
import argparse
from itertools import combinations


def calculate_true_density(G_true: nx.DiGraph, nodes: set):
    """Calculates the exact absolute density of a node set."""
    n = len(nodes)
    if n < 2:
        return 0.0
    subg = G_true.subgraph(nodes)
    return subg.number_of_edges() / (n * (n - 1))


def generate_fast_scale_free_dag(
    n_total=100_000, n_community=20, p_community=0.8, m_edges=10, seed=42
):
    print(f"Generating {n_total}-node Barabási-Albert DAG...")
    rng = random.Random(seed)

    G_bg = nx.barabasi_albert_graph(n_total, m=m_edges, seed=rng)
    G = nx.DiGraph()
    G.add_nodes_from(range(n_total))

    G.add_edges_from((min(u, v), max(u, v)) for u, v in G_bg.edges())

    # Randomly sample non-contiguous nodes, strictly avoiding the massive hubs near index 0
    valid_pool = range(n_total // 4, n_total)
    community_nodes = rng.sample(valid_pool, n_community)
    community = set(community_nodes)
    q_node = rng.choice(community_nodes)

    print(f"Planting dense community of size {n_community} around node {q_node}...")
    G.add_edges_from(
        (u, v)
        for u, v in combinations(community_nodes, 2)
        if rng.random() < p_community
    )

    return G, q_node, community


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Densest Subgraph Test Data")
    parser.add_argument(
        "--out_dir", type=str, required=True, help="Target directory for output files"
    )
    parser.add_argument(
        "--n_nodes",
        type=int,
        default=1_000_000,
        help="Total number of nodes in the background graph",
    )
    parser.add_argument(
        "--m_edges",
        type=int,
        default=10,
        help="Number of edges to attach from a new node (Barabasi-Albert 'm')",
    )
    parser.add_argument(
        "--n_community",
        type=int,
        default=20,
        help="Size of the planted dense community",
    )
    parser.add_argument(
        "--p_community",
        type=float,
        default=0.8,
        help="Edge probability within the planted community",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for topology generation"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    edges_file = os.path.join(args.out_dir, "edge.csv")
    community_file = os.path.join(args.out_dir, "gt_comm.csv")
    metadata_file = os.path.join(args.out_dir, "metadata.json")

    t_start = time.time()
    G, q_node, true_community = generate_fast_scale_free_dag(
        n_total=args.n_nodes,
        n_community=args.n_community,
        p_community=args.p_community,
        m_edges=args.m_edges,
        seed=args.seed,
    )

    # Calculate the exact density of the planted community
    true_density = calculate_true_density(G, true_community)

    print(f"Writing edge list to {edges_file}...")
    df_edges = pd.DataFrame(G.edges(), columns=["source", "target"])
    df_edges.to_csv(edges_file, index=False)

    print(f"Writing true community nodes to {community_file}...")
    # Wrap in pandas to easily apply the exact header requested
    df_comm = pd.DataFrame(sorted(true_community), columns=["node_id"])
    df_comm.to_csv(community_file, index=False)

    print(f"Writing run metadata to {metadata_file}...")
    with open(metadata_file, "w") as f:
        json.dump(
            {
                "nodes": args.n_nodes,
                "edges": G.number_of_edges(),
                "m_edges": args.m_edges,
                "n_community": args.n_community,
                "p_community": args.p_community,
                "seed": args.seed,
                "planted_density": round(true_density, 6),
                "planted_size": len(true_community),
            },
            f,
            indent=2,
        )

    t_end = time.time()

    print(f"\nGraph successfully exported in {t_end - t_start:.2f}s.")
    print("=" * 50)
    print("GENERATED GRAPH FILES")
    print("=" * 50)
    print(f"Output Dir : {args.out_dir}")
    print(f"Edges File : {edges_file}")
    print(f"Comm File  : {community_file}")
    print(f"Meta File  : {metadata_file}")
    print("=" * 50)
