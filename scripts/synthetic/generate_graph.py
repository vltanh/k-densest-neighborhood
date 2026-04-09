import networkx as nx
import pandas as pd
import random
import time
import os
import argparse
import json
from itertools import combinations


def calculate_true_density(G_true: nx.DiGraph, nodes: set):
    """Calculates the exact absolute directed density of a node set."""
    n = len(nodes)
    if n < 2:
        return 0.0
    subg = G_true.subgraph(nodes)
    return subg.number_of_edges() / (n * (n - 1))


def generate_general_directed_graph(
    n_total=100_000,
    n_community=20,
    p_community=0.8,
    m_edges=10,
    p_reciprocal=0.001,
    seed=42,
):
    print(f"Generating {n_total}-node Directed Graph skeleton...")
    rng = random.Random(seed)

    # 1. Background Skeleton
    G_bg = nx.barabasi_albert_graph(n_total, m=m_edges, seed=rng)
    G = nx.DiGraph()
    G.add_nodes_from(range(n_total))

    for u, v in G_bg.edges():
        if rng.random() < 0.5:
            G.add_edge(u, v)
        else:
            G.add_edge(v, u)

    # 2. Planted Community Skeleton
    valid_pool = range(n_total // 4, n_total)
    community_nodes = rng.sample(valid_pool, n_community)
    community = set(community_nodes)

    print(f"Planting skeleton for community of size {n_community}...")
    for u, v in combinations(community_nodes, 2):
        if rng.random() < p_community:
            # Assign base direction randomly
            if rng.random() < 0.5:
                G.add_edge(u, v)
            else:
                G.add_edge(v, u)

    # 3. Global Reciprocal Pass
    # We iterate over a static snapshot of the edges so we don't evaluate newly added cycles
    print(f"Applying uniform {p_reciprocal*100}% reciprocal cycle pass globally...")
    current_edges = list(G.edges())
    reciprocal_edges = []

    for u, v in current_edges:
        if rng.random() < p_reciprocal:
            if not G.has_edge(v, u):
                reciprocal_edges.append((v, u))

    G.add_edges_from(reciprocal_edges)

    return G, community


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Densest Subgraph Test Data (Uniform Cycles)"
    )
    parser.add_argument(
        "--out_dir", type=str, required=True, help="Target directory for output files"
    )
    parser.add_argument(
        "--n_nodes", type=int, default=1_000_000, help="Total number of nodes"
    )
    parser.add_argument(
        "--m_edges", type=int, default=10, help="Number of edges per new node (BA 'm')"
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
        help="Base edge probability within the planted community",
    )
    parser.add_argument(
        "--p_reciprocal",
        type=float,
        default=0.001,
        help="Global probability of an edge being a 2-cycle",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    edges_file = os.path.join(args.out_dir, "edge.csv")
    community_file = os.path.join(args.out_dir, "gt_comm.csv")
    metadata_file = os.path.join(args.out_dir, "metadata.json")

    t_start = time.time()
    G, true_community = generate_general_directed_graph(
        n_total=args.n_nodes,
        n_community=args.n_community,
        p_community=args.p_community,
        m_edges=args.m_edges,
        p_reciprocal=args.p_reciprocal,
        seed=args.seed,
    )

    true_density = calculate_true_density(G, true_community)

    print(f"Writing edge list to {edges_file}...")
    df_edges = pd.DataFrame(G.edges(), columns=["source", "target"])
    df_edges.to_csv(edges_file, index=False)

    print(f"Writing true community nodes to {community_file}...")
    df_comm = pd.DataFrame(sorted(true_community), columns=["node_id"])
    df_comm.to_csv(community_file, index=False)

    print(f"Writing run metadata to {metadata_file}...")
    metadata = {
        "nodes": args.n_nodes,
        "edges": G.number_of_edges(),
        "m_edges": args.m_edges,
        "n_community": args.n_community,
        "p_community": args.p_community,
        "p_reciprocal": args.p_reciprocal,
        "seed": args.seed,
        "planted_density": round(true_density, 6),
        "planted_size": len(true_community),
    }

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

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
