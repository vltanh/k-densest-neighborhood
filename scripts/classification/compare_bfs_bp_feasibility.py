import csv
import os
import subprocess
import tempfile
from collections import defaultdict, deque

import pandas as pd


def run_solver_nodes(q_node, k, edge_csv, bin_path, extra_args):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        out_csv = tmp.name
    cmd = [
        bin_path,
        "--mode",
        "sim",
        "--input",
        edge_csv,
        "--query",
        str(q_node),
        "--output",
        out_csv,
    ]
    if k is not None:
        cmd.extend(["--k", str(k)])
    cmd.extend(extra_args)
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        if not os.path.exists(out_csv):
            return []
        df = pd.read_csv(out_csv)
        return df["node_id"].astype(int).tolist()
    except subprocess.CalledProcessError:
        return []
    finally:
        if os.path.exists(out_csv):
            os.remove(out_csv)


def directed_density(nodes, out_neighbors):
    node_set = set(nodes)
    n = len(node_set)
    if n <= 1:
        return 0.0
    m = 0
    for u in node_set:
        for v in out_neighbors.get(u, ()):
            if v in node_set and v != u:
                m += 1
    return m / (n * (n - 1))


def edge_connectivity_at_least(nodes, undir_neighbors, kappa):
    node_set = set(nodes)
    if kappa <= 0:
        return True
    if len(node_set) <= 1:
        return False
    for source in node_set:
        for target in node_set:
            if source == target:
                continue
            residual = {}
            adj = defaultdict(list)
            for u in node_set:
                for v in undir_neighbors.get(u, ()):
                    if v not in node_set:
                        continue
                    residual[(u, v)] = residual.get((u, v), 0) + 1
                    if v not in adj[u]:
                        adj[u].append(v)
                    if u not in adj[v]:
                        adj[v].append(u)

            flow = 0
            while flow < kappa:
                parent = {source: source}
                queue = deque([source])
                while queue and target not in parent:
                    u = queue.popleft()
                    for v in adj.get(u, ()):
                        if v not in parent and residual.get((u, v), 0) > 0:
                            parent[v] = u
                            queue.append(v)
                if target not in parent:
                    break
                v = target
                while v != source:
                    u = parent[v]
                    residual[(u, v)] -= 1
                    residual[(v, u)] = residual.get((v, u), 0) + 1
                    v = u
                flow += 1
            if flow < kappa:
                return False
    return True


def main():
    dataset = "Cora"
    edge_csv = os.path.join("data", dataset, "edge.csv")
    node_csv = os.path.join("data", dataset, "nodes.csv")
    bin_path = "./solver/bin/solver"
    out_dir = os.path.join("exps", "classification", dataset, "feasibility_checks")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bfs_depth1_vs_bp_k5_kappa2_test.csv")

    df_nodes = pd.read_csv(node_csv)
    query_nodes = df_nodes[df_nodes["test"]]["node_id"].astype(int).tolist()
    df_edges = pd.read_csv(edge_csv)
    out_neighbors = defaultdict(set)
    undir_neighbors = defaultdict(set)
    for s, t in zip(df_edges["source"], df_edges["target"]):
        s = int(s)
        t = int(t)
        out_neighbors[s].add(t)
        undir_neighbors[s].add(t)
        undir_neighbors[t].add(s)

    rows = []
    k = 5
    kappa = 2
    for i, q_node in enumerate(query_nodes, 1):
        bfs_nodes = run_solver_nodes(
            q_node, None, edge_csv, bin_path, ["--bfs", "--bfs-depth", "1"]
        )
        bp_nodes = run_solver_nodes(
            q_node,
            k,
            edge_csv,
            bin_path,
            ["--bp", "--kappa", str(kappa), "--time-limit", "-1"],
        )
        bfs_density = directed_density(bfs_nodes, out_neighbors)
        bp_density = directed_density(bp_nodes, out_neighbors)
        bfs_size_feasible = len(bfs_nodes) >= k
        bfs_conn_feasible = edge_connectivity_at_least(
            bfs_nodes, undir_neighbors, kappa
        )
        bfs_feasible = bfs_size_feasible and bfs_conn_feasible
        rows.append(
            {
                "query_node": q_node,
                "bfs_size": len(bfs_nodes),
                "bp_size": len(bp_nodes),
                "bfs_density": bfs_density,
                "bp_density": bp_density,
                "bfs_size_feasible": bfs_size_feasible,
                "bfs_kappa_feasible": bfs_conn_feasible,
                "bfs_feasible": bfs_feasible,
                "bfs_beats_bp": bfs_feasible and bfs_density > bp_density + 1e-9,
                "density_gap_bfs_minus_bp": bfs_density - bp_density,
            }
        )
        if i % 100 == 0:
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"processed {i}/{len(query_nodes)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    feasible = df[df["bfs_feasible"]]
    print(f"saved {out_path}")
    print(f"queries: {len(df)}")
    print(f"bfs feasible for k={k},kappa={kappa}: {len(feasible)}")
    if len(feasible) > 0:
        print(f"bfs feasible rate: {len(feasible) / len(df):.4f}")
        print(f"bfs beats bp on feasible: {int(feasible['bfs_beats_bp'].sum())}")
        print(
            "avg density feasible bfs/bp:",
            feasible["bfs_density"].mean(),
            feasible["bp_density"].mean(),
        )
        print(
            "avg density gap:",
            feasible["density_gap_bfs_minus_bp"].mean(),
        )
        print(
            "max positive gap:",
            feasible["density_gap_bfs_minus_bp"].max(),
        )


if __name__ == "__main__":
    main()
