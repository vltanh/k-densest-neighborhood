import networkx as nx
import gurobipy as gp
from gurobipy import GRB
import random
import logging
import time

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DensestSubgraphSolver:
    def __init__(self, G, q, k, tolerance=1e-4):
        self.G = G
        self.q = q
        self.k = k
        self.tol = tolerance
        self.global_best_density = 0.0
        self.global_best_solution = set()
        self.start_time = time.time()

    def solve(self):
        """Main Dinkelbach Loop (Outer Parameter Update)"""
        lambda_val = 0.0

        # Heuristic Initialization: Start with BFS of size k
        active_set = self._get_initial_active_set()

        iteration = 0
        while True:
            iteration += 1
            logger.info(f"--- DINKELBACH ITERATION {iteration} ---")
            logger.info(f"    Current Lambda: {lambda_val:.6f}")
            logger.info(f"    Active Set Size (Start): {len(active_set)}")

            # 2. Solve Parametric Problem (Branch-and-Price)
            # Returns: (Best Nodes Found, Max Objective Value, Updated Active Set)
            best_nodes, best_obj, expanded_set = self._branch_and_bound(
                lambda_val, active_set
            )

            # Update Active Set (Warm Start for next iteration)
            active_set = expanded_set

            # 3. Check Pruning / Convergence
            logger.info(f"    Best Objective Found: {best_obj:.6f}")

            # If the max possible gain (best_obj) is <= 0, we cannot improve further.
            if best_obj <= self.tol:
                logger.info("    Converged: Objective function is non-positive.")
                break

            # 4. Update Lambda
            current_density = self._calculate_density(best_nodes)

            if current_density > self.global_best_density:
                self.global_best_density = current_density
                self.global_best_solution = best_nodes
                logger.info(
                    f"    >>> NEW BEST SOLUTION FOUND! Density: {current_density:.6f} | Size: {len(best_nodes)}"
                )

            if abs(current_density - lambda_val) <= self.tol:
                logger.info("    Converged: Lambda is stable.")
                break

            lambda_val = current_density

        total_time = time.time() - self.start_time
        logger.info(f"--- FINISHED in {total_time:.2f}s ---")
        return self.global_best_solution, self.global_best_density

    def _branch_and_bound(self, lambda_val, active_set):
        """
        Loop 2: Branch-and-Bound Tree
        """
        # Stack stores nodes: {'fixed': {id: 0/1}, 'forbidden': set()}
        stack = [{"fixed": {self.q: 1}, "forbidden": set()}]

        best_obj_local = -float("inf")
        best_sol_local = set()

        # Maintain local copy of active set
        current_active_set = set(active_set)

        nodes_explored = 0

        while stack:
            nodes_explored += 1
            node = stack.pop()

            # --- Loop 3: Column Generation (Solve Relaxation) ---
            x_sol, lp_obj = self._solve_and_price(
                current_active_set, node["fixed"], node["forbidden"], lambda_val
            )

            # 1. Pruning
            if lp_obj <= best_obj_local + self.tol:
                continue

            # 2. Integrality Check
            non_integral = [
                v for v, val in x_sol.items() if self.tol < val < 1.0 - self.tol
            ]

            if not non_integral:
                # Integer Solution Found
                # Calculate True Objective: F(S) = 2|E| - lambda(|S|^2 - |S|)
                s_nodes = [v for v, val in x_sol.items() if val > 0.5]
                s_size = len(s_nodes)

                if s_size >= 2:
                    subg = self.G.subgraph(s_nodes)
                    edges = subg.number_of_edges()
                    true_obj = 2 * edges - lambda_val * (s_size**2 - s_size)

                    if true_obj > best_obj_local:
                        best_obj_local = true_obj
                        best_sol_local = set(s_nodes)
                continue

            # 3. Branching (Most Fractional)
            branch_var = min(non_integral, key=lambda v: abs(x_sol[v] - 0.5))

            # Child 1: Force = 1
            child_one = {
                "fixed": node["fixed"].copy(),
                "forbidden": node["forbidden"].copy(),
            }
            child_one["fixed"][branch_var] = 1
            stack.append(child_one)

            # Child 2: Force = 0
            child_zero = {
                "fixed": node["fixed"].copy(),
                "forbidden": node["forbidden"].copy(),
            }
            child_zero["fixed"][branch_var] = 0
            child_zero["forbidden"].add(branch_var)  # Optimization: prevent re-pricing
            stack.append(child_zero)

        return best_sol_local, best_obj_local, current_active_set

    def _solve_and_price(self, active_set, fixed_vars, forbidden, lambda_val):
        """
        Solves Relaxation + Column Generation (Pricing) Loop
        """
        cg_iters = 0
        while True:
            cg_iters += 1

            # --- A. Build & Solve QP Relaxation ---
            model = gp.Model("RMP")
            model.setParam("OutputFlag", 0)

            x = {}
            active_list = list(active_set)

            # Create Variables
            for n in active_list:
                lb, ub = 0.0, 1.0
                if n in fixed_vars:
                    lb = ub = fixed_vars[n]
                x[n] = model.addVar(lb=lb, ub=ub, name=f"x_{n}")

            model.update()

            # Quadratic Objective: 2|E| - lambda * (sum(x)^2 - sum(x))
            # Gurobi maximizes: c'x + 0.5 x'Qx
            # We want: 2(z_edges) + lambda*sum(x) - lambda*(sum(x))^2

            obj_lin = gp.LinExpr()

            # Edge Terms (2 * z_uv)
            # Linearize edges z <= x_u, z <= x_v
            subgraph = self.G.subgraph(active_list)
            for u, v in subgraph.edges():
                z = model.addVar(lb=0, ub=1)
                model.addConstr(z <= x[u])
                model.addConstr(z <= x[v])
                obj_lin += 2 * z

            sum_x = gp.quicksum(x[n] for n in active_list)
            obj_lin += lambda_val * sum_x

            # Set Quadratic Objective
            model.setObjective(obj_lin - lambda_val * (sum_x * sum_x), GRB.MAXIMIZE)

            # Size Constraint
            model.addConstr(sum_x >= self.k, name="size_k")

            model.optimize()

            if model.Status == GRB.INFEASIBLE:
                return {}, -float("inf")

            lp_obj = model.ObjVal
            x_vals = {n: x[n].X for n in active_list}

            # --- B. Pricing (Find entering variable) ---
            # Search Boundary of Active Set
            candidates = set()
            for n in active_list:
                for neighbor in self.G.neighbors(n):
                    if neighbor not in active_set and neighbor not in forbidden:
                        candidates.add(neighbor)

            best_candidate = None
            max_rc = self.tol

            # Gradient Calculation
            current_sum_x = sum(x_vals.values())
            # Marginal Penalty: d/ds [ lambda(s^2 - s) ] = lambda(2s - 1)
            marginal_penalty = lambda_val * (2 * current_sum_x - 1)

            # Size Dual (Shadow Price)
            size_dual = model.getConstrByName("size_k").Pi

            for u in candidates:
                # Marginal Edge Gain: 2 * sum(x_v for v in Neighbors(u) intersection Active)
                neighbor_sum = sum(x_vals.get(v, 0) for v in self.G.neighbors(u))
                edge_gain = 2 * neighbor_sum

                reduced_cost = edge_gain - marginal_penalty + size_dual

                if reduced_cost > max_rc:
                    max_rc = reduced_cost
                    best_candidate = u

            # Logging for the user to see the "Density Barrier"
            if cg_iters % 5 == 0 or best_candidate:
                # Only log occasionally or when adding nodes to avoid clutter
                pass

            if best_candidate:
                active_set.add(best_candidate)
                # print(f"      [CG] Added Node {best_candidate}. RC={max_rc:.4f}. Active Size: {len(active_set)}")
                continue
            else:
                return x_vals, lp_obj

    def _get_initial_active_set(self):
        active = {self.q}
        queue = [self.q]
        visited = {self.q}
        while len(active) < self.k and queue:
            curr = queue.pop(0)
            for n in self.G.neighbors(curr):
                if n not in visited:
                    visited.add(n)
                    active.add(n)
                    queue.append(n)
                    if len(active) >= self.k:
                        break

        # Add strict neighbors to ensure connectivity
        for n in self.G.neighbors(self.q):
            active.add(n)
        return active

    def _calculate_density(self, nodes):
        if len(nodes) < 2:
            return 0.0
        subgraph = self.G.subgraph(nodes)
        return (2.0 * subgraph.number_of_edges()) / (len(nodes) * (len(nodes) - 1))


# --- Graph Generation & Execution ---


def generate_complex_graph(n_total=10000, n_community=20, p_community=0.9, seed=42):
    """
    Generates a scale-free graph with an injected dense community.

    Parameters:
    - n_total (int): Total number of nodes in the background graph.
    - n_community (int): Number of nodes in the dense subgraph to inject.
    - p_community (float): Target density for the injected subgraph (0.0 to 1.0).
    - seed (int): Random seed for reproducibility.

    Returns:
    - G (nx.Graph): The generated graph.
    - q (int): The query node (guaranteed to be inside the community).
    - community_nodes (set): The set of nodes belonging to the ground-truth community.
    """
    random.seed(seed)
    # 1. Base Graph: Barabasi-Albert (Scale-Free)
    # m=2 means new nodes attach to 2 existing nodes. This creates hubs.
    logger.info(f"Generating base graph (N={n_total})...")
    G = nx.barabasi_albert_graph(n=n_total, m=2, seed=seed)

    # 2. Select Nodes for Injection
    # We pick nodes from the "middle" of the ID range (e.g., 25% to 75%)
    # to avoid the massive hubs at the start (0, 1, 2...) and the leaves at the end.
    start_range = int(n_total * 0.25)
    end_range = int(n_total * 0.75)

    # Ensure range is valid
    if end_range - start_range < n_community:
        start_range, end_range = 0, n_total  # Fallback to full range

    community_nodes = set(random.sample(range(start_range, end_range), n_community))
    community_list = list(community_nodes)
    q = community_list[0]  # The query node is the first one in the list

    # 3. Inject Edges to Reach Target Density
    # Max possible edges in a subgraph of size k is k*(k-1)/2
    max_edges = (n_community * (n_community - 1)) // 2
    target_edges = int(max_edges * p_community)

    # Count existing edges within the chosen set (from the base graph generation)
    current_edges = 0
    subgraph_edges = []

    # Identify existing edges
    for i in range(n_community):
        for j in range(i + 1, n_community):
            u, v = community_list[i], community_list[j]
            if G.has_edge(u, v):
                current_edges += 1
            else:
                subgraph_edges.append((u, v))

    edges_needed = target_edges - current_edges

    if edges_needed > 0:
        logger.info(
            f"Injecting {edges_needed} edges to reach density {p_community:.2f}..."
        )
        # Shuffle potential non-edges and pick the needed amount
        random.shuffle(subgraph_edges)
        edges_to_add = subgraph_edges[:edges_needed]
        G.add_edges_from(edges_to_add)
    else:
        logger.info(
            f"Selected nodes already have density >= {p_community:.2f}. No edges added."
        )

    return G, q, community_nodes


def get_subgraph_density(G, nodes):
    """
    Computes the exact edge density of a subgraph induced by 'nodes'.
    Formula: 2|E| / (|S|(|S|-1))
    """
    n = len(nodes)
    if n < 2:
        return 0.0

    # Induce the subgraph to count internal edges
    subg = G.subgraph(nodes)
    m = subg.number_of_edges()

    density = (2.0 * m) / (n * (n - 1))
    return density


if __name__ == "__main__":
    # 1. Setup
    G, q, true_community = generate_complex_graph(
        n_total=1_000_000, n_community=100, p_community=0.9
    )

    # 2. Solve
    solver = DensestSubgraphSolver(G, q=q, k=10)
    found_nodes, density = solver.solve()

    # 3. Analysis
    true_density = get_subgraph_density(G, true_community)

    print("\n" + "=" * 40)
    print("COMPARISON")
    print("=" * 40)
    print(f"Ground Truth Density: {true_density:.4f} (Size: {len(true_community)})")
    print(f"Algorithm Density:    {density:.4f} (Size: {len(found_nodes)})")

    # 2. Verdict
    if density >= true_density - 1e-4:
        print(
            ">> SUCCESS: Algorithm found a community as dense or denser than the ground truth."
        )
    else:
        print(">> SUBOPTIMAL: The algorithm got stuck in a local optimum.")

    # Calculate Precision/Recall relative to the injected community
    intersection = len(true_community.intersection(found_nodes))
    precision = intersection / len(found_nodes)
    recall = intersection / len(true_community)

    print(f"Found Size: {len(found_nodes)}")
    print(f"True Community Size: {len(true_community)}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(
        f"Active Set Size (Final): {len(solver._get_initial_active_set())} -> Ended at optimized local size."
    )
