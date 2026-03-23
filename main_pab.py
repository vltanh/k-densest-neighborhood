import networkx as nx
import gurobipy as gp
from gurobipy import GRB
import random
import logging
import time
from itertools import combinations

# ==========================================
# 1. LOGGING CONFIGURATION
# ==========================================
# Change to logging.INFO to reduce verbosity, or logging.DEBUG for deep inspection
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress Gurobi's internal stdout to keep our logs clean
global_env = gp.Env(empty=True)
global_env.setParam("OutputFlag", 0)
global_env.start()


# ==========================================
# 2. GRAPH & ORACLE SIMULATION
# ==========================================
def generate_planted_dag(n_total=300, n_community=20, p_community=0.8, seed=42):
    """Generates a topological DAG with a dense contiguous community."""
    random.seed(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n_total))

    # Background DAG (Sparse forward edges)
    p_bg = 4.0 / n_total
    for i in range(n_total):
        for j in range(i + 1, n_total):
            if random.random() < p_bg:
                G.add_edge(i, j)

    # Plant Dense Community
    start_idx = n_total // 2
    community = set(range(start_idx, start_idx + n_community))
    q_node = start_idx

    for u in community:
        for v in range(u + 1, start_idx + n_community):
            if random.random() < p_community:
                G.add_edge(u, v)

    return G, q_node, community


class DAGOracle:
    """Strict online oracle. No global graph methods exposed."""

    def __init__(self, G: nx.DiGraph):
        self._G = G
        self.queries_made = 0
        self.revealed_nodes = set()

    def query(self, v: int):
        if v not in self._G:
            return [], []
        self.queries_made += 1
        self.revealed_nodes.add(v)
        return list(self._G.predecessors(v)), list(self._G.successors(v))


# ==========================================
# 3. STRAIGHTFORWARD COLUMN GENERATION SOLVER
# ==========================================
class StraightforwardCGSolver:
    def __init__(self, oracle: DAGOracle, q: int, k: int, tol=1e-5):
        self.oracle = oracle
        self.q = q
        self.k = k
        self.tol = tol

        # State Tracking
        self.V_active = set()
        self.F = set()  # Frontier (Known neighbors not yet active)
        self.E_known = set()  # Edges known to exist

        self._initialize_warm_start()

    def _initialize_warm_start(self):
        """BFS to get an initial feasible basis of size k and fully expose its frontier."""
        logger.info("Initializing feasible basis via Oracle BFS...")
        self.V_active.add(self.q)
        queue = [self.q]

        # 1. Expand V_active until size k
        while len(self.V_active) < self.k and queue:
            curr = queue.pop(0)
            in_n, out_n = self.oracle.query(curr)

            for neighbor in in_n + out_n:
                if neighbor not in self.V_active:
                    self.V_active.add(neighbor)
                    queue.append(neighbor)
                    if len(self.V_active) >= self.k:
                        break

        # 2. [THE FIX] We MUST query all active nodes to map their external edges.
        # Otherwise, the Frontier is completely blind.
        for v in list(self.V_active):
            if v not in self.oracle.revealed_nodes:
                in_n, out_n = self.oracle.query(v)
                for u in in_n:
                    self.E_known.add((u, v))
                for w in out_n:
                    self.E_known.add((v, w))
            else:
                # Even if queried in step 1, ensure its edges are in E_known
                in_n = list(
                    self.oracle._G.predecessors(v)
                )  # Using backdoor strictly for mapping knowns
                out_n = list(self.oracle._G.successors(v))
                for u in in_n:
                    self.E_known.add((u, v))
                for w in out_n:
                    self.E_known.add((v, w))

        # 3. Rebuild the Frontier rigorously
        self.F.clear()
        for u, w in self.E_known:
            if u in self.V_active and w not in self.V_active:
                self.F.add(w)
            if w in self.V_active and u not in self.V_active:
                self.F.add(u)

        logger.info(
            f"Warm start complete. V_active size: {len(self.V_active)}, Frontier size: {len(self.F)}"
        )

    def _expand_frontier(self, v, skip_query=False):
        """Queries oracle and updates the frontier/edges."""
        if not skip_query:
            in_n, out_n = self.oracle.query(v)
            for u in in_n:
                self.E_known.add((u, v))
            for w in out_n:
                self.E_known.add((v, w))

        # Any node involved in a known edge with V_active that isn't active goes to Frontier
        for u, w in self.E_known:
            if u in self.V_active and w not in self.V_active:
                self.F.add(w)
            if w in self.V_active and u not in self.V_active:
                self.F.add(u)

    def get_density(self, nodes):
        """Internal density based STRICTLY on edges revealed by the Oracle."""
        n = len(nodes)
        if n < 2:
            return 0.0
        edges = sum(1 for u, w in self.E_known if u in nodes and w in nodes)
        return edges / (n * (n - 1))

    def solve(self):
        """Main Dinkelbach Loop."""
        lambda_val = self.get_density(self.V_active)
        best_sol = set()
        iteration = 0

        logger.info("Starting Dinkelbach Optimization Loop...")

        while True:
            iteration += 1
            logger.info(
                f"\n=== DINKELBACH ITERATION {iteration} | Lambda = {lambda_val:.6f} ==="
            )

            # --- 1. Column Generation (Continuous Relaxation) ---
            cg_iter = 0
            while True:
                cg_iter += 1
                x_bar, pi, lp_obj = self._solve_rmp(lambda_val, continuous=True)

                # Pricing Oracle
                sum_x = sum(x_bar.values())

                # Dual subtraction: - pi (Gurobi pi is typically <= 0 for Maximize >= constr)
                # Dinkelbach derivative penalty: - 2 * lambda * sum_x
                omega = -(2 * lambda_val * sum_x) - pi

                max_rc = self.tol
                best_f = None

                for f in self.F:
                    # Sum of fractional active neighbors
                    frac_deg = sum(
                        x_bar[v]
                        for v in self.V_active
                        if (f, v) in self.E_known or (v, f) in self.E_known
                    )
                    rc = frac_deg + omega

                    if rc > max_rc:
                        max_rc = rc
                        best_f = f

                if best_f:
                    logger.debug(
                        f"  [CG {cg_iter:02d}] LP Obj: {lp_obj:.4f} | Pi: {pi:.4f} | Max RC: {max_rc:.4f} -> Added Node {best_f}"
                    )
                    self.V_active.add(best_f)
                    self.F.discard(best_f)
                    self._expand_frontier(best_f)
                else:
                    logger.debug(
                        f"  [CG {cg_iter:02d}] LP Relaxation Converged. LP Obj: {lp_obj:.4f}"
                    )
                    break

            # --- 2. Solve Exact Integer Program over Generated Columns ---
            logger.info(
                f"  [MIP] Solving Integer Program over {len(self.V_active)} generated columns..."
            )
            x_int, _, mip_obj = self._solve_rmp(lambda_val, continuous=False)

            current_sol = {v for v, val in x_int.items() if val > 0.5}
            current_density = self.get_density(current_sol)

            logger.info(
                f"  [MIP] MIP Obj: {mip_obj:.4f} | Integer Density Found: {current_density:.6f} | Size: {len(current_sol)}"
            )

            # --- 3. Check Convergence ---
            if current_density <= lambda_val + self.tol:
                logger.info(">>> CONVERGED: No denser integer solution found. <<<")
                break

            lambda_val = current_density
            best_sol = current_sol

        return best_sol, lambda_val

    def _solve_rmp(self, lambda_val, continuous=True):
        """Constructs and solves the McCormick Gurobi model."""
        model = gp.Model("RMP", env=global_env)
        model.setParam("OutputFlag", 0)

        if not continuous:
            model.setParam("TimeLimit", 30)  # 30 seconds max per MIP phase
            # model.setParam("MIPGap", 0.05)  # Accept a solution within 5% of optimal

        vtype = GRB.CONTINUOUS if continuous else GRB.BINARY

        # Variables
        x = {}
        for v in self.V_active:
            lb = 1.0 if v == self.q else 0.0
            x[v] = model.addVar(lb=lb, ub=1.0, vtype=vtype, name=f"x_{v}")

        obj_expr = gp.LinExpr()

        # Edge Variables (y_uv)
        for u, v in self.E_known:
            if u in self.V_active and v in self.V_active:
                y = model.addVar(lb=0, ub=1, vtype=vtype)
                model.addConstr(y <= x[u])
                model.addConstr(y <= x[v])
                obj_expr += y

        # McCormick Linearization Variables (w_uv) for |S|^2 penalty
        for u, v in combinations(self.V_active, 2):
            w = model.addVar(lb=0, ub=1, vtype=vtype)
            model.addConstr(w >= x[u] + x[v] - 1)
            obj_expr -= 2 * lambda_val * w

        # Constraints
        sum_x = gp.quicksum(x.values())
        size_constr = model.addConstr(sum_x >= self.k, name="SizeK")

        model.setObjective(obj_expr, GRB.MAXIMIZE)
        model.optimize()

        status = model.Status

        if status == GRB.OPTIMAL:
            # GRB.OPTIMAL means it either proved absolute optimality OR successfully hit your target MIPGap.
            if not continuous:
                logger.info(
                    f"  [Gurobi] MIP Solved to Target (Achieved Gap: {model.MIPGap:.2%})"
                )

        elif status == GRB.TIME_LIMIT:
            # Hit the 30-second TimeLimit. Check if it found at least one valid integer solution.
            if not continuous and model.SolCount > 0:
                logger.warning(
                    f"  [Gurobi] MIP Time Limit Reached! Using best found incumbent. (Gap: {model.MIPGap:.2%})"
                )
            else:
                logger.error("  [Gurobi] TIME LIMIT HIT - NO INTEGER SOLUTION FOUND.")
                return {}, 0.0, -float("inf")

        elif status == GRB.INFEASIBLE:
            logger.error("  [Gurobi] INFEASIBLE MODEL.")
            return {}, 0.0, -float("inf")

        elif status == GRB.INTERRUPTED:
            logger.warning("  [Gurobi] Run was manually interrupted.")

        else:
            logger.warning(
                f"  [Gurobi] Terminated with unexpected status code: {status}"
            )

        x_vals = {v: x[v].X for v in self.V_active}
        pi = size_constr.Pi if continuous else 0.0

        return x_vals, pi, model.ObjVal


def calculate_true_density(G_true, nodes):
    """External metric evaluator using the omniscient ground-truth graph."""
    n = len(nodes)
    if n < 2:
        return 0.0
    # Induce subgraph on the true graph to count all actual edges
    subg = G_true.subgraph(nodes)
    return subg.number_of_edges() / (n * (n - 1))


# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    start_time = time.time()

    # 1. Generate Graph
    G_dag, q_node, true_community = generate_planted_dag(
        n_total=300, n_community=20, p_community=0.8
    )
    logger.info(
        f"Graph Generated. Nodes: {G_dag.number_of_nodes()}, Directed Edges: {G_dag.number_of_edges()}"
    )

    # 2. Setup Oracle & Solver
    oracle = DAGOracle(G_dag)
    solver = StraightforwardCGSolver(oracle, q=q_node, k=100)

    # 3. Solve
    best_nodes, final_internal_density = solver.solve()

    # 4. External Metrics Evaluation
    true_community_density = calculate_true_density(G_dag, true_community)
    algorithm_true_density = calculate_true_density(G_dag, best_nodes)

    intersection = len(true_community.intersection(best_nodes))
    precision = intersection / len(best_nodes) if best_nodes else 0

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(
        f"True Community Density : {true_community_density:.4f} (Size: {len(true_community)})"
    )

    # Show both what the algorithm *thought* it had, and what it *actually* had
    print(
        f"Algorithm Internal Dens: {final_internal_density:.4f} (Based on revealed edges)"
    )
    print(
        f"Algorithm True Density : {algorithm_true_density:.4f} (Based on ground truth)"
    )

    print(f"Precision wrt Planted  : {precision:.2f} ({intersection} nodes matched)")
    print(f"Oracle Queries Made    : {oracle.queries_made} / {G_dag.number_of_nodes()}")
