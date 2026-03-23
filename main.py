import networkx as nx
import gurobipy as gp
from gurobipy import GRB
import random
import logging
import time
from itertools import combinations

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("gurobipy").setLevel(logging.ERROR)

# --- FLUSH GUROBI STARTUP LOGS ---
_ghost = gp.Model("ghost")
_ghost.setParam("OutputFlag", 0)


# ==========================================
# 1. GRAPH FACTORY & ORACLE
# ==========================================
def generate_fast_scale_free_dag(
    n_total=100_000, n_community=20, p_community=0.8, m_edges=100, seed=42
):
    """
    Generates a scale-free DAG.
    m_edges: The number of background edges (citations) each new node creates.
    """
    random.seed(seed)

    # 1. Fast Background Graph (Scale-Free)
    # m_edges controls the out-degree of our simulated papers
    G_bg = nx.barabasi_albert_graph(n_total, m=m_edges, seed=seed)

    G = nx.DiGraph()
    G.add_nodes_from(range(n_total))

    # 2. Enforce DAG Topology (Lower ID -> Higher ID prevents cycles)
    directed_edges = [(u, v) if u < v else (v, u) for u, v in G_bg.edges()]
    G.add_edges_from(directed_edges)

    # 3. Plant Dense Community
    start_idx = n_total // 2
    community = set(range(start_idx, start_idx + n_community))
    q_node = start_idx

    # 4. Inject Edges for Density
    edges_to_add = []
    for u, v in combinations(community, 2):
        if u > v:
            u, v = v, u

        if random.random() < p_community:
            edges_to_add.append((u, v))

    G.add_edges_from(edges_to_add)

    return G, q_node, community


class DAGOracle:
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


def calculate_true_density(G_true, nodes):
    n = len(nodes)
    if n < 2:
        return 0.0
    subg = G_true.subgraph(nodes)
    return subg.number_of_edges() / (n * (n - 1))


# ==========================================
# 2. FULL BRANCH-AND-PRICE SOLVER
# ==========================================
class FullBranchAndPriceSolver:
    def __init__(self, oracle: DAGOracle, q: int, k: int, tol=1e-5):
        self.oracle = oracle
        self.q = q
        self.k = k
        self.tol = tol

        self.V_active = set()
        self.F = set()
        self.E_known = set()

        self._initialize_warm_start()

    def _initialize_warm_start(self):
        """BFS to get initial feasible basis and expose frontier."""
        self.V_active.add(self.q)
        queue = [self.q]

        while len(self.V_active) < self.k and queue:
            curr = queue.pop(0)
            in_n, out_n = self.oracle.query(curr)
            for neighbor in in_n + out_n:
                if neighbor in in_n:
                    self.E_known.add((neighbor, curr))
                else:
                    self.E_known.add((curr, neighbor))

                if neighbor not in self.V_active:
                    self.V_active.add(neighbor)
                    queue.append(neighbor)
                    if len(self.V_active) >= self.k:
                        break

        # Fully map the active frontier
        for v in list(self.V_active):
            if v not in self.oracle.revealed_nodes:
                in_n, out_n = self.oracle.query(v)
                for u in in_n:
                    self.E_known.add((u, v))
                for w in out_n:
                    self.E_known.add((v, w))
            else:
                in_n = list(self.oracle._G.predecessors(v))
                out_n = list(self.oracle._G.successors(v))
                for u in in_n:
                    self.E_known.add((u, v))
                for w in out_n:
                    self.E_known.add((v, w))

        for u, w in self.E_known:
            if u in self.V_active and w not in self.V_active:
                self.F.add(w)
            if w in self.V_active and u not in self.V_active:
                self.F.add(u)

    def _compute_lookahead_bound(self, v1, lambda_val):
        """Purely combinatorial DAG bound. Cannot be cheated by fractional variables."""
        k_rem = max(0, self.k - len(v1))
        e_fixed = sum(1 for u, v in self.E_known if u in v1 and v in v1)

        # Max cross edges a frontier node can provide to the fixed set
        deltas = []
        for f in self.F:
            cross = sum(
                1 for v in v1 if (f, v) in self.E_known or (v, f) in self.E_known
            )
            deltas.append(cross)

        deltas.sort(reverse=True)
        top_cross = sum(deltas[:k_rem])

        # Combinatorial upper limit: Fixed + Max Cross + Max DAG Internal (s choose 2)
        raw_edges = e_fixed + top_cross + (k_rem * (k_rem - 1)) / 2.0

        s_size = max(self.k, len(v1))
        penalty = lambda_val * (s_size**2 - s_size)

        return raw_edges - penalty

    def get_density(self, nodes):
        n = len(nodes)
        if n < 2:
            return 0.0
        edges = sum(1 for u, w in self.E_known if u in nodes and w in nodes)
        return edges / (n * (n - 1))

    def solve(self):
        """Outer Dinkelbach Loop."""
        lambda_val = self.get_density(self.V_active)
        best_sol = set(self.V_active)
        iteration = 0

        while True:
            iteration += 1
            logger.info(
                f"\n=== DINKELBACH ITERATION {iteration} | Lambda = {lambda_val:.6f} ==="
            )

            # Inner Loop: Exact Branch-and-Price
            current_sol, param_obj = self._branch_and_price_tree(lambda_val)

            if not current_sol:
                logger.info(
                    ">>> CONVERGED: Subproblem yielded no strictly positive improvement."
                )
                break

            current_density = self.get_density(current_sol)
            logger.info(
                f"  [B&P] Tree Finished. Found Exact Integer Density: {current_density:.6f} | Size: {len(current_sol)}"
            )

            if current_density <= lambda_val + self.tol:
                logger.info(
                    ">>> CONVERGED: No denser integer solution found globally. <<<"
                )
                break

            lambda_val = current_density
            best_sol = current_sol

        return best_sol, lambda_val

    def _branch_and_price_tree(self, lambda_val):
        """Custom DFS Branch-and-Bound tree executing CG at every node."""
        stack = [{"v1": {self.q}, "v0": set()}]
        best_obj_global = 0.0  # Dinkelbach requires strictly positive improvement
        best_sol_global = set()

        nodes_processed = 0

        while stack:
            node = stack.pop()
            nodes_processed += 1

            u_local = self._compute_lookahead_bound(node["v1"], lambda_val)
            if u_local <= best_obj_global + self.tol:
                continue

            # 1. Column Generation at current branch
            x_bar, lp_obj = self._cg_loop(node["v1"], node["v0"], lambda_val)

            # 2. Prune by LP Bound
            if lp_obj <= best_obj_global + self.tol:
                continue

            # 3. Integrality Check
            fractional_vars = {
                v: val for v, val in x_bar.items() if self.tol < val < 1.0 - self.tol
            }

            if not fractional_vars:
                # Exact integer found
                s_nodes = [v for v, val in x_bar.items() if val > 0.5]
                if len(s_nodes) >= self.k:
                    # Calculate true parametric objective to avoid rounding drift
                    edges = sum(
                        1 for u, w in self.E_known if u in s_nodes and w in s_nodes
                    )
                    n_s = len(s_nodes)
                    true_obj = edges - lambda_val * (n_s**2 - n_s)

                    if true_obj > best_obj_global:
                        best_obj_global = true_obj
                        best_sol_global = set(s_nodes)
                        logger.info(
                            f"    [*] B&B Node {nodes_processed}: New Integer Incumbent | Parametric Obj: {true_obj:.4f}"
                        )
                continue

            # 4. Branching: Select the most fractional variable (closest to 0.5)
            branch_var = min(
                fractional_vars, key=lambda v: abs(fractional_vars[v] - 0.5)
            )

            # Left Child (Exclude)
            child_0 = {"v1": node["v1"].copy(), "v0": node["v0"].copy()}
            child_0["v0"].add(branch_var)
            stack.append(child_0)

            # Right Child (Include) - Explored first in DFS to find good lower bounds
            child_1 = {"v1": node["v1"].copy(), "v0": node["v0"].copy()}
            child_1["v1"].add(branch_var)
            stack.append(child_1)

        return best_sol_global, best_obj_global

    def _cg_loop(self, v1, v0, lambda_val):
        """Solves the continuous RMP to optimality, generating columns as needed."""
        while True:
            # Solve Restricted Master Problem
            model = gp.Model("RMP")
            model.setParam("OutputFlag", 0)

            x = {}
            for v in self.V_active:
                if v in v0:
                    continue  # Hard exclusion from LP
                lb = 1.0 if v in v1 else 0.0
                x[v] = model.addVar(lb=lb, ub=1.0, name=f"x_{v}")

            if not x:
                return {}, -float("inf")

            obj_expr = gp.LinExpr()

            for u, w in self.E_known:
                if u in x and w in x:
                    y = model.addVar(lb=0, ub=1)
                    model.addConstr(y <= x[u])
                    model.addConstr(y <= x[w])
                    obj_expr += y

            for u, w in combinations(x.keys(), 2):
                w_var = model.addVar(lb=0, ub=1)
                model.addConstr(w_var >= x[u] + x[w] - 1)
                obj_expr -= 2 * lambda_val * w_var

            sum_x = gp.quicksum(x.values())

            # Phase-I Slack to prevent premature infeasibility during branching
            slack = model.addVar(lb=0, ub=self.k)
            size_constr = model.addConstr(sum_x + slack >= self.k)

            BIG_M = 1e6
            obj_expr -= BIG_M * slack

            model.setObjective(obj_expr, GRB.MAXIMIZE)
            model.optimize()

            if model.Status == GRB.INFEASIBLE:
                return {}, -float("inf")

            lp_obj = model.ObjVal
            true_lp_obj = lp_obj + BIG_M * slack.X

            x_bar = {v: x[v].X for v in x}
            pi = size_constr.Pi

            # Pricing Oracle
            sum_x_bar = sum(x_bar.values())
            omega = -(2 * lambda_val * sum_x_bar) - pi

            max_rc = self.tol
            best_f = None

            for f in self.F:
                if f in v0:
                    continue  # Do not generate forbidden nodes

                frac_deg = sum(
                    x_bar[v]
                    for v in x
                    if (f, v) in self.E_known or (v, f) in self.E_known
                )
                rc = frac_deg + omega

                if rc > max_rc:
                    max_rc = rc
                    best_f = f

            if best_f:
                self.V_active.add(best_f)
                self.F.discard(best_f)

                # Expand frontier by querying the new node
                in_n, out_n = self.oracle.query(best_f)
                for u in in_n:
                    self.E_known.add((u, best_f))
                for w in out_n:
                    self.E_known.add((best_f, w))

                for u, w in self.E_known:
                    if u in self.V_active and w not in self.V_active:
                        self.F.add(w)
                    if w in self.V_active and u not in self.V_active:
                        self.F.add(u)
            else:
                return x_bar, true_lp_obj


# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("INITIALIZING EXACT FULL BRANCH-AND-PRICE")
    print("=" * 50)

    t_start = time.time()

    # 1. Graph Generation
    G_dag, q_node, true_community = generate_fast_scale_free_dag(
        n_total=100_000, m_edges=100, n_community=20, p_community=0.8
    )
    oracle = DAGOracle(G_dag)
    t_gen = time.time()
    logger.info(
        f"Graph Generated in {t_gen - t_start:.4f}s. Nodes: {G_dag.number_of_nodes()}, Directed Edges: {G_dag.number_of_edges()}"
    )

    # 2. Initialization & Oracle Warm Start
    solver = FullBranchAndPriceSolver(oracle, q=q_node, k=10)
    t_init = time.time()
    logger.info(
        f"Oracle and Solver Initialized (Warm Start complete) in {t_init - t_gen:.4f}s."
    )

    # 3. Algorithm Execution
    best_nodes, final_internal_density = solver.solve()
    t_solve = time.time()
    logger.info(f"Optimization Loop completed in {t_solve - t_init:.4f}s.")

    # 4. Metric Evaluation
    true_community_density = calculate_true_density(G_dag, true_community)
    algorithm_true_density = calculate_true_density(G_dag, best_nodes)
    intersection = len(true_community.intersection(best_nodes))
    precision = intersection / len(best_nodes) if best_nodes else 0
    t_metrics = time.time()

    # 5. Final Printout
    print("\n" + "=" * 50)
    print("TIMING BREAKDOWN")
    print("=" * 50)
    print(f"Graph Generation Time  : {t_gen - t_start:.4f}s")
    print(f"Warm Start & Init Time : {t_init - t_gen:.4f}s")
    print(f"B&P Optimization Time  : {t_solve - t_init:.4f}s")
    print(f"Metrics Eval Time      : {t_metrics - t_solve:.4f}s")
    print(f"Total Pipeline Time    : {t_metrics - t_start:.4f}s")

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(
        f"True Community Density : {true_community_density:.4f} (Size: {len(true_community)})"
    )
    print(
        f"Algorithm Internal Dens: {final_internal_density:.4f} (Based on revealed edges)"
    )
    print(
        f"Algorithm True Density : {algorithm_true_density:.4f} (Based on ground truth)"
    )
    print(f"Precision wrt Planted  : {precision:.2f} ({intersection} nodes matched)")
    print(f"Oracle Queries Made    : {oracle.queries_made} / {G_dag.number_of_nodes()}")
