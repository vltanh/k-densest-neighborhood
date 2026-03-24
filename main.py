import networkx as nx
import gurobipy as gp
from gurobipy import GRB
import random
import logging
import time
from itertools import combinations
from collections import deque

logger = logging.getLogger(__name__)


# ==========================================
# 1. GRAPH FACTORY & ORACLE
# ==========================================
def generate_fast_scale_free_dag(
    n_total=100_000, n_community=20, p_community=0.8, m_edges=100, seed=42
):
    """
    Generates a scale-free Directed Acyclic Graph (DAG) with a planted dense community.
    Uses an isolated random state to prevent global seed contamination.
    """
    rng = random.Random(seed)

    G_bg = nx.barabasi_albert_graph(n_total, m=m_edges, seed=rng)
    G = nx.DiGraph()
    G.add_nodes_from(range(n_total))

    # Enforce DAG topology by strictly directing edges from lower to higher IDs
    G.add_edges_from((min(u, v), max(u, v)) for u, v in G_bg.edges())

    # Randomize planted community placement to avoid the extreme hubs near index 0
    community_start_idx = rng.randint(n_total // 4, n_total - n_community)
    community_nodes = range(community_start_idx, community_start_idx + n_community)
    community = set(community_nodes)
    q_node = community_start_idx

    # Plant the dense target community
    G.add_edges_from(
        (u, v)
        for u, v in combinations(community_nodes, 2)
        if rng.random() < p_community
    )

    return G, q_node, community


class DAGOracle:
    """
    Simulates a localized 1-hop query interface for massive or remote graphs,
    caching responses to track exact query complexity.
    """

    def __init__(self, G: nx.DiGraph):
        self._G = G
        self.queries_made = 0
        self.revealed_nodes = set()
        self._cache = {}

    def query(self, v: int):
        if v not in self._G:
            return [], []
        if v not in self._cache:
            self.queries_made += 1
            self.revealed_nodes.add(v)
            self._cache[v] = (
                list(self._G.predecessors(v)),
                list(self._G.successors(v)),
            )
        return self._cache[v]


def calculate_true_density(G_true: nx.DiGraph, nodes: set):
    """Calculates the absolute density of a node set based on the underlying graph."""
    n = len(nodes)
    if n < 2:
        return 0.0
    subg = G_true.subgraph(nodes)
    return subg.number_of_edges() / (n * (n - 1))


# ==========================================
# 2. EXACT FULL BRANCH-AND-PRICE SOLVER
# ==========================================
class FullBranchAndPriceSolver:
    """
    Exact solver for the Maximum Density Subgraph problem with a size constraint.
    Utilizes Dinkelbach's fractional programming algorithm wrapped around a
    custom stateful Branch-and-Price (Column Generation) tree.
    """

    def __init__(
        self,
        oracle: DAGOracle,
        q: int,
        k: int,
        gurobi_env,
        tol=1e-6,
        bb_node_limit=10_000,
        bb_time_limit=300.0,
        bb_gap_tol=1e-4,
        dinkelbach_max_iter=50,
        cg_batch_fraction=0.1,
        cg_min_batch=5,
        cg_max_batch=50,
    ):
        self.oracle = oracle
        self.q = q
        self.k = k
        self.env = gurobi_env
        self.tol = tol

        self.bb_node_limit = bb_node_limit
        self.bb_time_limit = bb_time_limit
        self.bb_gap_tol = bb_gap_tol
        self.dinkelbach_max_iter = dinkelbach_max_iter

        self.cg_batch_fraction = cg_batch_fraction
        self.cg_min_batch = cg_min_batch
        self.cg_max_batch = cg_max_batch

        self.V_active = set()
        self.F = set()
        self.E_known = set()
        self.pending_edges = set()

        self._initialize_active_set()
        self._init_global_model()

    # ------------------------------------------
    # Initialization & State Management
    # ------------------------------------------
    def _initialize_active_set(self):
        """Executes a BFS to guarantee the initial active set strictly satisfies size k."""
        self.V_active.add(self.q)
        queue = deque([self.q])
        bfs_queried = set()

        while len(self.V_active) < self.k and queue:
            curr = queue.popleft()
            bfs_queried.add(curr)
            preds, succs = self.oracle.query(curr)

            for u in preds:
                self.E_known.add((u, curr))
                self.pending_edges.add((u, curr))
            for w in succs:
                self.E_known.add((curr, w))
                self.pending_edges.add((curr, w))

            for nb in preds + succs:
                if nb not in self.V_active:
                    self.V_active.add(nb)
                    queue.append(nb)
                    if len(self.V_active) >= self.k:
                        break

        # Map internal topology of the initialized active set, avoiding redundant queries
        for v in list(self.V_active):
            if v not in bfs_queried:
                preds, succs = self.oracle.query(v)
                for u in preds:
                    self.E_known.add((u, v))
                    self.pending_edges.add((u, v))
                for w in succs:
                    self.E_known.add((v, w))
                    self.pending_edges.add((v, w))

        # Compile the initial 1-hop frontier
        for u, v in self.E_known:
            if u in self.V_active and v not in self.V_active:
                self.F.add(v)
            if v in self.V_active and u not in self.V_active:
                self.F.add(u)

    def _init_global_model(self):
        """
        Instantiates the persistent Gurobi Restricted Master Problem (RMP).
        Variables and constraints are appended monotonically to prevent memory leaks.
        """
        self.rmp = gp.Model("Persistent_RMP", env=self.env)
        self.rmp.setParam("OutputFlag", 0)

        self.x_vars = {}
        self.y_vars = {}
        self.w_vars = {}

        self.synced_nodes = set()
        self._y_obj_terms = []
        self._w_obj_terms = []

        self.size_constr = self.rmp.addConstr(gp.LinExpr() >= self.k, name="size_k")
        self.last_lambda = -1.0

    # ------------------------------------------
    # Subgraph Combinatorics
    # ------------------------------------------
    def _count_edges_in(self, nodes):
        """Exact O(|S|^2) directed edge count via O(1) hash lookups."""
        node_list = list(nodes)
        n = len(node_list)
        edges = 0
        for i in range(n):
            for j in range(n):
                if i != j and (node_list[i], node_list[j]) in self.E_known:
                    edges += 1
        return edges

    def _density(self, nodes):
        """Calculates subgraph density restricted to locally queried edges."""
        n = len(nodes)
        if n < 2:
            return 0.0
        return self._count_edges_in(nodes) / (n * (n - 1))

    def _parametric_obj(self, nodes, lambda_val):
        """Evaluates the Dinkelbach exact quadratic objective."""
        n = len(nodes)
        return self._count_edges_in(nodes) - lambda_val * (n * n - n)

    def _expand_node(self, f):
        """Absorbs a frontier node into the active set and updates the boundary."""
        self.V_active.add(f)
        self.F.discard(f)
        preds, succs = self.oracle.query(f)

        for u in preds:
            self.E_known.add((u, f))
            self.pending_edges.add((u, f))
            if u not in self.V_active:
                self.F.add(u)
        for w in succs:
            self.E_known.add((f, w))
            self.pending_edges.add((f, w))
            if w not in self.V_active:
                self.F.add(w)

    # ------------------------------------------
    # Stateful RMP Management
    # ------------------------------------------
    def _sync_rmp_structure(self, lambda_val):
        """
        Executes a Delta-Sync, incrementally appending newly discovered nodes, edges,
        and McCormick envelopes to the Gurobi backend.
        """
        structural_changes = False
        new_nodes = self.V_active - self.synced_nodes

        # 1. Delta-Sync Nodes (x variables)
        for v in new_nodes:
            var = self.rmp.addVar(lb=0.0, ub=1.0, name=f"x_{v}")
            self.x_vars[v] = var
            self.rmp.chgCoeff(self.size_constr, var, 1.0)
            structural_changes = True

        # 2. Delta-Sync Edges (y variables)
        processed_edges = set()
        for u, v in self.pending_edges:
            if u in self.x_vars and v in self.x_vars:
                if (u, v) not in self.y_vars:
                    yvar = self.rmp.addVar(lb=0.0, ub=1.0, name=f"y_{u}_{v}")
                    self.rmp.addConstr(yvar <= self.x_vars[u])
                    self.rmp.addConstr(yvar <= self.x_vars[v])
                    self.y_vars[(u, v)] = yvar
                    self._y_obj_terms.append(yvar)
                    structural_changes = True
                processed_edges.add((u, v))

        self.pending_edges -= processed_edges

        # 3. Delta-Sync McCormick Envelopes (w variables)
        if new_nodes:
            for u in new_nodes:
                for v in self.synced_nodes:
                    u_key, v_key = (u, v) if u < v else (v, u)
                    if (u_key, v_key) not in self.w_vars:
                        wvar = self.rmp.addVar(
                            lb=0.0, ub=1.0, name=f"w_{u_key}_{v_key}"
                        )
                        self.rmp.addConstr(
                            wvar >= self.x_vars[u_key] + self.x_vars[v_key] - 1
                        )
                        self.w_vars[(u_key, v_key)] = wvar
                        self._w_obj_terms.append(wvar)
                        structural_changes = True

            for u, v in combinations(sorted(new_nodes), 2):
                if (u, v) not in self.w_vars:
                    wvar = self.rmp.addVar(lb=0.0, ub=1.0, name=f"w_{u}_{v}")
                    self.rmp.addConstr(wvar >= self.x_vars[u] + self.x_vars[v] - 1)
                    self.w_vars[(u, v)] = wvar
                    self._w_obj_terms.append(wvar)
                    structural_changes = True

            self.synced_nodes.update(new_nodes)

        # 4. Objective Realignment
        if structural_changes or lambda_val != self.last_lambda:
            self.rmp.update()

            obj_expr = gp.LinExpr()
            if self._y_obj_terms:
                obj_expr.addTerms([1.0] * len(self._y_obj_terms), self._y_obj_terms)
            if self._w_obj_terms:
                obj_expr.addTerms(
                    [-2.0 * lambda_val] * len(self._w_obj_terms), self._w_obj_terms
                )

            self.rmp.setObjective(obj_expr, GRB.MAXIMIZE)
            self.last_lambda = lambda_val

    def _apply_node_bounds(self, v1, v0):
        """O(|V|) bound mutation for zero-allocation B&B tree traversal."""
        for v, var in self.x_vars.items():
            if v in v1:
                var.LB = 1.0
                var.UB = 1.0
            elif v in v0:
                var.LB = 0.0
                var.UB = 0.0
            else:
                var.LB = 0.0
                var.UB = 1.0

    # ------------------------------------------
    # Column Generation (Pricing)
    # ------------------------------------------
    def _price_frontier(self, x_bar, pi, v0, lambda_val):
        """
        Evaluates the continuous reduced cost of all unexplored frontier nodes.
        Implements Dynamic Batch Pricing to mitigate frontier cascades.
        """
        sum_x_bar = sum(x_bar.values())
        omega = -2 * lambda_val * sum_x_bar - pi

        eligible_frontier = {f for f in self.F if f not in v0}
        frac_degrees = {f: 0.0 for f in eligible_frontier}

        for u, v in self.E_known:
            if u in frac_degrees and v in self.V_active:
                frac_degrees[u] += x_bar.get(v, 0.0)
            if v in frac_degrees and u in self.V_active:
                frac_degrees[v] += x_bar.get(u, 0.0)

        candidates = []
        for f, frac_deg in frac_degrees.items():
            rc = frac_deg + omega
            if rc > self.tol:
                candidates.append((rc, f))

        candidates.sort(reverse=True, key=lambda x: x[0])

        dynamic_limit = int(len(self.V_active) * self.cg_batch_fraction)
        batch_size = max(self.cg_min_batch, min(dynamic_limit, self.cg_max_batch))

        return [f for rc, f in candidates[:batch_size]]

    def _column_generation(self, v1, v0, lambda_val, t_start):
        """
        Solves the continuous RMP to optimality via delayed column generation.
        Maintains scope-safe cache returns to gracefully handle time-limit interrupts.
        """
        local_x_bar = None
        local_lp_obj = float("-inf")

        while True:
            if time.time() - t_start > self.bb_time_limit:
                logger.warning("    [!] Column Generation loop hit global time limit.")
                return local_x_bar, local_lp_obj

            self._sync_rmp_structure(lambda_val)
            self._apply_node_bounds(v1, v0)

            eligible_count = sum(1 for v in self.V_active if v not in v0)
            if eligible_count < self.k:
                return None, float("-inf")

            self.rmp.optimize()

            if self.rmp.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
                return None, float("-inf")

            local_x_bar = {v: var.X for v, var in self.x_vars.items()}
            pi = self.size_constr.Pi
            local_lp_obj = self.rmp.ObjVal

            top_f = self._price_frontier(local_x_bar, pi, v0, lambda_val)

            if top_f:
                for f in top_f:
                    self._expand_node(f)
            else:
                return local_x_bar, local_lp_obj

    # ------------------------------------------
    # Branch-and-Bound
    # ------------------------------------------
    def _branch_and_price(self, lambda_val):
        """
        Custom Depth-First Search executing Column Generation at every node.
        Includes dynamic upper bounding to safely prune shared column pools.
        """
        stack = [{"v1": {self.q}, "v0": set()}]
        best_int_obj = 0.0
        best_int_sol = None

        heuristic_global_ub = float("-inf")
        nodes_explored = 0
        t_start = time.time()

        while stack:
            if nodes_explored >= self.bb_node_limit:
                logger.info(f"    B&B node limit reached ({self.bb_node_limit})")
                break

            if time.time() - t_start > self.bb_time_limit:
                logger.info(f"    B&B time limit reached ({self.bb_time_limit}s)")
                break

            if best_int_obj > self.tol and heuristic_global_ub > float("-inf"):
                global_gap = (heuristic_global_ub - best_int_obj) / max(
                    abs(best_int_obj), self.tol
                )
                if global_gap <= self.bb_gap_tol:
                    logger.info(
                        f"    B&B global gap tolerance reached (gap={global_gap:.6f})"
                    )
                    break

            node = stack.pop()
            nodes_explored += 1

            x_bar, lp_obj = self._column_generation(
                node["v1"], node["v0"], lambda_val, t_start
            )

            if x_bar is None or lp_obj <= best_int_obj + self.tol:
                continue

            if lp_obj > heuristic_global_ub:
                heuristic_global_ub = lp_obj

            if best_int_obj > self.tol:
                local_gap = (lp_obj - best_int_obj) / max(abs(best_int_obj), self.tol)
                if local_gap <= self.bb_gap_tol:
                    continue

            fractional = {
                v: val for v, val in x_bar.items() if self.tol < val < 1.0 - self.tol
            }

            if not fractional:
                sol_nodes = {v for v, val in x_bar.items() if val > 0.5}
                if len(sol_nodes) >= self.k:
                    obj = self._parametric_obj(sol_nodes, lambda_val)
                    if obj > best_int_obj:
                        best_int_obj = obj
                        best_int_sol = sol_nodes
                        logger.info(
                            f"    B&B node {nodes_explored}: new incumbent "
                            f"obj={obj:.4f}, size={len(sol_nodes)}"
                        )
                continue

            branch_var = min(fractional, key=lambda v: abs(fractional[v] - 0.5))

            child_0 = {"v1": node["v1"].copy(), "v0": node["v0"].copy()}
            child_0["v0"].add(branch_var)
            stack.append(child_0)

            child_1 = {"v1": node["v1"].copy(), "v0": node["v0"].copy()}
            child_1["v1"].add(branch_var)
            stack.append(child_1)

        logger.info(f"    B&B finished: {nodes_explored} nodes explored")
        return best_int_sol, best_int_obj

    # ------------------------------------------
    # Main Outer Loop
    # ------------------------------------------
    def solve(self):
        """Outer Dinkelbach sequence seeking the root of the parametric objective."""
        best_sol = set(self.V_active)
        lambda_val = self._density(best_sol)
        logger.info(f"Initial density: {lambda_val:.6f} (size {len(best_sol)})")

        for t in range(1, self.dinkelbach_max_iter + 1):
            logger.info(f"=== DINKELBACH ITERATION {t} | lambda = {lambda_val:.6f} ===")

            sol, param_obj = self._branch_and_price(lambda_val)

            if sol is None or param_obj <= self.tol:
                logger.info(">>> CONVERGED: Z(lambda) <= 0, no improvement found. <<<")
                break

            new_density = self._density(sol)
            logger.info(f"  Found solution: density={new_density:.6f}, size={len(sol)}")

            if new_density <= lambda_val + self.tol:
                logger.info(">>> CONVERGED: density did not increase. <<<")
                break

            lambda_val = new_density
            best_sol = sol

        self.rmp.dispose()
        return best_sol, lambda_val


# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.getLogger("gurobipy").setLevel(logging.ERROR)

    global_env = gp.Env(empty=True)
    global_env.setParam("OutputFlag", 0)
    global_env.setParam("Method", 1)  # Dual Simplex for high numerical stability
    global_env.start()

    print("=" * 50)
    print("K-DENSEST NEIGHBORHOOD")
    print("=" * 50)

    t_start = time.time()

    G_dag, q_node, true_community = generate_fast_scale_free_dag(
        n_total=100_000, m_edges=5, n_community=20, p_community=0.8
    )
    oracle = DAGOracle(G_dag)
    t_gen = time.time()

    logger.info(
        f"Graph generated in {t_gen - t_start:.4f}s. "
        f"Nodes: {G_dag.number_of_nodes()}, Edges: {G_dag.number_of_edges()}"
    )

    solver = FullBranchAndPriceSolver(
        oracle,
        q=q_node,
        k=20,
        gurobi_env=global_env,
        bb_node_limit=100_000,
        bb_time_limit=300.0,
        bb_gap_tol=1e-4,
    )
    t_init = time.time()
    logger.info(f"Solver initialized in {t_init - t_gen:.4f}s.")

    best_nodes, final_density = solver.solve()
    t_solve = time.time()
    logger.info(f"Optimization completed in {t_solve - t_init:.4f}s.")

    true_density = calculate_true_density(G_dag, true_community)
    algo_true_density = calculate_true_density(G_dag, best_nodes)
    intersection = len(true_community & best_nodes)
    precision = intersection / len(best_nodes) if best_nodes else 0

    print("\n" + "=" * 50)
    print("TIMING")
    print("=" * 50)
    print(f"Graph generation  : {t_gen - t_start:.4f}s")
    print(f"Solver init       : {t_init - t_gen:.4f}s")
    print(f"Optimization      : {t_solve - t_init:.4f}s")
    print(f"Total             : {time.time() - t_start:.4f}s")

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"True community    : density={true_density:.4f}, size={len(true_community)}")
    print(f"Algorithm result  : density={final_density:.4f} (internal)")
    print(f"                    density={algo_true_density:.4f} (ground truth)")
    print(f"Precision         : {precision:.2f} ({intersection}/{len(best_nodes)})")
    print(f"Oracle queries    : {oracle.queries_made} / {G_dag.number_of_nodes()}")

    global_env.dispose()
