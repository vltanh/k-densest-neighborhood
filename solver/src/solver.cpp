#include "solver.hpp"
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <climits>
#include <limits>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/push_relabel_max_flow.hpp>

using namespace std;

namespace
{
double relative_gap(double best_bound, double incumbent, double tol)
{
    if (!std::isfinite(best_bound))
        return std::numeric_limits<double>::infinity();
    if (incumbent > tol)
        return std::max(0.0, (best_bound - incumbent) / std::max(std::fabs(incumbent), tol));
    return (best_bound <= tol) ? 0.0 : std::numeric_limits<double>::infinity();
}
}

FullBranchAndPriceSolver::FullBranchAndPriceSolver(IGraphOracle &oracle, int q, int k, GRBEnv &env,
                                                   double tol, int bb_node_limit, double bb_time_limit,
                                                   double bb_gap_tol, int dinkelbach_max_iter,
                                                   double cg_batch_fraction, int cg_min_batch, int cg_max_batch,
                                                   int kappa, double bb_hard_time_limit,
                                                   bool skip_materialize)
    : oracle(oracle), q(q), k(k), env(env), tol(tol), bb_node_limit(bb_node_limit),
      bb_time_limit(bb_time_limit), bb_hard_time_limit(bb_hard_time_limit),
      bb_gap_tol(bb_gap_tol), dinkelbach_max_iter(dinkelbach_max_iter),
      cg_batch_fraction(cg_batch_fraction), cg_min_batch(cg_min_batch), cg_max_batch(cg_max_batch),
      kappa(std::max(0, kappa)), skip_materialize(skip_materialize), rmp(nullptr)
{
    _initialize_active_set();
    _init_global_model();
}

FullBranchAndPriceSolver::~FullBranchAndPriceSolver()
{
    if (rmp)
        delete rmp;
}

// ── Graph Management ──────────────────────────────────────────────────────────

void FullBranchAndPriceSolver::_add_edge(int u, int v)
{
    if (u == v)
        return;

    if (!adj_out[u].count(v))
    {
        adj_out[u].insert(v);
        adj_in[v].insert(u);
        pending_edges.push_back({u, v});
    }
}

// Registers preds → v and v → succs into adj_out/adj_in and promotes any new
// non-active endpoint into the frontier F. Skips blacklisted endpoints.
void FullBranchAndPriceSolver::_ingest_neighbors(int v, const vector<int> &preds, const vector<int> &succs)
{
    for (int u : preds)
    {
        if (error_nodes.count(u))
            continue;
        _add_edge(u, v);
        if (V_active.find(u) == V_active.end())
            F.insert(u);
    }
    for (int w : succs)
    {
        if (error_nodes.count(w))
            continue;
        _add_edge(v, w);
        if (V_active.find(w) == V_active.end())
            F.insert(w);
    }
}

// BFS from q to seed V_active with ≥ k nodes. Every node popped from the queue
// is queried before insertion into V_active, so each active member has a
// materialised adjacency by the time the function returns. The post-loop sweep
// over V_active is a no-op safety net: it would re-query any active member
// missing from bfs_queried, which the current code path never produces.
// Throws std::runtime_error if the query node itself cannot be fetched.
void FullBranchAndPriceSolver::_initialize_active_set()
{
    queue<int> q_nodes;
    q_nodes.push(q);
    unordered_set<int> bfs_queried;

    while (V_active.size() < (size_t)k && !q_nodes.empty())
    {
        int curr = q_nodes.front();
        q_nodes.pop();

        if (error_nodes.count(curr))
            continue;
        bfs_queried.insert(curr);

        try
        {
            const auto &[preds, succs] = oracle.query(curr);
            queried_nodes.insert(curr);

            V_active.insert(curr);
            F.erase(curr);

            _ingest_neighbors(curr, preds, succs);

            for (int u : preds)
                if (V_active.find(u) == V_active.end() && !error_nodes.count(u))
                    q_nodes.push(u);
            for (int w : succs)
                if (V_active.find(w) == V_active.end() && !error_nodes.count(w))
                    q_nodes.push(w);
        }
        catch (const std::exception &e)
        {
            std::cerr << "[" << get_timestamp() << "] Blacklisting node " << oracle.mapper.get_str(curr) << " during initialization: " << e.what() << "\n";
            error_nodes.insert(curr);
            F.erase(curr);
            V_active.erase(curr);
        }
    }

    // Safety net: query any V_active member that is not in bfs_queried. The
    // current first-pass logic inserts into V_active only after a successful
    // query, so this loop body is unreachable in the present code; it is kept
    // as defense against future paths that might promote a node before
    // querying it.
    vector<int> active_copy(V_active.begin(), V_active.end());
    for (int v : active_copy)
    {
        if (bfs_queried.find(v) != bfs_queried.end() || error_nodes.count(v))
            continue;
        try
        {
            const auto &[preds, succs] = oracle.query(v);
            queried_nodes.insert(v);
            _ingest_neighbors(v, preds, succs);
        }
        catch (const std::exception &e)
        {
            std::cerr << "[" << get_timestamp() << "] Blacklisting node " << oracle.mapper.get_str(v) << " during post-init: " << e.what() << "\n";
            error_nodes.insert(v);
            V_active.erase(v);
        }
    }

    if (error_nodes.count(q))
        throw std::runtime_error("Fatal: The query node itself failed to fetch!");
}

// ── Gurobi Model Initialisation ───────────────────────────────────────────────

void FullBranchAndPriceSolver::_init_global_model()
{
    rmp = new GRBModel(env);
    rmp->set(GRB_IntParam_OutputFlag, 0);

    GRBLinExpr expr = 0;
    size_constr = rmp->addConstr(expr >= k, "size_k");

    // Fix the optimisation sense once. Per-variable Obj coefficients are set
    // at addVar time (y = 1, w = −2λ) and adjusted only when λ changes, so we
    // never need to rebuild the full objective expression afterwards.
    rmp->setObjective(GRBLinExpr(0.0), GRB_MAXIMIZE);
}

// ── Density Utilities ─────────────────────────────────────────────────────────

// Returns the number of directed edges with both endpoints in `nodes`.
int FullBranchAndPriceSolver::_count_edges_in(const unordered_set<int> &nodes)
{
    int edges = 0;
    for (int u : nodes)
    {
        auto it = adj_out.find(u);
        if (it != adj_out.end())
        {
            for (int v : it->second)
                if (nodes.find(v) != nodes.end())
                    edges++;
        }
    }
    return edges;
}

// Returns d(S) = |E(S)| / (|S|·(|S|−1)), the directed edge density of `nodes`.
// Returns 0 for sets smaller than 2.
double FullBranchAndPriceSolver::_density(const unordered_set<int> &nodes)
{
    double n = nodes.size();
    if (n < 2)
        return 0.0;
    return (double)_count_edges_in(nodes) / (n * (n - 1));
}

// Returns the Dinkelbach parametric objective f(S, λ) = |E(S)| − λ·|S|·(|S|−1).
double FullBranchAndPriceSolver::_parametric_obj(const unordered_set<int> &nodes, double lambda_val)
{
    double n = nodes.size();
    return (double)_count_edges_in(nodes) - lambda_val * (n * n - n);
}

// ── Incumbent Refinement ──────────────────────────────────────────────────────

// Greedily removes the worst node from `sol_nodes` until size = k or no single
// removal strictly improves the metric. The query node is never removed.
// maximize_density=true uses d(S) (for the BFS seed); false uses f(S, λ) (for
// B&B integer solutions).
//
// Implementation: maintains an internal-degree cache so the greedy step is O(|S|)
// per removal instead of O(|S|²·E). For both metrics, the node that maximises
// improvement when removed is exactly the node with the smallest internal degree
// (i.e. the fewest intra-set incident directed edges), since for fixed (E, n):
//   Δ density = 2E·(n-1)/(n(n-1)²(n-2)) − d_u/((n-1)(n-2))   (decreasing in d_u)
//   Δ f(S,λ)  = 2λ(n-1) − d_u                                 (decreasing in d_u)
void FullBranchAndPriceSolver::_prune_discrete_solution(unordered_set<int> &sol_nodes, double lambda_val, bool maximize_density, bool enforce_connectivity)
{
    if (sol_nodes.size() <= (size_t)k)
        return;

    // Precompute internal degree per node and the directed edge count E once.
    unordered_map<int, int> internal_deg;
    internal_deg.reserve(sol_nodes.size() * 2);
    for (int u : sol_nodes)
        internal_deg[u] = 0;

    int E = 0;
    for (int u : sol_nodes)
    {
        auto it = adj_out.find(u);
        if (it == adj_out.end())
            continue;
        for (int v : it->second)
        {
            if (sol_nodes.find(v) != sol_nodes.end())
            {
                ++E;
                ++internal_deg[u];
                ++internal_deg[v];
            }
        }
    }

    auto metric_for = [&](int n, int e) -> double
    {
        if (maximize_density)
            return (n < 2) ? 0.0 : (double)e / ((double)n * (n - 1));
        return (double)e - lambda_val * ((double)n * (n - 1));
    };

    int initial_size = (int)sol_nodes.size();
    double initial_metric = metric_for(initial_size, E);
    int nodes_removed = 0;

    cout << "[" << get_timestamp() << "]     > Pruning discrete solution (Initial Size: " << initial_size
         << " | " << (maximize_density ? "Density: " : "Param Obj: ") << fixed << setprecision(6) << initial_metric << ")" << endl;

    while (sol_nodes.size() > (size_t)k)
    {
        int n = (int)sol_nodes.size();

        int worst_node = -1;
        int worst_deg = INT_MAX;

        for (int u : sol_nodes)
        {
            if (u == q)
                continue;

            int d = internal_deg[u];

            // Only evaluate if this is a new candidate minimum
            if (d < worst_deg)
            {
                // ==========================================================
                // CONNECTIVITY PROTECTION (Micro-BFS)
                // ==========================================================
                if (enforce_connectivity)
                {
                    unordered_set<int> visited;
                    queue<int> test_q;
                    test_q.push(this->q);
                    visited.insert(this->q);

                    while (!test_q.empty())
                    {
                        int curr = test_q.front();
                        test_q.pop();

                        auto out_it = adj_out.find(curr);
                        if (out_it != adj_out.end())
                        {
                            for (int v : out_it->second)
                            {
                                if (v != u && sol_nodes.find(v) != sol_nodes.end() && visited.insert(v).second)
                                {
                                    test_q.push(v);
                                }
                            }
                        }
                        auto in_it = adj_in.find(curr);
                        if (in_it != adj_in.end())
                        {
                            for (int v : in_it->second)
                            {
                                if (v != u && sol_nodes.find(v) != sol_nodes.end() && visited.insert(v).second)
                                {
                                    test_q.push(v);
                                }
                            }
                        }
                    }

                    // If removing 'u' breaks connectivity, skip it!
                    if (visited.size() < sol_nodes.size() - 1)
                    {
                        continue;
                    }
                }
                // ==========================================================

                // If we get here, it's safe to remove and is the new minimum
                worst_deg = d;
                worst_node = u;
            }
        }

        if (worst_node == -1)
            break;

        double cur_metric = metric_for(n, E);
        double new_metric = metric_for(n - 1, E - worst_deg);
        double improvement = new_metric - cur_metric;
        if (improvement <= 1e-7)
            break;

        // Commit the removal: delete the node, shrink E, and decrement the
        // internal degree of every neighbour (once per incident directed edge).
        sol_nodes.erase(worst_node);
        E -= worst_deg;

        auto out_it = adj_out.find(worst_node);
        if (out_it != adj_out.end())
            for (int v : out_it->second)
                if (sol_nodes.find(v) != sol_nodes.end())
                    --internal_deg[v];
        auto in_it = adj_in.find(worst_node);
        if (in_it != adj_in.end())
            for (int v : in_it->second)
                if (sol_nodes.find(v) != sol_nodes.end())
                    --internal_deg[v];
        internal_deg.erase(worst_node);

        ++nodes_removed;
        cout << "[" << get_timestamp() << "]       - Pruned node " << oracle.mapper.get_str(worst_node)
             << " (Improvement: +" << fixed << setprecision(6) << improvement
             << " | New Size: " << sol_nodes.size() << ")" << endl;
    }

    if (nodes_removed > 0)
    {
        double final_metric = metric_for((int)sol_nodes.size(), E);
        cout << "[" << get_timestamp() << "]     > Pruning complete. Removed " << nodes_removed
             << " nodes. (Final Size: " << sol_nodes.size()
             << " | " << (maximize_density ? "Density: " : "Param Obj: ") << fixed << setprecision(6) << final_metric << ")" << endl;
    }
    else
    {
        cout << "[" << get_timestamp() << "]     > Pruning complete. No nodes removed (solution is already strictly minimal)." << endl;
    }
}

// Edge-connectivity verification for a pruned integer solution. Builds a flow
// network from adj_out restricted to sol_nodes (each directed edge contributes
// unit capacity, doubled for the undirected view) and runs q→t max-flow for
// every other vertex. Returns false on the first t whose max-flow falls below
// kappa.
bool FullBranchAndPriceSolver::_verify_kappa_connectivity(const std::unordered_set<int> &sol_nodes)
{
    if (kappa <= 0)
        return true;
    if (!sol_nodes.count(q))
        return false;
    if (sol_nodes.size() < 2)
        return false;

    std::unordered_map<int, int> g_to_b;
    std::vector<int> b_to_g;
    g_to_b.reserve(sol_nodes.size() * 2);
    b_to_g.reserve(sol_nodes.size());
    for (int node_id : sol_nodes)
    {
        g_to_b[node_id] = (int)b_to_g.size();
        b_to_g.push_back(node_id);
    }
    int N = (int)sol_nodes.size();

    typedef boost::adjacency_list_traits<boost::vecS, boost::vecS, boost::directedS> Traits;
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
                                  boost::property<boost::vertex_name_t, int>,
                                  boost::property<boost::edge_capacity_t, long long,
                                                  boost::property<boost::edge_residual_capacity_t, long long,
                                                                  boost::property<boost::edge_reverse_t, Traits::edge_descriptor>>>>
        FlowGraph;

    FlowGraph fg(N);
    auto capacity = boost::get(boost::edge_capacity, fg);
    auto rev = boost::get(boost::edge_reverse, fg);

    auto add_flow_edge = [&](int u, int v, long long cap)
    {
        auto e1 = boost::add_edge(u, v, fg).first;
        auto e2 = boost::add_edge(v, u, fg).first;
        capacity[e1] = cap;
        capacity[e2] = 0;
        rev[e1] = e2;
        rev[e2] = e1;
    };

    // Collect each undirected pair once. A symmetric directed input emits both
    // (u,v) and (v,u) in adj_out; both invocations of add_flow_edge would
    // otherwise double the unit capacity that models a single undirected edge.
    std::vector<std::pair<int, int>> undirected_pairs;
    {
        std::unordered_set<long long> seen;
        undirected_pairs.reserve(sol_nodes.size() * 4);
        for (int u : sol_nodes)
        {
            auto it = adj_out.find(u);
            if (it == adj_out.end())
                continue;
            for (int v : it->second)
            {
                if (u == v)
                    continue;
                if (!sol_nodes.count(v))
                    continue;
                int a = std::min(u, v);
                int b = std::max(u, v);
                long long key = ((long long)(unsigned int)a << 32) | (unsigned int)b;
                if (!seen.insert(key).second)
                    continue;
                undirected_pairs.emplace_back(a, b);
            }
        }
    }

    for (const auto &pr : undirected_pairs)
    {
        int bu = g_to_b[pr.first];
        int bv = g_to_b[pr.second];
        add_flow_edge(bu, bv, 1);
        add_flow_edge(bv, bu, 1);
    }

    int S = g_to_b[q];
    for (int target_node : sol_nodes)
    {
        if (target_node == q)
            continue;
        int T = g_to_b[target_node];
        // Reset residual capacities by recomputing from scratch each call: the
        // push_relabel implementation consumes the residual property, so we
        // build a fresh graph per target to keep verification self-contained.
        FlowGraph fg_iter(N);
        auto cap_iter = boost::get(boost::edge_capacity, fg_iter);
        auto rev_iter = boost::get(boost::edge_reverse, fg_iter);
        auto add_iter_edge = [&](int u, int v, long long cap)
        {
            auto e1 = boost::add_edge(u, v, fg_iter).first;
            auto e2 = boost::add_edge(v, u, fg_iter).first;
            cap_iter[e1] = cap;
            cap_iter[e2] = 0;
            rev_iter[e1] = e2;
            rev_iter[e2] = e1;
        };
        for (const auto &pr : undirected_pairs)
        {
            int bu = g_to_b[pr.first];
            int bv = g_to_b[pr.second];
            add_iter_edge(bu, bv, 1);
            add_iter_edge(bv, bu, 1);
        }
        long long flow = boost::push_relabel_max_flow(fg_iter, S, T);
        if (flow < kappa)
            return false;
    }
    return true;
}

// ── Dynamic Graph Expansion ───────────────────────────────────────────────────

// Promotes frontier node f to V_active, loading its edges via the oracle.
// On failure, f is blacklisted and dropped from F.
void FullBranchAndPriceSolver::_expand_node(int f)
{
    if (error_nodes.count(f))
    {
        F.erase(f);
        return;
    }

    try
    {
        const auto &[preds, succs] = oracle.query(f);
        queried_nodes.insert(f);

        V_active.insert(f);
        F.erase(f);
        _ingest_neighbors(f, preds, succs);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[" << get_timestamp() << "] Blacklisting node " << oracle.mapper.get_str(f) << " due to API error: " << e.what() << "\n";
        error_nodes.insert(f);
        F.erase(f);
        V_active.erase(f);
    }
}

// Fetch f's adjacency from the oracle and populate adj_out / adj_in without
// promoting f to V_active. Used at suspected CG convergence to materialise
// frontier-internal edges that pricing cannot see otherwise: f stays in F (no
// new LP variable), but the joint subset pricer can now evaluate f's true
// degree to other frontier members.
void FullBranchAndPriceSolver::_materialize_adjacency(int f)
{
    if (queried_nodes.count(f) || error_nodes.count(f))
        return;
    try
    {
        const auto &[preds, succs] = oracle.query(f);
        queried_nodes.insert(f);
        _ingest_neighbors(f, preds, succs);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[" << get_timestamp() << "] Blacklisting node " << oracle.mapper.get_str(f) << " during materialisation: " << e.what() << "\n";
        error_nodes.insert(f);
        F.erase(f);
    }
}

// Materialise every frontier node whose adjacency has not yet been queried.
// Returns the number of nodes queried so the caller can decide whether to
// rerun pricing on the newly-visible edges.
int FullBranchAndPriceSolver::_materialize_unqueried_frontier()
{
    std::vector<int> to_query;
    to_query.reserve(F.size());
    for (int f : F)
        if (!queried_nodes.count(f) && !error_nodes.count(f))
            to_query.push_back(f);
    int materialised = 0;
    for (int f : to_query)
    {
        _materialize_adjacency(f);
        if (queried_nodes.count(f))
            ++materialised;
    }
    return materialised;
}

// ── RMP Structure Helpers ─────────────────────────────────────────────────────

// Adds a continuous x_v ∈ [0,1] variable to the RMP for each node in V_active
// not yet in synced_nodes, and registers it in the size constraint Σ xᵥ ≥ k.
// Populates new_nodes with the added nodes. Returns true if any were added.
bool FullBranchAndPriceSolver::_register_new_nodes(vector<int> &new_nodes)
{
    for (int v : V_active)
        if (!synced_nodes.count(v))
            new_nodes.push_back(v);

    if (new_nodes.empty())
        return false;

    for (int v : new_nodes)
        x_vars[v] = rmp->addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "");
    rmp->update();
    for (int v : new_nodes)
        rmp->chgCoeff(size_constr, x_vars[v], 1.0);
    return true;
}

// Processes pending_edges: for each edge (u, v) where both endpoints already
// have x_vars, adds y_{uv} ∈ [0,1] with y_{uv} ≤ x_u and y_{uv} ≤ x_v.
// Each y contributes +1 to the objective, so we set its Obj coefficient at
// addVar time rather than rebuilding the objective later.
// Edges whose endpoints are not yet in the RMP remain in pending_edges.
// Returns true if any y_vars were added.
bool FullBranchAndPriceSolver::_register_pending_edges()
{
    bool changed = false;
    vector<pair<int, int>> remaining;
    vector<pair<pair<int, int>, GRBVar>> new_support_vars;
    vector<pair<pair<int, int>, GRBVar>> support_link_updates;

    for (auto const &uv : pending_edges)
    {
        if (x_vars.count(uv.first) && x_vars.count(uv.second))
        {
            if (!y_vars.count(uv))
            {
                GRBVar yvar = rmp->addVar(0.0, 1.0, 1.0, GRB_CONTINUOUS, "");
                rmp->addConstr(yvar <= x_vars[uv.first]);
                rmp->addConstr(yvar <= x_vars[uv.second]);
                y_vars[uv] = yvar;
                y_obj_terms.push_back(yvar);
                changed = true;

                if (kappa > 0)
                {
                    pair<int, int> support_uv = (uv.first < uv.second) ? uv : make_pair(uv.second, uv.first);
                    auto z_it = z_vars.find(support_uv);
                    if (z_it == z_vars.end())
                    {
                        GRBVar zvar = rmp->addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "");
                        rmp->addConstr(zvar <= x_vars[support_uv.first]);
                        rmp->addConstr(zvar <= x_vars[support_uv.second]);
                        GRBConstr link = rmp->addConstr(zvar - yvar <= 0.0);
                        z_vars[support_uv] = zvar;
                        z_link_constrs[support_uv] = link;
                        new_support_vars.push_back({support_uv, zvar});
                    }
                    else
                    {
                        support_link_updates.push_back({support_uv, yvar});
                    }
                }
            }
        }
        else
        {
            remaining.push_back(uv);
        }
    }
    pending_edges = std::move(remaining);

    if ((!new_support_vars.empty() || !support_link_updates.empty()) && kappa > 0)
    {
        rmp->update();

        for (const auto &[support_uv, yvar] : support_link_updates)
        {
            auto link_it = z_link_constrs.find(support_uv);
            if (link_it != z_link_constrs.end())
                rmp->chgCoeff(link_it->second, yvar, -1.0);
        }

        for (const auto &[support_uv, zvar] : new_support_vars)
        {
            for (const ConnectivityCut &cut : connectivity_cuts)
            {
                bool u_in_R = cut.source_side.count(support_uv.first);
                bool v_in_R = cut.source_side.count(support_uv.second);
                if (u_in_R != v_in_R)
                    rmp->chgCoeff(cut.constr, zvar, 1.0);
            }
        }
    }

    return changed;
}

// Adds a w_{uv} ∈ [0,1] variable for every unordered pair {u, v} that involves
// at least one node from new_nodes. w_{uv} linearises the product x_u · x_v via
// the McCormick lower bound w_{uv} ≥ x_u + x_v − 1. Each w contributes
// −2λ to the objective, so we set its Obj coefficient at addVar time using
// the current λ; _update_objective only has to bulk-adjust existing w vars
// when λ changes between Dinkelbach iterations.
// Updates synced_nodes. Returns true if any w_vars were added.
bool FullBranchAndPriceSolver::_register_pair_vars(const vector<int> &new_nodes, double lambda_val)
{
    if (new_nodes.empty())
        return false;

    const double w_obj_coeff = -2.0 * lambda_val;
    bool changed = false;

    // Pairs between new nodes and already-synced nodes
    for (int u : new_nodes)
    {
        for (int v : synced_nodes)
        {
            pair<int, int> uv = (u < v) ? make_pair(u, v) : make_pair(v, u);
            if (!w_vars.count(uv))
            {
                GRBVar wvar = rmp->addVar(0.0, 1.0, w_obj_coeff, GRB_CONTINUOUS, "");
                rmp->addConstr(wvar >= x_vars[uv.first] + x_vars[uv.second] - 1);
                w_vars[uv] = wvar;
                w_obj_terms.push_back(wvar);
                changed = true;
            }
        }
    }

    // Pairs within the batch of new nodes
    for (size_t i = 0; i < new_nodes.size(); i++)
    {
        for (size_t j = i + 1; j < new_nodes.size(); j++)
        {
            int u = min(new_nodes[i], new_nodes[j]);
            int v = max(new_nodes[i], new_nodes[j]);
            pair<int, int> uv = {u, v};
            if (!w_vars.count(uv))
            {
                GRBVar wvar = rmp->addVar(0.0, 1.0, w_obj_coeff, GRB_CONTINUOUS, "");
                rmp->addConstr(wvar >= x_vars[u] + x_vars[v] - 1);
                w_vars[uv] = wvar;
                w_obj_terms.push_back(wvar);
                changed = true;
            }
        }
    }

    for (int v : new_nodes)
        synced_nodes.insert(v);
    return changed;
}

// Adjusts the objective for a new density estimate λ. The target objective is:
//   maximise  Σ y_{uv}  −  2λ · Σ w_{uv}
//
// y-coefficients are fixed at 1 and set when each y is created, so we only
// need to touch the w-coefficients. New w vars are already registered with the
// right coefficient in _register_pair_vars, so when λ is unchanged this is a
// no-op; when λ changed between Dinkelbach iterations, we bulk-reset every
// existing w var's Obj attribute via per-var .set() (still O(|w|) Gurobi calls
// but O(1) expression construction — vs. the previous O(|y| + |w|) rebuild
// through a full GRBLinExpr on every BB node).
void FullBranchAndPriceSolver::_update_objective(double lambda_val)
{
    if (lambda_val == last_lambda)
        return;

    if (!w_obj_terms.empty())
    {
        rmp->update();
        const double new_coeff = -2.0 * lambda_val;
        for (GRBVar &wvar : w_obj_terms)
            wvar.set(GRB_DoubleAttr_Obj, new_coeff);
    }
    last_lambda = lambda_val;
}

// A zero-branch can leave fewer than k currently-active variables eligible
// even though discovered frontier nodes could repair feasibility. Promote just
// enough non-forbidden frontier nodes before solving the LP so branch nodes are
// not pruned before column generation has a chance to add replacement columns.
bool FullBranchAndPriceSolver::_ensure_active_feasibility(const unordered_set<int> &v0_set)
{
    auto eligible_active_count = [&]() -> int
    {
        int eligible = 0;
        for (int v : V_active)
            if (!v0_set.count(v) && !error_nodes.count(v))
                ++eligible;
        return eligible;
    };

    auto known_incident_score = [&](int f) -> int
    {
        int score = 0;
        auto out_it = adj_out.find(f);
        if (out_it != adj_out.end())
            for (int v : out_it->second)
                if (V_active.count(v) && !v0_set.count(v))
                    ++score;

        auto in_it = adj_in.find(f);
        if (in_it != adj_in.end())
            for (int u : in_it->second)
                if (V_active.count(u) && !v0_set.count(u))
                    ++score;
        return score;
    };

    int eligible = eligible_active_count();
    while (eligible < k)
    {
        int best_f = -1;
        int best_score = -1;
        for (int f : F)
        {
            if (v0_set.count(f) || error_nodes.count(f))
                continue;
            int score = known_incident_score(f);
            if (score > best_score || (score == best_score && (best_f < 0 || f < best_f)))
            {
                best_f = f;
                best_score = score;
            }
        }

        if (best_f < 0)
            return false;

        bool was_active = V_active.count(best_f);
        _expand_node(best_f);
        if (!was_active && V_active.count(best_f) && !v0_set.count(best_f))
        {
            ++eligible;
            ++stats.total_columns_added;
        }
        else
        {
            eligible = eligible_active_count();
        }
    }

    return true;
}

// Syncs RMP structure with the current active set. Newly created y/w vars
// receive their objective coefficient at addVar time, so the only remaining
// work is a cheap λ-delta check inside _update_objective.
void FullBranchAndPriceSolver::_sync_rmp_structure(double lambda_val)
{
    vector<int> new_nodes;
    _register_new_nodes(new_nodes);
    _register_pending_edges();
    _register_pair_vars(new_nodes, lambda_val);
    _update_objective(lambda_val);
}

// ── Branch-and-Bound Helpers ──────────────────────────────────────────────────

// Applies variable bounds for the current B&B node. First resets all previously
// fixed variables to their natural [0,1] bounds, then fixes each v ∈ v1 to 1
// and each v ∈ v0 to 0.
void FullBranchAndPriceSolver::_apply_node_bounds(const vector<int> &v1, const vector<int> &v0)
{
    for (int v : bound_fixed)
    {
        auto it = x_vars.find(v);
        if (it != x_vars.end())
        {
            it->second.set(GRB_DoubleAttr_LB, 0.0);
            it->second.set(GRB_DoubleAttr_UB, 1.0);
        }
    }

    bound_fixed.clear();

    for (int v : v1)
    {
        auto it = x_vars.find(v);
        if (it != x_vars.end())
        {
            it->second.set(GRB_DoubleAttr_LB, 1.0);
            it->second.set(GRB_DoubleAttr_UB, 1.0);
            bound_fixed.insert(v);
        }
    }

    for (int v : v0)
    {
        auto it = x_vars.find(v);
        if (it != x_vars.end())
        {
            it->second.set(GRB_DoubleAttr_LB, 0.0);
            it->second.set(GRB_DoubleAttr_UB, 0.0);
            bound_fixed.insert(v);
        }
    }

    rmp->update();
}

// Returns the top-scoring frontier nodes by reduced cost for column generation:
//   rc(f) = frac_deg(f) + ω,   where ω = −2λ · Σ x̄_v − π
// frac_deg(f) is the sum of x̄_u over active neighbours of f. Batch size is
// clamped to [cg_min_batch, cg_max_batch], sorted by decreasing rc.
// Joint pricing heuristic: find a high-density k-set containing q in
// (V_active ∪ F). Used when individual reduced-cost pricing has returned no
// improving column. For our bilinear objective, solo pricing is provably
// incomplete: a subset T of frontier nodes can have positive joint reduced
// cost even when every solo RC(t) is negative, because the new y_{t,t'}
// variables among T contribute to the objective only when T is added jointly.
// Exact subset pricing is NP-hard, so the heuristic picks the densest k-subset
// it can enumerate inside a bounded budget; only members not already in
// V_active are returned to the caller for expansion.
vector<int> FullBranchAndPriceSolver::_greedy_joint_pricing(const unordered_set<int> &v0_set)
{
    // Candidate pool: every node currently in V_active ∪ F that is not
    // forbidden (v0) or blacklisted (error_nodes) and is not q itself.
    unordered_set<int> pool;
    for (int v : V_active) pool.insert(v);
    for (int v : F) pool.insert(v);
    pool.erase(q);
    for (int v : v0_set) pool.erase(v);
    for (int v : error_nodes) pool.erase(v);

    vector<int> cand(pool.begin(), pool.end());
    std::sort(cand.begin(), cand.end());
    int n_c = (int)cand.size();
    int k_minus_1 = k - 1;
    if (n_c < k_minus_1)
        return {};

    auto count_edges_in = [&](const vector<int> &S) -> int {
        int m = 0;
        for (size_t i = 0; i < S.size(); ++i)
        {
            auto a_it = adj_out.find(S[i]);
            if (a_it == adj_out.end())
                continue;
            for (size_t j = 0; j < S.size(); ++j)
            {
                if (i == j)
                    continue;
                if (a_it->second.count(S[j]))
                    ++m;
            }
        }
        return m;
    };

    vector<int> best_S;
    int best_m = -1;

    // Budget for exhaustive enumeration. C(n_c, k-1) over the candidate pool
    // is bounded; for k <= 5 and pools below ~ 100 it stays under 5e5.
    constexpr long long ENUMERATE_BUDGET = 500000;

    long long combos_est = 1;
    {
        for (int i = 0; i < k_minus_1; ++i)
        {
            combos_est *= (long long)(n_c - i);
            combos_est /= (i + 1);
            if (combos_est < 0 || combos_est > ENUMERATE_BUDGET)
            {
                combos_est = ENUMERATE_BUDGET + 1;
                break;
            }
        }
    }

    if (combos_est <= ENUMERATE_BUDGET)
    {
        vector<int> idx(k_minus_1);
        for (int i = 0; i < k_minus_1; ++i)
            idx[i] = i;
        vector<int> S;
        S.reserve(k);
        while (true)
        {
            S.clear();
            S.push_back(q);
            for (int i : idx)
                S.push_back(cand[i]);
            int m = count_edges_in(S);
            if (m > best_m)
            {
                best_m = m;
                best_S = S;
            }
            int i = k_minus_1 - 1;
            while (i >= 0 && idx[i] == n_c - k_minus_1 + i)
                --i;
            if (i < 0)
                break;
            ++idx[i];
            for (int j = i + 1; j < k_minus_1; ++j)
                idx[j] = idx[j - 1] + 1;
        }
    }
    else
    {
        // Greedy fallback for pools too large to enumerate exhaustively.
        // Each round picks the candidate maximising edges to the current S
        // (gain), then breaking ties by candidate degree into the remaining
        // pool (favours dense-core nodes over peripheral leaves).
        unordered_set<int> S_set = {q};
        auto edge_to_S = [&](int v) -> int {
            int g = 0;
            auto a_it = adj_out.find(v);
            if (a_it != adj_out.end())
                for (int u : a_it->second)
                    if (S_set.count(u))
                        ++g;
            return g;
        };
        auto edge_to_pool = [&](int v) -> int {
            int g = 0;
            auto a_it = adj_out.find(v);
            if (a_it != adj_out.end())
                for (int u : a_it->second)
                    if (pool.count(u) && !S_set.count(u))
                        ++g;
            return g;
        };
        while ((int)S_set.size() < k)
        {
            int best_node = -1;
            int best_gain = -1;
            int best_sec = -1;
            for (int v : cand)
            {
                if (S_set.count(v))
                    continue;
                int g = edge_to_S(v);
                int s = edge_to_pool(v);
                if (g > best_gain ||
                    (g == best_gain && s > best_sec) ||
                    (g == best_gain && s == best_sec && (best_node == -1 || v < best_node)))
                {
                    best_node = v;
                    best_gain = g;
                    best_sec = s;
                }
            }
            if (best_node == -1)
                break;
            S_set.insert(best_node);
        }
        if ((int)S_set.size() == k)
        {
            best_S.assign(S_set.begin(), S_set.end());
            best_m = count_edges_in(best_S);
        }
    }

    if ((int)best_S.size() != k)
        return {};

    vector<int> to_expand;
    for (int v : best_S)
        if (V_active.find(v) == V_active.end())
            to_expand.push_back(v);
    return to_expand;
}

vector<int> FullBranchAndPriceSolver::_price_frontier(const unordered_map<int, double> &x_bar, double pi, const unordered_set<int> &v0_set, double lambda_val)
{
    double sum_x_bar = 0.0;
    for (const auto &[v, val] : x_bar)
        sum_x_bar += val;

    double omega = -2.0 * lambda_val * sum_x_bar - pi;
    vector<pair<double, int>> candidates;

    for (int f : F)
    {
        if (v0_set.find(f) != v0_set.end())
            continue;

        double frac_deg = 0.0;

        auto out_it = adj_out.find(f);
        if (out_it != adj_out.end())
        {
            for (int v : out_it->second)
            {
                auto x_it = x_bar.find(v);
                if (x_it != x_bar.end() && V_active.find(v) != V_active.end())
                    frac_deg += x_it->second;
            }
        }

        auto in_it = adj_in.find(f);
        if (in_it != adj_in.end())
        {
            for (int u : in_it->second)
            {
                auto x_it = x_bar.find(u);
                if (x_it != x_bar.end() && V_active.find(u) != V_active.end())
                    frac_deg += x_it->second;
            }
        }

        double rc = frac_deg + omega;
        if (rc > tol)
            candidates.push_back({rc, f});
    }

    int max_batch = std::max(0, cg_max_batch);
    int min_batch = std::max(0, std::min(cg_min_batch, max_batch));
    if (max_batch == 0)
        return {};

    size_t dynamic_limit = 0;
    if (cg_batch_fraction > 0.0)
        dynamic_limit = std::max<size_t>(1, (size_t)std::ceil((double)V_active.size() * cg_batch_fraction));

    size_t batch_size = std::max((size_t)min_batch, std::min(dynamic_limit, (size_t)max_batch));
    batch_size = min(batch_size, candidates.size());

    auto cmp = [](const pair<double, int> &a, const pair<double, int> &b)
    {
        if (std::fabs(a.first - b.first) > 1e-6)
            return a.first > b.first;
        return a.second > b.second;
    };

    if (batch_size < candidates.size())
    {
        nth_element(candidates.begin(), candidates.begin() + batch_size, candidates.end(), cmp);
        sort(candidates.begin(), candidates.begin() + batch_size, cmp);
    }
    else
    {
        sort(candidates.begin(), candidates.end(), cmp);
    }

    vector<int> top_f;
    for (size_t i = 0; i < batch_size; i++)
        top_f.push_back(candidates[i].second);
    return top_f;
}

// Separates violated BQP triangle inequalities from the current LP solution.
//
// For every triple {u, v, w} of fractional nodes (x̄ ∈ (0.1, 0.9)), checks the
// triangle inequality:
//   x̄_u + x̄_v + x̄_w − w̄_{uv} − w̄_{vw} − w̄_{uw} ≤ 1
// and adds the corresponding cut to the RMP if it is violated by more than 1e-4.
// Stops after 20 cuts. Returns the number of cuts added.
//
// Implementation notes:
//   * frac_nodes is sorted so we can index pairs by position (i < j) and store
//     w values in a dense n×n matrix, avoiding repeated unordered_map lookups
//     deep in the triple loop.
//   * Each w̄ is fetched from Gurobi exactly once (in the O(n²) prep stage)
//     instead of O(n³) times as before.
//   * x_sum ≤ 1 + ε cannot yield a violated cut (since w̄ ≥ 0), so we prune
//     the j- and k-loops as early as possible.
int FullBranchAndPriceSolver::_separate_bqp_cuts(const unordered_map<int, double> &x_bar)
{
    constexpr double kFracLo = 0.1;
    constexpr double kFracHi = 0.9;
    constexpr double kViolationTol = 1e-4;
    constexpr int kMaxCutsPerRound = 20;

    vector<int> frac_nodes;
    frac_nodes.reserve(x_bar.size());
    for (const auto &[v, val] : x_bar)
    {
        if (val > kFracLo && val < kFracHi)
            frac_nodes.push_back(v);
    }

    if (frac_nodes.size() < 3)
        return 0;
    sort(frac_nodes.begin(), frac_nodes.end());

    const int n = (int)frac_nodes.size();
    constexpr double kMissing = std::numeric_limits<double>::quiet_NaN();

    // Position-indexed cache of x̄_i and w̄_{i,j}. w_mat entries are only filled
    // for i < j; the diagonal and lower triangle are left as NaN and never read.
    vector<double> x_vals(n);
    for (int i = 0; i < n; ++i)
        x_vals[i] = x_bar.at(frac_nodes[i]);

    vector<double> w_mat((size_t)n * n, kMissing);
    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            pair<int, int> uv = {frac_nodes[i], frac_nodes[j]}; // already sorted (i<j → id_i<id_j)
            auto it = w_vars.find(uv);
            if (it != w_vars.end())
                w_mat[(size_t)i * n + j] = it->second.get(GRB_DoubleAttr_X);
        }
    }

    int cuts_added = 0;

    for (int i = 0; i < n; ++i)
    {
        const double xi = x_vals[i];
        for (int j = i + 1; j < n; ++j)
        {
            const double xj = x_vals[j];
            // Tightest upper bound on x_sum: xi + xj + (max remaining x̄).
            // Since all frac_nodes satisfy x̄ < kFracHi, the cut cannot be
            // violated if xi + xj + kFracHi already fits within 1 + slack.
            if (xi + xj + kFracHi <= 1.0 + kViolationTol)
                continue;

            const double wij = w_mat[(size_t)i * n + j];
            if (std::isnan(wij))
                continue;

            const double xij = xi + xj;

            for (int kk = j + 1; kk < n; ++kk)
            {
                const double xk = x_vals[kk];
                const double x_sum = xij + xk;
                if (x_sum <= 1.0 + kViolationTol)
                    continue;

                const double wjk = w_mat[(size_t)j * n + kk];
                if (std::isnan(wjk))
                    continue;
                const double wik = w_mat[(size_t)i * n + kk];
                if (std::isnan(wik))
                    continue;

                const double w_sum = wij + wjk + wik;
                if (x_sum - w_sum > 1.0 + kViolationTol)
                {
                    int u = frac_nodes[i], v = frac_nodes[j], w = frac_nodes[kk];
                    pair<int, int> uv = {u, v};
                    pair<int, int> vw = {v, w};
                    pair<int, int> uw = {u, w};
                    rmp->addConstr(x_vars[u] + x_vars[v] + x_vars[w] - w_vars[uv] - w_vars[vw] - w_vars[uw] <= 1.0);
                    if (++cuts_added >= kMaxCutsPerRound)
                        return cuts_added;
                }
            }
        }
    }
    return cuts_added;
}

// ── Column Generation ─────────────────────────────────────────────────────────

// Solves the LP relaxation at one B&B node via column generation + BQP cuts.
// Returns the LP solution x̄ and objective, or an empty map and −∞ if infeasible.
pair<unordered_map<int, double>, double> FullBranchAndPriceSolver::_column_generation(
    const vector<int> &v1, const vector<int> &v0, double lambda_val, double current_incumbent,
    chrono::high_resolution_clock::time_point t_start_bb, double net_start_bb)
{
    unordered_map<int, double> local_x_bar;
    double local_lp_obj = -1e9;
    double prev_lp_bound = 1e9;
    int consecutive_stalls = 0;
    bool joint_pricing_tried = false;

    unordered_set<int> v0_set(v0.begin(), v0.end());
    if (!_ensure_active_feasibility(v0_set))
        return {std::move(local_x_bar), -1e9};

    while (true)
    {
        auto t_now = chrono::high_resolution_clock::now();
        double wall = chrono::duration<double>(t_now - t_start_bb).count();
        double net = oracle.cumulative_network_time - net_start_bb;
        double effective_time = max(0.0, wall - net);

        // Same gate as the BB loop: the soft no-improvement cap fires only
        // when there is an incumbent worth not improving on.
        if (bb_time_limit >= 0.0 && current_incumbent > tol && effective_time > bb_time_limit)
            return {std::move(local_x_bar), local_lp_obj};

        double solve_elapsed_cg = chrono::duration<double>(t_now - t_start_solve).count();
        if (bb_hard_time_limit >= 0.0 && solve_elapsed_cg > bb_hard_time_limit)
        {
            last_hard_cap_hit = true;
            return {std::move(local_x_bar), local_lp_obj};
        }

        auto t0 = chrono::high_resolution_clock::now();
        _sync_rmp_structure(lambda_val);
        // Bounds must be applied after synchronizing the RMP. At the root, q is
        // fixed before x_q exists; applying bounds before variable registration
        // silently leaves x_q free.
        _apply_node_bounds(v1, v0);
        auto t1 = chrono::high_resolution_clock::now();
        stats.t_sync += chrono::duration<double>(t1 - t0).count();

        double lp_time_budget = -1.0;
        // Soft budget tracks the BB-local clock; only meaningful once we have
        // an incumbent (consistent with the gated soft cap above).
        if (bb_time_limit >= 0.0 && current_incumbent > tol)
            lp_time_budget = max(1e-3, bb_time_limit - effective_time);
        bool hard_limited_lp = false;
        if (bb_hard_time_limit >= 0.0)
        {
            double hard_remaining = max(
                1e-3, bb_hard_time_limit - solve_elapsed_cg);
            if (lp_time_budget < 0.0 || hard_remaining < lp_time_budget)
            {
                lp_time_budget = hard_remaining;
                hard_limited_lp = true;
            }
        }
        if (lp_time_budget >= 0.0)
            rmp->set(GRB_DoubleParam_TimeLimit, lp_time_budget);
        else
            rmp->set(GRB_DoubleParam_TimeLimit, GRB_INFINITY);

        rmp->optimize();
        stats.total_lp_solves++;
        auto t2 = chrono::high_resolution_clock::now();
        stats.t_lp_solve += chrono::duration<double>(t2 - t1).count();

        int status = rmp->get(GRB_IntAttr_Status);
        if (status == GRB_TIME_LIMIT && hard_limited_lp)
        {
            double solve_elapsed_after_lp = chrono::duration<double>(
                chrono::high_resolution_clock::now() - t_start_solve).count();
            if (solve_elapsed_after_lp >= bb_hard_time_limit - 1e-3)
                last_hard_cap_hit = true;
        }
        if (status != GRB_OPTIMAL && status != GRB_SUBOPTIMAL && status != GRB_TIME_LIMIT)
            return {std::move(local_x_bar), -1e9};
        if (status != GRB_OPTIMAL && rmp->get(GRB_IntAttr_SolCount) == 0)
            return {std::move(local_x_bar), -1e9};

        local_x_bar.clear();
        for (const auto &[v, var] : x_vars)
            local_x_bar[v] = var.get(GRB_DoubleAttr_X);
        local_lp_obj = rmp->get(GRB_DoubleAttr_ObjVal);

        if (status != GRB_OPTIMAL)
            return {std::move(local_x_bar), local_lp_obj};

        double pi = size_constr.get(GRB_DoubleAttr_Pi);

        auto t3 = chrono::high_resolution_clock::now();
        vector<int> top_f = _price_frontier(local_x_bar, pi, v0_set, lambda_val);
        auto t4 = chrono::high_resolution_clock::now();
        stats.t_pricing += chrono::duration<double>(t4 - t3).count();

        if (!top_f.empty())
        {
            stats.total_columns_added += top_f.size();
            for (int f : top_f)
                _expand_node(f);
            continue;
        }

        // Frontier-internal edges are invisible to solo pricing. Materialise
        // lazily so the next round sees true degrees; returns 0 once F is fully
        // queried.
        if (!skip_materialize && _materialize_unqueried_frontier() > 0)
            continue;

        // Individual reduced-cost pricing returned no improving column. Before
        // declaring LP-optimality, run one structural pass that scores frontier
        // nodes jointly: a greedy k-densest seed starting from {q} surfaces
        // clique-completer subsets whose members each have negative solo RC
        // under the current dual prices but together form a high-density set.
        // The flag prevents an infinite loop when the greedy result is already
        // contained in V_active (no progress) or when later LP iterations
        // converge to the same incumbent.
        if (!joint_pricing_tried)
        {
            joint_pricing_tried = true;
            auto t_jp_0 = chrono::high_resolution_clock::now();
            vector<int> joint = _greedy_joint_pricing(v0_set);
            auto t_jp_1 = chrono::high_resolution_clock::now();
            stats.t_pricing += chrono::duration<double>(t_jp_1 - t_jp_0).count();
            if (!joint.empty())
            {
                stats.total_columns_added += joint.size();
                for (int f : joint)
                    _expand_node(f);
                continue;
            }
        }

        // Track consecutive LP bound stalls (no positive-RC columns were found)
        if (prev_lp_bound < 1e8)
        {
            double bound_improvement = prev_lp_bound - local_lp_obj;
            if (bound_improvement < 1e-3)
                consecutive_stalls++;
            else
                consecutive_stalls = 0;
        }
        prev_lp_bound = local_lp_obj;

        int n_fractional = 0;
        for (const auto &[v, val] : local_x_bar)
            if (val > 0.1 && val < 0.9)
                n_fractional++;

        double gap = 1.0;
        if (current_incumbent > tol)
            gap = (local_lp_obj - current_incumbent) / current_incumbent;

        // Attempt BQP cuts only when the solution is fractional enough, the LP
        // bound is still improving, and the gap warrants the separation effort
        if (n_fractional >= 3 && consecutive_stalls < 2 && gap > 0.01)
        {
            auto t5 = chrono::high_resolution_clock::now();
            int cuts = _separate_bqp_cuts(local_x_bar);
            auto t6 = chrono::high_resolution_clock::now();
            stats.t_separation += chrono::duration<double>(t6 - t5).count();

            if (cuts > 0)
            {
                stats.total_cuts_added += cuts;
                continue;
            }
        }

        return {std::move(local_x_bar), local_lp_obj};
    }
}

// ── Branching ─────────────────────────────────────────────────────────────────

// Selects a branching variable using fractional internal degree relative to 2λ.
// Nodes with internal_deg < 2λ are "hanging" (branch zero-first, pick weakest);
// nodes with internal_deg ≥ 2λ are "core" (branch one-first, pick most fractional).
// Returns {branch_var, branch_zero_first}; branch_var = -1 if already integer.
pair<int, bool> FullBranchAndPriceSolver::_select_branch_var(const unordered_map<int, double> &x_bar, double lambda_val)
{
    int branch_var = -1;
    double min_diff = 1.0;
    double min_hanging_deg = 1e9;
    double max_core_deg = -1.0;
    bool found_hanging = false;

    for (const auto &[v, val] : x_bar)
    {
        if (val <= tol || val >= 1.0 - tol)
            continue;

        double diff = std::fabs(val - 0.5);
        double internal_deg = 0.0;

        auto out_it = adj_out.find(v);
        if (out_it != adj_out.end())
        {
            for (int u : out_it->second)
            {
                auto x_it = x_bar.find(u);
                if (x_it != x_bar.end())
                    internal_deg += x_it->second;
            }
        }

        auto in_it = adj_in.find(v);
        if (in_it != adj_in.end())
        {
            for (int u : in_it->second)
            {
                auto x_it = x_bar.find(u);
                if (x_it != x_bar.end())
                    internal_deg += x_it->second;
            }
        }

        bool is_hanging = (internal_deg < 2.0 * lambda_val);

        if (is_hanging)
        {
            if (!found_hanging || internal_deg < min_hanging_deg - 1e-3)
            {
                found_hanging = true;
                min_hanging_deg = internal_deg;
                min_diff = diff;
                branch_var = v;
            }
            else if (std::fabs(internal_deg - min_hanging_deg) <= 1e-3 && diff < min_diff)
            {
                min_diff = diff;
                branch_var = v;
            }
        }
        else if (!found_hanging)
        {
            if (diff < min_diff - 1e-3)
            {
                min_diff = diff;
                max_core_deg = internal_deg;
                branch_var = v;
            }
            else if (std::fabs(diff - min_diff) <= 1e-3 && internal_deg > max_core_deg)
            {
                max_core_deg = internal_deg;
                branch_var = v;
            }
        }
    }

    bool branch_zero_first = found_hanging;
    return {branch_var, branch_zero_first};
}

// ── Branch-and-Price ──────────────────────────────────────────────────────────

// Depth-first B&B to solve max f(S, λ) s.t. |S| ≥ k. bb_time_limit is a
// "time without improvement" budget: the clock resets on each new incumbent.
pair<unordered_set<int>, double> FullBranchAndPriceSolver::_branch_and_price(double lambda_val)
{
    vector<BBNode> stack;
    BBNode root;
    root.v1.push_back(q);
    stack.push_back(std::move(root));
    double best_int_obj = 0.0;
    unordered_set<int> best_int_sol;
    string gap_status = "exhausted";
    double interrupted_node_bound = -std::numeric_limits<double>::infinity();

    auto t_start_bb = chrono::high_resolution_clock::now();
    double net_start_bb = oracle.cumulative_network_time;

    auto max_stack_bound = [&]() -> double
    {
        double best = -std::numeric_limits<double>::infinity();
        for (const BBNode &open_node : stack)
            if (open_node.bound > best)
                best = open_node.bound;
        return best;
    };

    auto publish_gap = [&]()
    {
        double open_bound = max_stack_bound();
        double best_bound = best_int_obj;
        if (std::isfinite(open_bound))
            best_bound = std::max(best_bound, open_bound);
        if (std::isfinite(interrupted_node_bound))
            best_bound = std::max(best_bound, interrupted_node_bound);
        stats.final_bb_incumbent_obj = best_int_obj;
        stats.final_bb_best_bound = best_bound;
        stats.final_optimality_gap = relative_gap(best_bound, best_int_obj, tol);
        stats.final_open_nodes = (int)stack.size() + (std::isfinite(interrupted_node_bound) ? 1 : 0);
        stats.final_gap_status = gap_status;
    };

    while (!stack.empty())
    {
        if (bb_node_limit >= 0 && stats.total_bb_nodes >= bb_node_limit)
        {
            cout << "[" << get_timestamp() << "]     [!] B&B node limit reached." << endl;
            gap_status = "node_limit";
            break;
        }

        auto t_now = chrono::high_resolution_clock::now();
        double wall = chrono::duration<double>(t_now - t_start_bb).count();
        double net = oracle.cumulative_network_time - net_start_bb;
        double effective_time = max(0.0, wall - net);

        // Soft no-improvement cap is gated on the existence of an improving
        // incumbent. Before BP finds any integer point that beats f = 0, the
        // clock does not fire: in that regime BB must run to exhaustion so the
        // Dinkelbach loop can conclude "no S improves at lambda_t" correctly,
        // which is the condition that makes the outer loop exact.
        if (bb_time_limit >= 0.0 && best_int_obj > tol && effective_time > bb_time_limit)
        {
            cout << "[" << get_timestamp() << "]     [!] Iteration algorithmic time limit reached ("
                 << bb_time_limit << "s without improvement on an existing incumbent)." << endl;
            gap_status = "time_limit";
            break;
        }

        if (bb_hard_time_limit >= 0.0)
        {
            double solve_elapsed = chrono::duration<double>(t_now - t_start_solve).count();
            if (solve_elapsed > bb_hard_time_limit)
            {
                last_hard_cap_hit = true;
                cout << "[" << get_timestamp() << "]     [!] Hard wall-time cap reached ("
                     << bb_hard_time_limit << "s); returning current incumbent." << endl;
                gap_status = "hard_time_limit";
                break;
            }
        }

        double open_bound = max_stack_bound();
        if (bb_gap_tol >= 0.0 && best_int_obj > tol && std::isfinite(open_bound))
        {
            double gap = relative_gap(std::max(open_bound, best_int_obj), best_int_obj, tol);
            if (gap <= bb_gap_tol)
            {
                gap_status = "gap_tolerance";
                break;
            }
        }

        BBNode node = std::move(stack.back());
        stack.pop_back();
        if (std::isfinite(node.bound) && node.bound <= best_int_obj + tol)
            continue;
        stats.total_bb_nodes++;

        auto [x_bar, lp_obj] = _column_generation(node.v1, node.v0, lambda_val, best_int_obj, t_start_bb, net_start_bb);
        if (last_hard_cap_hit)
        {
            gap_status = "hard_time_limit";
            interrupted_node_bound = (std::isfinite(lp_obj) && lp_obj > -1e8) ? lp_obj : node.bound;
            break;
        }

        if (x_bar.empty() || lp_obj <= best_int_obj + tol)
            continue;

        auto [branch_var, branch_zero_first] = _select_branch_var(x_bar, lambda_val);

        if (branch_var == -1)
        {
            // ====================================================================
            // kappa-connectivity check (Menger's theorem)
            // ====================================================================

            if (this->kappa == 0)
            {
                std::unordered_set<int> sol_nodes;
                for (const auto &[v, val] : x_bar)
                {
                    if (val > 0.5)
                        sol_nodes.insert(v);
                }

                _prune_discrete_solution(sol_nodes, lambda_val, false, false);

                if (sol_nodes.size() >= (size_t)k)
                {
                    double int_obj = _parametric_obj(sol_nodes, lambda_val);
                    if (int_obj > best_int_obj + tol)
                    {
                        best_int_obj = int_obj;
                        best_int_sol = std::move(sol_nodes);
                        t_start_bb = chrono::high_resolution_clock::now();
                        net_start_bb = oracle.cumulative_network_time;
                    }
                }

                continue;
            }

            std::unordered_set<int> sol_nodes;
            for (const auto &[v, val] : x_bar)
            {
                if (val > 0.5)
                    sol_nodes.insert(v);
            }

            // Map Gurobi node IDs to Boost contiguous indices
            std::unordered_map<int, int> g_to_b;
            std::vector<int> b_to_g;
            for (int node_id : sol_nodes)
            {
                g_to_b[node_id] = b_to_g.size();
                b_to_g.push_back(node_id);
            }
            int N = sol_nodes.size();

            bool is_k_connected = true;

            if (sol_nodes.count(this->q))
            {
                // Explicitly scoped Boost Graph definitions
                typedef boost::adjacency_list_traits<boost::vecS, boost::vecS, boost::directedS> Traits;
                typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
                                              boost::property<boost::vertex_name_t, int>,
                                              boost::property<boost::edge_capacity_t, long long,
                                                              boost::property<boost::edge_residual_capacity_t, long long,
                                                                              boost::property<boost::edge_reverse_t, Traits::edge_descriptor>>>>
                    FlowGraph;

                int S = g_to_b[this->q];

                for (int target_node : sol_nodes)
                {
                    if (target_node == this->q)
                        continue;

                    // 1. Build the capacity network for the current integer solution
                    FlowGraph fg(N);
                    auto capacity = boost::get(boost::edge_capacity, fg);
                    auto rev = boost::get(boost::edge_reverse, fg);

                    auto add_flow_edge = [&](int u, int v, long long cap)
                    {
                        auto e1 = boost::add_edge(u, v, fg).first;
                        auto e2 = boost::add_edge(v, u, fg).first;
                        capacity[e1] = cap;
                        capacity[e2] = 0;
                        rev[e1] = e2;
                        rev[e2] = e1;
                    };

                    // Symmetric directed inputs emit reciprocal y_vars for each
                    // undirected edge; collapse to one unit-capacity undirected
                    // edge per unordered pair so flow capacity stays at 1 in
                    // each direction.
                    std::unordered_set<long long> seen_uv;
                    for (const auto &[edge, y_var] : y_vars)
                    {
                        if (y_var.get(GRB_DoubleAttr_X) <= 0.5)
                            continue;
                        if (!sol_nodes.count(edge.first) || !sol_nodes.count(edge.second))
                            continue;
                        int a = std::min(edge.first, edge.second);
                        int b = std::max(edge.first, edge.second);
                        long long key = ((long long)(unsigned int)a << 32) | (unsigned int)b;
                        if (!seen_uv.insert(key).second)
                            continue;
                        int u = g_to_b[a];
                        int v = g_to_b[b];
                        add_flow_edge(u, v, 1);
                        add_flow_edge(v, u, 1);
                    }

                    // 2. Compute Max-Flow (Number of edge-disjoint paths)
                    int T = g_to_b[target_node];
                    long long flow = boost::push_relabel_max_flow(fg, S, T);

                    // 3. Separation: If flow < \kappa, find the Min-Cut
                    if (flow < this->kappa)
                    {
                        is_k_connected = false;

                        std::unordered_set<int> reachable_boost;
                        std::queue<int> bq;
                        std::vector<bool> vis(N, false);

                        bq.push(S);
                        vis[S] = true;
                        reachable_boost.insert(S);

                        auto res = boost::get(boost::edge_residual_capacity, fg);
                        while (!bq.empty())
                        {
                            int curr = bq.front();
                            bq.pop();
                            for (auto e : boost::make_iterator_range(boost::out_edges(curr, fg)))
                            {
                                if (res[e] > 0 && !vis[boost::target(e, fg)])
                                {
                                    vis[boost::target(e, fg)] = true;
                                    bq.push(boost::target(e, fg));
                                    reachable_boost.insert(boost::target(e, fg));
                                }
                            }
                        }

                        std::unordered_set<int> reachable_gurobi;
                        for (int b_id : reachable_boost)
                        {
                            reachable_gurobi.insert(b_to_g[b_id]);
                        }

                        // 4. Inject Cut-Set Inequality. The verifier uses the
                        // undirected support graph, so the cut uses z_vars and
                        // counts a reciprocal citation pair only once.
                        GRBLinExpr cut_expr = 0;
                        for (const auto &[support_uv, z_var] : z_vars)
                        {
                            bool u_in_R = reachable_gurobi.count(support_uv.first);
                            bool v_in_R = reachable_gurobi.count(support_uv.second);

                            // Edge crosses the Min-Cut
                            if (u_in_R != v_in_R)
                            {
                                cut_expr += z_var;
                            }
                        }

                        // Force the LP to select enough edges crossing this cut
                        GRBConstr cut = rmp->addConstr(cut_expr >= this->kappa * x_vars[target_node], "k_connectivity_cut");
                        connectivity_cuts.push_back({std::move(reachable_gurobi), target_node, cut});
                        stats.total_cuts_added++;
                        break;
                    }
                }
            }

            if (!is_k_connected)
            {
                stats.total_bb_nodes--;
                node.bound = lp_obj;
                stack.push_back(std::move(node));
                continue;
            }
            // ====================================================================

            // Integer solution kappa-feasible. Prune to improve the parametric objective; if
            // pruning breaks edge-connectivity, revert to the pre-prune set (already verified).
            std::unordered_set<int> pre_prune_sol = sol_nodes;
            _prune_discrete_solution(sol_nodes, lambda_val, false, true);

            if (this->kappa > 0)
            {
                bool prune_ok = _verify_kappa_connectivity(sol_nodes);
                if (!prune_ok)
                {
                    cout << "[" << get_timestamp() << "]     [!] Post-prune kappa verification failed; reverting to pre-prune set." << endl;
                    sol_nodes = std::move(pre_prune_sol);
                }
                else
                {
                    this->last_kappa_verified = true;
                }
            }

            if (sol_nodes.size() >= (size_t)k)
            {
                double obj = _parametric_obj(sol_nodes, lambda_val);
                if (obj > best_int_obj)
                {
                    best_int_obj = obj;
                    best_int_sol = sol_nodes;
                    cout << "[" << get_timestamp() << "]     > Incumbent updated at Node " << stats.total_bb_nodes
                         << " | Obj: " << fixed << setprecision(4) << obj
                         << " | Size: " << sol_nodes.size() << endl;

                    t_start_bb = chrono::high_resolution_clock::now();
                    net_start_bb = oracle.cumulative_network_time;
                }
            }
            continue;
        }

        BBNode child_0 = node;
        BBNode child_1 = std::move(node);
        child_0.v0.push_back(branch_var);
        child_1.v1.push_back(branch_var);
        child_0.bound = lp_obj;
        child_1.bound = lp_obj;

        if (branch_zero_first)
        {
            stack.push_back(std::move(child_1));
            stack.push_back(std::move(child_0));
        }
        else
        {
            stack.push_back(std::move(child_0));
            stack.push_back(std::move(child_1));
        }
    }

    publish_gap();
    return {best_int_sol, best_int_obj};
}

// ── Dinkelbach Outer Loop ─────────────────────────────────────────────────────

// Dinkelbach outer loop: iteratively solves the parametric subproblem and
// updates λ = d(S) until convergence. Returns the best solution and its density.
pair<unordered_set<int>, double> FullBranchAndPriceSolver::solve()
{
    auto t_start_global = chrono::high_resolution_clock::now();
    t_start_solve = t_start_global;

    unordered_set<int> best_sol = V_active;

    // Prune the BFS seed to a near-optimal starting point before the first
    // Dinkelbach iteration, giving a tighter initial λ
    unordered_set<int> pre_prune_seed = best_sol;
    _prune_discrete_solution(best_sol, 0.0, true, kappa > 0);
    if (kappa > 0 && !_verify_kappa_connectivity(best_sol))
        best_sol = std::move(pre_prune_seed);

    bool best_sol_feasible = (kappa <= 0) || _verify_kappa_connectivity(best_sol);
    double lambda_val = best_sol_feasible ? _density(best_sol) : 0.0;

    cout << fixed << setprecision(6);
    cout << "[" << get_timestamp() << "] Init Active Set | Size: " << best_sol.size() << " | Density: " << lambda_val << endl;
    if (kappa > 0 && !best_sol_feasible)
    {
        cout << "[" << get_timestamp() << "]   Initial seed is not kappa-feasible; starting Dinkelbach at lambda=0." << endl;
    }
    cout << "--------------------------------------------------" << endl;

    for (int t = 1; dinkelbach_max_iter < 0 || t <= dinkelbach_max_iter; t++)
    {
        cout << "[" << get_timestamp() << "] === DINKELBACH ITERATION " << t << " | Lambda = " << lambda_val << " ===" << endl;

        auto t_start_iter = chrono::high_resolution_clock::now();
        int bb_nodes_before = stats.total_bb_nodes;
        int lp_solves_before = stats.total_lp_solves;

        auto [sol, param_obj] = _branch_and_price(lambda_val);

        auto t_end_iter = chrono::high_resolution_clock::now();
        double iter_time = chrono::duration<double>(t_end_iter - t_start_iter).count();
        int iter_bb_nodes = stats.total_bb_nodes - bb_nodes_before;
        int iter_lp_solves = stats.total_lp_solves - lp_solves_before;

        cout << "[" << get_timestamp() << "]   -> Iteration Finished in " << fixed << setprecision(3) << iter_time << "s" << endl;
        cout << "[" << get_timestamp() << "]   -> Nodes Explored : " << iter_bb_nodes << " (Total: " << stats.total_bb_nodes << ")" << endl;
        cout << "[" << get_timestamp() << "]   -> LP Solves      : " << iter_lp_solves << " (Total: " << stats.total_lp_solves << ")" << endl;

        stats.lambda_trajectory.push_back({
            t,
            lambda_val,
            iter_time,
            iter_bb_nodes,
            iter_lp_solves,
            stats.final_bb_incumbent_obj,
            stats.final_bb_best_bound,
            stats.final_optimality_gap,
        });

        if (sol.empty() || param_obj <= tol)
        {
            cout << "[" << get_timestamp() << "]   Status            : Converged (No improvement found)" << endl;
            if (!best_sol_feasible && kappa > 0)
            {
                cout << "[" << get_timestamp() << "]   Status            : No kappa-feasible incumbent found." << endl;
                best_sol.clear();
                lambda_val = 0.0;
            }
            break;
        }

        double new_density = _density(sol);
        cout << "[" << get_timestamp() << "]   Found Solution    : Size: " << sol.size() << " | New Density: " << new_density << endl;

        if (new_density <= lambda_val + tol)
        {
            cout << "[" << get_timestamp() << "]   Status            : Converged (Density bound reached)" << endl;
            if (!best_sol_feasible && kappa > 0)
            {
                best_sol = sol;
                lambda_val = new_density;
                best_sol_feasible = true;
            }
            break;
        }

        lambda_val = new_density;
        best_sol = sol;
        best_sol_feasible = true;

        if (bb_hard_time_limit >= 0.0)
        {
            double solve_elapsed = chrono::duration<double>(
                chrono::high_resolution_clock::now() - t_start_solve).count();
            if (solve_elapsed > bb_hard_time_limit)
            {
                last_hard_cap_hit = true;
                cout << "[" << get_timestamp() << "]   Status            : Hard wall-time cap reached ("
                     << bb_hard_time_limit << "s); returning current incumbent." << endl;
                break;
            }
        }
    }

    auto t_end_global = chrono::high_resolution_clock::now();
    stats.t_total = chrono::duration<double>(t_end_global - t_start_global).count();

    if (kappa > 0)
    {
        bool final_ok = !best_sol.empty() && _verify_kappa_connectivity(best_sol);
        last_kappa_verified = final_ok;
        last_kappa_verify_failed = !final_ok;
    }

    return {best_sol, lambda_val};
}
