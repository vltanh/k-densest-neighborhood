#include "solver.hpp"
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <chrono>

using namespace std;

FullBranchAndPriceSolver::FullBranchAndPriceSolver(IGraphOracle &oracle, int q, int k, GRBEnv &env,
                                                   double tol, int bb_node_limit, double bb_time_limit,
                                                   double bb_gap_tol, int dinkelbach_max_iter,
                                                   double cg_batch_fraction, int cg_min_batch, int cg_max_batch)
    : oracle(oracle), q(q), k(k), env(env), tol(tol), bb_node_limit(bb_node_limit),
      bb_time_limit(bb_time_limit), bb_gap_tol(bb_gap_tol), dinkelbach_max_iter(dinkelbach_max_iter),
      cg_batch_fraction(cg_batch_fraction), cg_min_batch(cg_min_batch), cg_max_batch(cg_max_batch), rmp(nullptr)
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

// BFS from q to seed V_active with ≥ k nodes. A second pass queries nodes that
// were added as neighbours but never expanded, completing their adjacency lists.
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

            V_active.insert(curr);
            F.erase(curr);

            for (int u : preds)
            {
                if (error_nodes.count(u))
                    continue;
                _add_edge(u, curr);
                if (V_active.find(u) == V_active.end())
                    F.insert(u);
            }
            for (int w : succs)
            {
                if (error_nodes.count(w))
                    continue;
                _add_edge(curr, w);
                if (V_active.find(w) == V_active.end())
                    F.insert(w);
            }

            for (int u : preds)
            {
                if (V_active.find(u) == V_active.end() && !error_nodes.count(u))
                    q_nodes.push(u);
            }
            for (int w : succs)
            {
                if (V_active.find(w) == V_active.end() && !error_nodes.count(w))
                    q_nodes.push(w);
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "[" << get_timestamp() << "] Blacklisting node " << oracle.mapper.get_str(curr) << " during initialization: " << e.what() << "\n";
            error_nodes.insert(curr);
            F.erase(curr);
            V_active.erase(curr);
        }
    }

    // Second pass: query active nodes that were added as neighbours during BFS
    // but never expanded themselves, so their own adjacency lists are complete.
    vector<int> active_copy(V_active.begin(), V_active.end());
    for (int v : active_copy)
    {
        if (bfs_queried.find(v) == bfs_queried.end() && !error_nodes.count(v))
        {
            try
            {
                const auto &[preds, succs] = oracle.query(v);
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
            catch (const std::exception &e)
            {
                std::cerr << "[" << get_timestamp() << "] Blacklisting node " << oracle.mapper.get_str(v) << " during post-init: " << e.what() << "\n";
                error_nodes.insert(v);
                V_active.erase(v);
            }
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
void FullBranchAndPriceSolver::_prune_discrete_solution(unordered_set<int> &sol_nodes, double lambda_val, bool maximize_density)
{
    if (sol_nodes.size() <= (size_t)k)
        return;

    bool changed = true;
    int initial_size = sol_nodes.size();
    double initial_metric = maximize_density ? _density(sol_nodes) : _parametric_obj(sol_nodes, lambda_val);
    int nodes_removed = 0;

    cout << "[" << get_timestamp() << "]     > Pruning discrete solution (Initial Size: " << initial_size
         << " | " << (maximize_density ? "Density: " : "Param Obj: ") << fixed << setprecision(6) << initial_metric << ")" << endl;

    while (changed && sol_nodes.size() > (size_t)k)
    {
        changed = false;
        int worst_node = -1;
        double best_improvement = 1e-7;

        double current_metric = maximize_density ? _density(sol_nodes) : _parametric_obj(sol_nodes, lambda_val);

        vector<int> candidate_nodes(sol_nodes.begin(), sol_nodes.end());

        for (int u : candidate_nodes)
        {
            if (u == q)
                continue;

            sol_nodes.erase(u);
            double new_metric = maximize_density ? _density(sol_nodes) : _parametric_obj(sol_nodes, lambda_val);
            sol_nodes.insert(u);

            double improvement = new_metric - current_metric;
            if (improvement > best_improvement)
            {
                best_improvement = improvement;
                worst_node = u;
            }
        }

        if (worst_node != -1)
        {
            sol_nodes.erase(worst_node);
            changed = true;
            nodes_removed++;

            cout << "[" << get_timestamp() << "]       - Pruned node " << oracle.mapper.get_str(worst_node)
                 << " (Improvement: +" << fixed << setprecision(6) << best_improvement
                 << " | New Size: " << sol_nodes.size() << ")" << endl;
        }
    }

    if (nodes_removed > 0)
    {
        double final_metric = maximize_density ? _density(sol_nodes) : _parametric_obj(sol_nodes, lambda_val);
        cout << "[" << get_timestamp() << "]     > Pruning complete. Removed " << nodes_removed
             << " nodes. (Final Size: " << sol_nodes.size()
             << " | " << (maximize_density ? "Density: " : "Param Obj: ") << fixed << setprecision(6) << final_metric << ")" << endl;
    }
    else
    {
        cout << "[" << get_timestamp() << "]     > Pruning complete. No nodes removed (solution is already strictly minimal)." << endl;
    }
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

        V_active.insert(f);
        F.erase(f);

        for (int u : preds)
        {
            if (error_nodes.count(u))
                continue;
            _add_edge(u, f);
            if (V_active.find(u) == V_active.end())
                F.insert(u);
        }
        for (int w : succs)
        {
            if (error_nodes.count(w))
                continue;
            _add_edge(f, w);
            if (V_active.find(w) == V_active.end())
                F.insert(w);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "[" << get_timestamp() << "] Blacklisting node " << oracle.mapper.get_str(f) << " due to API error: " << e.what() << "\n";
        error_nodes.insert(f);
        F.erase(f);
        V_active.erase(f);
    }
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
// Edges whose endpoints are not yet in the RMP remain in pending_edges.
// Returns true if any y_vars were added.
bool FullBranchAndPriceSolver::_register_pending_edges()
{
    bool changed = false;
    vector<pair<int, int>> remaining;
    for (auto const &uv : pending_edges)
    {
        if (x_vars.count(uv.first) && x_vars.count(uv.second))
        {
            if (!y_vars.count(uv))
            {
                GRBVar yvar = rmp->addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "");
                rmp->addConstr(yvar <= x_vars[uv.first]);
                rmp->addConstr(yvar <= x_vars[uv.second]);
                y_vars[uv] = yvar;
                y_obj_terms.push_back(yvar);
                changed = true;
            }
        }
        else
        {
            remaining.push_back(uv);
        }
    }
    pending_edges = std::move(remaining);
    return changed;
}

// Adds a w_{uv} ∈ [0,1] variable for every unordered pair {u, v} that involves
// at least one node from new_nodes. w_{uv} linearises the product x_u · x_v via
// the McCormick lower bound w_{uv} ≥ x_u + x_v − 1. Updates synced_nodes.
// Returns true if any w_vars were added.
bool FullBranchAndPriceSolver::_register_pair_vars(const vector<int> &new_nodes)
{
    if (new_nodes.empty())
        return false;

    bool changed = false;

    // Pairs between new nodes and already-synced nodes
    for (int u : new_nodes)
    {
        for (int v : synced_nodes)
        {
            pair<int, int> uv = (u < v) ? make_pair(u, v) : make_pair(v, u);
            if (!w_vars.count(uv))
            {
                GRBVar wvar = rmp->addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "");
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
                GRBVar wvar = rmp->addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "");
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

// Rebuilds and sets the Gurobi objective for the current density estimate λ:
//   maximise  Σ y_{uv}  −  2λ · Σ w_{uv}
//
// The y-sum counts directed edges inside S; the w-sum approximates |S|·(|S|−1)
// via the linearised product variables, so the objective tracks the parametric
// problem f(S, λ) = |E(S)| − λ·|S|·(|S|−1).
void FullBranchAndPriceSolver::_update_objective(double lambda_val)
{
    rmp->update();
    GRBLinExpr obj_expr = 0;
    if (!y_obj_terms.empty())
    {
        vector<double> coeffs(y_obj_terms.size(), 1.0);
        obj_expr.addTerms(coeffs.data(), y_obj_terms.data(), y_obj_terms.size());
    }
    if (!w_obj_terms.empty())
    {
        vector<double> coeffs(w_obj_terms.size(), -2.0 * lambda_val);
        obj_expr.addTerms(coeffs.data(), w_obj_terms.data(), w_obj_terms.size());
    }
    rmp->setObjective(obj_expr, GRB_MAXIMIZE);
    last_lambda = lambda_val;
}

// Syncs RMP structure with the current active set; rebuilds the objective only
// when structure or lambda changed since the last call.
void FullBranchAndPriceSolver::_sync_rmp_structure(double lambda_val)
{
    vector<int> new_nodes;
    bool changed = _register_new_nodes(new_nodes);
    changed |= _register_pending_edges();
    changed |= _register_pair_vars(new_nodes);
    if (changed || lambda_val != last_lambda)
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

    int dynamic_limit = V_active.size() * cg_batch_fraction;
    size_t batch_size = max((size_t)cg_min_batch, min((size_t)dynamic_limit, (size_t)cg_max_batch));
    batch_size = min(batch_size, candidates.size());

    auto cmp = [](const pair<double, int> &a, const pair<double, int> &b)
    {
        if (abs(a.first - b.first) > 1e-6)
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
int FullBranchAndPriceSolver::_separate_bqp_cuts(const unordered_map<int, double> &x_bar)
{
    vector<int> frac_nodes;
    for (const auto &[v, val] : x_bar)
    {
        if (val > 0.1 && val < 0.9)
            frac_nodes.push_back(v);
    }

    if (frac_nodes.size() < 3)
        return 0;
    sort(frac_nodes.begin(), frac_nodes.end());

    int n = frac_nodes.size();
    int cuts_added = 0;

    for (int idx1 = 0; idx1 < n; idx1++)
    {
        for (int idx2 = idx1 + 1; idx2 < n; idx2++)
        {
            for (int idx3 = idx2 + 1; idx3 < n; idx3++)
            {
                int u = frac_nodes[idx1], v = frac_nodes[idx2], w = frac_nodes[idx3];

                pair<int, int> uv = (u < v) ? make_pair(u, v) : make_pair(v, u);
                pair<int, int> vw = (v < w) ? make_pair(v, w) : make_pair(w, v);
                pair<int, int> uw = (u < w) ? make_pair(u, w) : make_pair(w, u);

                if (w_vars.find(uv) == w_vars.end() || w_vars.find(vw) == w_vars.end() || w_vars.find(uw) == w_vars.end())
                    continue;

                double x_sum = x_bar.at(u) + x_bar.at(v) + x_bar.at(w);
                double w_sum = w_vars[uv].get(GRB_DoubleAttr_X) + w_vars[vw].get(GRB_DoubleAttr_X) + w_vars[uw].get(GRB_DoubleAttr_X);

                if (x_sum - w_sum > 1.0 + 1e-4)
                {
                    rmp->addConstr(x_vars[u] + x_vars[v] + x_vars[w] - w_vars[uv] - w_vars[vw] - w_vars[uw] <= 1.0);
                    cuts_added++;
                    if (cuts_added >= 20)
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

    // Feasibility pre-check: enough active nodes remain after excluding v0
    if (V_active.size() < (size_t)(k + v0.size()))
        return {std::move(local_x_bar), -1e9};

    unordered_set<int> v0_set(v0.begin(), v0.end());
    int eligible = 0;
    for (int v : V_active)
        if (v0_set.find(v) == v0_set.end())
            eligible++;
    if (eligible < k)
        return {std::move(local_x_bar), -1e9};

    auto t0 = chrono::high_resolution_clock::now();
    _apply_node_bounds(v1, v0);
    auto t1 = chrono::high_resolution_clock::now();
    stats.t_sync += chrono::duration<double>(t1 - t0).count();

    while (true)
    {
        auto t_now = chrono::high_resolution_clock::now();
        double wall = chrono::duration<double>(t_now - t_start_bb).count();
        double net = oracle.cumulative_network_time - net_start_bb;
        double effective_time = max(0.0, wall - net);

        if (effective_time > bb_time_limit)
            return {std::move(local_x_bar), local_lp_obj};

        t0 = chrono::high_resolution_clock::now();
        _sync_rmp_structure(lambda_val);
        t1 = chrono::high_resolution_clock::now();
        stats.t_sync += chrono::duration<double>(t1 - t0).count();

        double remaining_budget = max(1e-3, bb_time_limit - effective_time);
        rmp->set(GRB_DoubleParam_TimeLimit, remaining_budget);

        rmp->optimize();
        stats.total_lp_solves++;
        auto t2 = chrono::high_resolution_clock::now();
        stats.t_lp_solve += chrono::duration<double>(t2 - t1).count();

        int status = rmp->get(GRB_IntAttr_Status);
        if (status != GRB_OPTIMAL && status != GRB_SUBOPTIMAL && status != GRB_TIME_LIMIT)
            return {std::move(local_x_bar), -1e9};

        local_x_bar.clear();
        for (const auto &[v, var] : x_vars)
            local_x_bar[v] = var.get(GRB_DoubleAttr_X);
        double pi = size_constr.get(GRB_DoubleAttr_Pi);
        local_lp_obj = rmp->get(GRB_DoubleAttr_ObjVal);

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

        double diff = abs(val - 0.5);
        double internal_deg = 0.0;

        if (adj_out.count(v))
            for (int u : adj_out.at(v))
                if (x_bar.count(u))
                    internal_deg += x_bar.at(u);

        if (adj_in.count(v))
            for (int u : adj_in.at(v))
                if (x_bar.count(u))
                    internal_deg += x_bar.at(u);

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
            else if (abs(internal_deg - min_hanging_deg) <= 1e-3 && diff < min_diff)
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
            else if (abs(diff - min_diff) <= 1e-3 && internal_deg > max_core_deg)
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
    vector<BBNode> stack = {{{q}, {}}};
    double best_int_obj = 0.0;
    unordered_set<int> best_int_sol;
    double heuristic_global_ub = -1e9;

    auto t_start_bb = chrono::high_resolution_clock::now();
    double net_start_bb = oracle.cumulative_network_time;

    while (!stack.empty())
    {
        if (stats.total_bb_nodes >= bb_node_limit)
        {
            cout << "[" << get_timestamp() << "]     [!] B&B node limit reached." << endl;
            break;
        }

        auto t_now = chrono::high_resolution_clock::now();
        double wall = chrono::duration<double>(t_now - t_start_bb).count();
        double net = oracle.cumulative_network_time - net_start_bb;
        double effective_time = max(0.0, wall - net);

        if (effective_time > bb_time_limit)
        {
            cout << "[" << get_timestamp() << "]     [!] Iteration algorithmic time limit reached ("
                 << bb_time_limit << "s without improvement)." << endl;
            break;
        }

        if (best_int_obj > tol && heuristic_global_ub > -1e8)
        {
            double gap = (heuristic_global_ub - best_int_obj) / max(abs(best_int_obj), tol);
            if (gap <= bb_gap_tol)
                break;
        }

        BBNode node = std::move(stack.back());
        stack.pop_back();
        stats.total_bb_nodes++;

        auto [x_bar, lp_obj] = _column_generation(node.v1, node.v0, lambda_val, best_int_obj, t_start_bb, net_start_bb);

        if (x_bar.empty() || lp_obj <= best_int_obj + tol)
            continue;
        if (lp_obj > heuristic_global_ub)
            heuristic_global_ub = lp_obj;

        auto [branch_var, branch_zero_first] = _select_branch_var(x_bar, lambda_val);

        if (branch_var == -1)
        {
            // Integer solution — prune to improve the parametric objective before
            // accepting as incumbent
            unordered_set<int> sol_nodes;
            for (const auto &[v, val] : x_bar)
                if (val > 0.5)
                    sol_nodes.insert(v);

            _prune_discrete_solution(sol_nodes, lambda_val, false);

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

    return {best_int_sol, best_int_obj};
}

// ── Dinkelbach Outer Loop ─────────────────────────────────────────────────────

// Dinkelbach outer loop: iteratively solves the parametric subproblem and
// updates λ = d(S) until convergence. Returns the best solution and its density.
pair<unordered_set<int>, double> FullBranchAndPriceSolver::solve()
{
    auto t_start_global = chrono::high_resolution_clock::now();

    unordered_set<int> best_sol = V_active;

    // Prune the BFS seed to a near-optimal starting point before the first
    // Dinkelbach iteration, giving a tighter initial λ
    _prune_discrete_solution(best_sol, 0.0, true);

    double lambda_val = _density(best_sol);

    cout << fixed << setprecision(6);
    cout << "[" << get_timestamp() << "] Init Active Set | Size: " << best_sol.size() << " | Density: " << lambda_val << endl;
    cout << "--------------------------------------------------" << endl;

    for (int t = 1; t <= dinkelbach_max_iter; t++)
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

        if (sol.empty() || param_obj <= tol)
        {
            cout << "[" << get_timestamp() << "]   Status            : Converged (No improvement found)" << endl;
            break;
        }

        double new_density = _density(sol);
        cout << "[" << get_timestamp() << "]   Found Solution    : Size: " << sol.size() << " | New Density: " << new_density << endl;

        if (new_density <= lambda_val + tol)
        {
            cout << "[" << get_timestamp() << "]   Status            : Converged (Density bound reached)" << endl;
            break;
        }

        lambda_val = new_density;
        best_sol = sol;
    }

    auto t_end_global = chrono::high_resolution_clock::now();
    stats.t_total = chrono::duration<double>(t_end_global - t_start_global).count();

    return {best_sol, lambda_val};
}
