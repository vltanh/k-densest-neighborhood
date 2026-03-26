#include "solver.hpp"
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cmath>

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
                {
                    q_nodes.push(u);
                }
            }
            for (int w : succs)
            {
                if (V_active.find(w) == V_active.end() && !error_nodes.count(w))
                {
                    q_nodes.push(w);
                }
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

    // Pass 2: Fill out edges for remaining V_active
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
    {
        throw std::runtime_error("Fatal: The query node itself failed to fetch!");
    }
}

void FullBranchAndPriceSolver::_init_global_model()
{
    rmp = new GRBModel(env);
    rmp->set(GRB_IntParam_OutputFlag, 0);

    GRBLinExpr expr = 0;
    size_constr = rmp->addConstr(expr >= k, "size_k");
}

int FullBranchAndPriceSolver::_count_edges_in(const unordered_set<int> &nodes)
{
    int edges = 0;
    for (int u : nodes)
    {
        auto it = adj_out.find(u);
        if (it != adj_out.end())
        {
            for (int v : it->second)
            {
                if (nodes.find(v) != nodes.end())
                {
                    edges++;
                }
            }
        }
    }
    return edges;
}

double FullBranchAndPriceSolver::_density(const unordered_set<int> &nodes)
{
    double n = nodes.size();
    if (n < 2)
        return 0.0;
    return (double)_count_edges_in(nodes) / (n * (n - 1));
}

double FullBranchAndPriceSolver::_parametric_obj(const unordered_set<int> &nodes, double lambda_val)
{
    double n = nodes.size();
    return (double)_count_edges_in(nodes) - lambda_val * (n * n - n);
}

void FullBranchAndPriceSolver::_expand_node(int f)
{
    // Abort immediately if this node was previously blacklisted
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
        // Banish the node completely
        error_nodes.insert(f);
        F.erase(f);
        V_active.erase(f);
    }
}

void FullBranchAndPriceSolver::_sync_rmp_structure(double lambda_val)
{
    bool structural_changes = false;
    vector<int> new_nodes;

    for (int v : V_active)
    {
        if (synced_nodes.find(v) == synced_nodes.end())
        {
            new_nodes.push_back(v);
        }
    }

    for (int v : new_nodes)
    {
        GRBVar var = rmp->addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "");
        x_vars[v] = var;
        structural_changes = true;
    }

    if (structural_changes)
        rmp->update();

    for (int v : new_nodes)
    {
        rmp->chgCoeff(size_constr, x_vars[v], 1.0);
    }

    vector<pair<int, int>> remaining_pending;

    for (auto const &uv : pending_edges)
    {
        if (x_vars.find(uv.first) != x_vars.end() && x_vars.find(uv.second) != x_vars.end())
        {
            if (y_vars.find(uv) == y_vars.end())
            {
                GRBVar yvar = rmp->addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "");
                rmp->addConstr(yvar <= x_vars[uv.first]);
                rmp->addConstr(yvar <= x_vars[uv.second]);
                y_vars[uv] = yvar;
                y_obj_terms.push_back(yvar);
                structural_changes = true;
            }
        }
        else
        {
            remaining_pending.push_back(uv);
        }
    }
    pending_edges = std::move(remaining_pending);

    if (!new_nodes.empty())
    {
        for (int u : new_nodes)
        {
            for (int v : synced_nodes)
            {
                pair<int, int> uv = (u < v) ? make_pair(u, v) : make_pair(v, u);
                if (w_vars.find(uv) == w_vars.end())
                {
                    GRBVar wvar = rmp->addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "");
                    rmp->addConstr(wvar >= x_vars[uv.first] + x_vars[uv.second] - 1);
                    w_vars[uv] = wvar;
                    w_obj_terms.push_back(wvar);
                    structural_changes = true;
                }
            }
        }

        for (size_t i = 0; i < new_nodes.size(); i++)
        {
            for (size_t j = i + 1; j < new_nodes.size(); j++)
            {
                int u = min(new_nodes[i], new_nodes[j]);
                int v = max(new_nodes[i], new_nodes[j]);
                pair<int, int> uv = {u, v};
                if (w_vars.find(uv) == w_vars.end())
                {
                    GRBVar wvar = rmp->addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "");
                    rmp->addConstr(wvar >= x_vars[u] + x_vars[v] - 1);
                    w_vars[uv] = wvar;
                    w_obj_terms.push_back(wvar);
                    structural_changes = true;
                }
            }
        }

        for (int v : new_nodes)
            synced_nodes.insert(v);
    }

    if (structural_changes || lambda_val != last_lambda)
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
}

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

pair<unordered_map<int, double>, double> FullBranchAndPriceSolver::_column_generation(const vector<int> &v1, const vector<int> &v0, double lambda_val, double current_incumbent, chrono::high_resolution_clock::time_point t_start)
{
    unordered_map<int, double> local_x_bar;
    double local_lp_obj = -1e9;
    double prev_lp_bound = 1e9;
    int consecutive_stalls = 0;

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
        if (chrono::duration<double>(t_now - t_start).count() > bb_time_limit)
            return {std::move(local_x_bar), local_lp_obj};

        t0 = chrono::high_resolution_clock::now();
        _sync_rmp_structure(lambda_val);
        t1 = chrono::high_resolution_clock::now();
        stats.t_sync += chrono::duration<double>(t1 - t0).count();

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
        {
            if (val > 0.1 && val < 0.9)
                n_fractional++;
        }

        double gap = 1.0;
        if (current_incumbent > tol)
            gap = (local_lp_obj - current_incumbent) / current_incumbent;

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

pair<unordered_set<int>, double> FullBranchAndPriceSolver::_branch_and_price(double lambda_val)
{
    vector<BBNode> stack = {{{q}, {}}};
    double best_int_obj = 0.0;
    unordered_set<int> best_int_sol;
    double heuristic_global_ub = -1e9;

    auto t_start = chrono::high_resolution_clock::now();

    while (!stack.empty())
    {
        if (stats.total_bb_nodes >= bb_node_limit)
        {
            cout << "[" << get_timestamp() << "]     [!] B&B node limit reached." << endl;
            break;
        }

        auto t_now = chrono::high_resolution_clock::now();
        if (chrono::duration<double>(t_now - t_start).count() > bb_time_limit)
        {
            cout << "[" << get_timestamp() << "]     [!] B&B time limit reached." << endl;
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

        auto [x_bar, lp_obj] = _column_generation(node.v1, node.v0, lambda_val, best_int_obj, t_start);

        if (x_bar.empty() || lp_obj <= best_int_obj + tol)
            continue;
        if (lp_obj > heuristic_global_ub)
            heuristic_global_ub = lp_obj;

        vector<int> fractional;
        int branch_var = -1;
        double min_diff = 1.0;
        int max_deg = -1;

        for (const auto &[v, val] : x_bar)
        {
            if (val > tol && val < 1.0 - tol)
            {
                fractional.push_back(v);
                double diff = abs(val - 0.5);

                int deg = 0;
                if (adj_out.count(v))
                    deg += adj_out[v].size();
                if (adj_in.count(v))
                    deg += adj_in[v].size();

                bool better = false;
                if (diff < min_diff - 1e-3)
                {
                    better = true;
                }
                else if (abs(diff - min_diff) <= 1e-3)
                {
                    if (deg > max_deg)
                    {
                        better = true;
                    }
                    else if (deg == max_deg && v < branch_var)
                    {
                        better = true;
                    }
                }

                if (better)
                {
                    min_diff = diff;
                    max_deg = deg;
                    branch_var = v;
                }
            }
        }

        if (fractional.empty())
        {
            unordered_set<int> sol_nodes;
            for (const auto &[v, val] : x_bar)
                if (val > 0.5)
                    sol_nodes.insert(v);

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
                }
            }
            continue;
        }

        BBNode child_0 = node;
        BBNode child_1 = std::move(node);

        child_0.v0.push_back(branch_var);
        child_1.v1.push_back(branch_var);

        stack.push_back(std::move(child_0));
        stack.push_back(std::move(child_1));
    }

    return {best_int_sol, best_int_obj};
}

pair<unordered_set<int>, double> FullBranchAndPriceSolver::solve()
{
    auto t_start_global = chrono::high_resolution_clock::now();

    unordered_set<int> best_sol = V_active;
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