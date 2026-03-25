#include <iostream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <cmath>
#include <string>
#include <iomanip>
#include <ctime>
#include "gurobi_c++.h"

using namespace std;

// ==========================================
// 1. TIMESTAMP LOGGER
// ==========================================
string get_timestamp() {
    auto now = chrono::system_clock::now();
    auto ms = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()) % 1000;
    time_t now_c = chrono::system_clock::to_time_t(now);
    tm parts;
    
    #if defined(_WIN32) || defined(_WIN64)
        localtime_s(&parts, &now_c);
    #else
        localtime_r(&now_c, &parts);
    #endif
    
    char buf[24];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &parts);
    
    ostringstream oss;
    oss << buf << "," << setfill('0') << setw(3) << ms.count();
    return oss.str();
}

// Standard hash for std::pair of integers
struct pair_hash {
    template <class T1, class T2>
    size_t operator () (const pair<T1, T2>& p) const {
        auto h1 = hash<T1>{}(p.first);
        auto h2 = hash<T2>{}(p.second);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

// ==========================================
// 2. STATS TRACKER
// ==========================================
struct SolverStats {
    int total_bb_nodes = 0;
    int total_lp_solves = 0;
    int total_columns_added = 0;
    int total_cuts_added = 0;
    
    double t_sync = 0.0;
    double t_lp_solve = 0.0;
    double t_pricing = 0.0;
    double t_separation = 0.0;
    double t_total = 0.0;
};

// ==========================================
// 3. DYNAMIC STRING-TO-INT ID MAPPER
// ==========================================
class IDMapper {
public:
    unordered_map<string, int> str_to_id;
    vector<string> id_to_str;

    int get_or_create_id(const string& s) {
        auto it = str_to_id.find(s);
        if (it != str_to_id.end()) return it->second;
        int id = id_to_str.size();
        str_to_id[s] = id;
        id_to_str.push_back(s);
        return id;
    }

    string get_str(int id) const { return id_to_str[id]; }
    int size() const { return id_to_str.size(); }
};

// ==========================================
// 4. LAZY ORACLE
// ==========================================
class DAGOracle {
private:
    unordered_map<string, vector<string>> db_adj_out;
    unordered_map<string, vector<string>> db_adj_in;

public:
    IDMapper mapper;
    int queries_made = 0;
    unordered_map<int, pair<vector<int>, vector<int>>> _cache;

    DAGOracle() {}

    void add_db_edge(const string& u, const string& v) {
        db_adj_out[u].push_back(v);
        db_adj_in[v].push_back(u);
    }

    // Zero-copy const reference return for high-performance retrieval
    const pair<vector<int>, vector<int>>& query(int v_int) {
        if (_cache.find(v_int) != _cache.end()) {
            return _cache[v_int];
        }
        queries_made++;
        
        string v_str = mapper.get_str(v_int);
        
        vector<int> int_preds;
        if (db_adj_in.find(v_str) != db_adj_in.end()) {
            for (const string& u_str : db_adj_in[v_str]) {
                int_preds.push_back(mapper.get_or_create_id(u_str));
            }
        }
        
        vector<int> int_succs;
        if (db_adj_out.find(v_str) != db_adj_out.end()) {
            for (const string& w_str : db_adj_out[v_str]) {
                int_succs.push_back(mapper.get_or_create_id(w_str));
            }
        }
        
        _cache[v_int] = {int_preds, int_succs};
        return _cache[v_int];
    }
};

// ==========================================
// 5. SOLVER STATE STRUCTURE
// ==========================================
struct BBNode {
    vector<int> v1; 
    vector<int> v0;
};

// ==========================================
// 6. EXACT FULL BRANCH-AND-PRICE SOLVER
// ==========================================
class FullBranchAndPriceSolver {
private:
    DAGOracle& oracle;
    int q;
    int k;
    GRBEnv& env;
    double tol;
    int bb_node_limit;
    double bb_time_limit;
    double bb_gap_tol;
    int dinkelbach_max_iter;
    double cg_batch_fraction;
    int cg_min_batch, cg_max_batch;

    unordered_set<int> V_active;
    unordered_set<int> F;
    unordered_set<pair<int, int>, pair_hash> E_known;
    
    unordered_map<int, unordered_set<int>> adj_out;
    unordered_map<int, unordered_set<int>> adj_in;
    unordered_set<pair<int, int>, pair_hash> pending_edges;

    GRBModel* rmp;
    unordered_map<int, GRBVar> x_vars;
    unordered_map<pair<int, int>, GRBVar, pair_hash> y_vars;
    unordered_map<pair<int, int>, GRBVar, pair_hash> w_vars;
    
    unordered_set<int> synced_nodes;
    vector<GRBVar> y_obj_terms;
    vector<GRBVar> w_obj_terms;
    unordered_set<int> bound_fixed;

    GRBConstr size_constr;
    double last_lambda = -1.0;

    void _add_edge(int u, int v) {
        pair<int, int> edge = {u, v};
        if (E_known.find(edge) == E_known.end()) {
            E_known.insert(edge);
            adj_out[u].insert(v);
            adj_in[v].insert(u);
            pending_edges.insert(edge);
        }
    }

    void _initialize_active_set() {
        V_active.insert(q);
        queue<int> q_nodes;
        q_nodes.push(q);
        unordered_set<int> bfs_queried;

        while (V_active.size() < (size_t)k && !q_nodes.empty()) {
            int curr = q_nodes.front();
            q_nodes.pop();
            bfs_queried.insert(curr);
            
            const auto& [preds, succs] = oracle.query(curr);
            
            for (int u : preds) {
                _add_edge(u, curr);
                if (V_active.find(u) == V_active.end()) F.insert(u);
            }
            for (int w : succs) {
                _add_edge(curr, w);
                if (V_active.find(w) == V_active.end()) F.insert(w);
            }

            for (int u : preds) {
                if (V_active.find(u) == V_active.end()) {
                    V_active.insert(u);
                    F.erase(u);
                    q_nodes.push(u);
                    if (V_active.size() >= (size_t)k) break;
                }
            }
            if (V_active.size() >= (size_t)k) break;
            
            for (int w : succs) {
                if (V_active.find(w) == V_active.end()) {
                    V_active.insert(w);
                    F.erase(w);
                    q_nodes.push(w);
                    if (V_active.size() >= (size_t)k) break;
                }
            }
        }

        for (int v : V_active) {
            if (bfs_queried.find(v) == bfs_queried.end()) {
                const auto& [preds, succs] = oracle.query(v);
                for (int u : preds) {
                    _add_edge(u, v);
                    if (V_active.find(u) == V_active.end()) F.insert(u);
                }
                for (int w : succs) {
                    _add_edge(v, w);
                    if (V_active.find(w) == V_active.end()) F.insert(w);
                }
            }
        }
    }

    void _init_global_model() {
        rmp = new GRBModel(env);
        rmp->set(GRB_IntParam_OutputFlag, 0);
        
        GRBLinExpr expr = 0;
        size_constr = rmp->addConstr(expr >= k, "size_k");
    }

    int _count_edges_in(const unordered_set<int>& nodes) {
        int edges = 0;
        for (int u : nodes) {
            if (adj_out.find(u) != adj_out.end()) {
                for (int v : adj_out[u]) {
                    if (nodes.find(v) != nodes.end()) {
                        edges++;
                    }
                }
            }
        }
        return edges;
    }

    double _density(const unordered_set<int>& nodes) {
        double n = nodes.size();
        if (n < 2) return 0.0;
        return (double)_count_edges_in(nodes) / (n * (n - 1));
    }

    double _parametric_obj(const unordered_set<int>& nodes, double lambda_val) {
        double n = nodes.size();
        return (double)_count_edges_in(nodes) - lambda_val * (n * n - n);
    }

    void _expand_node(int f) {
        V_active.insert(f);
        F.erase(f);
        const auto& [preds, succs] = oracle.query(f);

        for (int u : preds) {
            _add_edge(u, f);
            if (V_active.find(u) == V_active.end()) F.insert(u);
        }
        for (int w : succs) {
            _add_edge(f, w);
            if (V_active.find(w) == V_active.end()) F.insert(w);
        }
    }

    void _sync_rmp_structure(double lambda_val) {
        bool structural_changes = false;
        vector<int> new_nodes;
        
        for (int v : V_active) {
            if (synced_nodes.find(v) == synced_nodes.end()) {
                new_nodes.push_back(v);
            }
        }

        for (int v : new_nodes) {
            GRBVar var = rmp->addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "x_" + to_string(v));
            x_vars[v] = var;
            structural_changes = true;
        }
        
        if (structural_changes) rmp->update(); 

        for (int v : new_nodes) {
            rmp->chgCoeff(size_constr, x_vars[v], 1.0);
        }

        vector<pair<int, int>> edges_to_remove;
        vector<pair<int, int>> new_y_pairs;
        
        for (auto const& uv : pending_edges) {
            if (x_vars.find(uv.first) != x_vars.end() && x_vars.find(uv.second) != x_vars.end()) {
                if (y_vars.find(uv) == y_vars.end()) {
                    GRBVar yvar = rmp->addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "y_" + to_string(uv.first) + "_" + to_string(uv.second));
                    y_vars[uv] = yvar;
                    y_obj_terms.push_back(yvar);
                    new_y_pairs.push_back(uv);
                    structural_changes = true;
                }
                edges_to_remove.push_back(uv);
            }
        }
        
        if (!new_y_pairs.empty()) {
            rmp->update();
            for (auto const& uv : new_y_pairs) {
                rmp->addConstr(y_vars[uv] <= x_vars[uv.first]);
                rmp->addConstr(y_vars[uv] <= x_vars[uv.second]);
            }
        }
        
        for (auto const& edge : edges_to_remove) pending_edges.erase(edge);

        if (!new_nodes.empty()) {
            vector<pair<int, int>> new_w_pairs;
            
            for (int u : new_nodes) {
                for (int v : synced_nodes) {
                    pair<int, int> uv = (u < v) ? make_pair(u, v) : make_pair(v, u);
                    if (w_vars.find(uv) == w_vars.end()) {
                        GRBVar wvar = rmp->addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "w_" + to_string(uv.first) + "_" + to_string(uv.second));
                        w_vars[uv] = wvar;
                        w_obj_terms.push_back(wvar);
                        new_w_pairs.push_back(uv);
                        structural_changes = true;
                    }
                }
            }

            for (size_t i = 0; i < new_nodes.size(); i++) {
                for (size_t j = i + 1; j < new_nodes.size(); j++) {
                    int u = min(new_nodes[i], new_nodes[j]);
                    int v = max(new_nodes[i], new_nodes[j]);
                    pair<int, int> uv = {u, v};
                    if (w_vars.find(uv) == w_vars.end()) {
                        GRBVar wvar = rmp->addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "w_" + to_string(u) + "_" + to_string(v));
                        w_vars[uv] = wvar;
                        w_obj_terms.push_back(wvar);
                        new_w_pairs.push_back(uv);
                        structural_changes = true;
                    }
                }
            }
            
            if (!new_w_pairs.empty()) {
                rmp->update();
                for (auto const& uv : new_w_pairs) {
                    rmp->addConstr(w_vars[uv] >= x_vars[uv.first] + x_vars[uv.second] - 1);
                }
            }
            
            for (int v : new_nodes) synced_nodes.insert(v);
        }

        if (structural_changes || lambda_val != last_lambda) {
            rmp->update();
            GRBLinExpr obj_expr = 0;
            
            if (!y_obj_terms.empty()) {
                vector<double> coeffs(y_obj_terms.size(), 1.0);
                obj_expr.addTerms(coeffs.data(), y_obj_terms.data(), y_obj_terms.size());
            }
            
            if (!w_obj_terms.empty()) {
                vector<double> coeffs(w_obj_terms.size(), -2.0 * lambda_val);
                obj_expr.addTerms(coeffs.data(), w_obj_terms.data(), w_obj_terms.size());
            }
            
            rmp->setObjective(obj_expr, GRB_MAXIMIZE);
            last_lambda = lambda_val;
        }
    }

    void _apply_node_bounds(const vector<int>& v1, const vector<int>& v0) {
        for (int v : bound_fixed) {
            if (x_vars.find(v) != x_vars.end()) {
                x_vars[v].set(GRB_DoubleAttr_LB, 0.0);
                x_vars[v].set(GRB_DoubleAttr_UB, 1.0);
            }
        }
        
        bound_fixed.clear();
        
        for (int v : v1) {
            if (x_vars.find(v) != x_vars.end()) {
                x_vars[v].set(GRB_DoubleAttr_LB, 1.0);
                x_vars[v].set(GRB_DoubleAttr_UB, 1.0);
                bound_fixed.insert(v);
            }
        }
        
        for (int v : v0) {
            if (x_vars.find(v) != x_vars.end()) {
                x_vars[v].set(GRB_DoubleAttr_LB, 0.0);
                x_vars[v].set(GRB_DoubleAttr_UB, 0.0);
                bound_fixed.insert(v);
            }
        }
        
        rmp->update();
    }

    vector<int> _price_frontier(const unordered_map<int, double>& x_bar, double pi, const unordered_set<int>& v0_set, double lambda_val) {
        double sum_x_bar = 0.0;
        for (const auto& [v, val] : x_bar) sum_x_bar += val;
        
        double omega = -2.0 * lambda_val * sum_x_bar - pi;
        vector<pair<double, int>> candidates;

        for (int f : F) {
            if (v0_set.find(f) != v0_set.end()) continue;
            
            double frac_deg = 0.0;
            if (adj_out.find(f) != adj_out.end()) {
                for (int v : adj_out[f]) {
                    if (V_active.find(v) != V_active.end() && x_bar.find(v) != x_bar.end()) frac_deg += x_bar.at(v);
                }
            }
            if (adj_in.find(f) != adj_in.end()) {
                for (int u : adj_in[f]) {
                    if (V_active.find(u) != V_active.end() && x_bar.find(u) != x_bar.end()) frac_deg += x_bar.at(u);
                }
            }
            
            double rc = frac_deg + omega;
            if (rc > tol) candidates.push_back({rc, f});
        }

        int dynamic_limit = V_active.size() * cg_batch_fraction;
        size_t batch_size = max((size_t)cg_min_batch, min((size_t)dynamic_limit, (size_t)cg_max_batch));
        batch_size = min(batch_size, candidates.size());

        partial_sort(candidates.begin(), candidates.begin() + batch_size, candidates.end(), 
                     [](const pair<double, int>& a, const pair<double, int>& b) { return a.first > b.first; });

        vector<int> top_f;
        for (size_t i = 0; i < batch_size; i++) top_f.push_back(candidates[i].second);
        return top_f;
    }

    int _separate_bqp_cuts(const unordered_map<int, double>& x_bar) {
        vector<int> frac_nodes;
        for (const auto& [v, val] : x_bar) {
            if (val > 0.1 && val < 0.9) frac_nodes.push_back(v);
        }
        
        int n = frac_nodes.size();
        if (n < 3) return 0;
        
        int cuts_added = 0;
        
        for (int idx1 = 0; idx1 < n; idx1++) {
            for (int idx2 = idx1 + 1; idx2 < n; idx2++) {
                for (int idx3 = idx2 + 1; idx3 < n; idx3++) {
                    int u = frac_nodes[idx1], v = frac_nodes[idx2], w = frac_nodes[idx3];
                    
                    pair<int, int> uv = (u < v) ? make_pair(u, v) : make_pair(v, u);
                    pair<int, int> vw = (v < w) ? make_pair(v, w) : make_pair(w, v);
                    pair<int, int> uw = (u < w) ? make_pair(u, w) : make_pair(w, u);
                    
                    if (w_vars.find(uv) == w_vars.end() || w_vars.find(vw) == w_vars.end() || w_vars.find(uw) == w_vars.end()) continue;
                    
                    double x_sum = x_bar.at(u) + x_bar.at(v) + x_bar.at(w);
                    double w_sum = w_vars[uv].get(GRB_DoubleAttr_X) + w_vars[vw].get(GRB_DoubleAttr_X) + w_vars[uw].get(GRB_DoubleAttr_X);
                    
                    if (x_sum - w_sum > 1.0 + 1e-4) {
                        rmp->addConstr(x_vars[u] + x_vars[v] + x_vars[w] - w_vars[uv] - w_vars[vw] - w_vars[uw] <= 1.0);
                        cuts_added++;
                        if (cuts_added >= 20) return cuts_added;
                    }
                }
            }
        }
        return cuts_added;
    }

    pair<unordered_map<int, double>, double> _column_generation(const vector<int>& v1, const vector<int>& v0, double lambda_val, auto t_start) {
        unordered_map<int, double> local_x_bar;
        double local_lp_obj = -1e9;

        // Hoist the hash set creation out of the while loop
        unordered_set<int> v0_set(v0.begin(), v0.end());
        int eligible = 0;
        for (int v : V_active) if (v0_set.find(v) == v0_set.end()) eligible++;
        
        if (eligible < k) return {std::move(local_x_bar), -1e9};

        while (true) {
            auto t_now = chrono::high_resolution_clock::now();
            if (chrono::duration<double>(t_now - t_start).count() > bb_time_limit) return {std::move(local_x_bar), local_lp_obj};

            auto t0 = chrono::high_resolution_clock::now();
            _sync_rmp_structure(lambda_val);
            _apply_node_bounds(v1, v0);
            auto t1 = chrono::high_resolution_clock::now();
            stats.t_sync += chrono::duration<double>(t1 - t0).count();

            rmp->optimize();
            stats.total_lp_solves++;
            auto t2 = chrono::high_resolution_clock::now();
            stats.t_lp_solve += chrono::duration<double>(t2 - t1).count();

            int status = rmp->get(GRB_IntAttr_Status);
            if (status != GRB_OPTIMAL && status != GRB_SUBOPTIMAL && status != GRB_TIME_LIMIT) return {std::move(local_x_bar), -1e9};

            local_x_bar.clear();
            for (const auto& [v, var] : x_vars) local_x_bar[v] = var.get(GRB_DoubleAttr_X);
            double pi = size_constr.get(GRB_DoubleAttr_Pi);
            local_lp_obj = rmp->get(GRB_DoubleAttr_ObjVal);

            auto t3 = chrono::high_resolution_clock::now();
            vector<int> top_f = _price_frontier(local_x_bar, pi, v0_set, lambda_val);
            auto t4 = chrono::high_resolution_clock::now();
            stats.t_pricing += chrono::duration<double>(t4 - t3).count();

            if (!top_f.empty()) {
                stats.total_columns_added += top_f.size();
                for (int f : top_f) _expand_node(f);
                continue;
            }

            int n_fractional = 0;
            for (const auto& [v, val] : local_x_bar) if (val > 0.1 && val < 0.9) n_fractional++;
            
            if (n_fractional > k) {
                auto t5 = chrono::high_resolution_clock::now();
                int cuts = _separate_bqp_cuts(local_x_bar);
                auto t6 = chrono::high_resolution_clock::now();
                stats.t_separation += chrono::duration<double>(t6 - t5).count();
                
                if (cuts > 0) {
                    stats.total_cuts_added += cuts;
                    continue;
                }
            }

            return {std::move(local_x_bar), local_lp_obj};
        }
    }

    pair<unordered_set<int>, double> _branch_and_price(double lambda_val) {
        vector<BBNode> stack = {{ {q}, {} }};
        double best_int_obj = 0.0;
        unordered_set<int> best_int_sol;
        double heuristic_global_ub = -1e9;
        
        auto t_start = chrono::high_resolution_clock::now();

        while (!stack.empty()) {
            if (stats.total_bb_nodes >= bb_node_limit) {
                cout << "[" << get_timestamp() << "]     [!] B&B node limit reached." << endl;
                break;
            }
            
            auto t_now = chrono::high_resolution_clock::now();
            if (chrono::duration<double>(t_now - t_start).count() > bb_time_limit) {
                cout << "[" << get_timestamp() << "]     [!] B&B time limit reached." << endl;
                break;
            }

            if (best_int_obj > tol && heuristic_global_ub > -1e8) {
                double gap = (heuristic_global_ub - best_int_obj) / max(abs(best_int_obj), tol);
                if (gap <= bb_gap_tol) break;
            }

            BBNode node = std::move(stack.back());
            stack.pop_back();
            stats.total_bb_nodes++;

            auto [x_bar, lp_obj] = _column_generation(node.v1, node.v0, lambda_val, t_start);

            if (x_bar.empty() || lp_obj <= best_int_obj + tol) continue;
            if (lp_obj > heuristic_global_ub) heuristic_global_ub = lp_obj;

            vector<int> fractional;
            int branch_var = -1;
            double min_diff = 1.0;

            for (const auto& [v, val] : x_bar) {
                if (val > tol && val < 1.0 - tol) {
                    fractional.push_back(v);
                    double diff = abs(val - 0.5);
                    if (diff < min_diff) {
                        min_diff = diff;
                        branch_var = v;
                    }
                }
            }

            if (fractional.empty()) {
                unordered_set<int> sol_nodes;
                for (const auto& [v, val] : x_bar) if (val > 0.5) sol_nodes.insert(v);
                
                if (sol_nodes.size() >= (size_t)k) {
                    double obj = _parametric_obj(sol_nodes, lambda_val);
                    if (obj > best_int_obj) {
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

public:
    SolverStats stats;

    FullBranchAndPriceSolver(DAGOracle& oracle, int q, int k, GRBEnv& env,
                             double tol=1e-6, int bb_node_limit=100000, double bb_time_limit=300.0,
                             double bb_gap_tol=1e-4, int dinkelbach_max_iter=50,
                             double cg_batch_fraction=0.1, int cg_min_batch=5, int cg_max_batch=50)
        : oracle(oracle), q(q), k(k), env(env), tol(tol), bb_node_limit(bb_node_limit),
          bb_time_limit(bb_time_limit), bb_gap_tol(bb_gap_tol), dinkelbach_max_iter(dinkelbach_max_iter),
          cg_batch_fraction(cg_batch_fraction), cg_min_batch(cg_min_batch), cg_max_batch(cg_max_batch), rmp(nullptr) 
    {
        _initialize_active_set();
        _init_global_model();
    }

    ~FullBranchAndPriceSolver() {
        if (rmp) delete rmp;
    }

    pair<unordered_set<int>, double> solve() {
        auto t_start_global = chrono::high_resolution_clock::now();
        
        unordered_set<int> best_sol = V_active;
        double lambda_val = _density(best_sol);
        
        cout << fixed << setprecision(6);
        cout << "[" << get_timestamp() << "] Init Active Set | Size: " << best_sol.size() << " | Density: " << lambda_val << endl;
        cout << "--------------------------------------------------" << endl;

        for (int t = 1; t <= dinkelbach_max_iter; t++) {
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

            if (sol.empty() || param_obj <= tol) {
                cout << "[" << get_timestamp() << "]   Status            : Converged (No improvement found)" << endl;
                break;
            }

            double new_density = _density(sol);
            cout << "[" << get_timestamp() << "]   Found Solution    : Size: " << sol.size() << " | New Density: " << new_density << endl;

            if (new_density <= lambda_val + tol) {
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
};

// ==========================================
// 7. MAIN EXECUTION
// ==========================================
int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <edge.csv> <q_node_string_id> <k_target>" << endl;
        return 1;
    }

    string filename = argv[1];
    string q_node_str = argv[2];
    int k_target = stoi(argv[3]);

    try {
        GRBEnv env = GRBEnv(true);
        env.set("OutputFlag", "0");
        env.set("Method", "1"); 
        env.start();

        cout << "==================================================" << endl;
        cout << "K-DENSEST NEIGHBORHOOD (LAZY API SIMULATOR)" << endl;
        cout << "==================================================" << endl;

        auto t_io_start = chrono::high_resolution_clock::now();
        
        DAGOracle oracle;
        ifstream infile(filename);
        if (!infile.is_open()) {
            cerr << "Error: Could not open file " << filename << endl;
            return 1;
        }

        string line;
        size_t edge_count = 0;
        if (getline(infile, line)) { 
            while (getline(infile, line)) {
                size_t comma_pos = line.find(',');
                if (comma_pos != string::npos) {
                    string u_str = line.substr(0, comma_pos);
                    string v_str = line.substr(comma_pos + 1);
                    if (!v_str.empty() && v_str.back() == '\r') v_str.pop_back();
                    
                    oracle.add_db_edge(u_str, v_str);
                    edge_count++;
                }
            }
        }
        infile.close();

        int q_node_int = oracle.mapper.get_or_create_id(q_node_str);

        auto t_io_end = chrono::high_resolution_clock::now();
        cout << "[" << get_timestamp() << "] Sim DB Loaded     | Edges: " << edge_count << " | IO Time: " 
             << fixed << setprecision(2) << chrono::duration<double>(t_io_end - t_io_start).count() << "s" << endl;
        cout << "--------------------------------------------------" << endl;

        FullBranchAndPriceSolver solver(oracle, q_node_int, k_target, env);
        auto [best_nodes, final_density] = solver.solve();
        
        cout << "==================================================" << endl;
        cout << "OPTIMIZATION STATISTICS" << endl;
        cout << "==================================================" << endl;
        
        cout << left << setw(25) << "B&B Nodes Explored" << ": " << solver.stats.total_bb_nodes << endl;
        cout << left << setw(25) << "Total LP Solves" << ": " << solver.stats.total_lp_solves << endl;
        cout << left << setw(25) << "Columns Generated" << ": " << solver.stats.total_columns_added << endl;
        cout << left << setw(25) << "BQP Cuts Added" << ": " << solver.stats.total_cuts_added << endl;
        
        cout << left << setw(25) << "API Queries Made" << ": " << oracle.queries_made << endl;
        cout << left << setw(25) << "Unique Nodes Mapped" << ": " << oracle.mapper.size() << endl;
        cout << "--------------------------------------------------" << endl;
        
        cout << "TIMING BREAKDOWN" << endl;
        cout << left << setw(25) << "Model Sync Time" << ": " << fixed << setprecision(3) << solver.stats.t_sync << "s" << endl;
        cout << left << setw(25) << "Gurobi LP Time" << ": " << fixed << setprecision(3) << solver.stats.t_lp_solve << "s" << endl;
        cout << left << setw(25) << "Pricing Time" << ": " << fixed << setprecision(3) << solver.stats.t_pricing << "s" << endl;
        cout << left << setw(25) << "Separation Time" << ": " << fixed << setprecision(3) << solver.stats.t_separation << "s" << endl;
        cout << "--------------------------------------------------" << endl;
        cout << left << setw(25) << "Total Solver Time" << ": " << fixed << setprecision(3) << solver.stats.t_total << "s" << endl;
        
        cout << "==================================================" << endl;
        cout << "FINAL SOLUTION" << endl;
        cout << "==================================================" << endl;
        cout << left << setw(25) << "Density" << ": " << fixed << setprecision(6) << final_density << endl;
        cout << left << setw(25) << "Size" << ": " << best_nodes.size() << endl;
        cout << "Nodes:" << endl;
        
        for (int node : best_nodes) {
            cout << oracle.mapper.get_str(node) << endl;
        }

    } catch(const GRBException& e) {
        cerr << "[" << get_timestamp() << "] Gurobi Error code = " << e.getErrorCode() << endl;
        cerr << e.getMessage() << endl;
    } catch(...) {
        cerr << "[" << get_timestamp() << "] Exception during optimization" << endl;
    }

    return 0;
}