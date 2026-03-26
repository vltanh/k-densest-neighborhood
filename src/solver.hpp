#pragma once
#include "common.hpp"
#include "oracle.hpp"
#include "gurobi_c++.h"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <utility>
#include <chrono>

class FullBranchAndPriceSolver
{
private:
    IGraphOracle &oracle;
    int q;
    int k;
    GRBEnv &env;
    double tol;
    int bb_node_limit;
    double bb_time_limit;
    double bb_gap_tol;
    int dinkelbach_max_iter;
    double cg_batch_fraction;
    int cg_min_batch;
    int cg_max_batch;

    std::unordered_set<int> V_active;
    std::unordered_set<int> F;
    std::unordered_set<int> error_nodes;

    std::unordered_map<int, std::unordered_set<int>> adj_out;
    std::unordered_map<int, std::unordered_set<int>> adj_in;
    std::vector<std::pair<int, int>> pending_edges;

    GRBModel *rmp;
    std::unordered_map<int, GRBVar> x_vars;
    std::unordered_map<std::pair<int, int>, GRBVar, pair_hash> y_vars;
    std::unordered_map<std::pair<int, int>, GRBVar, pair_hash> w_vars;

    std::unordered_set<int> synced_nodes;
    std::vector<GRBVar> y_obj_terms;
    std::vector<GRBVar> w_obj_terms;
    std::unordered_set<int> bound_fixed;

    GRBConstr size_constr;
    double last_lambda = -1.0;

    void _add_edge(int u, int v);
    void _initialize_active_set();
    void _init_global_model();
    int _count_edges_in(const std::unordered_set<int> &nodes);
    double _density(const std::unordered_set<int> &nodes);
    double _parametric_obj(const std::unordered_set<int> &nodes, double lambda_val);
    void _expand_node(int f);

    // _sync_rmp_structure helpers
    bool _register_new_nodes(std::vector<int> &new_nodes);
    bool _register_pending_edges();
    bool _register_pair_vars(const std::vector<int> &new_nodes);
    void _update_objective(double lambda_val);

    void _sync_rmp_structure(double lambda_val);
    void _apply_node_bounds(const std::vector<int> &v1, const std::vector<int> &v0);
    std::vector<int> _price_frontier(const std::unordered_map<int, double> &x_bar, double pi, const std::unordered_set<int> &v0_set, double lambda_val);
    int _separate_bqp_cuts(const std::unordered_map<int, double> &x_bar);
    std::pair<std::unordered_map<int, double>, double> _column_generation(
        const std::vector<int> &v1,
        const std::vector<int> &v0,
        double lambda_val,
        double current_incumbent,
        std::chrono::high_resolution_clock::time_point t_start_bb,
        double net_start_bb);
    std::pair<int, bool> _select_branch_var(const std::unordered_map<int, double> &x_bar, double lambda_val);
    std::pair<std::unordered_set<int>, double> _branch_and_price(double lambda_val);
    void _prune_discrete_solution(std::unordered_set<int> &sol_nodes, double lambda_val, bool maximize_density);

public:
    SolverStats stats;

    FullBranchAndPriceSolver(IGraphOracle &oracle, int q, int k, GRBEnv &env,
                             double tol = 1e-6, int bb_node_limit = 100000, double bb_time_limit = 60.0,
                             double bb_gap_tol = 1e-4, int dinkelbach_max_iter = 50,
                             double cg_batch_fraction = 0.1, int cg_min_batch = 5, int cg_max_batch = 50);
    ~FullBranchAndPriceSolver();

    std::pair<std::unordered_set<int>, double> solve();
};