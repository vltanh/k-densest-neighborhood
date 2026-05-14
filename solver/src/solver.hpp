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
    double bb_hard_time_limit;
    double bb_gap_tol;
    int dinkelbach_max_iter;
    double cg_batch_fraction;
    int cg_min_batch;
    int cg_max_batch;
    int kappa;
    bool stream_incumbents;
    // When true, frontier-internal edges among unqueried F members stay
    // invisible to pricing. Trades correctness on F-internal cliques for zero
    // extra oracle hits per CG iter; intended for expensive oracles (OpenAlex).
    bool skip_materialize;

    std::unordered_set<int> V_active;
    std::unordered_set<int> F;
    std::unordered_set<int> error_nodes;
    // Set of nodes whose adjacency has been fetched from the oracle. Distinct
    // from V_active (which adds the node to the LP variable set): a node may
    // be materialized to fill in edges among the frontier without becoming an
    // LP variable. _expand_node materialises and adds to V_active; the new
    // _materialize_adjacency only materialises.
    std::unordered_set<int> queried_nodes;

    std::unordered_map<int, std::unordered_set<int>> adj_out;
    std::unordered_map<int, std::unordered_set<int>> adj_in;
    std::vector<std::pair<int, int>> pending_edges;

    GRBModel *rmp;
    std::unordered_map<int, GRBVar> x_vars;
    std::unordered_map<std::pair<int, int>, GRBVar, pair_hash> y_vars;
    std::unordered_map<std::pair<int, int>, GRBVar, pair_hash> w_vars;
    // Undirected support variables used only for rooted connectivity cuts.
    // z_{uv}=1 is allowed when at least one materialized directed edge between
    // u and v is selected. This keeps kappa cuts consistent with the
    // undirected support graph used by the max-flow verifier.
    std::unordered_map<std::pair<int, int>, GRBVar, pair_hash> z_vars;
    std::unordered_map<std::pair<int, int>, GRBConstr, pair_hash> z_link_constrs;

    struct ConnectivityCut
    {
        std::unordered_set<int> source_side;
        int target;
        GRBConstr constr;
    };
    std::vector<ConnectivityCut> connectivity_cuts;

    std::unordered_set<int> synced_nodes;
    std::vector<GRBVar> y_obj_terms;
    std::vector<GRBVar> w_obj_terms;
    std::unordered_set<int> bound_fixed;

    GRBConstr size_constr;
    double last_lambda = -1.0;

    void _add_edge(int u, int v);
    void _ingest_neighbors(int v, const std::vector<int> &preds, const std::vector<int> &succs);
    void _initialize_active_set();
    void _init_global_model();
    int _count_edges_in(const std::unordered_set<int> &nodes);
    double _density(const std::unordered_set<int> &nodes);
    double _parametric_obj(const std::unordered_set<int> &nodes, double lambda_val);
    void _expand_node(int f);
    void _materialize_adjacency(int f);
    int _materialize_unqueried_frontier();

    // _sync_rmp_structure helpers
    bool _register_new_nodes(std::vector<int> &new_nodes);
    bool _register_pending_edges();
    bool _register_pair_vars(const std::vector<int> &new_nodes, double lambda_val);
    void _update_objective(double lambda_val);
    bool _ensure_active_feasibility(const std::unordered_set<int> &v0_set);

    void _sync_rmp_structure(double lambda_val);
    void _apply_node_bounds(const std::vector<int> &v1, const std::vector<int> &v0);
    std::vector<int> _price_frontier(const std::unordered_map<int, double> &x_bar, double pi, const std::unordered_set<int> &v0_set, double lambda_val);
    std::vector<int> _greedy_joint_pricing(const std::unordered_set<int> &v0_set);
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
    void _prune_discrete_solution(std::unordered_set<int> &sol_nodes, double lambda_val, bool maximize_density, bool enforce_connectivity = false);
    bool _verify_kappa_connectivity(const std::unordered_set<int> &sol_nodes);
    void _record_incumbent(const std::unordered_set<int> &sol_nodes, double param_obj, double lambda_val);

public:
    SolverStats stats;
    bool last_kappa_verified = false;
    bool last_kappa_verify_failed = false;
    // True when the hard wall-time cap fired during this solve. The returned
    // incumbent is the best one found before the cap and may be suboptimal.
    bool last_hard_cap_hit = false;

    FullBranchAndPriceSolver(IGraphOracle &oracle, int q, int k, GRBEnv &env,
                             double tol = 1e-6, int bb_node_limit = -1, double bb_time_limit = -1.0,
                             double bb_gap_tol = -1.0, int dinkelbach_max_iter = -1,
                             double cg_batch_fraction = 1.0, int cg_min_batch = 0, int cg_max_batch = 50,
                             int kappa = 0, double bb_hard_time_limit = -1.0,
                             bool skip_materialize = false, bool stream_incumbents = false);
    // Solve start wall clock. Public so the Dinkelbach outer loop and the BB
    // inner loop share the same origin when checking the hard wall-time cap.
    std::chrono::high_resolution_clock::time_point t_start_solve;
    ~FullBranchAndPriceSolver();

    std::pair<std::unordered_set<int>, double> solve();
};
