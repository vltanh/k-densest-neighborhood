#include "exact_connected_avg_degree.hpp"
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <iomanip>

using namespace std;

// ============================================================================
// GUROBI LAZY CONSTRAINT CALLBACK (Cut-Set Separation)
// ============================================================================
class AvgDegConnectivityCallback : public GRBCallback
{
private:
    int query_node;
    const unordered_map<int, GRBVar> &x_vars;
    const unordered_map<pair<int, int>, GRBVar, pair_hash> &y_vars;

public:
    AvgDegConnectivityCallback(int q,
                               const unordered_map<int, GRBVar> &x,
                               const unordered_map<pair<int, int>, GRBVar, pair_hash> &y)
        : query_node(q), x_vars(x), y_vars(y) {}

protected:
    void callback() override
    {
        if (where == GRB_CB_MIPSOL)
        {
            unordered_set<int> selected_nodes;
            unordered_map<int, vector<int>> selected_adj;

            for (const auto &[node, var] : x_vars)
            {
                if (getSolution(var) > 0.5)
                {
                    selected_nodes.insert(node);
                }
            }

            for (const auto &[edge, var] : y_vars)
            {
                if (getSolution(var) > 0.5)
                {
                    selected_adj[edge.first].push_back(edge.second);
                    selected_adj[edge.second].push_back(edge.first);
                }
            }

            unordered_set<int> reachable;
            queue<int> q_bfs;

            if (selected_nodes.count(query_node))
            {
                q_bfs.push(query_node);
                reachable.insert(query_node);
            }

            while (!q_bfs.empty())
            {
                int curr = q_bfs.front();
                q_bfs.pop();
                for (int neighbor : selected_adj[curr])
                {
                    if (reachable.insert(neighbor).second)
                    {
                        q_bfs.push(neighbor);
                    }
                }
            }

            for (int node : selected_nodes)
            {
                if (reachable.find(node) == reachable.end())
                {
                    GRBLinExpr cut_expr = 0;
                    for (const auto &[edge, var] : y_vars)
                    {
                        bool u_in_C = reachable.count(edge.first);
                        bool v_in_C = reachable.count(edge.second);
                        if (u_in_C != v_in_C)
                        {
                            cut_expr += var;
                        }
                    }
                    addLazy(cut_expr >= x_vars.at(node));
                    break; // One cut per disconnected island is sufficient
                }
            }
        }
    }
};

// ============================================================================
// MAIN SOLVER (DINKELBACH + ILP)
// ============================================================================
vector<int> ExactConnectedAvgDegree::solve(int query_node, int depth)
{

    // 1. Extract the Local Graph via BFS
    unordered_set<int> V_local;
    vector<pair<int, int>> E_local; // Undirected edges
    queue<pair<int, int>> q_bfs;

    q_bfs.push({query_node, 0});
    V_local.insert(query_node);

    while (!q_bfs.empty())
    {
        auto [u, d] = q_bfs.front();
        q_bfs.pop();

        if (depth >= 0 && d >= depth)
            continue;

        pair<vector<int>, vector<int>> edges;
        try
        {
            edges = oracle_->query(u);
        }
        catch (...)
        {
            continue;
        }

        auto process_neighbor = [&](int v)
        {
            if (V_local.insert(v).second)
            {
                q_bfs.push({v, d + 1});
            }
            int min_uv = min(u, v);
            int max_uv = max(u, v);
            E_local.push_back({min_uv, max_uv});
        };

        for (int v : edges.second)
            process_neighbor(v);
        for (int v : edges.first)
            process_neighbor(v);
    }

    // Deduplicate edges
    unordered_set<pair<int, int>, pair_hash> unique_edges(E_local.begin(), E_local.end());

    cout << "[" << get_timestamp() << "] Local Graph Extracted | V: "
         << V_local.size() << " | E: " << unique_edges.size() << endl;

    // 2. Dinkelbach's Algorithm
    double lambda = 0.0;
    vector<int> best_nodes;
    double tol = 1e-5;
    int iter = 1;

    while (true)
    {
        GRBModel model(*env_);
        model.set(GRB_IntParam_OutputFlag, 0); // Silence Gurobi output
        model.set(GRB_IntParam_LazyConstraints, 1);
        model.set(GRB_DoubleParam_TimeLimit, 300.0); // 5 minute limit per iteration

        unordered_map<int, GRBVar> x;
        unordered_map<pair<int, int>, GRBVar, pair_hash> y;

        // Variables: x_u \in {0,1}
        for (int u : V_local)
        {
            x[u] = model.addVar(0.0, 1.0, -lambda, GRB_BINARY, "x_" + to_string(u));
        }

        // Variables: y_uv \in {0,1}
        for (const auto &edge : unique_edges)
        {
            y[edge] = model.addVar(0.0, 1.0, 1.0, GRB_BINARY,
                                   "y_" + to_string(edge.first) + "_" + to_string(edge.second));

            // Logical Constraints: y_uv <= x_u, y_uv <= x_v
            model.addConstr(y[edge] <= x[edge.first]);
            model.addConstr(y[edge] <= x[edge.second]);
        }

        // Anchor Constraint
        model.addConstr(x[query_node] == 1.0);

        model.set(GRB_IntAttr_ModelSense, GRB_MAXIMIZE);

        // Attach Connectivity Callback
        AvgDegConnectivityCallback cb(query_node, x, y);
        model.setCallback(&cb);

        cout << "[" << get_timestamp() << "] === DINKELBACH ITERATION " << iter
             << " | Lambda = " << fixed << setprecision(5) << lambda << " ===" << endl;

        model.optimize();

        if (model.get(GRB_IntAttr_Status) == GRB_TIME_LIMIT)
        {
            cout << "    [!] Gurobi Time Limit Reached during Dinkelbach iteration." << endl;
            break;
        }

        double F_lambda = model.get(GRB_DoubleAttr_ObjVal);
        cout << "    > Parametric Objective F(lambda): " << F_lambda << endl;

        if (F_lambda < tol)
        {
            cout << "    > Converged! Exact connected maximum found." << endl;
            break;
        }

        // Extract new solution
        double total_edges = 0;
        double total_nodes = 0;
        best_nodes.clear();

        for (const auto &[node, var] : x)
        {
            if (var.get(GRB_DoubleAttr_X) > 0.5)
            {
                best_nodes.push_back(node);
                total_nodes++;
            }
        }
        for (const auto &[edge, var] : y)
        {
            if (var.get(GRB_DoubleAttr_X) > 0.5)
                total_edges++;
        }

        // Update Lambda
        lambda = total_edges / total_nodes;
        iter++;
    }

    return best_nodes;
}
