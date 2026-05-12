#include "average_degree_solver.hpp"
#include "bfs_expand.hpp"
#include "grow_to_k.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/push_relabel_max_flow.hpp>
#include <queue>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <limits>

using namespace boost;

namespace
{
constexpr long long SCALE = 1000000LL;
constexpr long long INF_CAP = std::numeric_limits<long long>::max() / 4;
constexpr int MAX_BISECT_ITERS = 100;
constexpr double BISECT_TOL = 1e-9;
}

// Layer-by-layer BFS to depth d. Records each layer's directed-out adjacency
// via expand_bfs_layer so lg.edges captures every (u,v) directed edge where
// both endpoints lie in lg.nodes. Undirected adj, queried, and error_nodes are
// passed out for downstream consumers (grow_to_k_with_oracle).
AverageDegreeSolver::LocalGraph AverageDegreeSolver::explore_neighborhood(
    int start, int depth,
    std::unordered_set<int> &queried,
    std::unordered_set<int> &error_nodes,
    std::unordered_map<int, std::unordered_set<int>> &undirected_adj)
{
    LocalGraph lg;
    std::unordered_set<int> visited;
    visited.insert(start);
    undirected_adj[start];

    std::vector<int> frontier{start};
    std::vector<std::pair<int, std::vector<int>>> all_directed;

    int strict_depth = (depth < 0) ? std::numeric_limits<int>::max() : depth;
    for (int layer = 0; layer <= strict_depth && !frontier.empty(); ++layer)
    {
        std::vector<std::pair<int, std::vector<int>>> layer_directed;
        auto next_frontier = expand_bfs_layer(frontier, oracle_, start,
                                              visited, queried, error_nodes, undirected_adj,
                                              &layer_directed);
        for (auto &p : layer_directed)
        {
            int u = p.first;
            if (lg.id_map.find(u) == lg.id_map.end())
            {
                lg.id_map[u] = (int)lg.nodes.size();
                lg.nodes.push_back(u);
            }
            all_directed.push_back(std::move(p));
        }
        frontier = next_frontier;
    }

    for (const auto &[u, succs] : all_directed)
    {
        auto it_u = lg.id_map.find(u);
        if (it_u == lg.id_map.end())
            continue;
        int i = it_u->second;
        for (int v : succs)
        {
            auto it_v = lg.id_map.find(v);
            if (it_v != lg.id_map.end())
                lg.edges.push_back({i, it_v->second});
        }
    }
    return lg;
}

std::vector<int> AverageDegreeSolver::solve(int query_node, int depth, int k)
{
    std::unordered_set<int> queried;
    std::unordered_set<int> error_nodes;
    std::unordered_map<int, std::unordered_set<int>> undirected_adj;
    LocalGraph lg = explore_neighborhood(query_node, depth,
                                         queried, error_nodes, undirected_adj);

    if (lg.id_map.find(query_node) == lg.id_map.end())
    {
        std::cerr << "[" << get_timestamp() << "] Error: Query node could not be fetched.\n";
        return {};
    }

    const int n = (int)lg.nodes.size();
    const int query_idx = lg.id_map[query_node];

    std::unordered_set<std::pair<int, int>, pair_hash> edge_set(lg.edges.begin(), lg.edges.end());
    std::vector<std::pair<int, int>> directed_edges(edge_set.begin(), edge_set.end());

    using Traits = adjacency_list_traits<vecS, vecS, directedS>;
    using FlowGraph = adjacency_list<vecS, vecS, directedS,
                                     property<vertex_name_t, int>,
                                     property<edge_capacity_t, long long,
                                              property<edge_residual_capacity_t, long long,
                                                       property<edge_reverse_t, Traits::edge_descriptor>>>>;

    auto solve_threshold = [&](double lambda_val)
    {
        FlowGraph g(n + (int)directed_edges.size() + 2);
        auto capacity = get(edge_capacity, g);
        auto rev = get(edge_reverse, g);

        auto add_flow_edge = [&](int u, int v, long long cap)
        {
            auto e1 = add_edge(u, v, g).first;
            auto e2 = add_edge(v, u, g).first;
            capacity[e1] = cap;
            capacity[e2] = 0;
            rev[e1] = e2;
            rev[e2] = e1;
        };

        const int source = n + (int)directed_edges.size();
        const int sink = source + 1;

        for (int i = 0; i < n; ++i)
        {
            long long node_cost = (long long)llround(lambda_val * SCALE);
            if (node_cost > 0)
                add_flow_edge(i, sink, node_cost);
        }

        for (size_t idx = 0; idx < directed_edges.size(); ++idx)
        {
            const auto &[u_idx, v_idx] = directed_edges[idx];
            const int item = n + (int)idx;
            add_flow_edge(source, item, SCALE);
            add_flow_edge(item, u_idx, INF_CAP);
            add_flow_edge(item, v_idx, INF_CAP);
        }

        add_flow_edge(source, query_idx, INF_CAP);

        push_relabel_max_flow(g, source, sink);

        auto res = get(edge_residual_capacity, g);
        std::vector<bool> vis(n + (int)directed_edges.size() + 2, false);
        std::queue<int> q_nodes;
        q_nodes.push(source);
        vis[source] = true;

        while (!q_nodes.empty())
        {
            int u = q_nodes.front();
            q_nodes.pop();
            for (auto e : make_iterator_range(out_edges(u, g)))
            {
                int v = target(e, g);
                if (res[e] > 0 && !vis[v])
                {
                    vis[v] = true;
                    q_nodes.push(v);
                }
            }
        }

        std::vector<int> selected_nodes;
        int selected_edges = 0;
        for (int i = 0; i < n; ++i)
            if (vis[i])
                selected_nodes.push_back(lg.nodes[i]);
        for (size_t idx = 0; idx < directed_edges.size(); ++idx)
            if (vis[n + (int)idx])
                selected_edges++;

        double value = (double)selected_edges - lambda_val * (double)selected_nodes.size();
        return std::make_pair(selected_nodes, std::make_pair(selected_edges, value));
    };

    double low = 0.0;
    double high = (double)directed_edges.size();
    std::vector<int> best_nodes;
    double best_density = -1.0;
    bool low_probed = false;
    bool high_probed = false;

    auto consider = [&](const std::vector<int> &nodes, int edges)
    {
        if (nodes.empty())
            return;
        double density = (double)edges / nodes.size();
        if (density > best_density)
        {
            best_density = density;
            best_nodes = nodes;
        }
    };

    for (int iter = 0; iter < MAX_BISECT_ITERS; ++iter)
    {
        double mid = (low + high) / 2.0;
        auto [nodes_mid, pair_mid] = solve_threshold(mid);
        int edges_mid = pair_mid.first;
        double value_mid = pair_mid.second;
        consider(nodes_mid, edges_mid);

        if (value_mid > 0.0)
        {
            low = mid;
            low_probed = true;
            high_probed = false;
        }
        else
        {
            high = mid;
            high_probed = true;
            low_probed = false;
        }

        if (high - low <= BISECT_TOL)
            break;
    }

    if (!low_probed)
    {
        auto [nodes_low, pair_low] = solve_threshold(low);
        consider(nodes_low, pair_low.first);
    }
    if (!high_probed)
    {
        auto [nodes_high, pair_high] = solve_threshold(high);
        consider(nodes_high, pair_high.first);
    }

    if (best_nodes.empty())
        best_nodes.push_back(query_node);

    if (k > 0 && (int)best_nodes.size() < k)
    {
        std::unordered_set<int> S(best_nodes.begin(), best_nodes.end());
        grow_to_k_with_oracle(S, oracle_, undirected_adj,
                              queried, error_nodes, k, query_node);
        best_nodes.assign(S.begin(), S.end());
    }

    return best_nodes;
}
