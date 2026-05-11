#include "average_degree_solver.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/push_relabel_max_flow.hpp>
#include <queue>
#include <cmath>
#include <iostream>
#include <set>
#include <algorithm>
#include <unordered_set>
#include <limits>

using namespace boost;

typedef adjacency_list_traits<vecS, vecS, directedS> Traits;
typedef adjacency_list<vecS, vecS, directedS,
                       property<vertex_name_t, int>,
                       property<edge_capacity_t, long long,
                                property<edge_residual_capacity_t, long long,
                                         property<edge_reverse_t, Traits::edge_descriptor>>>>
    Graph;

AverageDegreeSolver::LocalGraph AverageDegreeSolver::explore_neighborhood(int start, int depth)
{
    LocalGraph lg;
    std::queue<std::pair<int, int>> q;
    std::set<int> visited;
    std::set<int> error_nodes;

    q.push({start, 0});
    visited.insert(start);

    while (!q.empty())
    {
        auto [u, d] = q.front();
        q.pop();

        std::pair<std::vector<int>, std::vector<int>> edges;
        try
        {
            edges = oracle_->query(u);
        }
        catch (const std::exception &e)
        {
            if (error_nodes.insert(u).second)
            {
                std::cerr << "[" << get_timestamp() << "] Blacklisting node "
                          << oracle_->mapper.get_str(u) << " due to API error: " << e.what() << std::endl;
            }
            continue;
        }

        lg.id_map[u] = lg.nodes.size();
        lg.nodes.push_back(u);

        if (depth < 0 || d < depth)
        {
            for (int v : edges.second)
            {
                if (visited.insert(v).second)
                    q.push({v, d + 1});
            }
            for (int v : edges.first)
            {
                if (visited.insert(v).second)
                    q.push({v, d + 1});
            }
        }
    }

    for (size_t i = 0; i < lg.nodes.size(); ++i)
    {
        int u = lg.nodes[i];
        std::pair<std::vector<int>, std::vector<int>> edges;
        try
        {
            edges = oracle_->query(u);
        }
        catch (...)
        {
            continue;
        }
        for (int v : edges.second)
        {
            if (lg.id_map.count(v))
            {
                lg.edges.push_back({(int)i, lg.id_map[v]});
            }
        }
    }
    return lg;
}

std::vector<int> AverageDegreeSolver::solve(int query_node, int depth)
{
    LocalGraph lg = explore_neighborhood(query_node, depth);

    if (lg.id_map.find(query_node) == lg.id_map.end())
    {
        std::cerr << "[" << get_timestamp() << "] Error: Query node could not be fetched.\n";
        return {};
    }

    int n = (int)lg.nodes.size();
    int query_idx = lg.id_map[query_node];

    std::unordered_set<std::pair<int, int>, pair_hash> edge_set(lg.edges.begin(), lg.edges.end());
    std::vector<std::pair<int, int>> directed_edges(edge_set.begin(), edge_set.end());

    using Traits = adjacency_list_traits<vecS, vecS, directedS>;
    using FlowGraph = adjacency_list<vecS, vecS, directedS,
                                     property<vertex_name_t, int>,
                                     property<edge_capacity_t, long long,
                                              property<edge_residual_capacity_t, long long,
                                                       property<edge_reverse_t, Traits::edge_descriptor>>>>;

    const long long SCALE = 1000000LL;
    const long long INF_CAP = std::numeric_limits<long long>::max() / 4;

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

        int source = n + (int)directed_edges.size();
        int sink = source + 1;

        for (int i = 0; i < n; ++i)
        {
            long long node_cost = (long long)llround(lambda_val * SCALE);
            if (node_cost > 0)
                add_flow_edge(i, sink, node_cost);
        }

        for (size_t idx = 0; idx < directed_edges.size(); ++idx)
        {
            const auto &[u_idx, v_idx] = directed_edges[idx];
            int item = n + (int)idx;
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
        {
            if (vis[i])
                selected_nodes.push_back(lg.nodes[i]);
        }
        for (size_t idx = 0; idx < directed_edges.size(); ++idx)
        {
            if (vis[n + (int)idx])
                selected_edges++;
        }

        double value = (double)selected_edges - lambda_val * (double)selected_nodes.size();
        return std::make_pair(selected_nodes, std::make_pair(selected_edges, value));
    };

    double low = 0.0;
    double high = (double)directed_edges.size();
    double tol = 1e-9;
    std::vector<int> best_nodes;
    double best_density = -1.0;

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

    for (int iter = 0; iter < 100; ++iter)
    {
        double mid = (low + high) / 2.0;
        auto [nodes_mid, pair_mid] = solve_threshold(mid);
        int edges_mid = pair_mid.first;
        double value_mid = pair_mid.second;
        consider(nodes_mid, edges_mid);

        if (value_mid > 0.0)
        {
            low = mid;
        }
        else
        {
            high = mid;
        }

        if (high - low <= tol)
            break;
    }

    auto [nodes_low, pair_low] = solve_threshold(low);
    consider(nodes_low, pair_low.first);
    auto [nodes_high, pair_high] = solve_threshold(high);
    consider(nodes_high, pair_high.first);

    if (best_nodes.empty())
        best_nodes.push_back(query_node);

    return best_nodes;
}

std::vector<int> AverageDegreeSolver::solve_at_least_k_core(int query_node, int depth)
{
    LocalGraph lg = explore_neighborhood(query_node, depth);

    auto query_it = lg.id_map.find(query_node);
    if (query_it == lg.id_map.end())
    {
        std::cerr << "[" << get_timestamp() << "] Error: Query node could not be fetched.\n";
        return {};
    }

    int n = (int)lg.nodes.size();
    if (n == 0)
        return {};

    int query_idx = query_it->second;
    std::vector<std::map<int, double>> adj(n);

    for (const auto &e : lg.edges)
    {
        if (e.first == e.second)
            continue;
        int u = std::min(e.first, e.second);
        int v = std::max(e.first, e.second);
        adj[u][v] += 1.0;
        adj[v][u] += 1.0;
    }

    std::vector<double> degree(n, 0.0);
    double internal_weight = 0.0;
    for (int u = 0; u < n; ++u)
    {
        for (const auto &[v, w] : adj[u])
            degree[u] += w;
    }
    for (int u = 0; u < n; ++u)
    {
        for (const auto &[v, w] : adj[u])
            if (u < v)
                internal_weight += w;
    }

    std::vector<char> active(n, true);
    int active_count = n;
    bool query_active = true;
    double best_density = -1.0;
    std::vector<int> best_nodes;

    auto snapshot_active = [&]()
    {
        std::vector<int> result;
        result.reserve(active_count);
        for (int i = 0; i < n; ++i)
        {
            if (active[i])
                result.push_back(lg.nodes[i]);
        }
        return result;
    };

    while (active_count > 0 && query_active)
    {
        if (active_count >= k_)
        {
            double density = internal_weight / active_count;
            if (density > best_density)
            {
                best_density = density;
                best_nodes = snapshot_active();
            }
        }

        int remove_idx = -1;
        double min_degree = std::numeric_limits<double>::infinity();
        for (int i = 0; i < n; ++i)
        {
            if (active[i] && degree[i] < min_degree)
            {
                min_degree = degree[i];
                remove_idx = i;
            }
        }

        if (remove_idx < 0)
            break;

        active[remove_idx] = false;
        active_count--;
        if (remove_idx == query_idx)
            query_active = false;

        for (const auto &[v, w] : adj[remove_idx])
        {
            if (active[v])
            {
                degree[v] -= w;
                internal_weight -= w;
            }
        }
        degree[remove_idx] = 0.0;
    }

    if (!best_nodes.empty())
        return best_nodes;

    return {query_node};
}
