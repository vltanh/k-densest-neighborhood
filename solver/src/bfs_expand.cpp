#include "bfs_expand.hpp"
#include "common.hpp"
#include <iostream>

std::vector<int> expand_bfs_layer(
    const std::vector<int> &frontier,
    IGraphOracle *oracle,
    int seed_for_log,
    std::unordered_set<int> &visited,
    std::unordered_set<int> &queried,
    std::unordered_set<int> &error_nodes,
    std::unordered_map<int, std::unordered_set<int>> &adj,
    std::vector<std::pair<int, std::vector<int>>> *directed_out)
{
    std::vector<int> next_frontier;
    for (int u : frontier)
    {
        if (error_nodes.count(u) || queried.count(u))
            continue;
        std::pair<std::vector<int>, std::vector<int>> edges;
        try
        {
            edges = oracle->query(u);
        }
        catch (const std::exception &e)
        {
            if (error_nodes.insert(u).second)
            {
                std::cerr << "[" << get_timestamp() << "] BFS oracle query failed for "
                          << u << " (seed " << seed_for_log << "): " << e.what() << std::endl;
            }
            continue;
        }
        queried.insert(u);

        if (directed_out)
            directed_out->push_back({u, edges.second});

        auto record = [&](int v)
        {
            if (visited.insert(v).second)
            {
                adj[v];
                next_frontier.push_back(v);
            }
            adj[u].insert(v);
            adj[v].insert(u);
        };

        for (int v : edges.first)
            record(v);
        for (int v : edges.second)
            record(v);
    }
    return next_frontier;
}
