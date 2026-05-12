#include "bfs_solver.hpp"
#include "grow_to_k.hpp"
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <vector>
#include <iostream>

using namespace std;

vector<int> BFSSolver::solve(int query_node, int depth, int k)
{
    unordered_set<int> visited;
    unordered_set<int> error_nodes;
    unordered_map<int, unordered_set<int>> adj;
    visited.insert(query_node);
    adj[query_node];

    queue<pair<int, int>> q;
    q.push({query_node, 0});

    // Expand while either the depth cap is unmet or the pool is still below k.
    // Past the depth cap, expansion continues from already-enqueued frontier
    // nodes only (no further descendants are pushed), so depth keeps its
    // "minimum layers" meaning while k enforces a minimum pool size.
    while (!q.empty())
    {
        auto [u, d] = q.front();
        q.pop();

        bool depth_cap_reached = (depth >= 0 && d >= depth);
        bool pool_target_met = (k <= 0) || ((int)visited.size() >= k);
        if (depth_cap_reached && pool_target_met)
            continue;

        std::pair<std::vector<int>, std::vector<int>> edges;
        try
        {
            edges = oracle_->query(u);
        }
        catch (const std::exception &e)
        {
            // Skip this node, keep expanding the rest of the frontier. A single
            // 404 or rate-limit on one neighbour shouldn't void the whole BFS.
            if (error_nodes.insert(u).second)
            {
                std::cerr << "[" << get_timestamp() << "] BFS oracle query failed for "
                          << u << " (seed " << query_node << "): " << e.what() << std::endl;
            }
            continue;
        }

        auto record_neighbor = [&](int v)
        {
            if (visited.insert(v).second)
            {
                adj[v];
                q.push({v, d + 1});
            }
            adj[u].insert(v);
            adj[v].insert(u);
        };

        for (int v : edges.first)
            record_neighbor(v);
        for (int v : edges.second)
            record_neighbor(v);
    }

    vector<int> pool(visited.begin(), visited.end());

    if (k <= 0)
        return pool;

    unordered_set<int> S;
    S.insert(query_node);
    grow_to_k(S, pool, adj, k);
    return vector<int>(S.begin(), S.end());
}
