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
    unordered_map<int, unordered_set<int>> adj;
    visited.insert(query_node);
    adj[query_node];

    queue<pair<int, int>> q;
    q.push({query_node, 0});

    while (!q.empty())
    {
        auto [u, d] = q.front();
        q.pop();

        if (depth >= 0 && d >= depth)
            continue;

        std::pair<std::vector<int>, std::vector<int>> edges;
        try
        {
            edges = oracle_->query(u);
        }
        catch (const std::exception &e)
        {
            std::cerr << "[" << get_timestamp() << "] BFS baseline failed on query "
                      << query_node << ": " << e.what() << std::endl;
            return {};
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
