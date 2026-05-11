#include "bfs_solver.hpp"
#include <unordered_set>
#include <queue>
#include <vector>
#include <iostream>

using namespace std;

vector<int> BFSSolver::solve(int query_node, int depth)
{
    unordered_set<int> nodes;
    nodes.insert(query_node);
    queue<pair<int, int>> q;
    unordered_set<int> visited;
    visited.insert(query_node);
    q.push({query_node, 0});

    while (!q.empty())
    {
        auto [u, d] = q.front();
        q.pop();

        if (depth >= 0 && d >= depth)
            continue;

        try
        {
            const auto &[preds, succs] = oracle_->query(u);
            for (int v : preds)
            {
                if (visited.insert(v).second)
                    q.push({v, d + 1});
                nodes.insert(v);
            }
            for (int v : succs)
            {
                if (visited.insert(v).second)
                    q.push({v, d + 1});
                nodes.insert(v);
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "[" << get_timestamp() << "] BFS baseline failed on query "
                      << query_node << ": " << e.what() << std::endl;
            return {};
        }
    }

    return vector<int>(nodes.begin(), nodes.end());
}
