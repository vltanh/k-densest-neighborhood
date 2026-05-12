#include "bfs_solver.hpp"
#include "bfs_expand.hpp"
#include "grow_to_k.hpp"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <limits>

using namespace std;

vector<int> BFSSolver::solve(int query_node, int depth, int k)
{
    unordered_set<int> visited;
    unordered_set<int> queried;
    unordered_set<int> error_nodes;
    unordered_map<int, unordered_set<int>> adj;
    visited.insert(query_node);
    adj[query_node];

    unordered_set<int> S{query_node};
    vector<int> frontier{query_node};

    // Closed ball of radius `depth`: query every layer 0..depth, add layer
    // 0..depth-1's discoveries to S (those are at distance ≤ depth). The
    // final layer's discoveries (distance depth+1) stay in adj as candidates
    // but not in S.
    int strict_depth = (depth < 0) ? std::numeric_limits<int>::max() : depth;
    for (int layer = 0; layer <= strict_depth && !frontier.empty(); ++layer)
    {
        auto next_frontier = expand_bfs_layer(frontier, oracle_, query_node,
                                              visited, queried, error_nodes, adj);
        if (layer < strict_depth)
            for (int u : next_frontier)
                S.insert(u);
        frontier = next_frontier;
    }

    if (k > 0 && (int)S.size() < k)
        grow_to_k_with_oracle(S, oracle_, adj, queried, error_nodes, k, query_node);

    return vector<int>(S.begin(), S.end());
}
