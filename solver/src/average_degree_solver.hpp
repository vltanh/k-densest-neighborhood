#pragma once

#include "common.hpp"
#include "oracle.hpp"
#include "subgraph_quality.hpp"
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>

class AverageDegreeSolver
{
public:
    explicit AverageDegreeSolver(IGraphOracle *oracle) : oracle_(oracle) {}

    // k: target subgraph size. If the Goldberg+bisection optimum has |S*| < k,
    //    grow_to_k_with_oracle picks one extra node per step by maximum
    //    edges-into-S (querying the oracle as needed) until |S| == k or the
    //    reachable graph is exhausted. k <= 0 disables growth and returns the
    //    unconstrained optimum.
    std::vector<int> solve(int query_node, int exploration_depth = 3, int k = -1);

private:
    IGraphOracle *oracle_;

    struct LocalGraph
    {
        std::vector<int> nodes;
        std::map<int, int> id_map;
        std::vector<std::pair<int, int>> edges;
    };

    LocalGraph explore_neighborhood(
        int start_node, int depth,
        std::unordered_set<int> &queried,
        std::unordered_set<int> &error_nodes,
        std::unordered_map<int, std::unordered_set<int>> &undirected_adj);
};
