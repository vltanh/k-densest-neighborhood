#pragma once

#include "common.hpp"
#include "oracle.hpp"
#include "subgraph_quality.hpp"
#include <vector>
#include <set>
#include <map>

class AverageDegreeSolver
{
public:
    AverageDegreeSolver(IGraphOracle *oracle, int k) : oracle_(oracle), k_(k) {}

    // Main solver: returns only the selected node set.
    std::vector<int> solve(int query_node, int exploration_depth = 3);
    std::vector<int> solve_at_least_k_core(int query_node, int exploration_depth = 3);

private:
    IGraphOracle *oracle_;
    int k_;

    struct LocalGraph
    {
        std::vector<int> nodes;
        std::map<int, int> id_map;
        std::vector<std::pair<int, int>> edges;
    };

    LocalGraph explore_neighborhood(int start_node, int depth);
};
