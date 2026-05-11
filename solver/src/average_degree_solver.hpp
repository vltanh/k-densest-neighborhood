#pragma once

#include "common.hpp"
#include "oracle.hpp"
#include "subgraph_quality.hpp"
#include <vector>
#include <map>

class AverageDegreeSolver
{
public:
    explicit AverageDegreeSolver(IGraphOracle *oracle) : oracle_(oracle) {}

    std::vector<int> solve(int query_node, int exploration_depth = 3);

private:
    IGraphOracle *oracle_;

    struct LocalGraph
    {
        std::vector<int> nodes;
        std::map<int, int> id_map;
        std::vector<std::pair<int, int>> edges;
    };

    LocalGraph explore_neighborhood(int start_node, int depth);
};
