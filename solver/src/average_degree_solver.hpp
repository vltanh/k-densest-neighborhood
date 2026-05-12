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

    // k: target subgraph size. If the Goldberg+bisection optimum has |S*| < k,
    //    grow_to_k greedily enlarges S* with the max-edges-in rule until size k
    //    (or pool exhaustion). k <= 0 disables growth and returns the
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

    LocalGraph explore_neighborhood(int start_node, int depth);
};
