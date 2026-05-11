#pragma once

#include "common.hpp"
#include "oracle.hpp"
#include <vector>
#include <unordered_set>
#include <unordered_map>

class ConnectedGreedyBaseline
{
public:
    ConnectedGreedyBaseline(IGraphOracle *oracle, int k) : oracle_(oracle), k_(k) {}

    // Main solver: returns only the selected node set.
    std::vector<int> solve(int query_node, int exploration_depth = 3);

private:
    IGraphOracle *oracle_;
    int k_;
};
