#pragma once

#include "common.hpp"
#include "oracle.hpp"
#include <vector>

class BFSSolver
{
public:
    explicit BFSSolver(IGraphOracle *oracle) : oracle_(oracle) {}

    // depth: strict BFS expansion depth from query (depth < 0 means unlimited).
    // k: result targeting. Returns the closed ball of radius depth when its
    //    size is >= k (so the depth-d layer is never truncated). When the ball
    //    is smaller than k, grow_to_k_with_oracle picks one extra node per
    //    step by maximum edges-into-S (querying the oracle as needed) until
    //    |S| == k or the reachable graph is exhausted. k <= 0 disables
    //    targeting.
    std::vector<int> solve(int query_node, int depth = 1, int k = -1);

private:
    IGraphOracle *oracle_;
};
