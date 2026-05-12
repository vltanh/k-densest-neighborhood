#pragma once

#include "common.hpp"
#include "oracle.hpp"
#include <vector>

class BFSSolver
{
public:
    explicit BFSSolver(IGraphOracle *oracle) : oracle_(oracle) {}

    // depth: BFS expansion depth from query (depth < 0 means unlimited).
    // k: if > 0, the visited pool is reduced to exactly k nodes (or fewer if the
    //    component is too small) via the shared grow_to_k greedy heuristic
    //    seeded with {query}. If k <= 0, returns the full visited pool.
    std::vector<int> solve(int query_node, int depth = 1, int k = -1);

private:
    IGraphOracle *oracle_;
};
