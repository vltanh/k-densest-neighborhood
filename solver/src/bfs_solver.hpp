#pragma once

#include "common.hpp"
#include "oracle.hpp"
#include <vector>

class BFSSolver
{
public:
    explicit BFSSolver(IGraphOracle *oracle) : oracle_(oracle) {}

    std::vector<int> solve(int query_node, int depth = 1);

private:
    IGraphOracle *oracle_;
};
