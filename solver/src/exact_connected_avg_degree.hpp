#pragma once

#include "common.hpp"
#include "oracle.hpp"
#include "gurobi_c++.h"
#include <vector>

class ExactConnectedAvgDegree
{
public:
    ExactConnectedAvgDegree(IGraphOracle *oracle, GRBEnv *env)
        : oracle_(oracle), env_(env) {}

    std::vector<int> solve(int query_node, int exploration_depth = 3);

private:
    IGraphOracle *oracle_;
    GRBEnv *env_;
};
