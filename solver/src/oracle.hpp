#pragma once
#include "common.hpp"
#include <utility>
#include <vector>

class IGraphOracle
{
public:
    IDMapper mapper;
    int queries_made = 0;

    // Base properties for all Oracles to ensure the Solver can query them safely
    double cumulative_network_time = 0.0;
    int max_in_edges = 0;

    virtual ~IGraphOracle() = default;

    // Forces derived classes to implement the exact DAG query signature
    virtual const std::pair<std::vector<int>, std::vector<int>> &query(int v_int) = 0;
};