#pragma once
#include "common.hpp"
#include <utility>
#include <vector>
#include <atomic>

class IGraphOracle
{
public:
    IDMapper mapper;

    std::atomic<int> queries_made{0};
    std::atomic<double> cumulative_network_time{0.0};

    int max_in_edges = 1500;

    virtual ~IGraphOracle() = default;

    // Forces derived classes to implement the exact DAG query signature
    virtual std::pair<std::vector<int>, std::vector<int>> query(int v_int) = 0;
};