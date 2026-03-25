#pragma once
#include "common.hpp"
#include <utility>

class IGraphOracle {
public:
    IDMapper mapper;
    int queries_made = 0;
    
    virtual ~IGraphOracle() = default;
    
    // Forces derived classes to implement your exact DAG query signature
    virtual const std::pair<std::vector<int>, std::vector<int>>& query(int v_int) = 0;
};