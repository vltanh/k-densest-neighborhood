#pragma once
#include "oracle.hpp"
#include <unordered_map>
#include <vector>
#include <string>

class SimulationOracle : public IGraphOracle
{
    std::unordered_map<std::string, std::vector<std::string>> db_adj_out;
    std::unordered_map<std::string, std::vector<std::string>> db_adj_in;
    std::unordered_map<int, std::pair<std::vector<int>, std::vector<int>>> _cache;

public:
    explicit SimulationOracle(int max_in = 1500);
    void add_db_edge(const std::string &u, const std::string &v);
    const std::pair<std::vector<int>, std::vector<int>> &query(int v_int) override;
};
