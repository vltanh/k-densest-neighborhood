#pragma once
#include "oracle.hpp"
#include <unordered_map>
#include <vector>
#include <string>

class SimulationOracle : public IGraphOracle {
private:
    std::unordered_map<std::string, std::vector<std::string>> db_adj_out;
    std::unordered_map<std::string, std::vector<std::string>> db_adj_in;
    std::unordered_map<int, std::pair<std::vector<int>, std::vector<int>>> _cache;

public:
    SimulationOracle() {}

    void add_db_edge(const std::string& u, const std::string& v) {
        db_adj_out[u].push_back(v);
        db_adj_in[v].push_back(u);
    }

    // Your exact query() logic goes here...
    const std::pair<std::vector<int>, std::vector<int>>& query(int v_int) override {
        auto [it, inserted] = _cache.try_emplace(v_int);
        if (!inserted) return it->second;

        queries_made++;
        std::string v_str = mapper.get_str(v_int);
        
        std::vector<int> int_preds;
        auto in_it = db_adj_in.find(v_str);
        if (in_it != db_adj_in.end()) {
            for (const std::string& u_str : in_it->second) {
                int_preds.push_back(mapper.get_or_create_id(u_str));
            }
        }
        
        std::vector<int> int_succs;
        auto out_it = db_adj_out.find(v_str);
        if (out_it != db_adj_out.end()) {
            for (const std::string& w_str : out_it->second) {
                int_succs.push_back(mapper.get_or_create_id(w_str));
            }
        }
        
        it->second = {int_preds, int_succs};
        return it->second;
    }
};