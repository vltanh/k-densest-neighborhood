#include "simulation_oracle.hpp"

SimulationOracle::SimulationOracle(int max_in)
{
    this->max_in_edges = max_in;
}

void SimulationOracle::add_db_edge(const std::string &u, const std::string &v)
{
    db_adj_out[u].push_back(v);
    db_adj_in[v].push_back(u);
}

std::pair<std::vector<int>, std::vector<int>> SimulationOracle::query(int v_int)
{
    std::lock_guard<std::mutex> lock(cache_mtx);
    auto [it, inserted] = _cache.try_emplace(v_int);
    if (!inserted)
        return it->second;

    queries_made++;
    std::string v_str = mapper.get_str(v_int);

    std::vector<int> int_preds;
    auto in_it = db_adj_in.find(v_str);
    if (in_it != db_adj_in.end())
    {
        int fetched_in = 0;
        for (const std::string &u_str : in_it->second)
        {
            if (fetched_in >= max_in_edges)
                break;
            int_preds.push_back(mapper.get_or_create_id(u_str));
            fetched_in++;
        }
    }

    std::vector<int> int_succs;
    auto out_it = db_adj_out.find(v_str);
    if (out_it != db_adj_out.end())
    {
        for (const std::string &w_str : out_it->second)
            int_succs.push_back(mapper.get_or_create_id(w_str));
    }

    it->second = {int_preds, int_succs};
    return it->second;
}
