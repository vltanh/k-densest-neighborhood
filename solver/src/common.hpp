#pragma once
#include <string>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <vector>
#include <unordered_map>
#include <limits>

inline std::string get_timestamp()
{
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    time_t now_c = std::chrono::system_clock::to_time_t(now);
    tm parts;
#if defined(_WIN32) || defined(_WIN64)
    localtime_s(&parts, &now_c);
#else
    localtime_r(&now_c, &parts);
#endif
    char buf[24];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &parts);
    std::ostringstream oss;
    oss << buf << "," << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

struct pair_hash
{
    template <class T1, class T2>
    size_t operator()(const std::pair<T1, T2> &p) const
    {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

struct DinkelbachIter
{
    int iter;
    double lambda;
    double iter_time_s;
    int bb_nodes;
    int lp_solves;
    double bb_incumbent_obj;
    double bb_best_bound;
    double optimality_gap;
};

struct SolverStats
{
    int total_bb_nodes = 0;
    int total_lp_solves = 0;
    int total_columns_added = 0;
    int total_cuts_added = 0;
    double t_sync = 0.0, t_lp_solve = 0.0, t_pricing = 0.0, t_separation = 0.0, t_total = 0.0;
    double final_bb_incumbent_obj = 0.0;
    double final_bb_best_bound = 0.0;
    double final_optimality_gap = 0.0;
    int final_open_nodes = 0;
    std::string final_gap_status = "not_run";
    std::vector<DinkelbachIter> lambda_trajectory;
};

class IDMapper
{
public:
    std::unordered_map<std::string, int> str_to_id;
    std::vector<std::string> id_to_str;

    int get_or_create_id(const std::string &s)
    {
        auto it = str_to_id.find(s);
        if (it != str_to_id.end())
            return it->second;
        int id = id_to_str.size();
        str_to_id[s] = id;
        id_to_str.push_back(s);
        return id;
    }
    std::string get_str(int id) const { return id_to_str[id]; }
    int size() const { return id_to_str.size(); }
};

struct BBNode
{
    std::vector<int> v1;
    std::vector<int> v0;
    double bound = std::numeric_limits<double>::infinity();
};
