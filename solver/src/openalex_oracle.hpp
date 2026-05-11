#pragma once
#include "oracle.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <cstddef>

class OpenAlexOracle : public IGraphOracle
{
    std::unordered_map<int, std::pair<std::vector<int>, std::vector<int>>> _cache;

    static std::size_t WriteCallback(void *contents, std::size_t size, std::size_t nmemb, void *userp);
    std::string fetch_url(const std::string &url, int max_retries = 3);
    std::string extract_id(const std::string &full_url);
    std::string url_encode(const std::string &value);

public:
    explicit OpenAlexOracle(int max_in = 0);
    const std::pair<std::vector<int>, std::vector<int>> &query(int v_int) override;
};
