#pragma once
#include "oracle.hpp"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <stdexcept>
#include <algorithm>

using json = nlohmann::json;

class OpenAlexOracle : public IGraphOracle
{
private:
    std::unordered_map<int, std::pair<std::vector<int>, std::vector<int>>> _cache;

    static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp)
    {
        ((std::string *)userp)->append((char *)contents, size * nmemb);
        return size * nmemb;
    }

    std::string fetch_url(const std::string &url, int max_retries = 3)
    {
        std::string readBuffer;
        CURL *curl = curl_easy_init();
        if (!curl)
            return readBuffer;

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "KDensestSolver/1.0 (mailto:your_email@illinois.edu)");
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);

        for (int attempt = 1; attempt <= max_retries; ++attempt)
        {
            readBuffer.clear();

            CURLcode res = curl_easy_perform(curl);
            long http_code = 0;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

            if (res == CURLE_OK && http_code == 200)
            {
                break;
            }
            else
            {
                std::cerr << "[" << get_timestamp() << "] HTTP Request failed (Attempt "
                          << attempt << "/" << max_retries << ") for " << url << "\n"
                          << "    -> cURL Error: " << curl_easy_strerror(res)
                          << " | HTTP Code: " << http_code << "\n";

                if (!readBuffer.empty())
                {
                    std::string snippet = readBuffer.substr(0, 250);
                    std::replace(snippet.begin(), snippet.end(), '\n', ' ');
                    std::cerr << "    -> Server Msg: " << snippet
                              << (readBuffer.size() > 250 ? "..." : "") << "\n";
                }

                if (attempt < max_retries)
                {
                    int sleep_ms = 1000 * (1 << (attempt - 1));
                    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
                }
                else
                {
                    // CRITICAL FIX: Throw after exhausting retries!
                    curl_easy_cleanup(curl);
                    throw std::runtime_error("HTTP fetch failed after max retries.");
                }
            }
        }

        curl_easy_cleanup(curl);
        return readBuffer;
    }

    std::string extract_id(const std::string &full_url)
    {
        size_t pos = full_url.find_last_of('/');
        if (pos != std::string::npos)
            return full_url.substr(pos + 1);
        return full_url;
    }

public:
    OpenAlexOracle() {}

    const std::pair<std::vector<int>, std::vector<int>> &query(int v_int) override
    {
        auto [it, inserted] = _cache.try_emplace(v_int);
        if (!inserted)
            return it->second;

        queries_made++;
        std::string v_str = mapper.get_str(v_int);
        std::vector<int> int_preds, int_succs;

        std::string out_url = "https://api.openalex.org/works/" + v_str;
        std::string out_response = fetch_url(out_url);

        try
        {
            if (!out_response.empty())
            {
                auto j_out = json::parse(out_response);
                if (j_out.contains("referenced_works"))
                {
                    for (const auto &ref : j_out["referenced_works"])
                    {
                        std::string ref_id = extract_id(ref.get<std::string>());
                        int_succs.push_back(mapper.get_or_create_id(ref_id));
                    }
                }
            }
        }
        catch (const std::exception &e)
        {
            // CRITICAL FIX: Rethrow the parse error so the solver can blacklist the node
            throw std::runtime_error("JSON Parse Error (Out-Edges): " + std::string(e.what()));
        }

        std::string in_url = "https://api.openalex.org/works?filter=cites:" + v_str + "&per-page=200";
        std::string in_response = fetch_url(in_url);

        try
        {
            if (!in_response.empty())
            {
                auto j_in = json::parse(in_response);
                if (j_in.contains("results"))
                {
                    for (const auto &result : j_in["results"])
                    {
                        if (result.contains("id"))
                        {
                            std::string citing_id = extract_id(result["id"].get<std::string>());
                            int_preds.push_back(mapper.get_or_create_id(citing_id));
                        }
                    }
                }
            }
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("JSON Parse Error (In-Edges): " + std::string(e.what()));
        }

        it->second = {int_preds, int_succs};
        return it->second;
    }
};