#include "openalex_oracle.hpp"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <thread>
#include <stdexcept>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <chrono>

using json = nlohmann::json;

OpenAlexOracle::OpenAlexOracle(int max_in)
{
    this->max_in_edges = max_in;
}

std::size_t OpenAlexOracle::WriteCallback(void *contents, std::size_t size, std::size_t nmemb, void *userp)
{
    ((std::string *)userp)->append((char *)contents, size * nmemb);
    return size * nmemb;
}

std::string OpenAlexOracle::fetch_url(const std::string &url, int max_retries)
{
    std::string readBuffer;
    CURL *curl = curl_easy_init();
    if (!curl)
        return readBuffer;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "KDensestSolver/1.0 (mailto:vltanh@illinois.edu)");
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
            break;

        if (res == CURLE_OK && http_code == 404)
        {
            curl_easy_cleanup(curl);
            throw std::runtime_error("404 Not Found");
        }

        std::cerr << "[" << get_timestamp() << "] HTTP Request failed (Attempt "
                  << attempt << "/" << max_retries << ") for " << url << "\n"
                  << "    -> cURL Error: " << curl_easy_strerror(res)
                  << " | HTTP Code: " << http_code << "\n";

        if (attempt < max_retries)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000 * (1 << (attempt - 1))));
        }
        else
        {
            curl_easy_cleanup(curl);
            throw std::runtime_error("HTTP fetch failed after max retries.");
        }
    }

    curl_easy_cleanup(curl);
    return readBuffer;
}

std::string OpenAlexOracle::extract_id(const std::string &full_url)
{
    size_t pos = full_url.find_last_of('/');
    if (pos != std::string::npos)
        return full_url.substr(pos + 1);
    return full_url;
}

std::string OpenAlexOracle::url_encode(const std::string &value)
{
    std::ostringstream escaped;
    escaped.fill('0');
    escaped << std::hex;
    for (char c : value)
    {
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~')
            escaped << c;
        else
        {
            escaped << std::uppercase;
            escaped << '%' << std::setw(2) << int((unsigned char)c);
            escaped << std::nouppercase;
        }
    }
    return escaped.str();
}

const std::pair<std::vector<int>, std::vector<int>> &OpenAlexOracle::query(int v_int)
{
    auto [it, inserted] = _cache.try_emplace(v_int);
    if (!inserted)
        return it->second;

    queries_made++;
    std::string v_str = mapper.get_str(v_int);
    std::vector<int> int_preds, int_succs;

    auto io_start = std::chrono::high_resolution_clock::now();

    // Outgoing edges
    std::string out_response = fetch_url("https://api.openalex.org/works/" + v_str);
    try
    {
        if (!out_response.empty())
        {
            auto j_out = json::parse(out_response);
            if (j_out.contains("referenced_works"))
            {
                for (const auto &ref : j_out["referenced_works"])
                    int_succs.push_back(mapper.get_or_create_id(extract_id(ref.get<std::string>())));
            }
        }
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error("JSON Parse Error (Out-Edges): " + std::string(e.what()));
    }

    // Incoming edges — cursor-paginated
    int fetched_in = 0;
    std::string cursor = "*";
    while (fetched_in < max_in_edges && !cursor.empty())
    {
        std::string in_url = "https://api.openalex.org/works?filter=cites:" + v_str +
                             "&select=id&per-page=200&cursor=" + url_encode(cursor);
        std::string in_response = fetch_url(in_url);
        if (in_response.empty())
            break;

        try
        {
            auto j_in = json::parse(in_response);
            if (!j_in.contains("results") || j_in["results"].empty())
                break;

            for (const auto &result : j_in["results"])
            {
                if (fetched_in >= max_in_edges)
                    break;
                if (result.contains("id"))
                {
                    int_preds.push_back(mapper.get_or_create_id(extract_id(result["id"].get<std::string>())));
                    fetched_in++;
                }
            }

            if (fetched_in < max_in_edges && j_in.contains("meta") &&
                j_in["meta"].contains("next_cursor") && !j_in["meta"]["next_cursor"].is_null())
                cursor = j_in["meta"]["next_cursor"].get<std::string>();
            else
                cursor = "";
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("JSON Parse Error (In-Edges Paginated): " + std::string(e.what()));
        }
    }

    cumulative_network_time += std::chrono::duration<double>(
                                   std::chrono::high_resolution_clock::now() - io_start)
                                   .count();

    it->second = {int_preds, int_succs};
    return it->second;
}
