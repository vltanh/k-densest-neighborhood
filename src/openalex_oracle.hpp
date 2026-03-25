#pragma once
#include "oracle.hpp"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <iostream>

using json = nlohmann::json;

class OpenAlexOracle : public IGraphOracle {
private:
    std::unordered_map<int, std::pair<std::vector<int>, std::vector<int>>> _cache;

    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
        ((std::string*)userp)->append((char*)contents, size * nmemb);
        return size * nmemb;
    }

public:
    OpenAlexOracle() {}

    const std::pair<std::vector<int>, std::vector<int>>& query(int v_int) override {
        auto [it, inserted] = _cache.try_emplace(v_int);
        if (!inserted) return it->second;

        queries_made++;
        std::string v_str = mapper.get_str(v_int);
        std::vector<int> int_preds, int_succs;

        // OpenAlex API: Get what this paper cites (out-edges / succs)
        std::string url = "https://api.openalex.org/works/" + v_str;
        std::string readBuffer;

        CURL* curl = curl_easy_init();
        if (curl) {
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_USERAGENT, "KDensestSolver/1.0 (mailto:your_email@illinois.edu)");
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
            
            CURLcode res = curl_easy_perform(curl);
            curl_easy_cleanup(curl);

            if (res == CURLE_OK) {
                try {
                    auto j = json::parse(readBuffer);
                    if (j.contains("referenced_works")) {
                        for (const auto& ref : j["referenced_works"]) {
                            std::string ref_id = ref.get<std::string>();
                            size_t pos = ref_id.find_last_of('/');
                            if (pos != std::string::npos) ref_id = ref_id.substr(pos + 1);
                            int_succs.push_back(mapper.get_or_create_id(ref_id));
                        }
                    }
                } catch (...) {
                    std::cerr << "JSON Parse error for " << v_str << "\n";
                }
            }
        }

        // Note: To get in-edges (preds), you would need a 2nd API call here: 
        // https://api.openalex.org/works?filter=cites:W... 
        // For brevity, only succs are fetched in this template.

        it->second = {int_preds, int_succs};
        return it->second;
    }
};