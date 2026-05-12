#include "grow_to_k.hpp"
#include "bfs_expand.hpp"

void grow_to_k(std::unordered_set<int> &S,
               const std::vector<int> &pool,
               const std::unordered_map<int, std::unordered_set<int>> &adj,
               int k)
{
    if ((int)S.size() >= k)
        return;

    std::unordered_set<int> candidates;
    candidates.reserve(pool.size());
    for (int v : pool)
        if (!S.count(v))
            candidates.insert(v);

    while ((int)S.size() < k && !candidates.empty())
    {
        int best = -1;
        int best_gain = -1;
        for (int v : candidates)
        {
            int gain = 0;
            auto it = adj.find(v);
            if (it != adj.end())
            {
                for (int u : it->second)
                    if (S.count(u))
                        ++gain;
            }
            if (gain > best_gain || (gain == best_gain && (best < 0 || v < best)))
            {
                best_gain = gain;
                best = v;
            }
        }
        if (best < 0)
            break;
        S.insert(best);
        candidates.erase(best);
    }
}

void grow_to_k_with_oracle(std::unordered_set<int> &S,
                           IGraphOracle *oracle,
                           std::unordered_map<int, std::unordered_set<int>> &adj,
                           std::unordered_set<int> &queried,
                           std::unordered_set<int> &error_nodes,
                           int k,
                           int seed_for_log)
{
    if ((int)S.size() >= k)
        return;

    std::unordered_set<int> candidates;
    for (int u : S)
    {
        auto it = adj.find(u);
        if (it == adj.end())
            continue;
        for (int v : it->second)
            if (!S.count(v))
                candidates.insert(v);
    }

    auto query_one = [&](int u)
    {
        if (queried.count(u) || error_nodes.count(u))
            return;
        std::vector<int> frontier{u};
        std::unordered_set<int> local_visited{u};
        auto discovered = expand_bfs_layer(frontier, oracle, seed_for_log,
                                           local_visited, queried, error_nodes, adj);
        for (int v : discovered)
            if (!S.count(v))
                candidates.insert(v);
    };

    while ((int)S.size() < k)
    {
        if (candidates.empty())
        {
            int probe = -1;
            for (int u : S)
                if (!queried.count(u) && !error_nodes.count(u))
                {
                    probe = u;
                    break;
                }
            if (probe < 0)
                return;
            query_one(probe);
            continue;
        }
        int best = -1;
        int best_gain = -1;
        for (int v : candidates)
        {
            int gain = 0;
            auto it = adj.find(v);
            if (it != adj.end())
                for (int u : it->second)
                    if (S.count(u))
                        ++gain;
            if (gain > best_gain || (gain == best_gain && (best < 0 || v < best)))
            {
                best_gain = gain;
                best = v;
            }
        }
        if (best < 0)
            return;
        S.insert(best);
        candidates.erase(best);
        query_one(best);
    }
}
