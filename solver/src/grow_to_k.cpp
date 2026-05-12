#include "grow_to_k.hpp"

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
