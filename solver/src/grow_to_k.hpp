#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

// Greedy at-least-k growth: enlarge S to size k by repeatedly adding the pool
// node with maximum |N(v) intersect S|, breaking ties by smallest raw id.
// `adj` is an undirected adjacency map over the pool; missing entries are
// treated as empty. Returns early if pool is exhausted before reaching k.
void grow_to_k(std::unordered_set<int> &S,
               const std::vector<int> &pool,
               const std::unordered_map<int, std::unordered_set<int>> &adj,
               int k);
