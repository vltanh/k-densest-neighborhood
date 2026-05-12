#pragma once

#include "oracle.hpp"
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

// Oracle-driven one-by-one growth: at each step, picks the candidate (any
// node adjacent to S but not in S) with maximum |N(v) intersect S|, ties to
// smallest raw id, inserts it into S, and queries it so its own neighbours
// become future candidates. When no candidate is reachable from already-
// known adjacency, probes an unqueried S member to discover more. Updates
// `adj`, `queried`, and `error_nodes` in place. Returns when |S| reaches k
// or the reachable graph is exhausted.
void grow_to_k_with_oracle(std::unordered_set<int> &S,
                           IGraphOracle *oracle,
                           std::unordered_map<int, std::unordered_set<int>> &adj,
                           std::unordered_set<int> &queried,
                           std::unordered_set<int> &error_nodes,
                           int k,
                           int seed_for_log);
