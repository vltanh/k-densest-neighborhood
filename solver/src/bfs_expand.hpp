#pragma once

#include "oracle.hpp"
#include <unordered_map>
#include <unordered_set>
#include <vector>

// One BFS layer. Queries every node in `frontier` (skipping those already
// marked in `error_nodes`), records every neighbour into `visited` and
// `undirected_adj`, and returns the set of newly visited nodes (the next
// frontier). When `directed_out` is non-null, appends
// (queried_node, list_of_out_neighbours) per successful query so the caller
// can also build a directed graph from the same oracle responses.
//
// `seed_for_log` is the original query node id, used only for diagnostic
// messages on oracle errors.
std::vector<int> expand_bfs_layer(
    const std::vector<int> &frontier,
    IGraphOracle *oracle,
    int seed_for_log,
    std::unordered_set<int> &visited,
    std::unordered_set<int> &queried,
    std::unordered_set<int> &error_nodes,
    std::unordered_map<int, std::unordered_set<int>> &undirected_adj,
    std::vector<std::pair<int, std::vector<int>>> *directed_out = nullptr);
