#include "connected_greedy_baseline.hpp"
#include <queue>
#include <iostream>
#include <algorithm>
#include <set>

using namespace std;

vector<int> ConnectedGreedyBaseline::solve(int query_node, int depth)
{
    // ========================================================================
    // 1. Explore Neighborhood (Standard BFS)
    // ========================================================================
    unordered_set<int> visited;
    queue<pair<int, int>> q_bfs;
    unordered_set<int> error_nodes;

    q_bfs.push({query_node, 0});
    visited.insert(query_node);

    while (!q_bfs.empty())
    {
        auto [u, d] = q_bfs.front();
        q_bfs.pop();

        if (depth >= 0 && d >= depth)
            continue;

        pair<vector<int>, vector<int>> edges;
        try
        {
            edges = oracle_->query(u);
        }
        catch (const std::exception &e)
        {
            if (error_nodes.insert(u).second)
            {
                cerr << "[" << get_timestamp() << "] Blacklisting node "
                     << oracle_->mapper.get_str(u) << " due to API error: " << e.what() << endl;
            }
            continue;
        }

        for (int v : edges.second)
        {
            if (visited.insert(v).second)
                q_bfs.push({v, d + 1});
        }
        for (int v : edges.first)
        {
            if (visited.insert(v).second)
                q_bfs.push({v, d + 1});
        }
    }

    // ========================================================================
    // 2. Build Internal Adjacency for the Explored Set
    // ========================================================================
    unordered_map<int, vector<int>> adj;
    for (int u : visited)
    {
        try
        {
            auto edges = oracle_->query(u);
            for (int v : edges.second)
            {
                if (visited.count(v))
                {
                    adj[u].push_back(v);
                    adj[v].push_back(u); // Treat as undirected for connectivity
                }
            }
        }
        catch (...)
        {
            continue;
        }
    }

    // ========================================================================
    // 3. Connectivity-Aware Greedy Peeling
    // ========================================================================
    unordered_set<int> current_S = visited;
    vector<int> best_S;
    double best_avg_density = -1.0;

    while (current_S.size() > 1)
    {
        // A. Evaluate Current Subgraph
        int internal_edges = 0;
        unordered_map<int, int> internal_deg;
        for (int u : current_S)
            internal_deg[u] = 0;

        for (int u : current_S)
        {
            for (int v : adj[u])
            {
                if (current_S.count(v))
                {
                    internal_edges++;
                    internal_deg[u]++;
                }
            }
        }
        internal_edges /= 2; // Correct for undirected double-counting

        double current_density = (double)internal_edges / current_S.size();

        // Track the best valid subset seen so far
        if (current_S.size() >= (size_t)k_ && current_density > best_avg_density)
        {
            best_avg_density = current_density;
            best_S.assign(current_S.begin(), current_S.end());
        }

        // B. Sort nodes by internal degree (Ascending)
        vector<pair<int, int>> candidates;
        for (int u : current_S)
        {
            if (u != query_node)
            {
                candidates.push_back({internal_deg[u], u});
            }
        }
        sort(candidates.begin(), candidates.end());

        // C. Find the weakest node that does NOT break connectivity
        bool node_removed = false;
        for (auto const &[deg, candidate_node] : candidates)
        {

            // Micro-BFS to check articulation
            unordered_set<int> test_visited;
            queue<int> test_q;
            test_q.push(query_node);
            test_visited.insert(query_node);

            while (!test_q.empty())
            {
                int curr = test_q.front();
                test_q.pop();
                for (int neighbor : adj[curr])
                {
                    if (neighbor != candidate_node && current_S.count(neighbor) && test_visited.insert(neighbor).second)
                    {
                        test_q.push(neighbor);
                    }
                }
            }

            // If the graph remains connected, it's safe to peel!
            if (test_visited.size() == current_S.size() - 1)
            {
                current_S.erase(candidate_node);
                node_removed = true;
                break; // Break candidate loop, go to next peeling iteration
            }
        }

        // If no node can be safely removed (they are all load-bearing bridges), stop.
        if (!node_removed)
        {
            break;
        }
    }

    // Fallback if the graph was so sparse we couldn't even find a k-sized component
    if (best_S.empty())
    {
        best_S.assign(current_S.begin(), current_S.end());
    }

    return best_S;
}
