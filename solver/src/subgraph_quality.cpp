#include "subgraph_quality.hpp"
#include <queue>
#include <unordered_map>
#include <unordered_set>

SubgraphQualities compute_subgraph_qualities(const std::vector<int> &nodes, IGraphOracle *oracle)
{
    SubgraphQualities q;
    q.nodes = nodes;
    q.num_nodes = nodes.size();
    q.num_edges = 0;
    q.boundary_edges_out = 0;
    q.outgoing_volume = 0;
    q.weak_components = 0;
    q.largest_weak_component_size = 0;
    q.avg_degree_density = 0.0;
    q.avg_total_internal_degree = 0.0;
    q.edge_density = 0.0;
    q.outgoing_conductance = 0.0;
    q.expansion = 0.0;
    q.weak_component_ratio = 0.0;
    q.reciprocity = 0.0;

    if (q.num_nodes == 0)
        return q;

    std::unordered_set<int> node_set(nodes.begin(), nodes.end());
    std::unordered_set<std::pair<int, int>, pair_hash> internal_edges;
    std::unordered_map<int, std::vector<int>> weak_adj;

    for (int u : nodes)
    {
        try
        {
            auto edges = oracle->query(u);
            for (int v : edges.second)
            {
                if (u == v)
                    continue;

                q.outgoing_volume++;
                if (node_set.count(v))
                {
                    q.num_edges++;
                    internal_edges.insert({u, v});
                    weak_adj[u].push_back(v);
                    weak_adj[v].push_back(u);
                }
                else
                {
                    q.boundary_edges_out++;
                }
            }
        }
        catch (...)
        {
        }
    }

    std::unordered_set<int> seen;
    for (int start : nodes)
    {
        if (!seen.insert(start).second)
            continue;

        q.weak_components++;
        int component_size = 0;
        std::queue<int> bfs;
        bfs.push(start);
        while (!bfs.empty())
        {
            int u = bfs.front();
            bfs.pop();
            component_size++;
            for (int v : weak_adj[u])
            {
                if (seen.insert(v).second)
                    bfs.push(v);
            }
        }
        if (component_size > q.largest_weak_component_size)
            q.largest_weak_component_size = component_size;
    }

    int reciprocal_edges = 0;
    for (const auto &edge : internal_edges)
    {
        if (internal_edges.count({edge.second, edge.first}))
            reciprocal_edges++;
    }

    q.avg_degree_density = (double)q.num_edges / q.num_nodes;
    q.avg_total_internal_degree = (2.0 * q.num_edges) / q.num_nodes;
    if (q.num_nodes > 1)
        q.edge_density = (double)q.num_edges / ((double)q.num_nodes * (q.num_nodes - 1.0));
    if (q.outgoing_volume > 0)
        q.outgoing_conductance = (double)q.boundary_edges_out / q.outgoing_volume;
    q.expansion = (double)q.boundary_edges_out / q.num_nodes;
    q.weak_component_ratio = (double)q.largest_weak_component_size / q.num_nodes;
    if (q.num_edges > 0)
        q.reciprocity = (double)reciprocal_edges / q.num_edges;

    return q;
}
