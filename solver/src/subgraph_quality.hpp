#pragma once

#include "oracle.hpp"
#include <vector>

struct SubgraphQualities
{
    std::vector<int> nodes;
    int num_nodes;
    int num_edges;                 // Directed internal edges.
    int boundary_edges_out;         // Directed edges from S to V\S.
    int outgoing_volume;            // Directed edges from S to V.
    int weak_components;            // Components in S after ignoring edge direction.
    int largest_weak_component_size;
    double avg_degree_density;      // Directed internal edges per selected node.
    double avg_total_internal_degree;
    double edge_density;            // Directed density: |E(S)| / (|S| * (|S|-1)).
    double outgoing_conductance;    // boundary_edges_out / outgoing_volume.
    double expansion;               // boundary_edges_out / |S|.
    double weak_component_ratio;    // largest weak component size / |S|.
    double reciprocity;             // Fraction of internal directed edges with reverse edge present.
};

SubgraphQualities compute_subgraph_qualities(const std::vector<int> &nodes, IGraphOracle *oracle);
