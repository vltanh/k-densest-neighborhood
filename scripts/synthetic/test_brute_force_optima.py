"""Unit tests for the brute-force optima enumerator on small fixed graphs.

Each fixture has a closed-form expected optimum for the avg-degree and
edge-density objectives so the enumerator pinpoints regressions in either the
bit-mask iteration, the connectivity check, or the kappa enumeration."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bruteforce_verify import (  # noqa: E402
    _bitmask_adjacency,
    _edge_connectivity_in_subset,
    brute_force_optima,
)


def _build(n: int, edges_und):
    edges_dir = []
    for u, v in edges_und:
        edges_dir.append((u, v))
        edges_dir.append((v, u))
    return _bitmask_adjacency(n, edges_dir)


class BruteForceOptimaTests(unittest.TestCase):
    # Conventions: brute_force_optima reads adj_out, which the helper builds as
    # a symmetric directed adjacency (each undirected edge {u,v} -> u->v AND
    # v->u). So an undirected K_n has 2 * C(n,2) = n*(n-1) directed edges, and
    # edge_density = m / (n*(n-1)) equals 1.0 on any clique. Tests below use
    # this directed convention throughout.

    def test_triangle_optima_k2(self):
        adj = _build(3, [(0, 1), (1, 2), (0, 2)])
        opt = brute_force_optima(adj, 3, q=0, k_set=[2], kappa_set=[0, 1, 2])
        # K3 is a clique: density = 1.0. {0,1} is also a clique (density 1.0)
        # and is enumerated first, so the bit-mask iterator's first-found tie
        # keeps the 2-node set; size and node count reflect that.
        self.assertAlmostEqual(opt["bp"][2]["score"], 1.0)
        # avg_degree on K3 = 6 (directed edges) / 3 (nodes) = 2.0.
        self.assertAlmostEqual(opt["avgdeg"]["score"], 2.0)
        # kappa>=1 with k=2: {0,1} is connected (lambda=1); pick that or {0,1,2}.
        self.assertAlmostEqual(opt["bp_kappa"][2][1]["score"], 1.0)
        # kappa=2 requires lambda(S) >= 2: only K3 satisfies, density still 1.0.
        self.assertAlmostEqual(opt["bp_kappa"][2][2]["score"], 1.0)
        self.assertEqual(opt["bp_kappa"][2][2]["size"], 3)

    def test_k4_optima(self):
        adj = _build(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        opt = brute_force_optima(adj, 4, q=0, k_set=[2, 3, 4], kappa_set=[0, 1, 2, 3])
        # Every clique inside K4 has density 1.0.
        self.assertAlmostEqual(opt["bp"][4]["score"], 1.0)
        # avg_degree of K4 = 12 / 4 = 3.0.
        self.assertAlmostEqual(opt["avgdeg"]["score"], 3.0)
        # K4 has edge-connectivity 3; kappa=3 feasible.
        self.assertAlmostEqual(opt["bp_kappa"][4][3]["score"], 1.0)
        self.assertEqual(opt["bp_kappa"][4][3]["size"], 4)

    def test_k4_minus_edge_optima(self):
        # K4 minus edge (2, 3): 5 undirected edges = 10 directed.
        adj = _build(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])
        opt = brute_force_optima(adj, 4, q=0, k_set=[3, 4], kappa_set=[0, 1, 2, 3])
        # m=10, n=4: density = 10/12.
        self.assertAlmostEqual(opt["bp"][4]["score"], 10.0 / 12.0)
        # Subsets of size 3 containing 0: {0,1,2}, {0,1,3}, {0,2,3}; the first
        # two are K3 (density 1.0). Best score is 1.0.
        self.assertAlmostEqual(opt["bp"][3]["score"], 1.0)
        # kappa=2 on the full 4-set is feasible (every pair has 2 edge-disjoint
        # paths). Density remains 10/12.
        self.assertAlmostEqual(opt["bp_kappa"][4][2]["score"], 10.0 / 12.0)
        # kappa=3 infeasible on any subset containing 0 (max edge-connectivity
        # in this graph is 2). best_bp_kappa[4][3] remains the sentinel -1.0.
        self.assertEqual(opt["bp_kappa"][4][3]["score"], -1.0)

    def test_two_k3_shared_edge(self):
        # Two K3 sharing edge (0, 1): undirected edges
        # (0,1), (0,2), (1,2), (0,3), (1,3). 5 undirected = 10 directed.
        adj = _build(4, [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)])
        opt = brute_force_optima(adj, 4, q=0, k_set=[3, 4], kappa_set=[0, 1, 2])
        # Full set density = 10 / 12.
        self.assertAlmostEqual(opt["bp"][4]["score"], 10.0 / 12.0)
        # k=3: subsets {0,1,2} and {0,1,3} are K3 with density 1.0.
        self.assertAlmostEqual(opt["bp"][3]["score"], 1.0)
        # avg_degree on full set = 10 / 4 = 2.5
        self.assertAlmostEqual(opt["avgdeg"]["score"], 2.5)

    def test_edge_connectivity_helper_matches_known_values(self):
        # K3 has edge-connectivity 2.
        adj = _build(3, [(0, 1), (1, 2), (0, 2)])
        self.assertEqual(_edge_connectivity_in_subset(adj, 0b111, 0), 2)
        # P3 (path) has edge-connectivity 1.
        adj = _build(3, [(0, 1), (1, 2)])
        self.assertEqual(_edge_connectivity_in_subset(adj, 0b111, 0), 1)
        # Disconnected {0, 2}: no path -> 0.
        adj = _build(3, [(0, 1)])
        self.assertEqual(_edge_connectivity_in_subset(adj, 0b101, 0), 0)


if __name__ == "__main__":
    unittest.main()
