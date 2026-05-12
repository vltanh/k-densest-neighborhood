import math
import os
import sys
import tempfile
import unittest
from collections import Counter
from unittest.mock import patch

import pandas as pd

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "synthetic"))


try:
    from solver_utils import (
        _bfs_with_forbidden,
        argmax_label,
        compute_per_class_breakdown,
        compute_subgraph_quality,
        effective_params,
        evaluate_nodes,
        method_extra_args,
        run_solver,
    )
except ModuleNotFoundError as exc:
    compute_subgraph_quality = None
    compute_per_class_breakdown = None
    run_solver = None
    evaluate_nodes = None
    argmax_label = None
    _bfs_with_forbidden = None
    method_extra_args = None
    effective_params = None
    SOLVER_UTILS_IMPORT_ERROR = exc
else:
    SOLVER_UTILS_IMPORT_ERROR = None


@unittest.skipIf(
    compute_subgraph_quality is None,
    f"solver quality dependencies unavailable: {SOLVER_UTILS_IMPORT_ERROR}",
)
class SolverQualityTests(unittest.TestCase):
    def test_undir_internal_ncut_uses_symmetric_pymincut_edges(self):
        mincut_neighbors = {
            0: {1},
            1: {0, 2},
            2: {1, 3},
            3: {2},
        }

        out_neighbors = {
            0: {1},
            1: {2},
            2: {3},
            3: set(),
        }

        qualities = compute_subgraph_quality(
            [0, 1, 2, 3], out_neighbors, mincut_neighbors
        )

        # Path 0-1-2-3, min cut isolates a leaf: cut=1, vol_a=1, vol_b=5.
        # NCut = 1 * (1+5) / (1*5) = 1.2.
        self.assertTrue(math.isclose(qualities["undir_internal_ncut"], 1.2))


@unittest.skipIf(
    run_solver is None,
    f"solver utility dependencies unavailable: {SOLVER_UTILS_IMPORT_ERROR}",
)
class SolverCommandTests(unittest.TestCase):
    def test_classification_run_solver_forwards_max_in_edges(self):
        calls = []

        class Result:
            returncode = 0
            stdout = (
                'JSON_RESULT:{"nodes":[],"oracle":{"queries_made":0},'
                '"lambda_trajectory":[],"kappa_verified":null,'
                '"kappa_verify_failed":null,"stats":null,"qualities":null}\n'
            )
            stderr = ""

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return Result()

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("_solver_runner.subprocess.run", side_effect=fake_run):
                run_solver(
                    q_node=7,
                    k=5,
                    edge_csv="edges.csv",
                    bin_path="solver",
                    tmp_dir=tmp_dir,
                    extra_args=["--bp"],
                    max_in_edges=25,
                )

        self.assertIn("--max-in-edges", calls[0])
        idx = calls[0].index("--max-in-edges")
        self.assertEqual(calls[0][idx + 1], "25")
        self.assertIn("--emit-json", calls[0])


@unittest.skipIf(
    argmax_label is None,
    f"solver utility dependencies unavailable: {SOLVER_UTILS_IMPORT_ERROR}",
)
class ArgmaxLabelTests(unittest.TestCase):
    def test_argmax_label_breaks_ties_by_label_value(self):
        self.assertEqual(argmax_label(Counter({2: 3, 1: 3, 3: 1})), 1)

    def test_argmax_label_returns_none_for_empty_counter(self):
        self.assertIsNone(argmax_label(Counter()))

    def test_argmax_label_prefers_majority(self):
        self.assertEqual(argmax_label(Counter({0: 1, 1: 5})), 1)


@unittest.skipIf(
    _bfs_with_forbidden is None,
    f"solver utility dependencies unavailable: {SOLVER_UTILS_IMPORT_ERROR}",
)
class ForbiddenBfsTests(unittest.TestCase):
    def _line_graph(self):
        import networkx as nx

        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        return G

    def test_forbidden_intermediate_blocks_reach(self):
        G = self._line_graph()
        distances = _bfs_with_forbidden(G, 0, cutoff=10, forbidden={2})
        self.assertEqual(distances, {0: 0, 1: 1})

    def test_no_forbidden_returns_all_reachable(self):
        G = self._line_graph()
        distances = _bfs_with_forbidden(G, 0, cutoff=10, forbidden=set())
        self.assertEqual(distances, {0: 0, 1: 1, 2: 2, 3: 3})

    def test_source_in_forbidden_still_expands(self):
        G = self._line_graph()
        distances = _bfs_with_forbidden(G, 0, cutoff=10, forbidden={0})
        # Source is visited at distance 0 so its non-forbidden neighbours
        # remain reachable; the caller filters the source from the vote.
        self.assertEqual(distances, {0: 0, 1: 1, 2: 2, 3: 3})


@unittest.skipIf(
    evaluate_nodes is None,
    f"solver utility dependencies unavailable: {SOLVER_UTILS_IMPORT_ERROR}",
)
class LeakageGuardTests(unittest.TestCase):
    def _write_leak_dataset(self, tmpdir):
        edges = [(0, 1), (1, 2)]
        pd.DataFrame(edges, columns=["source", "target"]).to_csv(
            os.path.join(tmpdir, "edge.csv"), index=False
        )
        df_nodes = pd.DataFrame(
            {
                "node_id": [0, 1, 2, 3, 4],
                "label": [9, 0, 5, 7, 7],
                "train": [False, False, True, True, True],
                "val": [False, False, False, False, False],
                "test": [True, True, False, False, False],
            }
        )
        return df_nodes, os.path.join(tmpdir, "edge.csv")

    def test_forbidden_test_node_blocks_label_leak(self):
        with tempfile.TemporaryDirectory() as td:
            df_nodes, edge_csv = self._write_leak_dataset(td)

            def fake_run_one_query(*args, **kwargs):
                return {
                    "returncode": 0,
                    "pred_nodes": [],
                    "oracle_queries": math.nan,
                    "wall_time": None,
                }

            with patch("solver_utils.run_one_query", side_effect=fake_run_one_query):
                _, _, y_pred_no = evaluate_nodes(
                    [0], k=None, edge_csv=edge_csv, df_nodes=df_nodes,
                    bin_path="solver", tmp_dir=td, max_workers=1,
                    show_progress=False, return_query_nodes=True,
                )
                _, _, y_pred_fb = evaluate_nodes(
                    [0], k=None, edge_csv=edge_csv, df_nodes=df_nodes,
                    bin_path="solver", tmp_dir=td, max_workers=1,
                    show_progress=False, return_query_nodes=True,
                    forbidden_nodes={1, 2},
                )

        # Without forbidden, the fallback BFS hops through the test node 1 to
        # reach the train node 2 and adopts label 5 (the leak).
        self.assertEqual(y_pred_no[0], 5)
        # With the forbidden set, BFS cannot use node 1 as a stepping stone, no
        # train labels are reachable, and the global majority (7) is returned.
        self.assertEqual(y_pred_fb[0], 7)


@unittest.skipIf(
    compute_subgraph_quality is None,
    f"solver quality dependencies unavailable: {SOLVER_UTILS_IMPORT_ERROR}",
)
class Lambda2AndMixingTests(unittest.TestCase):
    """Cluster-quality primitives on small fixtures with closed-form answers."""

    @staticmethod
    def _undir(edges):
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        return adj

    def test_lambda2_on_triangle(self):
        # Normalised Laplacian of K_n: eigenvalues are 0 then n/(n-1) (mult n-1).
        adj = self._undir([(0, 1), (1, 2), (2, 0)])
        q = compute_subgraph_quality([0, 1, 2], {n: set() for n in adj}, adj)
        self.assertAlmostEqual(q["algebraic_connectivity_lambda2"], 1.5, places=6)

    def test_lambda2_on_k4(self):
        adj = self._undir([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        q = compute_subgraph_quality([0, 1, 2, 3], {n: set() for n in adj}, adj)
        self.assertAlmostEqual(q["algebraic_connectivity_lambda2"], 4.0 / 3.0, places=6)

    def test_lambda2_on_c4(self):
        adj = self._undir([(0, 1), (1, 2), (2, 3), (3, 0)])
        q = compute_subgraph_quality([0, 1, 2, 3], {n: set() for n in adj}, adj)
        self.assertAlmostEqual(q["algebraic_connectivity_lambda2"], 1.0, places=6)

    def test_lambda2_on_k13_star(self):
        # K_{1,3}: centre 0, leaves 1, 2, 3. Normalised Laplacian eigenvalues
        # for a complete bipartite K_{m,n} are 0, 1 (mult m+n-2), 2.
        adj = self._undir([(0, 1), (0, 2), (0, 3)])
        q = compute_subgraph_quality([0, 1, 2, 3], {n: set() for n in adj}, adj)
        self.assertAlmostEqual(q["algebraic_connectivity_lambda2"], 1.0, places=6)

    def test_lambda2_on_disconnected_two_k2(self):
        adj = self._undir([(0, 1), (2, 3)])
        q = compute_subgraph_quality([0, 1, 2, 3], {n: set() for n in adj}, adj)
        # Disconnected: two zero eigenvalues, so lambda_2 = 0.
        self.assertAlmostEqual(q["algebraic_connectivity_lambda2"], 0.0, places=8)

    def test_mixing_param_on_k13_center_full_subgraph(self):
        # Full K_{1,3} has no boundary; mixing_param = 0 / (0 + 2*3) = 0.
        adj = self._undir([(0, 1), (0, 2), (0, 3)])
        q = compute_subgraph_quality([0, 1, 2, 3], {n: set() for n in adj}, adj)
        self.assertAlmostEqual(q["mixing_param"], 0.0, places=8)

    def test_mixing_param_on_k13_center_partial_subgraph(self):
        # K_{1,3} but S = {centre, one leaf}; the other two leaves are external.
        # internal_und_edges = 1, boundary = 2 -> mixing = 2 / (2 + 2) = 0.5.
        adj = self._undir([(0, 1), (0, 2), (0, 3)])
        q = compute_subgraph_quality([0, 1], {n: set() for n in adj}, adj)
        self.assertAlmostEqual(q["mixing_param"], 0.5, places=8)


@unittest.skipIf(
    compute_per_class_breakdown is None,
    f"solver quality dependencies unavailable: {SOLVER_UTILS_IMPORT_ERROR}",
)
class PerClassBreakdownTests(unittest.TestCase):
    def test_breakdown_on_k13_with_known_labels(self):
        # K_{1,3}: centre 0 with label 7; leaves: 1 (label 7, train), 2 (label 7,
        # train), 3 (label 1, train).
        mincut = {0: {1, 2, 3}, 1: {0}, 2: {0}, 3: {0}}
        out = {0: {1, 2, 3}, 1: {0}, 2: {0}, 3: {0}}
        labels = [7, 7, 7, 1]
        train_mask = [False, True, True, True]
        out_dict = compute_per_class_breakdown(
            [0, 1, 2, 3], out, mincut, labels, train_mask, query_node=0
        )
        # Internal undirected edges: (0,1), (0,2), (0,3) -> 3 total. Same-label
        # pairs: (0,1), (0,2). Ratio = 2/3.
        self.assertAlmostEqual(
            out_dict["within_class_internal_edges_ratio"], 2.0 / 3.0, places=8
        )
        # Train neighbours of S excluding the query: {1, 2, 3}. Label counts:
        # 7 -> 2, 1 -> 1. Vote share for query label 7 = 2/3.
        self.assertAlmostEqual(out_dict["true_class_vote_share"], 2.0 / 3.0, places=8)
        self.assertEqual(out_dict["n_train_neighbours"], 3)
        self.assertEqual(out_dict["size_bucket"], "small")


@unittest.skipIf(
    method_extra_args is None,
    f"solver utility dependencies unavailable: {SOLVER_UTILS_IMPORT_ERROR}",
)
class MethodExtraArgsTests(unittest.TestCase):
    def test_avgdeg_extra_args_ignores_gurobi_seed(self):
        self.assertEqual(method_extra_args("avgdeg"), ["--avgdeg"])
        self.assertEqual(
            method_extra_args("avgdeg", {"anything": 1}, gurobi_seed=99),
            ["--avgdeg"],
        )

    def test_bfs_extra_args_uses_depth(self):
        self.assertEqual(
            method_extra_args("bfs", {"bfs_depth": 2}),
            ["--bfs", "--bfs-depth", "2"],
        )
        # gurobi_seed must not leak into bfs invocation
        self.assertNotIn(
            "--gurobi-seed",
            method_extra_args("bfs", {"bfs_depth": 1}, gurobi_seed=7),
        )

    def test_bp_extra_args_carries_kappa_and_seed(self):
        argv = method_extra_args(
            "bp",
            {"kappa": 2, "time_limit": -1, "dinkelbach_iter": -1},
            gurobi_seed=42,
        )
        self.assertIn("--bp", argv)
        self.assertIn("--kappa", argv)
        idx = argv.index("--kappa")
        self.assertEqual(argv[idx + 1], "2")
        self.assertIn("--gurobi-seed", argv)
        idx = argv.index("--gurobi-seed")
        self.assertEqual(argv[idx + 1], "42")

    def test_bp_extra_args_without_gurobi_seed(self):
        argv = method_extra_args("bp", {"kappa": 0})
        self.assertNotIn("--gurobi-seed", argv)


@unittest.skipIf(
    effective_params is None,
    f"solver utility dependencies unavailable: {SOLVER_UTILS_IMPORT_ERROR}",
)
class EffectiveParamsHashTests(unittest.TestCase):
    def test_forbidden_changes_hash(self):
        _, h1 = effective_params({"k": 5}, weighting="uniform", max_fallback_hops=10, forbidden_nodes={1, 2})
        _, h2 = effective_params({"k": 5}, weighting="uniform", max_fallback_hops=10, forbidden_nodes={1, 3})
        self.assertNotEqual(h1, h2)

    def test_weighting_changes_hash(self):
        _, h1 = effective_params({"k": 5}, weighting="uniform", max_fallback_hops=10)
        _, h2 = effective_params({"k": 5}, weighting="distance", max_fallback_hops=10)
        self.assertNotEqual(h1, h2)

    def test_same_inputs_same_hash(self):
        _, h1 = effective_params({"k": 5}, weighting="uniform", max_fallback_hops=10, forbidden_nodes={1, 2})
        _, h2 = effective_params({"k": 5}, weighting="uniform", max_fallback_hops=10, forbidden_nodes={2, 1})
        self.assertEqual(h1, h2)


class SplitMetaIdempotencyTests(unittest.TestCase):
    """prepare_data shuffle + split logic must be deterministic on a fixed
    seed so split_meta.json hashes match across regenerations."""

    def test_deterministic_shuffle_and_hash(self):
        import numpy as np

        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from split_utils import sha256_node_set

        pool = list(range(0, 200))
        rng_a = np.random.default_rng(42)
        a = list(pool)
        rng_a.shuffle(a)
        rng_b = np.random.default_rng(42)
        b = list(pool)
        rng_b.shuffle(b)
        self.assertEqual(a, b)

        split_a = sorted(a[: len(a) // 2])
        split_b = sorted(b[: len(b) // 2])
        self.assertEqual(sha256_node_set(split_a), sha256_node_set(split_b))


if __name__ == "__main__":
    unittest.main()
