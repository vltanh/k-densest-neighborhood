import math
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "synthetic"))

from tune_methods import (
    best_rows_by_method,
    build_tuning_stages,
    filter_completed_stages,
    limit_nodes,
    parse_float_list,
    parse_int_list,
)


try:
    from solver_utils import compute_subgraph_quality, run_solver
except ModuleNotFoundError as exc:
    compute_subgraph_quality = None
    run_solver = None
    SOLVER_UTILS_IMPORT_ERROR = exc
else:
    SOLVER_UTILS_IMPORT_ERROR = None

from benchmark_solvers import run_solver as run_benchmark_solver


@unittest.skipIf(
    compute_subgraph_quality is None,
    f"solver quality dependencies unavailable: {SOLVER_UTILS_IMPORT_ERROR}",
)
class SolverQualityTests(unittest.TestCase):
    def test_undir_internal_norm_min_cut_uses_symmetric_pymincut_edges(self):
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

        self.assertEqual(qualities["undir_internal_norm_min_cut_computed"], 1)
        self.assertTrue(math.isclose(qualities["undir_internal_norm_min_cut"], 1.0))


@unittest.skipIf(
    run_solver is None,
    f"solver utility dependencies unavailable: {SOLVER_UTILS_IMPORT_ERROR}",
)
class SolverCommandTests(unittest.TestCase):
    def test_classification_run_solver_forwards_max_in_edges(self):
        calls = []

        class Result:
            stdout = "API Queries Made : 0\n"

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return Result()

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("solver_utils.subprocess.run", side_effect=fake_run):
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

    def test_synthetic_benchmark_does_not_pass_rejected_flags_to_bfs(self):
        calls = []

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return Result()

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("benchmark_solvers.subprocess.run", side_effect=fake_run):
                run_benchmark_solver(
                    bin_path="solver",
                    edge_csv="edges.csv",
                    query_node="1",
                    k=5,
                    method="bfs",
                    time_limit=1.0,
                    node_limit=10,
                    baseline_depth=-1,
                    bfs_depth=2,
                    output_dir=tmp_dir,
                )

        self.assertNotIn("--k", calls[0])
        self.assertNotIn("--baseline-depth", calls[0])
        self.assertIn("--bfs-depth", calls[0])

    def test_synthetic_benchmark_passes_k_only_to_k_based_methods(self):
        calls = []

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return Result()

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("benchmark_solvers.subprocess.run", side_effect=fake_run):
                run_benchmark_solver(
                    bin_path="solver",
                    edge_csv="edges.csv",
                    query_node="1",
                    k=5,
                    method="bp",
                    time_limit=1.0,
                    node_limit=10,
                    baseline_depth=-1,
                    bfs_depth=2,
                    output_dir=tmp_dir,
                )

        self.assertIn("--k", calls[0])
        self.assertNotIn("--baseline-depth", calls[0])


class TuningConfigTests(unittest.TestCase):
    def test_tuning_config_builder_covers_requested_families(self):
        stages = build_tuning_stages(
            "all",
            k_values=[3, 5],
            kappa_values=[0, 2],
            bfs_depth_min=1,
            bfs_depth_max=2,
            bp_time_limits=[10.0, -1.0],
        )

        configs = [config for _, stage_configs in stages for config in stage_configs]

        self.assertEqual([name for name, _ in stages], ["avgdeg", "bfs", "bp"])
        self.assertEqual(
            len([c for c in configs if c["method"] == "avgdeg"]), 1
        )
        self.assertEqual(len([c for c in configs if c["method"] == "bfs"]), 2)
        self.assertEqual(len([c for c in configs if c["method"] == "bp"]), 8)
        self.assertEqual(
            {
                tuple(c["extra_args"])
                for c in configs
                if c["method"] == "bp" and c["k"] == 5 and c["kappa"] == 2
            },
            {
                ("--bp", "--kappa", "2", "--time-limit", "10.0"),
                ("--bp", "--kappa", "2", "--time-limit", "-1.0"),
            },
        )

    def test_tuning_resume_filters_completed_configs_with_nan_fields(self):
        stages = build_tuning_stages(
            "all",
            k_values=[3],
            kappa_values=[2],
            bfs_depth_min=1,
            bfs_depth_max=1,
            bp_time_limits=[60.0],
        )
        completed_rows = [
            {
                "method": "avgdeg",
                "k": math.nan,
                "kappa": math.nan,
                "depth": math.nan,
            },
            {"method": "bfs", "k": math.nan, "kappa": math.nan, "depth": 1},
        ]

        remaining = filter_completed_stages(stages, completed_rows)
        remaining_configs = [
            config for _, stage_configs in remaining for config in stage_configs
        ]

        self.assertEqual(
            [(name, len(configs)) for name, configs in remaining],
            [("avgdeg", 0), ("bfs", 0), ("bp", 1)],
        )
        self.assertEqual(remaining_configs[0]["method"], "bp")

    def test_tuning_parsers_and_node_limit_are_reproducible(self):
        self.assertEqual(parse_int_list("3, 5,,7"), [3, 5, 7])
        self.assertEqual(parse_float_list("60,-1,0.5"), [60.0, -1.0, 0.5])
        self.assertEqual(
            limit_nodes(list(range(10)), 4, seed=123),
            limit_nodes(list(range(10)), 4, seed=123),
        )
        self.assertEqual(limit_nodes([1, 2], 0, seed=123), [1, 2])

    def test_best_rows_by_method_uses_requested_metric(self):
        rows = [
            {"method": "bp", "f1": 0.4, "accuracy": 0.9},
            {"method": "bp", "f1": 0.7, "accuracy": 0.8},
            {"method": "bfs", "f1": 0.5, "accuracy": 0.6},
        ]

        best = best_rows_by_method(rows, "f1")

        self.assertEqual(best["bp"]["f1"], 0.7)
        self.assertEqual(best["bfs"]["accuracy"], 0.6)


if __name__ == "__main__":
    unittest.main()
