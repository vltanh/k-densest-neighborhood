import json
import os
import sys
import unittest
import warnings
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from _solver_runner import invoke_solver, parse_solver_json


_PAYLOAD = {
    "schema_version": "1.0",
    "method": "bp",
    "query_node": "0",
    "k": 5,
    "kappa": 2,
    "bfs_depth": None,
    "nodes": ["7", "11", "13"],
    "size": 3,
    "lambda_final": 0.5,
    "lambda_trajectory": [
        {"iter": 1, "lambda": 0.0, "iter_time_s": 0.01, "bb_nodes": 5, "lp_solves": 7}
    ],
    "kappa_verified": True,
    "kappa_verify_failed": False,
    "stats": {"total_bb_nodes": 5, "total_lp_solves": 7},
    "qualities": {"num_nodes": 3, "edge_density": 0.5},
    "oracle": {"queries_made": 9, "unique_nodes_mapped": 4,
               "cumulative_network_time_s": 0.0, "quality_extra_queries": 0},
    "io": {"input_edge_count": 100, "io_time_s": 0.01},
    "config": {"gurobi_seed": 42},
    "wall_time_s": 0.05,
}


def _stdout_with_payload(payload=None):
    payload = payload or _PAYLOAD
    return "log line\nJSON_RESULT:" + json.dumps(payload) + "\n"


class InvokeSolverTests(unittest.TestCase):
    def test_invoke_solver_passes_emit_json(self):
        captured = {}

        class Result:
            returncode = 0
            stdout = _stdout_with_payload()
            stderr = ""

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return Result()

        with patch("_solver_runner.subprocess.run", side_effect=fake_run):
            invoke_solver(
                bin_path="solver",
                edge_csv="edges.csv",
                query="0",
                extra_args=["--bp", "--k", "5"],
            )

        self.assertIn("--emit-json", captured["cmd"])
        self.assertNotIn("--output", captured["cmd"])

    def test_invoke_solver_parses_stdout_payload(self):
        class Result:
            returncode = 0
            stdout = _stdout_with_payload()
            stderr = ""

        with patch("_solver_runner.subprocess.run", return_value=Result()):
            result = invoke_solver(
                bin_path="solver", edge_csv="edges.csv", query="0",
                as_int_nodes=True,
            )

        self.assertEqual(result["pred_nodes"], [7, 11, 13])
        self.assertEqual(result["oracle_queries"], 9)
        self.assertTrue(result["kappa_verified"])
        self.assertFalse(result["kappa_verify_failed"])
        self.assertEqual(result["stats"]["total_bb_nodes"], 5)
        self.assertEqual(result["qualities"]["num_nodes"], 3)
        self.assertEqual(result["solver_json"]["schema_version"], "1.0")
        self.assertEqual(len(result["lambda_trajectory"]), 1)

    def test_invoke_solver_falls_back_on_missing_json(self):
        class Result:
            returncode = 0
            stdout = "API Queries Made : 4\nNodes:\n1 2 3\n"
            stderr = ""

        with patch("_solver_runner.subprocess.run", return_value=Result()):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = invoke_solver(
                    bin_path="solver", edge_csv="edges.csv", query="0"
                )

        self.assertEqual(result["oracle_queries"], 4)
        self.assertEqual(result["pred_nodes"], [])
        self.assertIsNone(result["kappa_verified"])
        self.assertIsNone(result["stats"])
        self.assertIsNone(result["solver_json"])
        self.assertTrue(any("JSON_RESULT" in str(item.message) for item in w))

    def test_parse_solver_json_handles_prefix_lines(self):
        stdout = "header\nlog line\nJSON_RESULT:" + json.dumps({"size": 7}) + "\ntrailer"
        parsed = parse_solver_json(stdout)
        self.assertEqual(parsed["size"], 7)


if __name__ == "__main__":
    unittest.main()
