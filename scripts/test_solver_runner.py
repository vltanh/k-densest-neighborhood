import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from _solver_runner import SolverInvocationError, invoke_solver, parse_solver_json


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SOLVER_BIN = os.path.join(_REPO_ROOT, "solver", "bin", "solver")
_SYNTHETIC_TRAP_EDGE_CSV = os.path.join(
    _REPO_ROOT, "data", "synthetic", "bf", "n20", "p050", "s4", "edge.csv"
)
_GUROBI_LICENSE = "/home/vltanh/gurobi.lic"

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
    "incumbent_trajectory": [
        {"bb_node": 3, "lambda": 0.0, "param_obj": 2.0, "density": 0.5, "size": 3, "nodes": ["7", "11", "13"]}
    ],
    "kappa_verified": True,
    "kappa_verify_failed": False,
    "optimality_gap": 0.0,
    "bb_incumbent_obj": 1.0,
    "bb_best_bound": 1.0,
    "gap_status": "exhausted",
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
        self.assertEqual(result["optimality_gap"], 0.0)
        self.assertEqual(result["gap_status"], "exhausted")
        self.assertEqual(result["stats"]["total_bb_nodes"], 5)
        self.assertEqual(result["qualities"]["num_nodes"], 3)
        self.assertEqual(result["solver_json"]["schema_version"], "1.0")
        self.assertEqual(len(result["lambda_trajectory"]), 1)
        self.assertEqual(len(result["incumbent_trajectory"]), 1)

    def test_invoke_solver_rejects_missing_json(self):
        class Result:
            returncode = 0
            stdout = "API Queries Made : 4\nNodes:\n1 2 3\n"
            stderr = ""

        with patch("_solver_runner.subprocess.run", return_value=Result()):
            with self.assertRaises(SolverInvocationError):
                invoke_solver(
                    bin_path="solver", edge_csv="edges.csv", query="0"
                )

    def test_invoke_solver_rejects_nonzero_exit(self):
        class Result:
            returncode = 2
            stdout = ""
            stderr = "license error"

        with patch("_solver_runner.subprocess.run", return_value=Result()):
            with self.assertRaises(SolverInvocationError):
                invoke_solver(
                    bin_path="solver", edge_csv="edges.csv", query="0"
                )

    def test_parse_solver_json_handles_prefix_lines(self):
        stdout = "header\nlog line\nJSON_RESULT:" + json.dumps({"size": 7}) + "\ntrailer"
        parsed = parse_solver_json(stdout)
        self.assertEqual(parsed["size"], 7)

    def test_json_output_path_does_not_reuse_stale_payload(self):
        stale_payload = dict(_PAYLOAD)
        stale_payload["nodes"] = ["999"]
        fresh_payload = dict(_PAYLOAD)
        fresh_payload["nodes"] = ["7"]

        class Result:
            returncode = 0
            stdout = _stdout_with_payload(fresh_payload)
            stderr = ""

        with tempfile.TemporaryDirectory() as td:
            dump_path = os.path.join(td, "solver.json")
            with open(dump_path, "w") as f:
                json.dump(stale_payload, f)

            with patch("_solver_runner.subprocess.run", return_value=Result()):
                result = invoke_solver(
                    bin_path="solver",
                    edge_csv="edges.csv",
                    query="0",
                    as_int_nodes=True,
                    json_output_path=dump_path,
                )

        self.assertEqual(result["pred_nodes"], [7])


@unittest.skipUnless(
    os.path.exists(_SOLVER_BIN)
    and os.path.exists(_SYNTHETIC_TRAP_EDGE_CSV)
    and os.path.exists(_GUROBI_LICENSE),
    "solver binary, synthetic fixture, or Gurobi license unavailable",
)
class BpRootQueryRegressionTests(unittest.TestCase):
    def test_bp_root_fix_keeps_query_in_solution_on_synthetic_clique_trap(self):
        old_license = os.environ.get("GRB_LICENSE_FILE")
        os.environ["GRB_LICENSE_FILE"] = _GUROBI_LICENSE
        try:
            result = invoke_solver(
                bin_path=_SOLVER_BIN,
                edge_csv=_SYNTHETIC_TRAP_EDGE_CSV,
                query="6",
                extra_args=["--bp", "--k", "3", "--kappa", "0", "--gurobi-seed", "42"],
                as_int_nodes=True,
            )
        finally:
            if old_license is None:
                os.environ.pop("GRB_LICENSE_FILE", None)
            else:
                os.environ["GRB_LICENSE_FILE"] = old_license

        self.assertEqual(result["returncode"], 0, msg=result["stderr"][-1000:])
        self.assertIn(6, result["pred_nodes"])


if __name__ == "__main__":
    unittest.main()
