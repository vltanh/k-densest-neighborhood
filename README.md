## Densest Community Search

This repository implements exact and heuristic solvers for the **K-Densest Subgraph** problem on directed graphs, along with a node classification application using the discovered dense communities as neighborhoods.

---

## C++ Solver

Source code lives in `src/` and is built with CMake. The executable is placed at `bin/solver`.

### Dependencies

- **Gurobi** — set the `GUROBI_HOME` environment variable to your Gurobi installation (e.g., `export GUROBI_HOME=/path/to/gurobi1301/linux64`).
- **libcurl** — required for the OpenAlex live-API mode.
- **nlohmann/json** — automatically downloaded by CMake during the first build.

### Build

```bash
bash build.sh
```

### Usage

The solver supports two operating modes selected with `--mode`.

**Simulation mode** (local CSV graph):

```bash
./bin/solver --mode sim --input <edge.csv> --query <node_id> --k <k> [--output <out.csv>]
```

**OpenAlex mode** (live citation API):

```bash
./bin/solver --mode openalex --query <openalex_work_id> --k <k> [--output <out.csv>]
```

Required arguments:

- `--mode <sim|openalex>`: Mode of operation.
- `--query <node_id>`: String ID of the target query node.
- `--k <int>`: Target subgraph size (k ≥ 2).
- `--input <edge.csv>`: Path to the edge list CSV (`source`, `target` columns) — required for `sim` mode.

Optional arguments:

- `--output <out.csv>`: Save resulting community node IDs to this file (`node_id` column).
- `--time-limit <float>`: Max Branch-and-Bound time in seconds (default: `600.0`).
- `--node-limit <int>`: Max B&B nodes to explore (default: `100000`).
- `--gap-tol <float>`: Early-stopping relative gap tolerance (default: `1e-4`).
- `--dinkelbach-iter <int>`: Max Dinkelbach iterations (default: `50`).
- `--cg-batch-frac <float>`: Fraction of active set priced per iteration (default: `0.1`).
- `--cg-min-batch <int>`: Minimum columns added per pricing round (default: `5`).
- `--cg-max-batch <int>`: Maximum columns added per pricing round (default: `50`).
- `--tol <float>`: Numerical tolerance for zero-checks (default: `1e-6`).
- `--help`, `-h`: Print the help menu and exit.

---

## Synthetic Graph Generation

### Generate a Graph

```bash
python scripts/generate_graph.py --out_dir <output_dir> [options]
```

- `--out_dir`: Directory to write output files (`edge.csv`, `gt_comm.csv`, `metadata.json`).
- `--n_nodes`: Total number of nodes (default: `1000000`).
- `--m_edges`: Barabási–Albert attachment parameter (default: `10`).
- `--n_community`: Size of the planted dense community (default: `20`).
- `--p_community`: Edge probability within the planted community (default: `0.8`).
- `--p_reciprocal`: Global probability of adding a reciprocal edge (2-cycle) (default: `0.001`).
- `--seed`: Random seed (default: `42`).

Output files:
- `edge.csv` — directed edge list with columns `source`, `target`.
- `gt_comm.csv` — ground truth community with column `node_id`.
- `metadata.json` — graph statistics and generation parameters.

### Evaluate Solver Against Ground Truth

```bash
python scripts/evaluate_solver.py --gt <gt_comm.csv> --pred <pred_comm.csv>
```

- `--gt`: Path to the ground truth community CSV (`gt_comm.csv`).
- `--pred`: Path to the predicted community CSV produced by the solver.

Reports precision, recall, F1, and Jaccard similarity.

---

## Node Classification (CitationFull Datasets)

Uses the K-Densest community around each query node as its neighborhood for label propagation (majority vote). The shared solver execution and voting logic lives in `kdcs/solver_utils.py`.

### 1. Prepare Data

Downloads a CitationFull dataset and exports edge and node split files to `data/<dataset>/`. The split is **temporal/inductive**: nodes that have been cited ("foundational" papers) form the training set, while purely-citing ("new") papers are evenly divided into validation and test sets.

```bash
python scripts/prepare_data.py --dataset <dataset_name>
```

- `--dataset`: One of `Cora`, `Cora_ML`, `CiteSeer`, `DBLP`, or `PubMed` (default: `Cora_ML`).

Output files written to `data/<dataset>/`:
- `edge.csv` — directed edge list with columns `source`, `target`.
- `nodes.csv` — node metadata with columns `node_id`, `label`, `train`, `val`, `test`.

### 2. Tune Hyperparameters

Sweeps `k` over the validation split to find the best community size.

```bash
python scripts/tune.py --dataset <dataset_name> --k_min <min_k> --k_max <max_k> --k_step <step_k> --optimize <metric> --weighting <weight_strategy>
```

- `--dataset`: Dataset name matching a prepared `data/<dataset>/` directory (default: `Cora`).
- `--k_min`: Minimum k to evaluate (default: `5`).
- `--k_max`: Maximum k to evaluate (default: `25`).
- `--k_step`: Step size for k sweep (default: `5`).
- `--optimize`: Metric to maximize when selecting the best k (`accuracy`, `f1`, `precision`, `recall`; default: `accuracy`).
- `--weighting`: Voting weight strategy (`uniform` or `distance`; default: `uniform`).
- `--bin_path`: Path to the compiled solver binary (default: `./bin/solver`).
- `--workers`: Number of parallel workers (default: number of CPU cores).

### 3. Final Evaluation

Evaluates classification on a specific dataset split.

```bash
python scripts/evaluate.py --dataset <dataset_name> --split <split_name> --k <best_k> --weighting <weight_strategy>
```

- `--dataset`: Dataset name matching a prepared `data/<dataset>/` directory (default: `Cora`).
- `--split`: One of `train`, `val`, or `test`.
- `--k`: The optimal k found from tuning.
- `--weighting`: Voting weight strategy (`uniform` or `distance`; default: `uniform`).
- `--bin_path`: Path to the compiled solver binary (default: `./bin/solver`).
- `--workers`: Number of parallel workers (default: number of CPU cores).

Reports accuracy, macro precision, recall, F1, and a per-class classification report. When the solver returns a neighborhood with no training nodes (label starvation), a concentric BFS fallback is triggered automatically; the fallback rate is printed at the end of each run.

### 4. Baseline: Concentric BFS

Classifies each node by majority voting over the nearest training-set ring reachable via BFS on the undirected graph.

```bash
python scripts/baseline_bfs.py --dataset <dataset_name> --split <split_name> --max_hops <max_hops> --workers <num_workers>
```

- `--dataset`: Dataset name matching a prepared `data/<dataset>/` directory (default: `Cora`).
- `--split`: One of `train`, `val`, or `test`.
- `--max_hops`: Maximum BFS search depth (default: `10`).
- `--workers`: Number of parallel workers (default: number of CPU cores).