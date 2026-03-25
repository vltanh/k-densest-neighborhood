## Densest Community Search

This repository implements exact and heuristic solvers for the **K-Densest Subgraph** problem on directed graphs, along with a node classification application using the discovered dense communities as neighborhoods.

---

## C++ Solver

### Build

```bash
g++ -m64 -O3 -std=c++17 -I${GUROBI_HOME}/include/ solver.cpp -L${GUROBI_HOME}/lib/ -lgurobi_c++ -lgurobi130 -o bin/solver
```

Make sure to:
- set the `GUROBI_HOME` environment variable to the path where Gurobi is installed (e.g., `export GUROBI_HOME=/path/to/gurobi1301/linux64`), and
- replace `130` with the actual version number of Gurobi you have installed.

### Usage

```bash
./bin/solver <input_file> <q_node> <k> <output_file>
```

- `<input_file>`: Path to the input graph file in CSV format with columns `source` and `target` representing edges.
- `<q_node>`: The query node ID for which to find the densest community.
- `<k>`: Minimum community size to search for.
- `<output_file>`: Path to the output CSV file where the predicted community (`node_id` column) will be saved.

---

## Synthetic Graph Generation

### Generate a Graph

```bash
python generate_graph.py --out_dir <output_dir> [options]
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
python evaluate_solver.py --gt <gt_comm.csv> --pred <pred_comm.csv>
```

- `--gt`: Path to the ground truth community CSV (`gt_comm.csv`).
- `--pred`: Path to the predicted community CSV produced by the solver.

Reports precision, recall, F1, and Jaccard similarity.

---

## Node Classification (CitationFull Datasets)

Uses the K-Densest community around each query node as its neighborhood for label propagation (majority vote). The shared solver execution and voting logic lives in `solver_utils.py`.

### 1. Prepare Data

Downloads a CitationFull dataset and exports edge and node split files to `data/<dataset>/`. The split is **temporal/inductive**: nodes that have been cited ("foundational" papers) form the training set, while purely-citing ("new") papers are evenly divided into validation and test sets.

```bash
python prepare_data.py --dataset <dataset_name>
```

- `--dataset`: One of `Cora`, `Cora_ML`, `CiteSeer`, `DBLP`, or `PubMed` (default: `Cora_ML`).

Output files written to `data/<dataset>/`:
- `edge.csv` — directed edge list with columns `source`, `target`.
- `nodes.csv` — node metadata with columns `node_id`, `label`, `train`, `val`, `test`.

### 2. Tune Hyperparameters

Sweeps `k` over the validation split to find the best community size.

```bash
python tune.py --dataset <dataset_name> --k_min <min_k> --k_max <max_k> --k_step <step_k> --optimize <metric> --weighting <weight_strategy>
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
python evaluate.py --dataset <dataset_name> --split <split_name> --k <best_k> --weighting <weight_strategy>
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
python baseline_bfs.py --dataset <dataset_name> --split <split_name> --max_hops <max_hops> --workers <num_workers>
```

- `--dataset`: Dataset name matching a prepared `data/<dataset>/` directory (default: `Cora`).
- `--split`: One of `train`, `val`, or `test`.
- `--max_hops`: Maximum BFS search depth (default: `10`).
- `--workers`: Number of parallel workers (default: number of CPU cores).