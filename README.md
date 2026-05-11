## Densest Community Search

![Example of the Web GUI](docs/imgs/example.jpg)

This repository implements exact and heuristic solvers for the **K-Densest Subgraph** problem on directed graphs, along with a node classification application using the discovered dense communities as neighborhoods.

### Project Layout

```
k-densest-neighborhood/
├── solver/                 # C++ branch-and-price solver (CMake)
│   ├── src/                # C++ sources and headers
│   ├── CMakeLists.txt
│   ├── build.sh            # One-shot build script
│   └── bin/solver          # Compiled executable (after build)
├── backend/                # FastAPI service wrapping the solver
│   └── server.py
├── frontend/               # React + Vite + D3 web GUI
├── scripts/
│   ├── classification/     # Node-classification pipeline on CitationFull datasets
│   └── synthetic/          # Synthetic graph generation and ground-truth evaluation
└── docs/
    └── imgs/               # README and documentation assets
```

---

## Web GUI

An interactive browser-based explorer powered by the C++ solver and the OpenAlex live citation API, built with React, D3.js, and Tailwind CSS.

### Features

- Configure the query paper (OpenAlex ID), target community size *k*, and the branch-and-price tuning parameters exposed by the backend.
- Live telemetry panel streams solver log output in real time so you can monitor convergence as it happens.
- Interactive D3 force-directed graph showing **core** nodes (numbered circles) and **frontier** ghost nodes (their immediate citation neighbourhood).
- Paper ledger table listing each core paper's title, authors, venue, year, and citation count.
- **Details** modal with the full abstract; **Bib** button fetches BibTeX via DOI.
- SVG export of the current graph viewport.
- Stop button terminates the solver mid-run; each browser tab gets its own independent session.
- Sidebar width and ledger height are continuously drag-resizable; double-click either divider to collapse or restore it.

### Setup

**1. Build the C++ solver** (see [C++ Solver](#c-solver) below for prerequisites):

```bash
bash solver/build.sh
```

**2. Install backend dependencies:**

```bash
pip install fastapi uvicorn httpx pydantic
```

**3. Install frontend dependencies** (Node ≥ 18 required):

```bash
cd frontend
npm install
```

### Running

Start the API server (must be running before the frontend):

```bash
python backend/server.py
# Listening on http://0.0.0.0:8000
```

In a separate terminal, start the development server:

```bash
cd frontend
npm run dev
# Open http://localhost:5173
```

For a production build:

```bash
cd frontend
npm run build   # output in frontend/dist/
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/extract` | Run the solver; request body includes `session_id`, `query_node`, `k`, `max_in_edges`, and all solver tuning fields; streams NDJSON `log`/`result`/`error` packets |
| `POST` | `/api/extract-sim` | Run the same branch-and-price path against a prepared local CitationFull dataset |
| `POST` | `/api/stop?session_id=<id>` | Terminate the solver process for the given session |
| `GET` | `/api/datasets` | List prepared local datasets available to the simulation UI |
| `GET` | `/api/bibtex?doi=<doi>` | Fetch BibTeX for a paper via its DOI |

The `VITE_API_URL` environment variable overrides the default backend address (`http://127.0.0.1:8000`) for the frontend build.

The browser UI currently runs the default branch-and-price solver path. The C++ CLI also exposes BFS and average-degree baselines for experiments.

---

## C++ Solver

Source code lives in `solver/src/` and is built with CMake. The executable is placed at `solver/bin/solver`.

### Algorithm

Given a directed graph G = (V, E) and an integer k, the solver finds a subset S ⊆ V with |S| ≥ k that maximises the **edge density** d(S) = |E(S)| / (|S|·(|S|−1)), where E(S) are all directed edges with both endpoints in S.

The solver combines three nested algorithms:

**Dinkelbach's algorithm (outer loop)** — reduces the fractional-objective problem to a sequence of parametric subproblems. At each iteration t, given the current density estimate λₜ, it solves:

> maximise  |E(S)| − λₜ · |S|·(|S|−1)  subject to  |S| ≥ k

and updates λₜ₊₁ = d(Sₜ). Convergence is superlinear; the loop terminates when the parametric objective reaches zero.

**Branch-and-Price (middle loop)** — solves each parametric subproblem to integer optimality. The LP relaxation is solved at each B&B node via column generation. The B&B tree is explored depth-first with an early-exit gap tolerance.

Branching uses a domain-aware variable selection rule. Each fractional node v is classified by its *fractional internal degree* — the weighted sum of LP values of its neighbours in the current solution:
- **Hanging node** (internal degree < 2λ): likely to reduce density if included. The weakest such node is branched *zero-first* (exclusion explored first).
- **Core node** (internal degree ≥ 2λ): well-embedded in the current solution. The most-fractional such node is branched *one-first* (inclusion explored first).

When a B&B integer solution is found, a greedy node-removal pass is applied before accepting it as incumbent: nodes are removed one at a time (never the query node) as long as removal strictly improves the parametric objective, stopping at size k. This can tighten the incumbent and prune more of the remaining tree.

**Column generation (inner loop)** — instead of exposing all nodes to the LP at once, the solver maintains an *active set* and a *frontier*. At each CG iteration it solves the restricted master problem (RMP), then prices the frontier: a frontier node f enters the active set if its reduced cost

> rc(f) = deg_frac(f) − 2λ · Σ xᵥ − π

is positive, where deg_frac(f) is f's fractional degree into the current LP solution and π is the dual of the size constraint. Only the top-scoring batch of frontier nodes is added per round (controlled by `--cg-batch-frac`, `--cg-min-batch`, `--cg-max-batch`).

**BQP triangle cuts** — when the LP solution is fractional, the solver separates violated triangle inequalities on the product-linearisation variables:

> xᵤ + xᵥ + xw − wᵤᵥ − wᵥw − wᵤw ≤ 1

up to 20 cuts per round, tightening the LP bound before branching.

**Dynamic graph expansion** — nodes are fetched on demand from a pluggable oracle (local CSV or live OpenAlex API). Successor lists and up to `--max-in-edges` predecessors are retrieved lazily as the active set grows, so the solver works on implicit graphs without loading the entire edge list into memory. With the default `--max-in-edges 0`, incoming expansion is disabled.

### Dependencies

- **Gurobi** — set the `GUROBI_HOME` environment variable to your Gurobi installation (e.g., `export GUROBI_HOME=/path/to/gurobi1301/linux64`).
- **libcurl** — required for the OpenAlex live-API mode.
- **nlohmann/json** — automatically downloaded by CMake during the first build.

### Build

```bash
bash solver/build.sh
```

The script is location-independent: it `cd`s into its own directory, so it works from the repo root or from `solver/`.

### Usage

The solver supports two operating modes selected with `--mode`.

**Simulation mode** (local CSV graph):

```bash
./solver/bin/solver --mode sim --input <edge.csv> --query <node_id> --k <k> [--output <out.csv>]
```

**OpenAlex mode** (live citation API):

```bash
./solver/bin/solver --mode openalex --query <openalex_work_id> --k <k> [--output <out.csv>]
```

Required arguments:

- `--mode <sim|openalex>`: Mode of operation.
- `--query <node_id>`: String ID of the target query node.
- `--k <int>`: Target subgraph size (k ≥ 2).
- `--input <edge.csv>`: Path to the edge list CSV (`source`, `target` columns) — required for `sim` mode.

Optional arguments:

- `--output <out.csv>`: Save resulting community node IDs to this file (`node_id` column).
- `--time-limit <float>`: Algorithmic time budget in seconds, excluding network I/O (default: `60.0`). The clock resets each time a new incumbent is found, so this controls *time without improvement* rather than total B&B time. Use `-1` to disable it.
- `--node-limit <int>`: Max B&B nodes to explore per Dinkelbach iteration (default: `100000`).
- `--max-in-edges <int>`: Max incoming edges to fetch per node (default: `0`). Applies to both `sim` and `openalex` modes.
- `--gap-tol <float>`: Early-stopping relative gap tolerance for B&B (default: `1e-4`).
- `--dinkelbach-iter <int>`: Max Dinkelbach iterations (default: `50`).
- `--cg-batch-frac <float>`: Fraction of the active-set size to add per pricing round (default: `1.0`).
- `--cg-min-batch <int>`: Minimum columns added per pricing round (default: `50`).
- `--cg-max-batch <int>`: Maximum columns added per pricing round (default: `50`).
- `--tol <float>`: Numerical tolerance for zero-checks (default: `1e-6`).
- `--help`, `-h`: Print the help menu and exit.

Solver variant flags:
- `--bp`: Branch-and-price; uses `--k` and `--kappa`.
- `--avgdeg`: Exact query-anchored average-degree baseline; does not accept `--k` or `--baseline-depth` from the CLI.
- `--bfs`: BFS neighborhood baseline; uses `--bfs-depth`.
- `--conn-greedy`: Connected greedy baseline; uses `--k` and `--baseline-depth`.
- `--conn-avgdeg`: Exact connected average-degree baseline; uses `--baseline-depth`.

---

## Synthetic Graph Generation

### Generate a Graph

```bash
python scripts/synthetic/generate_graph.py --out_dir <output_dir> [options]
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
python scripts/synthetic/evaluate_solver.py --gt <gt_comm.csv> --pred <pred_comm.csv>
```

- `--gt`: Path to the ground truth community CSV (`gt_comm.csv`).
- `--pred`: Path to the predicted community CSV produced by the solver.

Reports precision, recall, F1, and Jaccard similarity.

### Benchmark Solver Variants

Runs the proposed branch-and-price solver and the baselines on the same planted-community query set, then reports per-run metrics and method-level summaries.

```bash
python scripts/synthetic/benchmark_solvers.py \
  --dataset_dir <data/dataset_name> \
  --k <community_size> \
  --output_csv <results.csv>
```

- `--dataset_dir`: Path to a synthetic dataset directory containing `edge.csv` and `gt_comm.csv`.
- `--k`: Target community size used for every run.
- `--methods`: Solver variants to include. Defaults to `bp bp_kappa avgdeg bfs`.
- `--baseline_depth`: BFS depth for `conn_greedy` and `conn_avgdeg`; defaults to `-1` for the full reachable graph.
- `--bfs_depth`: BFS depth for the BFS baseline; defaults to `1`.
- `--time_limit`: Per-run solver time limit for the branch-and-price solver.
- `--node_limit`: Per-run branch-and-bound node limit for the branch-and-price solver.
- `--query_limit`: Optional cap on the number of query nodes taken from `gt_comm.csv`.
- `--output_csv`: Optional path to persist the per-run table.

Method meanings:
- `bp`: proposed branch-and-price solver.
- `bp_kappa`: proposed solver with kappa-connectivity checks.
- `avgdeg`: exact query-anchored average-degree baseline over the full local reachable graph exposed by the oracle.
- `bfs`: BFS baseline with configurable hop depth.

---

## Node Classification (CitationFull Datasets)

Uses the K-Densest community around each query node as its neighborhood for label propagation (majority vote). The shared solver execution and voting logic lives in `scripts/classification/solver_utils.py`.

All scripts below live in `scripts/classification/` and should be run from the project root.

### 1. Prepare Data

Downloads a CitationFull dataset and exports edge and node split files to `data/<dataset>/`. The split is **temporal/inductive**: nodes that have been cited ("foundational" papers) form the training set, while purely-citing ("new") papers are evenly divided into validation and test sets.

```bash
python scripts/classification/prepare_data.py --dataset <dataset_name>
```

- `--dataset`: One of `Cora`, `Cora_ML`, `CiteSeer`, `DBLP`, or `PubMed` (default: `Cora_ML`).

Output files written to `data/<dataset>/`:
- `edge.csv` — directed edge list with columns `source`, `target`.
- `nodes.csv` — node metadata with columns `node_id`, `label`, `train`, `val`, `test`.

### 2. Tune Hyperparameters

For the current solver variants, use `tune_methods.py`. It evaluates branch-and-price, BFS, and exact average-degree configurations, resumes from partial CSVs, and writes both per-run metrics and selected best settings under `exps/classification/<dataset>/tmp_tune/`.

```bash
python scripts/classification/tune_methods.py \
  --dataset <dataset_name> \
  --family all \
  --k_values 3,4,5 \
  --kappa_values 0,1,2 \
  --bfs_depth_min 1 \
  --bfs_depth_max 3 \
  --bp_time_limit_values 60,-1 \
  --optimize f1 \
  --weighting distance
```

- `--family`: Solver family to tune (`all`, `bp`, `avgdeg`, `bfs`; default: `all`).
- `--k_values`: Comma-separated branch-and-price k values; overrides `--k_min`/`--k_max`.
- `--kappa_values`: Comma-separated kappa-connectivity settings.
- `--bp_time_limit_values`: Comma-separated BP time limits; use `-1` to disable the per-call limit.
- `--config_workers`: Number of configurations evaluated in parallel.
- `--workers`: Number of query nodes evaluated in parallel per configuration.
- `--limit_nodes`: Deterministic validation subset size for smoke runs; use `0` for the full split.
- `--seed`: Seed used by `--limit_nodes`.
- `--max_in_edges`: Maximum incoming edges to expose per queried node. The default `0` disables incoming expansion; the reported Cora runs used this default.
- `--force`: Ignore an existing partial CSV and recompute all configurations.

The legacy `tune.py` still sweeps only `k` for the default branch-and-price path:

```bash
python scripts/classification/tune.py --dataset <dataset_name> --k_min <min_k> --k_max <max_k> --k_step <step_k> --optimize <metric> --weighting <weight_strategy>
```

- `--dataset`: Dataset name matching a prepared `data/<dataset>/` directory (default: `Cora`).
- `--k_min`: Minimum k to evaluate (default: `5`).
- `--k_max`: Maximum k to evaluate (default: `25`).
- `--k_step`: Step size for k sweep (default: `5`).
- `--optimize`: Metric to maximize when selecting the best k (`accuracy`, `f1`, `precision`, `recall`; default: `accuracy`).
- `--weighting`: Voting weight strategy (`uniform` or `distance`; default: `uniform`).
- `--bin_path`: Path to the compiled solver binary (default: `./solver/bin/solver`).
- `--workers`: Number of parallel workers (default: number of CPU cores).
- `--limit_nodes`: Deterministic validation subset size for smoke runs; use `0` for the full split.

### 3. Final Evaluation

Evaluates classification on a specific dataset split.

```bash
python scripts/classification/evaluate.py --dataset <dataset_name> --split <split_name> --k <best_k> --weighting <weight_strategy>
```

- `--dataset`: Dataset name matching a prepared `data/<dataset>/` directory (default: `Cora`).
- `--split`: One of `train`, `val`, or `test`.
- `--k`: The optimal k found from tuning.
- `--weighting`: Voting weight strategy (`uniform` or `distance`; default: `uniform`).
- `--bin_path`: Path to the compiled solver binary (default: `./solver/bin/solver`).
- `--workers`: Number of parallel workers (default: number of CPU cores).
- `--limit_nodes`: Deterministic split subset size for smoke runs; use `0` for the full split.
- `--max_in_edges`: Maximum incoming edges to expose per queried node. Keep this at `0` to reproduce the reported outgoing-citation-only Cora experiments.

Reports accuracy, macro precision, recall, F1, and a per-class classification report. When the solver returns a neighborhood with no training nodes (label starvation), a concentric BFS fallback is triggered automatically; the fallback rate is printed at the end of each run. In the reported Cora fixed-method runs the fallback rate is `0%`, so fallback is a dead path rather than part of the observed method behavior.

For the fixed Cora comparisons used in the paper draft, run:

```bash
python scripts/classification/evaluate_fixed_methods.py --dataset Cora --split test
python scripts/classification/evaluate_fixed_methods.py --dataset Cora --split test --subset bfs_depth1_wrong
```

The full split writes to `exps/classification/Cora/test_fixed/`, while the BFS-hard subset writes to `exps/classification/Cora/test_fixed_bfs_depth1_wrong/`.

### 4. Classification Tests

Run syntax checks and the focused unit tests:

```bash
conda run -n dcs python -m py_compile scripts/classification/*.py scripts/synthetic/*.py
conda run -n dcs python -m unittest scripts.classification.test_solver_utils -v
```

The tuning helper tests use only the standard library plus pandas/scikit-learn already required by the classification scripts. Solver quality tests are skipped automatically when optional graph-analysis dependencies are unavailable.

### 5. Baseline: Concentric BFS

Classifies each node by majority voting over the nearest training-set ring reachable via BFS on the undirected graph.

```bash
python scripts/classification/baseline_bfs.py --dataset <dataset_name> --split <split_name> --max_hops <max_hops> --workers <num_workers>
```

- `--dataset`: Dataset name matching a prepared `data/<dataset>/` directory (default: `Cora`).
- `--split`: One of `train`, `val`, or `test`.
- `--max_hops`: Maximum BFS search depth (default: `10`).
- `--workers`: Number of parallel workers (default: number of CPU cores).
