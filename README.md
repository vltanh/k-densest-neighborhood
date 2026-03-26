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

### Example

```bash
./bin/solver --mode openalex --query w3182298045 --k 10 --time-limit 60
```

```
==================================================
K-DENSEST NEIGHBORHOOD (OPENALEX LIVE API)
==================================================
[2026-03-25 21:55:14,971] Init Active Set | Size: 10 | Density: 0.122222
--------------------------------------------------
[2026-03-25 21:55:14,971] === DINKELBACH ITERATION 1 | Lambda = 0.122222 ===
[2026-03-25 21:55:37,620] HTTP Request failed (Attempt 1/3) for https://api.openalex.org/works/W4285719527
    -> cURL Error: No error | HTTP Code: 404
    -> Server Msg: <!doctype html> <html lang=en> <title>404 Not Found</title> <h1>Not Found</h1> <p>The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.</p> 
[2026-03-25 21:55:38,728] HTTP Request failed (Attempt 2/3) for https://api.openalex.org/works/W4285719527
    -> cURL Error: No error | HTTP Code: 404
    -> Server Msg: <!doctype html> <html lang=en> <title>404 Not Found</title> <h1>Not Found</h1> <p>The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.</p> 
[2026-03-25 21:55:40,832] HTTP Request failed (Attempt 3/3) for https://api.openalex.org/works/W4285719527
    -> cURL Error: No error | HTTP Code: 404
    -> Server Msg: <!doctype html> <html lang=en> <title>404 Not Found</title> <h1>Not Found</h1> <p>The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.</p> 
[2026-03-25 21:55:40,833] Blacklisting node W4285719527 due to API error: HTTP fetch failed after max retries.
[2026-03-25 21:55:55,617]     > Incumbent updated at Node 7 | Obj: 93.8667 | Size: 34
[2026-03-25 21:55:56,314]     > Incumbent updated at Node 14 | Obj: 93.9333 | Size: 33
[2026-03-25 21:56:02,997]   -> Iteration Finished in 48.026s
[2026-03-25 21:56:02,997]   -> Nodes Explored : 45 (Total: 45)
[2026-03-25 21:56:02,997]   -> LP Solves      : 414 (Total: 414)
[2026-03-25 21:56:02,997]   Found Solution    : Size: 33 | New Density: 0.211
[2026-03-25 21:56:02,997] === DINKELBACH ITERATION 2 | Lambda = 0.211 ===
[2026-03-25 21:56:08,662]     > Incumbent updated at Node 81 | Obj: 1.4375 | Size: 22
[2026-03-25 21:56:13,748]     > Incumbent updated at Node 155 | Obj: 1.7538 | Size: 20
[2026-03-25 21:56:15,429]     > Incumbent updated at Node 182 | Obj: 1.7784 | Size: 19
[2026-03-25 21:56:20,624]     > Incumbent updated at Node 260 | Obj: 3.4318 | Size: 24
[2026-03-25 21:56:20,645]     > Incumbent updated at Node 263 | Obj: 4.4375 | Size: 22
[2026-03-25 21:56:20,674]     > Incumbent updated at Node 265 | Obj: 5.1458 | Size: 23
[2026-03-25 21:56:20,683]     > Incumbent updated at Node 267 | Obj: 5.4375 | Size: 22
[2026-03-25 21:56:21,031]     > Incumbent updated at Node 278 | Obj: 6.3068 | Size: 21
[2026-03-25 21:56:21,052]     > Incumbent updated at Node 281 | Obj: 6.4375 | Size: 22
[2026-03-25 21:56:21,057]     > Incumbent updated at Node 283 | Obj: 6.7538 | Size: 20
[2026-03-25 21:56:24,795]     > Incumbent updated at Node 336 | Obj: 6.7784 | Size: 19
[2026-03-25 21:56:26,007]     > Incumbent updated at Node 356 | Obj: 7.3068 | Size: 21
[2026-03-25 21:56:34,266]     > Incumbent updated at Node 464 | Obj: 7.4375 | Size: 22
[2026-03-25 21:56:34,277]     > Incumbent updated at Node 466 | Obj: 7.7538 | Size: 20
[2026-03-25 21:56:34,301]     > Incumbent updated at Node 467 | Obj: 9.3068 | Size: 21
[2026-03-25 21:56:35,206]     > Incumbent updated at Node 478 | Obj: 9.7538 | Size: 20
[2026-03-25 21:56:38,223]     > Incumbent updated at Node 515 | Obj: 9.7784 | Size: 19
[2026-03-25 21:57:00,323]     > Incumbent updated at Node 769 | Obj: 10.1458 | Size: 23
[2026-03-25 21:57:00,357]     > Incumbent updated at Node 772 | Obj: 10.4375 | Size: 22
[2026-03-25 21:57:01,192]     > Incumbent updated at Node 786 | Obj: 11.3068 | Size: 21
[2026-03-25 21:57:03,027]     [!] B&B time limit reached.
[2026-03-25 21:57:03,027]   -> Iteration Finished in 60.029s
[2026-03-25 21:57:03,027]   -> Nodes Explored : 765 (Total: 810)
[2026-03-25 21:57:03,027]   -> LP Solves      : 842 (Total: 1256)
[2026-03-25 21:57:03,027]   Found Solution    : Size: 21 | New Density: 0.238
[2026-03-25 21:57:03,027] === DINKELBACH ITERATION 3 | Lambda = 0.238 ===
[2026-03-25 21:57:08,361]     > Incumbent updated at Node 858 | Obj: 0.1429 | Size: 18
[2026-03-25 21:57:09,264]     > Incumbent updated at Node 873 | Obj: 1.2381 | Size: 17
[2026-03-25 21:57:16,965]     > Incumbent updated at Node 972 | Obj: 1.5714 | Size: 19
[2026-03-25 21:57:16,968]     > Incumbent updated at Node 973 | Obj: 2.2381 | Size: 17
[2026-03-25 21:57:19,384]     > Incumbent updated at Node 1009 | Obj: 3.1429 | Size: 18
[2026-03-25 21:57:19,457]     > Incumbent updated at Node 1012 | Obj: 4.5714 | Size: 19
[2026-03-25 21:57:20,303]     > Incumbent updated at Node 1027 | Obj: 5.1429 | Size: 18
[2026-03-25 21:57:20,305]     > Incumbent updated at Node 1028 | Obj: 5.2381 | Size: 17
[2026-03-25 21:57:34,555]     > Incumbent updated at Node 1204 | Obj: 6.2381 | Size: 17
[2026-03-25 21:58:03,085]     [!] B&B time limit reached.
[2026-03-25 21:58:03,085]   -> Iteration Finished in 60.059s
[2026-03-25 21:58:03,085]   -> Nodes Explored : 764 (Total: 1574)
[2026-03-25 21:58:03,085]   -> LP Solves      : 764 (Total: 2020)
[2026-03-25 21:58:03,085]   Found Solution    : Size: 17 | New Density: 0.261
[2026-03-25 21:58:03,085] === DINKELBACH ITERATION 4 | Lambda = 0.261 ===
[2026-03-25 21:58:13,979]     > Incumbent updated at Node 1710 | Obj: 0.3529 | Size: 16
[2026-03-25 21:58:14,322]     > Incumbent updated at Node 1718 | Obj: 1.1838 | Size: 15
[2026-03-25 21:58:34,613]     > Incumbent updated at Node 1998 | Obj: 1.4926 | Size: 14
[2026-03-25 21:59:03,114]     [!] B&B time limit reached.
[2026-03-25 21:59:03,114]   -> Iteration Finished in 60.028s
[2026-03-25 21:59:03,114]   -> Nodes Explored : 876 (Total: 2450)
[2026-03-25 21:59:03,114]   -> LP Solves      : 877 (Total: 2897)
[2026-03-25 21:59:03,114]   Found Solution    : Size: 14 | New Density: 0.269
[2026-03-25 21:59:03,114] === DINKELBACH ITERATION 5 | Lambda = 0.269 ===
[2026-03-25 22:00:03,163]     [!] B&B time limit reached.
[2026-03-25 22:00:03,163]   -> Iteration Finished in 60.049s
[2026-03-25 22:00:03,163]   -> Nodes Explored : 924 (Total: 3374)
[2026-03-25 22:00:03,163]   -> LP Solves      : 924 (Total: 3821)
[2026-03-25 22:00:03,163]   Status            : Converged (No improvement found)
==================================================
OPTIMIZATION STATISTICS
==================================================
B&B Nodes Explored       : 3374
Total LP Solves          : 3821
Columns Generated        : 39
BQP Cuts Added           : 8613
API Queries Made         : 49
Unique Nodes Mapped      : 2636
--------------------------------------------------
TIMING BREAKDOWN
Model Sync Time          : 0.460s
Gurobi LP Time           : 247.124s
Pricing Time             : 0.657s
Separation Time          : 0.785s
--------------------------------------------------
Total Solver Time        : 288.191s
==================================================
FINAL SOLUTION
==================================================
Density                  : 0.269231
Size                     : 14
Nodes:
W4294190884 W2950642167 W2093098337 W2587597110 W2066785331 W3137267085 w3182298045 W2038031975 W2194775991 W2112522387 W4280575696 W2906532173 W2012413612 W2155737223
```

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