# Repository Guidelines

## Project Shape

This repository combines four surfaces that share one solver contract:

- `solver/src/`: C++17 solver binary. `main.cpp` owns CLI validation and JSON/stream output. `solver.cpp` implements branch-and-price, Dinkelbach iterations, lazy frontier materialization, bounded joint pricing, BQP triangle cuts, rooted-kappa connectivity cuts/checks, and incumbent telemetry. `average_degree_solver.*`, `bfs_solver.*`, and `grow_to_k.*` are experiment baselines/post-processing.
- `scripts/_solver_runner.py`: Python single source of truth for invoking `solver/bin/solver` in simulation mode. It requires `--emit-json`; missing JSON or nonzero exits are hard failures.
- `scripts/classification/`: CitationFull-style data prep, per-query record generation, classification voting, quality metrics, and aggregation. Active workflow is `prepare_data.py`, `sweep_cluster_quality.py`, and the `aggregate_*.py` scripts.
- `backend/server.py` and `frontend/src/`: FastAPI plus React/Vite UI. The backend wraps the same solver binary and streams NDJSON packets to the frontend.

Generated artifacts live outside source:

- `data/`: prepared datasets and split metadata. Do not write experiment results here.
- `exps/`: experiment outputs, per-query `records.ndjson`, aggregate CSVs, and generated tables.
- `docs/paper/` and `docs/slides/`: paper and slide sources. LaTeX aux/log files are ignored; avoid committing generated side files unless explicitly requested.

## Current Solver Contract

Build the solver before any solver-backed test or experiment:

```bash
./solver/build.sh
```

The binary supports:

- `--bp`: branch-and-price. Requires `--k >= 2`, accepts `--kappa`, time/node/gap/Dinkelbach/CG controls, `--no-materialize`, `--gurobi-seed`, and optional `--stream-incumbents`. Omitting all variant flags still defaults to BP for legacy CLI compatibility, but scripts should pass `--bp` explicitly.
- `--avgdeg`: query-anchored average-degree baseline on the materialized local graph. Optional `--k` triggers grow-to-k when the returned optimum is smaller than requested. In the paper-style sweeps, `main.cpp` calls it with depth `-1`, so it explores all outgoing-reachable nodes exposed by the oracle convention.
- `--bfs`: BFS baseline. Uses `--bfs-depth`; optional `--k` triggers grow-to-k. The implementation queries layers `0..d`, returns the closed ball through depth `d`, and keeps the queried final layer's discoveries as grow-to-k candidates.

Important output contracts:

- `--emit-json` prints one `JSON_RESULT:<payload>` line unless `--json-output` is used. Python experiment code depends on structured JSON and intentionally fails if it is absent.
- `--json-output <path>` implies JSON emission and writes the raw payload to a file instead of stdout; callers clear stale files before invoking the binary.
- `--stream-incumbents` emits `INCUMBENT_JSON:<payload>` lines for BP incumbent updates. The backend maps these to `{type:"incumbent"}` NDJSON packets and the frontend telemetry panel consumes them.
- `--compute-qualities` can issue extra oracle queries after solving. Distinguish solve-query counts from quality-extra queries when interpreting records.

Gurobi is required for BP and for tests that exercise the real binary. In this environment use:

```bash
GRB_LICENSE_FILE=/home/vltanh/gurobi.lic
```

`solver/CMakeLists.txt` also requires `GUROBI_HOME`; it embeds a `SOLVER_BUILD_ID` of `<short-git-sha>_<UTC-build-time>` into JSON output. Classification sweep cache hashes include the solver binary hash, so rebuilding can intentionally invalidate records unless `--code-hash-override` is used for a deliberate resume of compatible records.

## Backend and Frontend Notes

Backend endpoints:

- `POST /api/extract`: OpenAlex live oracle.
- `POST /api/extract-sim`: local `data/<dataset>/edge.csv` simulation oracle.
- `POST /api/stop?session_id=...`: terminate the active solver subprocess for that session.
- `GET /api/datasets`: list prepared local simulation datasets.

The backend always forwards variant-safe CLI flags through `_variant_argv`; keep it aligned with `solver/src/main.cpp`. Streaming packets are newline-delimited JSON. Ordinary solver output is `{type:"log"}`, final graph data is `{type:"result"}`, post-solve metrics are `{type:"qualities"}`, and live BP incumbent updates are `{type:"incumbent"}`.

Frontend conventions:

- React + Vite app under `frontend/src/`.
- `useSubgraphExtractor.js` owns the streaming protocol and solver request shape.
- `telemetryParser.js` parses human-readable log lines; structured incumbent packets are handled separately.
- Components use PascalCase; hooks use `useX.js`; keep solver variant controls synchronized with `frontend/src/constants.js`.

## Classification Experiments

Use the `dcs` conda environment for classification scripts; the base Python may lack `networkx`, `pymincut`, or other graph dependencies.

```bash
conda run -n dcs python -m py_compile scripts/classification/*.py scripts/synthetic/*.py scripts/*.py
conda run -n dcs python -m unittest scripts.test_solver_runner scripts.classification.test_solver_utils -v
```

Active data prep:

```bash
conda run -n dcs python scripts/classification/prepare_data.py --dataset Cora_ML --source existing
```

The current split schema is hash-pinned in `data/<dataset>/split_meta.json`. The expected query pool is:

- pure source nodes,
- in the undirected support induced by nodes reachable from the query via outgoing edges, the bridge-free component containing the query has size at least 5.

Validation and test are label-stratified 50/50 from that pool under seed 42; odd label buckets are balanced by the current val/test sizes. `assert_split_meta_matches()` rejects stale `nodes.csv`, `edge.csv`, schema, pool criterion, split strategy, pool component threshold, or edge hashes. Trust `split_meta.json` and source code over older prose if counts disagree.

Current Cora_ML metadata at the time of this guide:

- 2,995 nodes and 8,416 directed edges.
- pure-source candidates: 1,249.
- query pool size 578.
- val 289, test 289, train 2,417.
- BFS-depth-1-wrong hard subset size 44.

Active sweep command for the paper-style Cora_ML grid:

```bash
GRB_LICENSE_FILE=/home/vltanh/gurobi.lic conda run -n dcs python scripts/classification/sweep_cluster_quality.py \
  --dataset Cora_ML \
  --family all \
  --bp-k 3,4,5 \
  --bp-kappa 0,1,2 \
  --bfs-depth 1 \
  --seeds 42 \
  --solver-time-limit 60 \
  --hard-time-limit 300 \
  --max-workers 8
```

The sweep writes resumable per-query records under `exps/classification/<dataset>/cluster_quality/<method>/<hash>/records.ndjson`. Deterministic methods use `seed=None`; BP records are per Gurobi seed. Effective cache params include classifier weighting, fallback-hop limit, forbidden val/test nodes, quality schema version, solver binary hash, and solver config such as `max_in_edges=0`. Completed solver outputs can be reused across split-hash changes when the graph, method, solver-facing params, solver config, and BP seed match; prediction labels and qualities are recomputed for the current split before writing fresh records. Aggregators validate split hashes and completeness by default. Use `--allow-partial` only for interim inspection of incomplete sweeps.

Classification records use inverse-distance voting over training-labelled returned nodes. Distances are measured in the undirected local support with validation/test nodes excluded as intermediates. If no training-labelled returned node is available, a directed fallback BFS searches up to 10 hops and then falls back to the global training-majority label; the current Cora_ML aggregate records have zero fallback uses.

Aggregation entry points:

- `aggregate_experiment.py`: broad, table-agnostic CSVs.
- `aggregate_to_latex.py`: render paper-style table snippets from wide aggregate CSVs.
- `aggregate_cluster_quality.py`: intrinsic quality summaries.
- `aggregate_cost.py`: cost summaries.
- `aggregate_classification.py` and `paired_bootstrap_ties.py`: classification and paired bootstrap views.

Legacy scripts such as `tune_methods.py`, `evaluate.py`, `baseline_bfs.py`, and the old tune/evaluate split workflow are not active. Prefer `rg --files scripts` and the active scripts above.

## Testing Checklist

There is no single universal suite. Choose checks by surface:

- C++ solver: `./solver/build.sh`.
- Solver/Python contract: `conda run -n dcs python -m unittest scripts.test_solver_runner -v`.
- Classification utilities: `conda run -n dcs python -m unittest scripts.classification.test_solver_utils -v`.
- Python syntax: `conda run -n dcs python -m py_compile scripts/classification/*.py scripts/synthetic/*.py scripts/*.py`.
- Frontend lint/build: `cd frontend && npm run lint` and `cd frontend && npm run build`.

For solver smoke tests, use small local CSVs and `--emit-json` before launching full Cora_ML sweeps. Full BP grids can take hours; rely on resumable records and check counts with `find exps/classification/<dataset>/cluster_quality -name records.ndjson -exec wc -l {} \;`.

## Coding Style and Safety

- C++ uses conventional headers/sources in `solver/src/`, `snake_case` filenames, PascalCase classes, and short method-specific CLI flags.
- Python scripts use module-level functions, `snake_case`, and explicit CSV/NDJSON files. Keep cache keys and split hashes deterministic.
- Frontend uses existing React component structure, D3 graph rendering, and lucide icons.
- Do not silently swallow solver failures in experiment code; bad solver JSON should fail loudly rather than cache empty communities.
- Do not commit `data/`, `exps/`, `solver/bin/`, `solver/build/`, `frontend/dist/`, `frontend/node_modules/`, or LaTeX aux/log artifacts unless explicitly requested.

## Commit and PR Style

Recent commits use short imperative summaries with simple scopes, for example:

- `solver: stream BP incumbent telemetry`
- `scripts: harden solver record caching`
- `scripts: hash-pin split metadata`

Keep commits focused by subsystem. PRs should include a concise summary, commands run, affected modules, and screenshots for frontend changes.
