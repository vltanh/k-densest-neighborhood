import os
import csv
import json
import math
import asyncio
import tempfile
import httpx
import random
import logging
import urllib.parse
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Tuple, List, Set

from pymincut.pygraph import PyGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


VARIANT_FLAGS = {
    "bp": "--bp",
    "avgdeg": "--avgdeg",
    "bfs": "--bfs",
}


class SolverParams(BaseModel):
    """Shared solver-side hyperparameters across both extraction endpoints."""

    k: int = Field(default=5, ge=0, description="Target subgraph size. BP requires k>=2. AvgDeg/BFS: k>0 grows the optimum to size k, k=0 returns the raw optimum.")
    variant: str = Field(default="bp", description="Solver variant: bp, avgdeg, bfs")
    kappa: int = Field(default=0, ge=0, description="Edge-connectivity threshold (BP only; 0 disables)")
    bfs_depth: int = Field(default=1, ge=0, description="BFS expansion depth (--bfs only)")

    time_limit: Optional[float] = -1.0
    hard_time_limit: Optional[float] = -1.0
    node_limit: Optional[int] = -1
    max_in_edges: Optional[int] = 0
    gap_tol: Optional[float] = -1.0
    dinkelbach_iter: Optional[int] = -1
    cg_batch_frac: Optional[float] = 1.0
    cg_min_batch: Optional[int] = 0
    cg_max_batch: Optional[int] = 50
    tol: Optional[float] = 1e-6
    no_materialize: Optional[bool] = False


class SolverRequest(SolverParams):
    session_id: str = Field(..., description="Unique ID for this extraction run")
    query_node: str = Field(..., pattern=r"^[a-zA-Z0-9]+$", description="OpenAlex ID")


_DATASET_NAME_PATTERN = r"^[A-Za-z0-9_]+$"


def _variant_argv(req: "SolverParams") -> list:
    """Translate a SolverParams payload into solver CLI arguments.

    Only forwards the per-variant flags that main.cpp accepts for the chosen variant;
    forbidden combinations would otherwise cause the binary to exit 1.
    """
    variant = (req.variant or "bp").lower()
    if variant not in VARIANT_FLAGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown variant '{req.variant}'. Allowed: {sorted(VARIANT_FLAGS)}",
        )

    args = [
        VARIANT_FLAGS[variant],
        "--time-limit", str(req.time_limit),
        "--hard-time-limit", str(req.hard_time_limit),
        "--node-limit", str(req.node_limit),
        "--max-in-edges", str(req.max_in_edges),
        "--gap-tol", str(req.gap_tol),
        "--dinkelbach-iter", str(req.dinkelbach_iter),
        "--cg-batch-frac", str(req.cg_batch_frac),
        "--cg-min-batch", str(req.cg_min_batch),
        "--cg-max-batch", str(req.cg_max_batch),
        "--tol", str(req.tol),
    ]

    if variant == "bp":
        if req.k < 2:
            raise HTTPException(
                status_code=400,
                detail="BP requires k>=2 (densest k-subgraph); got k="
                + str(req.k),
            )
        args += ["--k", str(req.k), "--kappa", str(req.kappa)]
        if req.no_materialize:
            args.append("--no-materialize")
    if variant == "avgdeg":
        if req.k > 0:
            args += ["--k", str(req.k)]
    if variant == "bfs":
        args += ["--bfs-depth", str(req.bfs_depth)]
        if req.k > 0:
            args += ["--k", str(req.k)]
    args.append("--compute-qualities")
    return args


app = FastAPI(title="KDensest Subgraph Explorer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_processes = {}


async def cleanup_solver_process(session_id: str, process):
    if process is None:
        return
    if active_processes.get(session_id) is process:
        active_processes.pop(session_id, None)
    if process.returncode is None:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()


@app.post("/api/stop")
async def stop_solver(session_id: str):
    process = active_processes.get(session_id)
    if process is None:
        return {"status": "no process running"}
    try:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        # Dict cleanup is owned by event_stream's finally (cleanup_solver_process).
        return {"status": "terminated"}
    except Exception as e:
        logger.error(f"Error terminating process {session_id}: {e}")
        return {"status": "error", "detail": str(e)}


@app.get("/api/bibtex")
async def get_bibtex(doi: str):
    if not doi or doi == "N/A":
        return {"error": "No DOI available."}
    clean_doi = doi.replace("https://doi.org/", "")
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(
                f"https://doi.org/{clean_doi}",
                headers={"Accept": "application/x-bibtex"},
                follow_redirects=True,
                timeout=10.0,
            )
            if res.status_code == 200:
                return {"bibtex": res.text}
            return {"error": f"Failed: {res.status_code}"}
    except Exception as e:
        logger.warning(f"BibTeX fetch failed for {clean_doi}: {e}")
        return {"error": str(e)}


async def fetch_paper_metadata(
    client: httpx.AsyncClient,
    node_id: str,
    semaphore: asyncio.Semaphore,
    max_in_edges: int,
):
    url = f"https://api.openalex.org/works/{node_id}"
    data_dict = {
        "id": node_id,
        "doi": "N/A",
        "citations": 0,
        "abstract": "No abstract.",
        "title": "Fetch Failed",
        "author": "Unknown",
        "year": "N/A",
        "journal": "N/A",
        "references": [],
        "cited_by": [],
    }

    try:
        async with semaphore:
            res = await client.get(url, timeout=10.0)
            if res.status_code == 200:
                data = res.json()
                data_dict["title"] = data.get("title", "Untitled")
                data_dict["doi"] = data.get("doi", "N/A")
                data_dict["citations"] = data.get("cited_by_count", 0)
                data_dict["year"] = data.get("publication_year", "N/A")
                data_dict["date"] = data.get("publication_date", "N/A")
                data_dict["concepts"] = [
                    c.get("display_name") for c in data.get("concepts", [])[:5]
                ]

                if data.get("authorships"):
                    author = (
                        data["authorships"][0]
                        .get("author", {})
                        .get("display_name", "Unknown")
                    )
                    data_dict["author"] = (
                        f"{author} et al." if len(data["authorships"]) > 1 else author
                    )

                loc = data.get("primary_location")
                if loc and loc.get("source"):
                    data_dict["journal"] = loc["source"].get(
                        "display_name", "Unknown Venue"
                    )

                data_dict["references"] = [
                    r.split("/")[-1] for r in data.get("referenced_works", [])
                ]

                inv_idx = data.get("abstract_inverted_index")
                if inv_idx:
                    max_pos = max([max(pos) for pos in inv_idx.values()])
                    words = [""] * (max_pos + 1)
                    for w, positions in inv_idx.items():
                        for p in positions:
                            words[p] = w
                    data_dict["abstract"] = " ".join(words).strip()

            fetched_in = 0
            cursor = "*"

            while fetched_in < max_in_edges and cursor:
                encoded_cursor = urllib.parse.quote(cursor)
                cite_url = f"https://api.openalex.org/works?filter=cites:{node_id}&select=id&per-page=200&cursor={encoded_cursor}"

                c_res = await client.get(cite_url, timeout=10.0)
                if c_res.status_code != 200:
                    break

                c_data = c_res.json()
                results = c_data.get("results", [])

                if not results:
                    break

                for c in results:
                    if fetched_in >= max_in_edges:
                        break
                    data_dict["cited_by"].append(c["id"].split("/")[-1])
                    fetched_in += 1

                meta = c_data.get("meta", {})
                cursor = meta.get("next_cursor") if fetched_in < max_in_edges else None

    except Exception as e:
        logger.warning(f"Metadata fetch failed for {node_id}: {str(e)}")

    return data_dict


# Resolve the compiled C++ solver relative to this file, so the backend can be
# launched from any working directory (e.g. `python backend/server.py`).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLVER_BIN = os.path.join(PROJECT_ROOT, "solver", "bin", "solver")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
SIM_DATASETS = ("Cora", "PubMed", "DBLP", "CiteSeer", "Cora_ML")


class SimSolverRequest(SolverParams):
    session_id: str = Field(..., description="Unique ID for this extraction run")
    dataset: str = Field(
        ...,
        pattern=_DATASET_NAME_PATTERN,
        description="Dataset name under data/",
    )
    query_node: int = Field(..., ge=0, description="Integer node id")
    ghost_sample_frac: Optional[float] = 0.1
    ghost_max_per_node: Optional[int] = 5


# Cache: dataset -> (nodes_df_dict, undirected adjacency, directed out-adjacency, num_classes, total undirected volume)
_sim_cache: Dict[
    str,
    Tuple[Dict[int, dict], Dict[int, List[int]], Dict[int, List[int]], int, int],
] = {}


def _load_sim_dataset(dataset: str):
    if dataset in _sim_cache:
        return _sim_cache[dataset]
    base = os.path.join(DATA_ROOT, dataset)
    nodes_csv = os.path.join(base, "nodes.csv")
    edge_csv = os.path.join(base, "edge.csv")
    if not (os.path.exists(nodes_csv) and os.path.exists(edge_csv)):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset} missing files")

    nodes: Dict[int, dict] = {}
    max_label = 0
    with open(nodes_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nid = int(row["node_id"])
            lbl = int(row["label"])
            max_label = max(max_label, lbl)
            split = (
                "train"
                if row.get("train", "").lower() == "true"
                else (
                    "val"
                    if row.get("val", "").lower() == "true"
                    else (
                        "test" if row.get("test", "").lower() == "true" else "unlabeled"
                    )
                )
            )
            nodes[nid] = {"id": nid, "label": lbl, "split": split}

    adj: Dict[int, List[int]] = {}
    adj_out: Dict[int, List[int]] = {}
    with open(edge_csv, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            s, t = int(row[0]), int(row[1])
            adj.setdefault(s, []).append(t)
            adj.setdefault(t, []).append(s)
            adj_out.setdefault(s, []).append(t)

    vol_full = sum(len(neigh) for neigh in adj.values())
    _sim_cache[dataset] = (nodes, adj, adj_out, max_label + 1, vol_full)
    return _sim_cache[dataset]


def _internal_ncut(nodes, internal_pairs) -> float:
    """Normalised cut of the induced undirected subgraph: cut * (vol_a + vol_b)
    / (vol_a * vol_b), where (part_a, part_b) is the min-cut and vol_x is the
    internal degree of part_x. Nodes are remapped to int indices so any
    hashable node type works. Returns NaN on PyGraph failure."""
    n = len(nodes)
    if n < 2 or not internal_pairs:
        return 0.0
    try:
        node_list = list(nodes)
        idx = {v: i for i, v in enumerate(node_list)}
        adj_internal: Dict[int, Set[int]] = {i: set() for i in range(n)}
        cluster_edges: List[Tuple[int, int]] = []
        for a, b in internal_pairs:
            ai, bi = idx[a], idx[b]
            cluster_edges.append((ai, bi))
            cluster_edges.append((bi, ai))
            adj_internal[ai].add(bi)
            adj_internal[bi].add(ai)
        part_a, part_b, cut_value = PyGraph(
            list(range(n)), cluster_edges
        ).mincut("noi", "bqueue", False)
        vol_a = sum(len(adj_internal[u]) for u in part_a)
        vol_b = sum(len(adj_internal[u]) for u in part_b)
        if vol_a > 0 and vol_b > 0:
            return cut_value * (vol_a + vol_b) / (vol_a * vol_b)
        return 0.0
    except Exception as e:
        logger.warning(f"int_ncut failed (n={n}, m_int={len(internal_pairs)}): {e}")
        return float("nan")


def _compute_openalex_qualities(core_ids: List[str], core_metadata: List[dict]) -> dict:
    """Directed avg degree, edge density, undirected NCut on the OpenAlex
    core set. cited_by is ignored: u->v with both in S is already in u's
    references list since every S node is queried."""
    S = set(core_ids)
    n = len(S)

    out_refs = {
        data["id"]: {r for r in data.get("references", []) if r != data["id"]}
        for data in core_metadata
    }

    m_dir = 0
    internal_pairs: Set[Tuple[str, str]] = set()
    for u in S:
        for v in out_refs.get(u, ()):
            if v in S:
                m_dir += 1
                if v != u:
                    internal_pairs.add((u, v) if u < v else (v, u))

    return {
        "avg_internal_degree": float(m_dir / n) if n > 0 else 0.0,
        "edge_density": float(m_dir / (n * (n - 1))) if n > 1 else 0.0,
        "int_ncut": float(_internal_ncut(core_ids, internal_pairs)),
    }


async def _spawn_solver_and_stream(session_id: str, cmd: List[str]):
    """Spawn solver subprocess, register in active_processes, stream stdout
    line by line as NDJSON {type:log} packets. Yields (None, line_ndjson) per
    log line and finally (process, None) so the caller can await wait() and
    inspect returncode."""
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    active_processes[session_id] = process
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        yield None, json.dumps(
            {"type": "log", "content": line.decode("utf-8").rstrip()}
        ) + "\n"
    yield process, None


def _compute_sim_qualities(
    core_ids: List[int],
    adj: Dict[int, List[int]],
    adj_out: Dict[int, List[int]],
    vol_full: int,
) -> dict:
    """Directed density, undirected conductance, undirected NCut on the
    induced subgraph defined by core_ids against cached sim adjacency.
    vol_full is the precomputed sum of undirected degrees for the full graph."""
    S = set(core_ids)
    n = len(S)

    m = 0
    for u in S:
        for v in adj_out.get(u, []):
            if v in S and v != u:
                m += 1

    internal_pairs: Set[Tuple[int, int]] = set()
    boundary_undirected_edges = 0
    for u in S:
        for v in adj.get(u, []):
            if v in S:
                if v == u:
                    continue
                internal_pairs.add((u, v) if u < v else (v, u))
            else:
                boundary_undirected_edges += 1

    vol_S = 2 * len(internal_pairs) + boundary_undirected_edges
    cond_denom = min(vol_S, vol_full - vol_S)
    ext_conductance = (
        boundary_undirected_edges / cond_denom if cond_denom > 0 else 0.0
    )

    return {
        "avg_internal_degree": float(m / n) if n > 0 else 0.0,
        "edge_density": float(m / (n * (n - 1))) if n > 1 else 0.0,
        "ext_conductance": float(ext_conductance),
        "int_ncut": float(_internal_ncut(S, internal_pairs)),
    }


@app.get("/api/datasets")
def list_datasets():
    out = []
    for d in SIM_DATASETS:
        base = os.path.join(DATA_ROOT, d)
        if os.path.exists(os.path.join(base, "nodes.csv")) and os.path.exists(
            os.path.join(base, "edge.csv")
        ):
            try:
                nodes, _adj, _adj_out, nclass, _vol = _load_sim_dataset(d)
                out.append({"name": d, "numNodes": len(nodes), "numClasses": nclass})
            except Exception as e:
                logger.warning(f"Failed loading {d}: {e}")
    return {"datasets": out}


@app.post("/api/extract-sim")
async def extract_sim(req: SimSolverRequest):
    if req.dataset not in SIM_DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {req.dataset}")
    bin_path = SOLVER_BIN
    if not os.path.exists(bin_path):
        raise HTTPException(status_code=500, detail="Solver missing.")

    nodes_meta, adj, adj_out, num_classes, vol_full = _load_sim_dataset(req.dataset)
    if req.query_node not in nodes_meta:
        raise HTTPException(
            status_code=400,
            detail=f"Query node {req.query_node} not in dataset (max id {max(nodes_meta)})",
        )

    edge_csv = os.path.join(DATA_ROOT, req.dataset, "edge.csv")
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".csv"
    ) as tmp_file:
        out_csv = tmp_file.name

    async def event_stream():
        process = None
        try:
            cmd = [
                bin_path,
                "--mode", "sim",
                "--input", edge_csv,
                "--query", str(req.query_node),
                "--output", out_csv,
            ] + _variant_argv(req)
            yield json.dumps(
                {
                    "type": "log",
                    "content": f"[sim:{req.dataset}] Executing: {' '.join(cmd)}",
                }
            ) + "\n"

            process = None
            async for proc, log_line in _spawn_solver_and_stream(req.session_id, cmd):
                if log_line is not None:
                    yield log_line
                else:
                    process = proc
            await process.wait()
            rc = process.returncode

            if rc != 0:
                msg = (
                    "Manually aborted."
                    if rc in [-15, -9]
                    else f"Failed (exit code {rc})"
                )
                yield json.dumps({"type": "error", "content": msg}) + "\n"
                return

            core_ids: List[int] = []
            if os.path.exists(out_csv):
                with open(out_csv, "r") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if row:
                            try:
                                core_ids.append(int(row[0].strip()))
                            except ValueError:
                                pass

            if not core_ids:
                yield json.dumps({"type": "error", "content": "Empty subgraph."}) + "\n"
                return

            qualities = _compute_sim_qualities(core_ids, adj, adj_out, vol_full)
            yield json.dumps({"type": "qualities", "content": qualities}) + "\n"

            core_set = set(core_ids)
            nodes_out = []
            for idx, nid in enumerate(core_ids):
                meta = nodes_meta.get(nid, {"label": -1, "split": "unlabeled"})
                deg = len(adj.get(nid, []))
                nodes_out.append(
                    {
                        "id": str(nid),
                        "rawId": nid,
                        "displayNum": idx + 1,
                        "label": meta["label"],
                        "split": meta["split"],
                        "degree": deg,
                        "type": "core",
                        "group": 1,
                    }
                )

            edges_out = []
            seen_directed = set()
            for nid in core_ids:
                for nb in adj_out.get(nid, []):
                    if nb in core_set and nb != nid:
                        key = (nid, nb)
                        if key in seen_directed:
                            continue
                        seen_directed.add(key)
                        edges_out.append(
                            {"source": str(nid), "target": str(nb), "type": "core"}
                        )

            # Ghost frontier sampling. Direction preserved from the original
            # directed graph: if nid -> nb, ghost is target; if nb -> nid,
            # ghost is source.
            ghost_set = set()
            frac = max(0.0, min(1.0, req.ghost_sample_frac or 0.0))
            cap = max(0, req.ghost_max_per_node or 0)
            for nid in core_ids:
                neighbors_outside = [n for n in adj.get(nid, []) if n not in core_set]
                if not neighbors_outside or frac <= 0 or cap <= 0:
                    continue
                sample_size = min(cap, max(1, int(len(neighbors_outside) * frac)))
                sample = random.sample(
                    neighbors_outside, min(sample_size, len(neighbors_outside))
                )
                out_set_nid = set(adj_out.get(nid, []))
                for nb in sample:
                    g_id = f"ghost_{nb}"
                    if g_id not in ghost_set:
                        gm = nodes_meta.get(nb, {"label": -1, "split": "unlabeled"})
                        nodes_out.append(
                            {
                                "id": g_id,
                                "rawId": nb,
                                "label": gm["label"],
                                "split": gm["split"],
                                "type": "ghost",
                                "group": 2,
                            }
                        )
                        ghost_set.add(g_id)
                    if nb in out_set_nid:
                        edges_out.append(
                            {"source": str(nid), "target": g_id, "type": "ghost"}
                        )
                    else:
                        edges_out.append(
                            {"source": g_id, "target": str(nid), "type": "ghost"}
                        )

            yield json.dumps(
                {
                    "type": "meta",
                    "content": {
                        "dataset": req.dataset,
                        "numClasses": num_classes,
                        "queryNode": req.query_node,
                        "queryLabel": nodes_meta[req.query_node]["label"],
                        "querySplit": nodes_meta[req.query_node]["split"],
                    },
                }
            ) + "\n"
            yield json.dumps(
                {"type": "result", "content": {"nodes": nodes_out, "edges": edges_out}}
            ) + "\n"
            yield json.dumps({"type": "log", "content": "Graph built!"}) + "\n"

        finally:
            await cleanup_solver_process(req.session_id, process)
            if os.path.exists(out_csv):
                os.remove(out_csv)

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/api/extract")
async def extract_subgraph(req: SolverRequest):
    bin_path = SOLVER_BIN
    if not os.path.exists(bin_path):
        raise HTTPException(status_code=500, detail="Solver missing.")
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".csv"
    ) as tmp_file:
        out_csv = tmp_file.name

    async def event_stream():
        process = None
        try:
            cmd = [
                bin_path,
                "--mode", "openalex",
                "--query", req.query_node,
                "--output", out_csv,
            ] + _variant_argv(req)
            yield json.dumps(
                {"type": "log", "content": f"Executing: {' '.join(cmd)}"}
            ) + "\n"

            process = None
            async for proc, log_line in _spawn_solver_and_stream(req.session_id, cmd):
                if log_line is not None:
                    yield log_line
                else:
                    process = proc
            await process.wait()
            rc = process.returncode

            if rc != 0:
                msg = (
                    "Manually aborted."
                    if rc in [-15, -9]
                    else f"Failed (exit code {rc})"
                )
                yield json.dumps({"type": "error", "content": msg}) + "\n"
                return

            yield json.dumps(
                {
                    "type": "log",
                    "content": "Solved. Fetching comprehensive OpenAlex metadata...",
                }
            ) + "\n"

            core_ids = []
            if os.path.exists(out_csv):
                with open(out_csv, "r") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if row:
                            core_ids.append(row[0].strip())

            if not core_ids:
                yield json.dumps({"type": "error", "content": "Empty subgraph."}) + "\n"
                return

            semaphore = asyncio.Semaphore(15)
            async with httpx.AsyncClient(
                headers={"User-Agent": "KDensestGUI"}
            ) as client:
                core_metadata = await asyncio.gather(
                    *[
                        fetch_paper_metadata(client, nid, semaphore, req.max_in_edges)
                        for nid in core_ids
                    ]
                )

            try:
                qualities = _compute_openalex_qualities(core_ids, core_metadata)
                yield json.dumps(
                    {"type": "qualities", "content": qualities}
                ) + "\n"
            except Exception as e:
                logger.warning(f"openalex qualities computation failed: {e}")

            nodes, edges = [], []
            core_id_set = set(core_ids)
            ghost_set = set()

            for idx, data in enumerate(core_metadata):
                nodes.append(
                    {
                        "id": data["id"],
                        "displayNum": idx + 1,
                        "doi": data["doi"],
                        "citations": data["citations"],
                        "abstract": data["abstract"],
                        "title": data["title"],
                        "author": data["author"],
                        "year": data["year"],
                        "journal": data["journal"],
                        "type": "core",
                        "group": 1,
                    }
                )

                # OUTGOING frontier
                out_refs = [
                    r
                    for r in data["references"]
                    if r not in core_id_set and r != data["id"]
                ]
                out_sample_size = max(1, int(len(out_refs) * 0.10)) if out_refs else 0
                out_sample = random.sample(out_refs, out_sample_size)
                for ref_id in out_sample:
                    g_id = f"ghost_{ref_id}"
                    if g_id not in ghost_set:
                        nodes.append({"id": g_id, "type": "ghost", "group": 2})
                        ghost_set.add(g_id)
                    edges.append(
                        {"source": data["id"], "target": g_id, "type": "ghost"}
                    )

                # INCOMING frontier
                in_refs = [
                    r
                    for r in data["cited_by"]
                    if r not in core_id_set and r != data["id"]
                ]
                in_sample_size = max(1, int(len(in_refs) * 0.10)) if in_refs else 0
                in_sample = in_refs[:in_sample_size]
                for ref_id in in_sample:
                    g_id = f"ghost_{ref_id}"
                    if g_id not in ghost_set:
                        nodes.append({"id": g_id, "type": "ghost", "group": 2})
                        ghost_set.add(g_id)
                    edges.append(
                        {"source": g_id, "target": data["id"], "type": "ghost"}
                    )

                # CORE edges
                for ref_id in data["references"]:
                    if ref_id in core_id_set and ref_id != data["id"]:
                        edges.append(
                            {"source": data["id"], "target": ref_id, "type": "core"}
                        )

            yield json.dumps(
                {"type": "result", "content": {"nodes": nodes, "edges": edges}}
            ) + "\n"
            yield json.dumps({"type": "log", "content": "Graph built!"}) + "\n"

        finally:
            await cleanup_solver_process(req.session_id, process)
            if os.path.exists(out_csv):
                os.remove(out_csv)

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
