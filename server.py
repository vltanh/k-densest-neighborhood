import os
import csv
import json
import asyncio
import tempfile
import httpx
import random
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

class SolverRequest(BaseModel):
    query_node: str = Field(..., description="OpenAlex ID")
    k: int = Field(..., ge=2, description="Target subgraph size")
    time_limit: Optional[float] = 600.0
    node_limit: Optional[int] = 100000
    gap_tol: Optional[float] = 1e-4
    dinkelbach_iter: Optional[int] = 50
    cg_batch_frac: Optional[float] = 0.1
    cg_min_batch: Optional[int] = 5
    cg_max_batch: Optional[int] = 50
    tol: Optional[float] = 1e-6

app = FastAPI(title="KDensest Subgraph Explorer API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

current_process = None

@app.post("/api/stop")
async def stop_solver():
    global current_process
    if current_process:
        try:
            current_process.terminate()
            return {"status": "terminated"}
        except Exception as e:
            return {"status": "error", "detail": str(e)}
    return {"status": "no process running"}

@app.get("/api/bibtex")
async def get_bibtex(doi: str):
    if not doi or doi == "N/A": return {"error": "No DOI available."}
    clean_doi = doi.replace("https://doi.org/", "")
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"https://doi.org/{clean_doi}", headers={"Accept": "application/x-bibtex"}, follow_redirects=True, timeout=10.0)
            if res.status_code == 200: return {"bibtex": res.text}
            return {"error": f"Failed: {res.status_code}"}
    except Exception as e: return {"error": str(e)}

async def fetch_paper_metadata(client: httpx.AsyncClient, node_id: str):
    url = f"https://api.openalex.org/works/{node_id}"
    data_dict = {
        "id": node_id, "doi": "N/A", "citations": 0, "abstract": "No abstract.",
        "title": "Fetch Failed", "author": "Unknown", "year": "N/A", "journal": "N/A",
        "references": [], "cited_by": []
    }
    
    try:
        # 1. Fetch main metadata (Outgoing edges)
        res = await client.get(url, timeout=10.0)
        if res.status_code == 200:
            data = res.json()
            data_dict["title"] = data.get("title", "Untitled")
            data_dict["doi"] = data.get("doi", "N/A")
            data_dict["citations"] = data.get("cited_by_count", 0)
            data_dict["year"] = data.get("publication_year", "N/A")
            
            if data.get("authorships"):
                author = data["authorships"][0].get("author", {}).get("display_name", "Unknown")
                data_dict["author"] = f"{author} et al." if len(data["authorships"]) > 1 else author
                
            loc = data.get("primary_location")
            if loc and loc.get("source"):
                data_dict["journal"] = loc["source"].get("display_name", "Unknown Venue")

            data_dict["references"] = [r.split("/")[-1] for r in data.get("referenced_works", [])]

            inv_idx = data.get("abstract_inverted_index")
            if inv_idx:
                max_pos = max([max(pos) for pos in inv_idx.values()])
                words = [""] * (max_pos + 1)
                for w, positions in inv_idx.items():
                    for p in positions: words[p] = w
                data_dict["abstract"] = " ".join(words).strip()
                
        # 2. Fetch Incoming Citations (Lightweight query)
        cite_url = f"https://api.openalex.org/works?filter=cites:{node_id}&select=id&per-page=50"
        c_res = await client.get(cite_url, timeout=5.0)
        if c_res.status_code == 200:
            c_data = c_res.json()
            data_dict["cited_by"] = [c["id"].split("/")[-1] for c in c_data.get("results", [])]
            
    except Exception:
        pass 
    
    return data_dict

@app.post("/api/extract")
async def extract_subgraph(req: SolverRequest):
    bin_path = os.path.join(os.getcwd(), "bin", "solver")
    if not os.path.exists(bin_path): raise HTTPException(status_code=500, detail="Solver missing.")
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv') as tmp_file: out_csv = tmp_file.name

    async def event_stream():
        global current_process
        try:
            cmd = [bin_path, "--mode", "openalex", "--query", req.query_node, "--k", str(req.k), "--output", out_csv, "--time-limit", str(req.time_limit), "--node-limit", str(req.node_limit), "--gap-tol", str(req.gap_tol), "--dinkelbach-iter", str(req.dinkelbach_iter), "--cg-batch-frac", str(req.cg_batch_frac), "--cg-min-batch", str(req.cg_min_batch), "--cg-max-batch", str(req.cg_max_batch), "--tol", str(req.tol)]
            yield json.dumps({"type": "log", "content": f"Executing: {' '.join(cmd)}"}) + "\n"

            current_process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            while True:
                line = await current_process.stdout.readline()
                if not line: break
                yield json.dumps({"type": "log", "content": line.decode('utf-8').rstrip()}) + "\n"

            await current_process.wait()
            rc = current_process.returncode
            current_process = None 
            if rc != 0:
                msg = "Manually aborted." if rc in [-15, -9] else f"Failed (exit code {rc})"
                yield json.dumps({"type": "error", "content": msg}) + "\n"
                return

            yield json.dumps({"type": "log", "content": "Solved. Fetching comprehensive OpenAlex metadata..."}) + "\n"

            core_ids = []
            if os.path.exists(out_csv):
                with open(out_csv, 'r') as f:
                    reader = csv.reader(f)
                    next(reader, None) 
                    for row in reader:
                        if row: core_ids.append(row[0].strip())

            if not core_ids:
                yield json.dumps({"type": "error", "content": "Empty subgraph."}) + "\n"
                return

            async with httpx.AsyncClient(headers={"User-Agent": "KDensestGUI"}) as client:
                core_metadata = await asyncio.gather(*[fetch_paper_metadata(client, nid) for nid in core_ids])

            nodes, edges = [], []
            core_id_set = set(core_ids)
            ghost_set = set() # Track unique ghosts
            
            for idx, data in enumerate(core_metadata):
                nodes.append({"id": data["id"], "displayNum": idx + 1, "doi": data["doi"], "citations": data["citations"], "abstract": data["abstract"], "title": data["title"], "author": data["author"], "year": data["year"], "journal": data["journal"], "type": "core", "group": 1})

                # Process OUTGOING frontier
                out_refs = [r for r in data["references"] if r not in core_id_set and r != data["id"]]
                out_sample_size = min(15, max(3, int(len(out_refs) * 0.15)))
                out_sample = random.sample(out_refs, min(len(out_refs), out_sample_size))
                for ref_id in out_sample:
                    g_id = f"ghost_{ref_id}"
                    if g_id not in ghost_set:
                        nodes.append({"id": g_id, "type": "ghost", "group": 2})
                        ghost_set.add(g_id)
                    edges.append({"source": data["id"], "target": g_id, "type": "ghost"})
                
                # Process INCOMING frontier
                in_refs = [r for r in data["cited_by"] if r not in core_id_set and r != data["id"]]
                in_sample_size = min(15, max(3, int(len(in_refs) * 0.15)))
                in_sample = random.sample(in_refs, min(len(in_refs), in_sample_size))
                for ref_id in in_sample:
                    g_id = f"ghost_{ref_id}"
                    if g_id not in ghost_set:
                        nodes.append({"id": g_id, "type": "ghost", "group": 2})
                        ghost_set.add(g_id)
                    edges.append({"source": g_id, "target": data["id"], "type": "ghost"})

                # Process CORE edges
                for ref_id in data["references"]:
                    if ref_id in core_id_set and ref_id != data["id"]:
                        edges.append({"source": data["id"], "target": ref_id, "type": "core"})

            yield json.dumps({"type": "result", "content": {"nodes": nodes, "edges": edges}}) + "\n"
            yield json.dumps({"type": "log", "content": "Graph built!"}) + "\n"

        finally:
            if os.path.exists(out_csv): os.remove(out_csv)

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)