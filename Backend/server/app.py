"""FastAPI service exposing GPU Personalized PageRank (graph_link) to the React frontend.

Mirrors the data contract in Frontend/src/lib/ppr/types.ts. JS `Map<NodeId, number>`
fields (ranks, degrees.in/out) are returned as plain JSON objects; the frontend
adapter rebuilds them into Maps.

Run on a CUDA host (graph_link must be importable — see ../graph_link/CLAUDE.md):

    pip install fastapi 'uvicorn[standard]'
    cd Backend/server
    uvicorn app:app --host 0.0.0.0 --port 8000

The Vite dev server proxies /api -> this service (Frontend/vite.config.ts). When the
backend runs on a remote GPU box, tunnel its port to the dev machine, e.g.:

    ssh -N -L 8000:localhost:8000 dayanb@<gpu-host>
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import graph_link

app = FastAPI(title="PPR Analyzer backend", version="1.0")

# Dev CORS — the Vite proxy keeps this same-origin, but allow direct calls too.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- request models (mirror Frontend/src/lib/ppr/types.ts) ----
class GraphNode(BaseModel):
    id: str


class GraphEdge(BaseModel):
    source: str
    target: str
    weight: float = 1.0


class Graph(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    format: str | None = None
    warnings: list[str] = Field(default_factory=list)


class PPRParams(BaseModel):
    alpha: float = 0.85
    maxIter: int = 100
    tolerance: float = 1e-6
    topX: int = 10
    seeds: list[str] = Field(default_factory=list)


class PPRRequest(BaseModel):
    graph: Graph
    params: PPRParams


@app.get("/api/health")
def health() -> dict:
    cuda = torch.cuda.is_available()
    return {
        "status": "ok",
        "cuda": cuda,
        "device": torch.cuda.get_device_name(0) if cuda else None,
    }


@app.post("/api/ppr")
def run_ppr(req: PPRRequest) -> dict:
    if not torch.cuda.is_available():
        raise HTTPException(status_code=503, detail="CUDA device not available on the server.")

    g, p = req.graph, req.params
    ids = [n.id for n in g.nodes]
    N = len(ids)
    if N == 0:
        raise HTTPException(status_code=422, detail="Graph has no nodes.")

    index_of = {nid: i for i, nid in enumerate(ids)}

    # Require at least one valid seed — this is *Personalized* PageRank.
    seed_idx = [index_of[s] for s in p.seeds if s in index_of]
    if not seed_idx:
        raise HTTPException(
            status_code=422,
            detail="Personalized PageRank requires at least one seed node.",
        )

    # Degrees are plain edge counts (matches Frontend/src/lib/ppr/degrees.ts).
    in_deg = {nid: 0.0 for nid in ids}
    out_deg = {nid: 0.0 for nid in ids}
    for e in g.edges:
        if e.source in out_deg:
            out_deg[e.source] += 1.0
        if e.target in in_deg:
            in_deg[e.target] += 1.0

    # Adjacency for the transition matrix: A[target, source] = weight, so
    # column-normalizing by each source's out-weight gives a column-stochastic A
    # (mass flows source -> target). normalize_transition_matrix() handles
    # dangling sources by spreading their column uniformly over all nodes.
    rows, cols, data = [], [], []
    for e in g.edges:
        si = index_of.get(e.source)
        ti = index_of.get(e.target)
        if si is None or ti is None or not np.isfinite(e.weight) or e.weight <= 0:
            continue
        rows.append(ti)
        cols.append(si)
        data.append(float(e.weight))

    A = sp.csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
        ),
        shape=(N, N),
    )
    A_norm = graph_link.normalize_transition_matrix(A)

    pbr = graph_link.csr_to_pbr(A_norm, block_rows=8, block_cols=8, min_nnz_per_block=1).to("cuda")
    sources = torch.tensor(seed_idx, dtype=torch.int32, device="cuda")

    X, iterations, converged, history = graph_link.run_personalized_pagerank(
        pbr,
        sources,
        alpha=p.alpha,
        max_iterations=p.maxIter,
        tolerance=p.tolerance,
        return_history=True,
    )

    # Uniform personalization over the seed set = mean of the per-seed columns
    # (PPR is linear in the teleport vector). Renormalize against fp drift.
    r = X.mean(dim=1)
    total = float(r.sum().item())
    if total > 0:
        r = r / total
    r = r.detach().cpu().numpy()

    return {
        "ranks": {ids[i]: float(r[i]) for i in range(N)},
        "iterations": int(iterations),
        "converged": bool(converged),
        "convergenceHistory": [float(h) for h in history],
        "degrees": {"in": in_deg, "out": out_deg},
    }
