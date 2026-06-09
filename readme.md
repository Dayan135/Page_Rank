# Page_Rank — GPU Personalized PageRank

A GPU-accelerated **Personalized PageRank (PPR)** engine built around a custom
sparse-matrix format, **Packed Block Representation (PBR)**. Many real-world
graphs (social, web) have block-sparse adjacency matrices; PBR exploits that
structure to beat the generic cuSPARSE SpMM primitive, and a CSR "remainder"
kernel mops up the nonzeros that don't fit the block pattern.

The repo ships the kernels, a Python API, an HTTP service, and a React app that
lets you upload a graph, run PPR on the GPU, and explore the results.

> ⚠️ **An NVIDIA GPU (CUDA) is required to run.** The backend builds and executes
> custom CUDA kernels — there is no CPU-only fallback for the server, and building
> or running it without an NVIDIA GPU will fail. (The frontend's in-browser mock
> needs no GPU, but it is not the real engine.)

## Architecture

```
Frontend/  React SPA ──HTTP──▶ Backend/server/  FastAPI
                                      │  graph_link.run_personalized_pagerank
                                      ▼
                          Backend/graph_link/  PBR + CUDA kernels (pip package)
```

| Layer | Path | What it is |
|---|---|---|
| Kernels + API | [`Backend/graph_link/`](Backend/graph_link/) | PBR format, CUDA SpMM (block + CSR remainder), PPR power iteration, pybind11, Python API |
| HTTP service | [`Backend/server/`](Backend/server/) | FastAPI `POST /api/ppr` wrapping `run_personalized_pagerank` |
| Tests | [`Tests/`](Tests/) | SpMM/PPR correctness + performance benchmarks |
| Web app | [`Frontend/`](Frontend/) | Upload → Configure → Results wizard, plus a Learn page on the math |

The PBR SpMM path is **manually synced** from the research repo
(`research_spdmm/fem`, branch `feat/pbr-csr-remainder`).

## Requirements

An **NVIDIA GPU with CUDA** is required to build and run the backend — compilation
and tests fail without one. On the BGU cluster, CUDA comes from the module system
(`module load cuda/12.5`) — see [`CLAUDE.md`](CLAUDE.md) for the exact environment
and the [server run guide](CLAUDE.md#running-the-ppr-server-gpu-cluster).

## Quick start

### 1. Backend (on a CUDA host)

```bash
# build the CUDA extension into the active conda env (e.g. pageRank_312)
module load cuda/12.5                       # cluster: no /usr/local/cuda
pip install -e Backend/graph_link/ --no-build-isolation

# run the HTTP service
pip install -r Backend/server/requirements.txt
cd Backend/server && uvicorn app:app --host 0.0.0.0 --port 8000
```

Health check: `curl localhost:8000/api/health` → `{"status":"ok","cuda":true,...}`.

### 2. Frontend

```bash
cd Frontend
npm install
npm run dev        # http://localhost:5173
```

Vite proxies `/api` → `http://localhost:8000` by default (override with
`VITE_PPR_TARGET`). If the backend runs on a remote GPU box, tunnel the port:

```bash
ssh -N -L 8000:localhost:8000 <user>@<gpu-host>
```

For a static build, set `VITE_PPR_API` to the backend's absolute URL.

## Using it

Upload a graph (edge list / COO triplets / adjacency CSV), pick **at least one
seed node** (Personalized PageRank is relative to the seeds), tune α / iterations
/ tolerance, and compute. Results come back as ranked cards, a sortable table,
charts (rank distribution, convergence, degree histogram), and an interactive
network graph. The **Learn** page explains the underlying math.

## Tests

```bash
pytest Tests/test_spmm.py -v            # SpMM vs scipy
pytest Tests/test_ppr_accuracy.py -v    # PPR vs NetworkX
cd Frontend && npm run test             # Vitest (frontend)
```

## The PBR format in one paragraph

A sparse matrix is tiled into fixed blocks (2×2, 4×4, 8×8). Each block that has
at least `min_nnz_per_block` nonzeros is stored compactly as a 64-bit bitmask
(`std::bitset<64>`) plus its packed values; the GPU kernel reconstructs the
multiply from the bitmask. Blocks too sparse to be worth a tile are dropped into
a **CSR remainder** handled by a separate warp-per-row kernel. Both kernels run
on independent CUDA streams and accumulate into the same output.

## More detail

Per-area developer guides live in `CLAUDE.md` files:
[root](CLAUDE.md) · [graph_link](Backend/graph_link/CLAUDE.md) ·
[Tests](Tests/CLAUDE.md) · [Frontend](Frontend/CLAUDE.md) ·
[server](Backend/server/README.md).
