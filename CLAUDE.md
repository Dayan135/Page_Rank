# Page_Rank — Project Guide for Claude

## What this project is

A GPU-accelerated **Personalized PageRank (PPR)** engine built around a custom sparse matrix format called **Packed Block Representation (PBR)**. The core insight is that many real-world graphs (social networks, web graphs) have block-sparse adjacency matrices — PBR exploits this structure to outperform the generic cuSPARSE SpMM primitive for block-structured matrices.

The project has four layers:
1. **`Backend/graph_link/`** — a pip-installable Python package containing custom CUDA kernels (C++/CUDA), pybind11 bindings, and a high-level Python API. The PBR SpMM path is manually synced from `research_spdmm/fem` (branch `feat/pbr-csr-remainder`): vectorized block kernel + a **CSR remainder** kernel.
2. **`Backend/server/`** — a FastAPI service wrapping `graph_link.run_personalized_pagerank`, consumed by the frontend (`POST /api/ppr`).
3. **`Tests/`** — correctness tests (`test_spmm.py`, `test_ppr_accuracy.py`) and performance benchmarks (`test_benchmark.py`, `benchmark_spdmm.py`).
4. **`Frontend/`** — a React SPA for uploading graphs, running PPR, and exploring results visually; wired to the backend via `src/lib/ppr/adapter.ts`.

## Remote machine requirement

**All compilation and testing must run on the remote GPU machine via SSH.**  
Do not attempt to build CUDA extensions or run tests locally — they will fail without a GPU.

## Build

```bash
# From repo root — must use --no-build-isolation so the build picks up the
# active conda/venv's PyTorch and CUDA installations.
pip install -e Backend/graph_link/ --no-build-isolation
```

After any change to `.cu` or `.cpp` files under `Backend/graph_link/`, rebuild before running tests.

## Environment (BGU GPU cluster)

There is **no `/usr/local/cuda`** on the cluster — load the toolkit from the
module system (`cuda/12.5` matches torch's cu121; the node default nvcc is 13.x,
too new) and derive `CUDA_HOME` from it. Conda env: `pageRank_312`.

```bash
module load cuda/12.5
export CC=$(which x86_64-conda-linux-gnu-cc)
export CXX=$(which x86_64-conda-linux-gnu-c++)
export CUDAHOSTCXX=$CXX
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH  # .so needs CXXABI_1.3.15 from conda libstdc++
export PYTHONPATH=$PYTHONPATH:/home/dayanb/SoftwareProjectPageRank/Backend/graph_link
```

Build + test on the cluster pin an RTX 3090 (`--gres=gpu:rtx_3090:1`, the
extension is sm_86-only). See the memory note `page-rank-cluster-run-setup`.

## Running tests

```bash
# Correctness (fast, ~5 s)
pytest Tests/test_spmm.py -v
pytest Tests/test_ppr_accuracy.py -v

# Performance benchmarks (slow — needs the flag)
pytest Tests/test_benchmark.py --benchmark-enable -v

# Standalone SpMM sweep
python Tests/benchmark_spdmm.py [--wandb]
```

All tests require a CUDA-capable GPU.

## Running the PPR server (GPU cluster)

The FastAPI server needs CUDA, so it runs as a long-running Slurm job on a GPU
node; a laptop reaches it through an SSH tunnel via the login node. Verified flow:

1. **One-time deps** — from the **login node** (compute nodes have no internet):
   `conda activate pageRank_312 && pip install fastapi 'uvicorn[standard]'`.
2. **Start the server** — submit a long job that runs uvicorn on a pinned RTX 3090.
   In the job: `module load cuda/12.5`, set `CUDA_HOME`/`LD_LIBRARY_PATH`/`PYTHONPATH`
   (see [`Backend/server/README.md`](Backend/server/README.md)), then
   `cd Backend/server && uvicorn app:app --host 0.0.0.0 --port 8000`.
   sbatch header: `--partition=rtx3090 --account=erant --qos=normal --gres=gpu:rtx_3090:1 --time=02:00:00`.
   The job logs its node; `squeue -u dayanb -j <id> -o %N` also shows it.
3. **Tunnel** from the laptop (compute nodes aren't directly SSH-able; the login
   node forwards to `<node>:8000` over the internal network):
   `ssh -N -L 8000:<node>:8000 dayanb@slurm.bgu.ac.il`.
4. **Client** — `cd Frontend && npm run dev`, open http://localhost:5173 (Vite
   proxies `/api` → localhost:8000 → tunnel → server).
5. **Stop** — `ssh dayanb@slurm.bgu.ac.il scancel <id>`, then close the tunnel and
   dev server.

If the server restarts on a different node, re-point the tunnel. **After any
cluster `git pull`** (which restores the stale tracked `.so`), rebuild before
serving: `rm -rf Backend/graph_link/build Backend/graph_link/graph_link_core*.so && pip install -e Backend/graph_link/ --no-build-isolation`.

## Key files to know

| Path | Purpose |
|---|---|
| `Backend/graph_link/kernels/pbr_kernel.cu` | PBR block SpMM (scalar + vectorized) and the CSR remainder kernel + launchers |
| `Backend/graph_link/kernels/ppr_kernel.cu` | PPR kernels (init, missing-mass, update) |
| `Backend/graph_link/kernels/pbr_kernels.hpp` | Launcher declarations (`launch_pbr_spmm`, `launch_csr_spmm`) |
| `Backend/graph_link/bindings/bindings_kernels.cu` | ATen wrappers: `pbr_spmm_cuda_dispatch` (allocates+returns Y, two streams) + PPR wrappers |
| `Backend/graph_link/graph_link/__init__.py` | High-level Python API (`csr_to_pbr`, `pbr_matmul`, `run_personalized_pagerank`) |
| `Backend/graph_link/graph_link/pbr_registry.py` | GPU metadata cache keyed by `id(pbr_mat)` |
| `Backend/server/app.py` | FastAPI service (`POST /api/ppr`) wrapping `run_personalized_pagerank` |
| `Tests/test_benchmark.py` | PPR performance benchmarks (pytest-benchmark) |
| `Tests/benchmark_spdmm.py` | Standalone SpMM sweep (CUDA events, optional W&B) |
| `Tests/matrices.py` | Synthetic matrix generators for benchmarking |

For deeper detail on `graph_link` internals see `Backend/graph_link/CLAUDE.md`.  
For test suite details see `Tests/CLAUDE.md`.  
For the React frontend see `Frontend/CLAUDE.md`.

---

## Frontend (React SPA)

A PPR analysis tool **wired to the GPU backend** through `Frontend/src/lib/ppr/adapter.ts`: the default `httpAdapter` POSTs to the FastAPI service in `Backend/server/`. The in-browser mock is kept behind the same adapter for tests/offline use, so `npm run dev` needs the backend reachable (or `VITE_PPR_API` set).

### Running locally

```bash
cd Frontend
npm install
npm run dev        # http://localhost:5173
```

### Other scripts

```bash
npm run build      # production bundle → dist/
npm run test       # Vitest, all suites (~40 tests)
npm run lint       # ESLint, 0 warnings allowed
npm run typecheck  # tsc --noEmit
```

### What it does

Three-step wizard: **Upload → Configure → Results**, plus a **Learn** page
explaining the PageRank / PPR math.

1. **Upload** — pick CSV format (edge list / COO triplets / adjacency matrix) then drop a file. Sample graphs in `Frontend/public/samples/` let you try it without a real file.
2. **Configure** — tune α (damping), max iterations, tolerance, top-X, and seed nodes. **At least one seed is required** (an "i" tooltip explains seeds; Compute is disabled until one is chosen).
3. **Results** — four tabs: top-ranked node cards, full sortable table, charts (rank distribution, convergence, degree histogram), interactive network graph (React Flow).

### CSV formats accepted

| Format | Required columns | Node IDs |
|---|---|---|
| Edge list | `source`, `target` (+ optional `weight`) | as-is from CSV |
| COO triplets | `row_idx`, `col_idx`, `value` | auto-named `n0`, `n1`, … |
| Adjacency matrix | header row + row labels | as-is from CSV |

### Backend connection

Wired via `httpAdapter` in `Frontend/src/lib/ppr/adapter.ts`. In dev, Vite
proxies `/api` → `VITE_PPR_TARGET` (default `http://localhost:8000`); for a
remote GPU box, SSH-tunnel the port. Run the service with
`uvicorn app:app` from `Backend/server/` (see its `README.md`). For a static
build, set `VITE_PPR_API` to the backend's absolute URL.

---

## Git workflow

After making changes, commit with a **meaningful message** that follows this format:

```
<type>(<scope>): <short imperative summary>

<body — explain WHY, not just what changed. Call out any non-obvious
trade-offs, performance implications, or design decisions.>
```

**Types:** `feat` (new capability), `fix` (bug), `perf` (performance), `refactor`, `test`, `docs`  
**Scopes:** `kernel`, `ppr`, `spmm`, `bindings`, `benchmark`, `tests`, `build`

Examples of good messages:
```
perf(ppr): move PPR iteration loop into C++ to eliminate per-iter CPU sync

The Python loop paid 3 GPU→CPU round-trips per iteration (Y.zero_(),
cudaStreamSynchronize x2, errors.max().item()). With 100 iterations that
is 300 blocking syncs per PPR run. The new run_ppr_cuda_loop binding runs
the entire power iteration in C++ using persistent streams and CUDA events,
checking convergence only every `check_interval` iterations.
```

```
fix(kernel): pass cudaStream_t to pbr_spmm kernel launches

launch_pbr_spmm had a stream parameter in the header and explicit
instantiations but the implementation was ignoring it, causing all block
SpMM kernels to launch on the default stream regardless of the caller's
stream argument.
```

**Push** to `origin main` after committing.
