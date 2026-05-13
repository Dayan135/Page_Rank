# Page_Rank — Project Guide for Claude

## What this project is

A GPU-accelerated **Personalized PageRank (PPR)** engine built around a custom sparse matrix format called **Packed Block Representation (PBR)**. The core insight is that many real-world graphs (social networks, web graphs) have block-sparse adjacency matrices — PBR exploits this structure to outperform the generic cuSPARSE SpMM primitive for block-structured matrices.

The project has three layers:
1. **`Backend/graph_link/`** — a pip-installable Python package containing custom CUDA kernels (C++/CUDA), pybind11 bindings, and a high-level Python API.
2. **`Tests/`** — correctness tests (`test_spmm.py`, `test_ppr_accuracy.py`) and performance benchmarks (`test_benchmark.py`, `benchmark_spdmm.py`).
3. **`Frontend/`** — a pipeline test harness.

## Build

```bash
# From repo root — must use --no-build-isolation so the build picks up the
# active conda/venv's PyTorch and CUDA installations.
pip install -e Backend/graph_link/ --no-build-isolation
```

After any change to `.cu` or `.cpp` files under `Backend/graph_link/`, rebuild before running tests.

## Environment (Linux GPU machine)

```bash
export CC=$(which x86_64-conda-linux-gnu-cc)
export CXX=$(which x86_64-conda-linux-gnu-c++)
export CUDAHOSTCXX=$CXX
export CUDA_HOME=/usr/local/cuda-12.5
export PYTHONPATH=$PYTHONPATH:/home/dayanb/SoftwareProjectPageRank/Backend/graph_link
```

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

## Key files to know

| Path | Purpose |
|---|---|
| `Backend/graph_link/kernels/pbr_kernel.cu` | All CUDA kernels: SpMM, PPR init, PPR update, full PPR loop |
| `Backend/graph_link/kernels/pbr_kernels.hpp` | Launcher declarations |
| `Backend/graph_link/bindings/bindings_kernels.cu` | pybind11 wrappers; `run_ppr_cuda_loop` orchestration |
| `Backend/graph_link/graph_link/__init__.py` | High-level Python API (`csr_to_pbr`, `pbr_matmul`, `run_personalized_pagerank`) |
| `Backend/graph_link/graph_link/pbr_registry.py` | GPU metadata cache keyed by `id(pbr_mat)` |
| `Tests/test_benchmark.py` | PPR performance benchmarks (pytest-benchmark) |
| `Tests/benchmark_spdmm.py` | Standalone SpMM sweep (CUDA events, optional W&B) |
| `Tests/matrices.py` | Synthetic matrix generators for benchmarking |

For deeper detail on `graph_link` internals see `Backend/graph_link/CLAUDE.md`.  
For test suite details see `Tests/CLAUDE.md`.

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
