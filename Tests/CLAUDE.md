# Tests — Developer Guide

> **Maintenance rule**: Keep this file in sync with the test suites. Any time a test file is added, removed, or its parametrize axes / assertions change, update the corresponding section here.

---

## Running the tests

```bash
# Correctness tests (fast, ~2–5 s)
pytest Tests/test_spmm.py -v
pytest Tests/test_ppr_accuracy.py -v

# Performance benchmarks (slow, needs --benchmark-enable)
pytest Tests/test_benchmark.py --benchmark-enable -v

# Save benchmark results for plotting
pytest Tests/test_benchmark.py --benchmark-enable --benchmark-json=output.json
python Tests/plot_benchmark.py          # reads output.json, writes ppr_benchmark_results.png
```

All tests require a CUDA-capable GPU. Tests are skipped automatically when CUDA is unavailable.

---

## test_spmm.py — SpMM correctness

**Purpose**: Verify that the PBR sparse-dense matrix multiplication (`pbr_matmul`) matches the scipy CSR reference (`A.dot(X)`) across a wide parameter grid.

**What it tests**: `graph_link.csr_to_pbr` → `.to(device)` → `graph_link.pbr_matmul` → compare against `scipy.sparse.csr_matrix.dot`.

**Parameter grid** (108 combinations):

| Parameter | Values | Notes |
|---|---|---|
| `N` | 64, 128, 1024 | Square matrix size |
| `num_features` | 8, 64, 256 | Columns of the dense X matrix |
| `density` | 0.05, 0.15 | Fraction of nonzeros in A |
| `block_size` | 2, 4, 8 | Block tile size (rows == cols) |
| `min_nnz` | 1, 4 | Min nnz per block to use compressed format; higher threshold → more goes to the CSR remainder |

**Tolerance**: `atol=1e-4` (float32 accumulation error).

**Critical cases**: `min_nnz=4` exercises the CSR remainder kernel heavily, especially for sparse matrices and small block sizes.

---

## test_ppr_accuracy.py — PPR correctness

**Purpose**: Verify that GPU Personalized PageRank (`run_personalized_pagerank`) matches NetworkX's `pagerank` reference for directed random graphs.

**What it tests**: directed Erdős-Rényi graph → `normalize_transition_matrix(A.T)` → `csr_to_pbr` → `run_personalized_pagerank` vs `nx.pagerank` (same α, same dangling redistribution).

**Parameter grid** (24 combinations):

| Parameter | Values | Notes |
|---|---|---|
| `N, p` | (50, 0.10), (100, 0.05) | Graph size and edge probability |
| `alpha` | 0.85, 0.90 | Teleportation probability (standard PPR damping) |
| `block_size` | 2, 4, 8 | PBR block tile size |
| `min_nnz` | 1, 4 | PBR remainder threshold |

**Sources tested per run**: nodes 0, N//4, N//2, N−1 (4 source columns per graph).

**Assertions per source**:
1. **Convergence**: must converge within 500 iterations.
2. **Mass conservation**: `sum(X[:, k]) ≈ 1.0` (atol=1e-4) — validates that the column-stochastic matrix and PPR update are correct end-to-end.
3. **L1 accuracy vs NetworkX**: `allclose(cuda_col, nx_col, atol=1e-3)`.

**NetworkX alignment**: `dangling={n: 1/N}` is passed to `nx.pagerank` so that dangling-node mass is redistributed uniformly, matching how `normalize_transition_matrix` handles zero-degree columns.

---

## test_benchmark.py — PPR performance

**Purpose**: Measure wall-clock throughput of PBR batched PPR vs PyTorch cuSPARSE baseline on large graphs. Uses `pytest-benchmark`.

**Graph**: clustered random graph (80% short-range edges to simulate locality, 20% random), 8 edges per node on average. Designed to favour the PBR format's block structure.

**Parameter grid**:

| Parameter | Values |
|---|---|
| `system_size` | 65 536, 262 144 |
| `features` | 1, 32, 128 |
| `block_size` (PBR only) | 2, 4 |

**Benchmarks**:

| Test | Backend | Notes |
|---|---|---|
| `test_benchmark_torch_ppr` | `torch.sparse.mm` (cuSPARSE) | Baseline; runs 100 PPR iterations, simplified update `X = α·A@X + (1-α)·e_s`, no convergence check |
| `test_benchmark_pbr_engine` | `run_personalized_pagerank` (custom CUDA) | PBR block + CSR remainder kernels; two CUDA streams in parallel |

Both backends operate on the same column-stochastic matrix produced by `normalize_transition_matrix`, ensuring a fair comparison.

Each benchmark: 20 rounds, 5 warmup rounds (`benchmark.pedantic`). `torch.cuda.synchronize()` is called inside the runner to get accurate wall-clock time.

---

## plot_benchmark.py — Result visualization

Not a pytest file. Run directly after producing `output.json` from `test_benchmark.py`.

Reads the pytest-benchmark JSON, parses `(algorithm, features, graph_size)` from test names, and produces a grouped bar chart (`ppr_benchmark_results.png`) comparing all algorithms across graph sizes and batch sizes.
