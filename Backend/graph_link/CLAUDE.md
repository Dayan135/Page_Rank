# graph_link — Developer Guide

## What this is

`graph_link` is a pip-installable Python library providing a **Packed Block
Representation (PBR)** sparse matrix format and a fast SpMM (sparse × dense)
via custom CUDA kernels. The primary use-case is GPU-accelerated Personalized
PageRank. The PBR multiplication path is **manually synced** from the research
repo (`research_spdmm/fem`, branch `feat/pbr-csr-remainder`) — when the kernel
changes there, update it here by hand.

---

## How to build

```bash
pip install -e Backend/graph_link/ --no-build-isolation
```

`--no-build-isolation` is required so the build uses the active conda/venv's
PyTorch + CUDA. Build system is **setuptools + `torch.utils.cpp_extension`**
(`setup.py`); `CMakeLists.txt` is a standalone alternative not used by pip.

> **Cluster note:** there is no `/usr/local/cuda` on the BGU cluster — load the
> toolkit from the module system (`module load cuda/12.5`, matching torch's
> cu121) and derive `CUDA_HOME` from `which nvcc`. See the root `CLAUDE.md`.

The compiled extension is `graph_link_core.cpython-312-x86_64-linux-gnu.so`.

---

## Directory layout

```
graph_link/
├── setup.py                   ← build config (the one pip uses)
├── pbr_matrix/
│   ├── pbr_matrix.hpp         ← data structures + CPU function decls
│   └── pbr_matrix.cpp         ← csr_to_pbr, pbr_to_csr, pbr_batched_matmul_cpu (at::Tensor)
├── kernels/
│   ├── pbr_kernels.hpp        ← launcher declarations (launch_pbr_spmm, launch_csr_spmm)
│   ├── pbr_kernel.cu          ← PBR block kernels + CSR remainder kernel + launchers
│   ├── ppr_kernels.hpp        ← PPR launcher declarations
│   ├── ppr_kernel.cu          ← PPR kernels (init / missing-mass / update)
│   └── kernels.hpp            ← CUDA error-check utility
├── bindings/
│   ├── bindings.cpp           ← PYBIND11_MODULE(graph_link_core, m); class + property getters
│   └── bindings_kernels.cu    ← ATen wrappers: pbr_spmm_cuda_dispatch + PPR wrappers
└── graph_link/                ← importable Python package
    ├── __init__.py            ← high-level API (pbr_matmul, csr_to_pbr, run_personalized_pagerank, …)
    ├── pbr_registry.py        ← GPU metadata cache: dict[id(pbr_mat)] → tensors
    ├── pbr_matrix_triton.py   ← Triton alternative kernel (experimental, block-only)
    └── cluster_permute_csr.py ← CSR reordering utility
```

An HTTP service that exposes this library to the React frontend lives in the
sibling [`../server/`](../server/) (FastAPI).

---

## Data structures — `pbr_matrix_t<index_t, scalar_t>` (`pbr_matrix.hpp`)

PBR splits a sparse matrix into compressed blocks plus a **CSR remainder**:

| Field | Type | Description |
|---|---|---|
| `rows`, `cols` | `index_t` | Matrix dimensions |
| `block_rows`, `block_cols` | `index_t` | Block tile size (equal; supported 2, 4, 8) |
| `block_codes` | `vector<std::bitset<64>>` | Bitmask per block — bit k set ⇒ k-th tile element is nonzero |
| `block_coords` | `vector<coord_t>` | Top-left (row, col) of each compressed block |
| `block_offsets` | `vector<index_t>` | Start index into `block_data` per block |
| `block_data` | `vector<scalar_t>` | Packed nonzeros for all compressed blocks |
| `remainder_indptr` | `vector<index_t>` | CSR row pointers (size `rows+1`) for leftover nnz |
| `remainder_col_ind` | `vector<index_t>` | CSR column indices of the remainder |
| `remainder_vals` | `vector<scalar_t>` | CSR values of the remainder |

**A block is compressed only if it has ≥ `min_nnz_per_block` nonzeros**; the
rest go to the CSR remainder. So `min_nnz=1` → tiny remainder; higher `min_nnz`
→ more mass in the remainder. (`coo_elem_t` still exists in the header but is
vestigial — the remainder is CSR, not COO.)

`csr_to_pbr` flushes the trailing partial stripe when `rows % block_rows != 0`
(a fix this copy carries that upstream research lacks).

---

## CUDA kernels (`kernels/pbr_kernel.cu`)

- **`pbr_spmm_zero_idle_kernel<index_t, scalar_t, BR, BC, TOTAL_SHARED, THREADS, FEAT_BLOCK>`**
  — scalar block kernel. Grid `(num_pbr_blocks, batch, ceil(features/FEAT_BLOCK))`.
  Loads X into shared memory, walks the bitmask, accumulates, `atomicAdd` to Y.
  The shared-mem partition `p` is floored to a power of two so inner `/p`,`%p`
  become shift/mask. `TOTAL_SHARED = 2048`.
- **`pbr_spmm_vec_kernel<…>`** — vectorized variant: each thread owns `L`
  contiguous features via `float4`/`double2`. Used when `features % L == 0`.
- **`csr_spmm_kernel<index_t, scalar_t>`** — the **CSR remainder** kernel
  (warp-per-row / CSR-Vector). One warp owns a row across all features, loading
  the row's structure once and reusing it across feature tiles. Replaced the old
  per-nnz COO kernel (~16× fewer L2/DRAM passes at features=512).

Launchers: `launch_pbr_spmm` (picks vec vs scalar by `features % L`; `block_codes`
is `const int64_t*`) and `launch_csr_spmm`. Both take a `cudaStream_t`.

---

## Wrapper layer (`bindings/bindings_kernels.cu`)

### `pbr_spmm_cuda_dispatch<index_t, scalar_t>(...) -> at::Tensor`
Main SpMM entry point. **Allocates and returns Y** (`[batch, rows, features]`).
Runs the block kernel and the CSR-remainder kernel on two **ATen stream-pool**
streams that fork from / rejoin the caller's current stream via CUDA events.
Exposed as `core.pbr_spmm_cuda_i32_f32 / i64_f32 / i32_f64 / i64_f64`.

The PPR kernels are also bound here (`init_ppr_cuda_*`, `missing_mass_cuda_*`,
`ppr_update_cuda_*`, `ppr_update_normalized_cuda_*`).

`bindings.cpp` exposes the matrix to Python via **property getters**
(`block_codes` as a uint64 array through `.to_ulong()`, flat `block_coords`,
`block_offsets`, `block_data`, `remainder_indptr`, `remainder_col_ind`,
`remainder_data`) plus `__copy__` (needed by the Python `.to()`), and
`accounted_blocks` / `compressed_nnz` / `remainder_nnz`. There is no `to_dict()`.

---

## Python API (`graph_link/__init__.py`)

### GPU metadata cache (`pbr_registry.py`)
`.to('cuda')` reads the matrix's property getters and stores CUDA tensors keyed
by `id(pbr_mat)`:

```python
{
    'codes':       int64  tensor,  # block bitmasks (read as int64_t in C++)
    'coords':      int32  tensor,  # flat [row0, col0, row1, col1, …]
    'offsets':     int32  tensor,  # per-block start index into data
    'data':        float32/64,     # packed block nonzeros
    'rem_indptr':  int32  tensor,  # CSR remainder row pointers (rows+1)
    'rem_col_ind': int32  tensor,  # CSR remainder column indices
    'rem_vals':    float32/64,     # CSR remainder values
}
```

### `csr_to_pbr(mat, block_rows=2, block_cols=2, min_nnz_per_block=2)`
`scipy.sparse.csr_matrix` or a CPU sparse-CSR `torch.Tensor` → `pbr_matrix_t`.

### `pbr_matmul(pbr_mat, x: Tensor) -> Tensor`
Unified entry. GPU path calls `pbr_batched_matmul_cuda` (returns Y); CPU path
calls `core.pbr_batched_matmul_cpu(pbr_mat, x)` (also returns a tensor). Squeezes
the batch dim for 2D input.

### `pbr_batched_matmul_cuda(pbr_mat, x) -> Tensor`
Selects the `core.pbr_spmm_cuda_*` variant by `meta` dtypes and returns
`Y = A @ x` (`[batch, rows, features]`). Passes block + CSR-remainder tensors.

### `normalize_transition_matrix(A_csr) -> sp.csr_matrix`
Column-stochastic normalization for PPR (dangling columns → uniform 1/N). Call
before `csr_to_pbr` when using `run_personalized_pagerank`.

### `run_personalized_pagerank(pbr_mat, source_nodes, alpha=0.85, max_iterations=100, tolerance=1e-6, return_history=False)`
Power iteration **X_{t+1} = α·A@X_t + (1−α)·e_s** (column-stochastic A, one
column per source). Each step allocates a fresh Y (the dispatch returns it).
Returns `(X, iterations, converged)`, or `(X, iterations, converged, history)`
when `return_history=True` — `history[t]` is the per-iteration max L1 update
error (used by the frontend convergence chart; free, reuses the existing sync).

---

## Template instantiation matrix

`{int32_t, int64_t} × {float, double}` → suffixes `i32_f32`, `i64_f32`,
`i32_f64`, `i64_f64`. Python uploads coords/offsets/remainder indices as
**int32**, so the `i32_*` variants are selected in practice (block_codes are
always read as int64).

---

## Tests

```
pytest Tests/test_spmm.py -v          # SpMM vs scipy
pytest Tests/test_ppr_accuracy.py -v  # PPR vs NetworkX
```

The `min_nnz=4` cases push significant mass into the **CSR remainder** and
exercise `csr_spmm_kernel`. See `Tests/CLAUDE.md`.

---

## Known design decisions

- **`min_nnz`**: small → almost everything compressed (tiny remainder); larger →
  blocks with few nnz fall into the CSR remainder. With `block_size=2, min_nnz=4`
  on a sparse matrix `accounted_blocks()` can be **0** — `launch_pbr_spmm` guards
  against a `grid.x=0` launch with an early return.
- **Stream overlap**: `pbr_spmm_cuda_dispatch` uses the ATen stream pool + events
  to overlap the block and CSR kernels and rejoin the caller's stream (no manual
  `cudaStreamCreate`/`Destroy`).
- **int32 indices on GPU**: coords/offsets/remainder indices are cast to int32 on
  transfer; values fit int32 for the graphs in scope.
- **`pagerank_engine.cpp`** in the root is dead code — not in the build.
