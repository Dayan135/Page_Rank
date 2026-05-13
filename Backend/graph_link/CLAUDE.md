# graph_link — Developer Guide

## What this is

`graph_link` is a Python library (pip-installable) that provides a **Packed Block Representation (PBR)** sparse matrix format and fast SpMM (sparse × dense matrix multiplication) via custom CUDA kernels. The primary use-case is GPU-accelerated Personalized PageRank.

---

## How to build

```bash
# From repo root (or from this directory with just '.')
pip install -e Backend/graph_link/ --no-build-isolation
```

`--no-build-isolation` is required so that the build uses the current conda/venv's PyTorch and CUDA rather than an isolated environment.

The build system is **setuptools + `torch.utils.cpp_extension`** (setup.py). CMakeLists.txt exists but is not used by pip — it is a standalone CMake alternative.

The compiled extension is `graph_link_core.cpython-312-x86_64-linux-gnu.so`, placed in the package root. After install the Python package `graph_link/` and the `.so` are both on sys.path via the editable link.

---

## Directory layout

```
graph_link/                    ← pip install root
├── setup.py                   ← build config (the one that matters for pip)
├── pyproject.toml             ← build-system declaration (setuptools)
├── CMakeLists.txt             ← standalone CMake (not used by pip)
│
├── pbr_matrix/
│   ├── pbr_matrix.hpp         ← core data structures + CPU functions declared
│   └── pbr_matrix.cpp         ← CPU implementations (csr_to_pbr, pbr_to_csr, pbr_batched_matmul_cpu)
│
├── kernels/
│   ├── pbr_kernels.hpp        ← CUDA launcher declarations
│   ├── pbr_kernel.cu          ← ALL CUDA kernels + launchers
│   └── kernels.hpp            ← CUDA error-check utilities
│
├── bindings/
│   ├── bindings.cpp           ← pybind11 module definition (PYBIND11_MODULE(graph_link_core, m))
│   └── bindings_kernels.cu    ← ATen-tensor wrappers that call CUDA launchers; bind_cuda_functions()
│
└── graph_link/                ← importable Python package
    ├── __init__.py            ← high-level API (pbr_matmul, csr_to_pbr, run_personalized_pagerank, …)
    ├── pbr_registry.py        ← GPU metadata cache: dict[id(pbr_mat)] → {codes, coords, offsets, data, rem_*}
    ├── pbr_matrix_triton.py   ← Triton alternative kernel (experimental)
    └── cluster_permute_csr.py ← CSR reordering utility
```

---

## Data structures

### `pbr_matrix_t<index_t, scalar_t>` (`pbr_matrix.hpp`)

The PBR format splits a sparse matrix into two parts:

| Field | Type | Description |
|---|---|---|
| `rows`, `cols` | `index_t` | Matrix dimensions |
| `block_rows`, `block_cols` | `index_t` | Block tile size (always equal, supported: 2, 4, 8) |
| `block_codes` | `vector<uint64_t>` | Bitmask per block — bit k set means the k-th element of that block tile is nonzero |
| `block_coords` | `vector<coord_t>` | Top-left (row, col) of each compressed block |
| `block_offsets` | `vector<index_t>` | Start index into `block_data` for each block |
| `block_data` | `vector<scalar_t>` | Packed nonzero values for all compressed blocks |
| `remainder_coo` | `vector<coo_elem_t>` | Elements from blocks that had fewer nnz than `min_nnz_per_block` — stored as plain COO (row, col, val) |

**A block is compressed only if it has ≥ `min_nnz_per_block` nonzeros.** Everything else goes into `remainder_coo`. This means:
- `min_nnz=1` → almost nothing in remainder → remainder is tiny
- `min_nnz=4` → blocks with 1–3 nnz go to remainder → remainder can be large

### `coo_elem_t<index_t, scalar_t>` (`pbr_matrix.hpp`)
Simple struct: `{ index_t row; index_t col; scalar_t val; }`.

---

## CUDA kernels (`kernels/pbr_kernel.cu`)

### `pbr_spmm_shared_reg_kernel<index_t, scalar_t, BLOCK_ROWS, BLOCK_COLS>`
Processes compressed blocks. Grid = `(num_pbr_blocks, batch_size, ceil(features/128))`.
- Loads a tile of X into shared memory (coalesced).
- Iterates over the bitmask to find nonzeros (uses `__ffsll`).
- Accumulates into registers, then `atomicAdd` to Y.
- Template-instantiated for block sizes 2×2, 4×4, 8×8.

### `coo_spmm_kernel<scalar_t>` *(added for remainder)*
Processes the COO remainder. Grid = `(nnz, batch_size, ceil(features/128))`.
- Each block handles one nonzero element across all features.
- `atomicAdd(&Y[row * features + feat], val * X[col * features + feat])`.
- Row/col arrays are always `int32_t`.

### Launchers
- `launch_pbr_spmm<index_t, scalar_t>(...)` — dispatches to the right block-size template; takes a `cudaStream_t stream` param.
- `launch_coo_spmm<scalar_t>(...)` — launches the COO kernel; takes a `cudaStream_t stream` param.

---

## Wrapper layer (`bindings/bindings_kernels.cu`)

### `pbr_full_spmm_cuda_wrapper<index_t, scalar_t>`
The main entry point called from Python. Does:
1. Creates two CUDA streams (`s_blocks`, `s_coo`).
2. Launches `launch_pbr_spmm` on `s_blocks`.
3. Launches `launch_coo_spmm` on `s_coo` (skipped if `rem_nnz == 0`).
4. `cudaStreamSynchronize` on both, then destroys them.

Exposed to Python as:
```
core.pbr_full_spmm_cuda_int32_float
core.pbr_full_spmm_cuda_int64_float
core.pbr_full_spmm_cuda_int32_double
core.pbr_full_spmm_cuda_int64_double
```

---

## Python API (`graph_link/__init__.py`)

### GPU metadata cache (`pbr_registry.py`)
When a PBR matrix is moved to GPU via `.to('cuda')`, its metadata is extracted and stored as PyTorch CUDA tensors in a global dict keyed by `id(pbr_mat)`:

```python
{
    'codes':    int64  tensor  # block bitmasks (reinterpreted as uint64 in C++)
    'coords':   int32  tensor  # flattened [row0, col0, row1, col1, …]
    'offsets':  int32  tensor  # per-block start index into data
    'data':     float32/64     # packed block nonzero values
    'rem_rows': int32  tensor  # COO remainder row indices
    'rem_cols': int32  tensor  # COO remainder col indices
    'rem_vals': float32/64     # COO remainder values
}
```

### `csr_to_pbr(mat, block_rows=8, block_cols=8, min_nnz_per_block=1)`
Accepts `scipy.sparse.csr_matrix` or a CPU `torch.Tensor` (sparse CSR). Returns a `pbr_matrix_t` C++ object.

### `pbr_matmul(pbr_mat, x: Tensor) → Tensor`
Unified entry point. Dispatches to GPU path (via `pbr_batched_matmul_cuda`) or CPU path. Handles 2D/3D input shapes.

### `pbr_batched_matmul_cuda(pbr_mat, x, y, batch_size, features)`
Dispatches to the correct `core.pbr_full_spmm_cuda_*` variant based on `meta['coords'].dtype` and `meta['data'].dtype`. Passes both the block and remainder tensors.

### `run_personalized_pagerank(pbr_mat, source_nodes, damping=0.85, max_iter=100, tol=1e-6)`
Full GPU loop: `init → (SpMM → missing_mass → ppr_update) × N`. Requires `pbr_mat.to('cuda')` first.

---

## `to_dict()` method (`pbr_matrix.hpp`)
Called by `_pbr_to_method` to extract all GPU-transferable arrays as numpy arrays:
- `block_codes`, `block_coords`, `block_offsets`, `block_data` — compressed block data
- `rem_rows`, `rem_cols`, `rem_vals` — remainder COO (may be empty arrays)

---

## Template instantiation matrix

| `index_t` | `scalar_t` | Python name suffix |
|---|---|---|
| `int32_t` | `float` | `int32_float` |
| `int64_t` | `float` | `int64_float` |
| `int32_t` | `double` | `int32_double` |
| `int64_t` | `double` | `int64_double` |

In practice, Python always casts coords/offsets/rem_rows/rem_cols to **int32**, so the `int32_*` variants are always selected.

---

## Tests

```
Tests/test_spmm.py
```

Single parametrized test `test_spmm_accuracy` covering:
- `N` ∈ {64, 128, 1024}
- `num_features` ∈ {8, 64, 256}
- `density` ∈ {0.05, 0.15}
- `block_size` ∈ {2, 4, 8} (rows == cols)
- `min_nnz` ∈ {1, 4}

Run with: `pytest Tests/test_spmm.py -v`

The `min_nnz=4` cases are the critical ones — they put significant data into `remainder_coo` and verify the COO kernel is working.

---

## Known design decisions

- **`min_nnz=1` vs `min_nnz=4`**: With `min_nnz=1`, almost all elements end up in compressed blocks and the remainder is tiny. With `min_nnz=4`, blocks with 1–3 nnz fall into the remainder, which can be substantial for sparse matrices. With `block_size=2, min_nnz=4` on a sparse matrix, `accounted_blocks()` can be **0** — `launch_pbr_spmm` guards against this with an early return to avoid launching a CUDA kernel with `grid.x=0` (invalid configuration).
- **Stream creation per call**: `pbr_full_spmm_cuda_wrapper` creates and destroys CUDA streams on every call. This is simple but has per-call overhead. For high-frequency calls, consider caching streams in the pbr registry.
- **Always int32 for COO indices**: Remainder row/col tensors are always cast to int32 on GPU transfer, consistent with how block coords are treated.
- **`pagerank_engine.cpp`** in the root is a standalone file, not compiled into the library.
