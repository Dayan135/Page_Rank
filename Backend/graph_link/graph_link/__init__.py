import numpy as np
import scipy.sparse as sp
import torch
import graph_link_core as core
import copy

from .pbr_registry import get_pbr_gpu_meta, set_pbr_gpu_meta, clear_pbr_gpu_meta
from .pbr_matrix_triton import pbr_batched_matmul_triton_gpu


pbr_batched_matmul_cpu = core.pbr_batched_matmul_cpu
# pbr_matmul = core.pbr_matmul
    



def _pbr_to_method(self, device='cuda'):
    # Standardize device string (e.g., handles 'cuda:0' vs 'cuda')
    target_device = 'cpu' if device is None else str(torch.device(device))
    
    # 1. Determine current device state
    existing_meta = get_pbr_gpu_meta(self)
    current_device = str(existing_meta['codes'].device) if existing_meta else 'cpu'
    
    # PyTorch Convention: If already on the requested device, return self exactly
    if current_device == target_device:
        return self

    # 2. Create a new object (The "Out-of-Place" requirement)
    # This creates a shallow copy of the C++ object so it gets a new id()
    try:
        new_pbr = copy.copy(self)
    except TypeError:
        raise NotImplementedError(
            "To support out-of-place .to(), your C++ PBRMatrix needs a copy constructor "
            "exposed to Python via pybind11."
        )

    # 3. Handle GPU transfer for the NEW object
    if target_device != 'cpu':
        # Block fields + CSR remainder (remainder_indptr has size rows+1; an
        # empty remainder is an all-zero indptr with empty col_ind / vals).
        gpu_meta = {
            'codes':       torch.as_tensor(self.block_codes,        device=device, dtype=torch.int64),
            'coords':      torch.as_tensor(self.block_coords,       device=device, dtype=torch.int32),
            'offsets':     torch.as_tensor(self.block_offsets,      device=device, dtype=torch.int32),
            'data':        torch.as_tensor(self.block_data,         device=device),
            'rem_indptr':  torch.as_tensor(self.remainder_indptr,   device=device, dtype=torch.int32),
            'rem_col_ind': torch.as_tensor(self.remainder_col_ind,  device=device, dtype=torch.int32),
            'rem_vals':    torch.as_tensor(self.remainder_data,     device=device),
        }

        # Register the GPU tensors under the new object's ID
        set_pbr_gpu_meta(new_pbr, gpu_meta)

    return new_pbr


# --- Inject the method into the C++ classes ---
for cls_name in ["PBRMatrixInt64Float", "PBRMatrixInt32Float", 
                 "PBRMatrixInt64Double", "PBRMatrixInt32Double"]:
    if hasattr(core, cls_name):
        setattr(getattr(core, cls_name), "to", _pbr_to_method)
        
    
def pbr_batched_matmul_cuda(pbr_mat, x: torch.Tensor) -> torch.Tensor:
    """
    Launches the PBR block SpMM and the CSR remainder SpMM on two CUDA streams
    in parallel via the combined C++ dispatch, which allocates and returns
    Y = A @ x with shape [batch, rows, features]. Callers squeeze the batch dim
    for 2D inputs.
    """
    meta = get_pbr_gpu_meta(pbr_mat)
    if meta is None:
        raise RuntimeError("PBRMatrix is not on GPU. Call .to('cuda') first.")
    index_dtype = meta['coords'].dtype
    data_dtype  = meta['data'].dtype
    if index_dtype == torch.int32 and data_dtype == torch.float32:
        kernel_func = core.pbr_spmm_cuda_i32_f32
    elif index_dtype == torch.int64 and data_dtype == torch.float32:
        kernel_func = core.pbr_spmm_cuda_i64_f32
    elif index_dtype == torch.int32 and data_dtype == torch.float64:
        kernel_func = core.pbr_spmm_cuda_i32_f64
    elif index_dtype == torch.int64 and data_dtype == torch.float64:
        kernel_func = core.pbr_spmm_cuda_i64_f64
    else:
        raise TypeError(f"Unsupported dtype combination: index={index_dtype}, data={data_dtype}")
    return kernel_func(
        pbr_mat.cols,
        pbr_mat.rows,
        pbr_mat.block_rows,
        pbr_mat.block_cols,
        meta['codes'],
        meta['coords'],
        meta['offsets'],
        meta['data'],
        meta['rem_indptr'],
        meta['rem_col_ind'],
        meta['rem_vals'],
        x,
    )


def pbr_matmul(pbr_mat, x: torch.Tensor):
    """
    Unified entry point for PBR MatMul. 
    Dispatches to the correct backend based on x.device.
    """
    # 1. Shape Inference
    if x.dim() == 2:
        batch, height, width = 1, x.shape[0], x.shape[1]
    elif x.dim() == 3:
        batch, height, width = x.shape
    else:
        raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

    # 2. Device Dispatch (GPU Path)
    if x.is_cuda:
        # Check if PBR metadata is already in the GPU registry
        meta = get_pbr_gpu_meta(pbr_mat)
        if meta is None:
            # First-time use: trigger the injected .to() method
            pbr_mat = pbr_mat.to(x.device)

        # Dispatch allocates and returns Y = [batch, rows, width]
        y = pbr_batched_matmul_cuda(pbr_mat, x)

        return y.squeeze(0) if x.dim() == 2 else y

    # 3. CPU Logic (no cuda!)
    # The C++ matmul allocates and returns Y = [batch, rows, width].
    y = core.pbr_batched_matmul_cpu(pbr_mat, x)

    return y.squeeze(0) if x.dim() == 2 else y


def csr_to_pbr(mat: sp.csr_matrix | torch.Tensor,
               block_rows: int = 8, block_cols: int = 8,
               min_nnz_per_block: int = 4):
    if isinstance(mat, torch.Tensor):
        if mat.device.type != 'cpu':
            raise ValueError('Input tensor must be on CPU')

        return core.csr_to_pbr(mat.crow_indices().numpy(),
                               mat.col_indices().numpy(),
                               mat.values().cpu().numpy(),
                               mat.shape[0], mat.shape[1],
                               block_rows, block_cols,
                               min_nnz_per_block)
    elif isinstance(mat, sp.csr_matrix):
        return core.csr_to_pbr(mat.indptr, mat.indices, mat.data,
                               mat.shape[0], mat.shape[1],
                               block_rows, block_cols,
                               min_nnz_per_block)
    else:
        raise TypeError(f'Unsupported type {type(mat)}')


def pbr_to_csr(pbr_mat, indptr, indices, data):
    return core.pbr_to_csr(pbr_mat, indptr, indices, data)


def pbr_analyze_csr(mat: sp.csr_matrix | torch.Tensor,
                    block_rows: int = 8, block_cols: int = 8,
                    min_nnz_coverage=1000):
    if isinstance(mat, torch.Tensor):
        if mat.device.type != 'cpu':
            raise ValueError('Input tensor must be on CPU')

        return core.pbr_analyze_csr(mat.crow_indices().numpy(),
                                    mat.col_indices().numpy(),
                                    mat.shape[0], mat.shape[1], mat._nnz(),
                                    block_rows, block_cols,
                                    min_nnz_coverage)
    elif isinstance(mat, sp.csr_matrix):
        return core.pbr_analyze_csr(mat.indptr, mat.indices,
                                    mat.shape[0], mat.shape[1], mat.nnz,
                                    block_rows, block_cols,
                                    min_nnz_coverage)
    else:
        raise TypeError(f'Unsupported type {type(mat)}')
    
    
def normalize_transition_matrix(A_csr: sp.csr_matrix) -> sp.csr_matrix:
    """
    Returns a column-stochastic transition matrix suitable for PPR.

    Each column is divided by its out-degree (column sum).
    Dangling nodes (zero out-degree) receive a uniform 1/N column so
    the matrix stays column-stochastic, conserving probability mass.
    """
    N = A_csr.shape[0]
    A = A_csr.tocsc().astype(np.float32)
    col_sums = np.asarray(A.sum(axis=0), dtype=np.float32).ravel()

    inv = np.zeros_like(col_sums)
    np.divide(1.0, col_sums, out=inv, where=col_sums > 0)
    A_norm = A.multiply(inv).tocsr()

    dangling = np.where(col_sums == 0)[0]
    if dangling.size > 0:
        r = np.tile(np.arange(N, dtype=np.int32), dangling.size)
        c = np.repeat(dangling.astype(np.int32), N)
        v = np.full(N * dangling.size, 1.0 / N, dtype=np.float32)
        A_norm = A_norm + sp.csr_matrix((v, (r, c)), shape=(N, N))

    return A_norm


def run_personalized_pagerank(
    pbr_mat,
    source_nodes: torch.Tensor,
    alpha: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
):
    """
    Batched Personalized PageRank via power iteration:

        X_{t+1} = alpha * A @ X_t + (1 - alpha) * e_s

    pbr_mat must be on GPU and built from a column-stochastic matrix
    (use normalize_transition_matrix() to prepare the adjacency matrix).

    Returns (X, iterations_run, converged).
    """
    meta = get_pbr_gpu_meta(pbr_mat)
    if meta is None:
        raise RuntimeError("PBRMatrix is not on GPU. Call .to('cuda') first.")

    if source_nodes.dtype != torch.int32:
        source_nodes = source_nodes.to(torch.int32)

    device = source_nodes.device
    N      = pbr_mat.rows
    K      = source_nodes.shape[0]
    dtype  = meta['data'].dtype

    # Single X buffer (updated in-place); each SpMM allocates its own Y.
    X      = torch.zeros((N, K), dtype=dtype, device=device)
    errors = torch.zeros(K,      dtype=dtype, device=device)

    if dtype == torch.float32:
        init_fn   = core.init_ppr_cuda_float
        update_fn = core.ppr_update_normalized_cuda_float
    else:
        init_fn   = core.init_ppr_cuda_double
        update_fn = core.ppr_update_normalized_cuda_double

    init_fn(X, source_nodes, N, K)

    for t in range(max_iterations):
        Y = pbr_batched_matmul_cuda(pbr_mat, X).squeeze(0)  # Y = A @ X, shape [N, K]
        update_fn(Y, X, source_nodes, alpha, N, K, errors)  # X = alpha*Y + (1-alpha)*e_s

        if errors.max().item() < tolerance:
            return X, t + 1, True

    return X, max_iterations, False


__all__ = [
    'csr_to_pbr',
    'pbr_to_csr',
    'pbr_analyze_csr',
    'pbr_matmul',
    'pbr_batched_matmul_cpu',
    'pbr_batched_matmul_triton_gpu',
    'pbr_batched_matmul_cuda',
    'get_pbr_gpu_meta',
    'normalize_transition_matrix',
    'run_personalized_pagerank',
]