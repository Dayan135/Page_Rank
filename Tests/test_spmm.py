import pytest
import numpy as np
import scipy.sparse as sp
import graph_link
import torch

def test_spmm_multi_feature_accuracy():
    """
    Verifies that A * X = Y for multiple features (columns in X).
    This tests the shared memory tiling and grid-stride loop logic.
    """
    # 1. Setup dimensions
    N = 128
    num_features = 32  
    
    # 2. Create a random Sparse Matrix A (CSR)
    # Using float32 as required by the compiled CUDA kernel
    A_csr = sp.random(N, N, density=0.1, format='csr', dtype=np.float32)
    
    # 3. Create random Dense Matrix X (N x num_features)
    X_dense = np.random.rand(N, num_features).astype(np.float32)
    
    # 4. Compute Golden Reference (CPU)
    Y_ref = A_csr.dot(X_dense)
    
    # 5. Convert CSR to PBR
    # Fix: Use 8x8 blocks and min_nnz=1 to avoid "all-zero" empty matrices
    pbr_mat = graph_link.csr_to_pbr(
        A_csr, 
        block_rows=8, 
        block_cols=8, 
        min_nnz_per_block=1
    )
    
    # Move PBR metadata to GPU registry
    pbr_mat = pbr_mat.to('cuda')
    
    # 6. Prepare Input X as a Torch Tensor on CUDA
    X_torch = torch.from_numpy(X_dense).to('cuda')
    
    # 7. Run SPMM using the unified entry point
    Y_cuda_torch = graph_link.pbr_matmul(pbr_mat, X_torch)
    
    # 8. Move back to CPU/Numpy for validation
    Y_cuda = Y_cuda_torch.cpu().numpy()
    
    # 9. Compare
    diff = np.abs(Y_ref - Y_cuda).sum()
    max_diff = np.max(np.abs(Y_ref - Y_cuda))
    print(f"\nTotal L1 Difference: {diff}")
    print(f"Max absolute Difference: {max_diff}")
    
    assert np.allclose(Y_cuda, Y_ref, atol=1e-4), f"SPMM Accuracy mismatch! Max diff: {max_diff}"

def test_spmm_large_feature_stride():
    """
    Tests the 'Grid-Stride' loop by using more features than 
    internal kernel constants (usually 128 or 512).
    """
    N = 64
    num_features = 1024 
    
    # Use Identity to verify basic data movement at large strides
    A_csr = sp.eye(N, format='csr', dtype=np.float32)
    X_dense = np.random.rand(N, num_features).astype(np.float32)
    
    Y_ref = A_csr.dot(X_dense)
    
    # Convert and move to GPU
    pbr_mat = graph_link.csr_to_pbr(A_csr, block_rows=8, block_cols=8, min_nnz_per_block=1)
    pbr_mat = pbr_mat.to('cuda')
    
    # Prepare X
    X_torch = torch.from_numpy(X_dense).to('cuda')
    
    # Run
    Y_cuda_torch = graph_link.pbr_matmul(pbr_mat, X_torch)
    Y_cuda = Y_cuda_torch.cpu().numpy()
    
    assert np.allclose(Y_cuda, Y_ref, atol=1e-4)