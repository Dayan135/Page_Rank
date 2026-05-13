import gc
import torch
import numpy as np
import scipy.sparse as sp
import pytest
from pytest import fixture, mark

import graph_link

@fixture
def damping(): return 0.85

@fixture
def max_iterations(): return 100

@fixture
def tolerance(): return 1e-6

@mark.parametrize('system_size', [65536, 262144]) 
@mark.parametrize('features', [1, 32, 128]) 
class TestBenchmarkPPRGPU:

    @fixture(autouse=True)
    def clean_vram(self):
        yield 
        gc.collect()
        torch.cuda.empty_cache()

    @fixture
    def transition_matrices(self, system_size):
        # 1. FAST, LOW-RAM GRAPH GENERATION
        nnz = system_size * 8
        
        # CREATE A GRAPH WITH HIGH LOCALITY (Like a real social network)
        # 80% of edges go to nearby nodes (dense blocks), 20% are random
        rows = np.random.randint(0, system_size, size=nnz, dtype=np.int32)
        
        # Create 'clustered' columns based on the rows
        offsets = np.random.normal(loc=0, scale=5, size=nnz).astype(np.int32)
        cols = np.clip(rows + offsets, 0, system_size - 1)
        
        # Add a little pure randomness (20%)
        random_mask = np.random.rand(nnz) > 0.8
        cols[random_mask] = np.random.randint(0, system_size, size=np.sum(random_mask))
        
        vals = np.random.rand(nnz).astype(np.float32)
        
        A_scipy = sp.coo_matrix((vals, (rows, cols)), shape=(system_size, system_size)).tocsr()

        # Convert to PyTorch CSR (GPU) for cuSPARSE baseline
        A_torch = torch.sparse_csr_tensor(
            torch.tensor(A_scipy.indptr, dtype=torch.int32),
            torch.tensor(A_scipy.indices, dtype=torch.int32),
            torch.tensor(A_scipy.data, dtype=torch.float32),
            size=(system_size, system_size),
            device='cuda'
        )
        return A_scipy, A_torch

    @fixture
    def source_nodes(self, system_size, features):
        return np.random.randint(0, system_size, size=features).astype(np.int32)

    # =========================================================================
    # 1. Torch cuSPARSE Baseline (GPU - Native PyTorch)
    # =========================================================================
    def test_benchmark_torch_ppr(self, transition_matrices, source_nodes, system_size, features, damping, max_iterations, tolerance, benchmark):
        _, A_torch = transition_matrices
        source_tensor = torch.tensor(source_nodes, dtype=torch.int64, device='cuda')

        @torch.no_grad() # Crucial: Prevent PyTorch from saving history and OOM-ing the VRAM
        def runner():
            X = torch.zeros((system_size, features), device='cuda')
            X[source_tensor, torch.arange(features)] = 1.0
            
            for _ in range(max_iterations):
                Y = torch.sparse.mm(A_torch, X)
                
                col_sums = torch.sum(Y, dim=0)
                X = damping * Y
                teleport = (damping * (1.0 - col_sums)) + (1.0 - damping)
                X[source_tensor, torch.arange(features)] += teleport
                
            torch.cuda.synchronize()

        benchmark.pedantic(runner, rounds=20, warmup_rounds=5)

    # =========================================================================
    # 2. The Challenger: Your PBR Batched Engine (GPU - Custom CUDA)
    # =========================================================================
    @mark.parametrize('block_size', [2,4]) 
    def test_benchmark_pbr_engine(self, transition_matrices, source_nodes, system_size, features, damping, max_iterations, tolerance, block_size, benchmark):
        A_scipy, _ = transition_matrices
        
        pbr_mat_obj = graph_link.csr_to_pbr(A_scipy, block_rows=block_size, block_cols=block_size, min_nnz_per_block=1)
        pbr_device = pbr_mat_obj.to('cuda')
        source_tensor = torch.tensor(source_nodes, dtype=torch.int32, device='cuda')

        @torch.no_grad()
        def runner():
            graph_link.run_personalized_pagerank(
                pbr_device, 
                source_tensor, 
                damping_factor=damping, 
                max_iterations=max_iterations, 
                tolerance=tolerance
            )
            torch.cuda.synchronize()

        benchmark.pedantic(runner, rounds=20, warmup_rounds=5)
        