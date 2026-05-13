import gc
import torch
import numpy as np
import scipy.sparse as sp
import pytest
from pytest import fixture, mark

import graph_link

@fixture
def alpha(): return 0.85

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
        nnz = system_size * 8

        # Clustered graph: 80% short-range edges (dense blocks), 20% random
        rows = np.random.randint(0, system_size, size=nnz, dtype=np.int32)
        offsets = np.random.normal(loc=0, scale=5, size=nnz).astype(np.int32)
        cols = np.clip(rows + offsets, 0, system_size - 1)
        random_mask = np.random.rand(nnz) > 0.8
        cols[random_mask] = np.random.randint(0, system_size, size=int(random_mask.sum()))
        vals = np.random.rand(nnz).astype(np.float32)

        A_scipy = sp.coo_matrix((vals, (rows, cols)), shape=(system_size, system_size)).tocsr()

        # Normalize to column-stochastic for PPR (both backends use the same matrix)
        A_norm = graph_link.normalize_transition_matrix(A_scipy)

        A_torch = torch.sparse_csr_tensor(
            torch.tensor(A_norm.indptr,  dtype=torch.int32),
            torch.tensor(A_norm.indices, dtype=torch.int32),
            torch.tensor(A_norm.data,    dtype=torch.float32),
            size=(system_size, system_size),
            device='cuda'
        )
        return A_norm, A_torch

    @fixture
    def source_nodes(self, system_size, features):
        return np.random.randint(0, system_size, size=features).astype(np.int32)

    # =========================================================================
    # 1a. Torch cuSPARSE — fixed iterations, no convergence check
    #     (pure SpMM throughput; single sync at the end)
    # =========================================================================
    def test_benchmark_torch_ppr(self, transition_matrices, source_nodes, system_size, features, alpha, max_iterations, benchmark):
        _, A_torch = transition_matrices
        source_tensor = torch.tensor(source_nodes, dtype=torch.int64, device='cuda')
        feat_idx = torch.arange(features, device='cuda')

        @torch.no_grad()
        def runner():
            X = torch.zeros((system_size, features), device='cuda')
            X[source_tensor, feat_idx] = 1.0

            for _ in range(max_iterations):
                Y = torch.sparse.mm(A_torch, X)
                X = alpha * Y
                X[source_tensor, feat_idx] += (1.0 - alpha)

            torch.cuda.synchronize()

        benchmark.pedantic(runner, rounds=20, warmup_rounds=5)

    # =========================================================================
    # 1b. Torch cuSPARSE — with per-iteration convergence check
    #     (fair comparison for test_benchmark_pbr_engine below)
    # =========================================================================
    def test_benchmark_torch_ppr_with_convergence(self, transition_matrices, source_nodes, system_size, features, alpha, max_iterations, tolerance, benchmark):
        _, A_torch = transition_matrices
        source_tensor = torch.tensor(source_nodes, dtype=torch.int64, device='cuda')
        feat_idx = torch.arange(features, device='cuda')

        @torch.no_grad()
        def runner():
            X = torch.zeros((system_size, features), device='cuda')
            X[source_tensor, feat_idx] = 1.0

            for _ in range(max_iterations):
                Y = torch.sparse.mm(A_torch, X)
                new_X = alpha * Y
                new_X[source_tensor, feat_idx] += (1.0 - alpha)
                errors = (new_X - X).abs().sum(dim=0)  # per-feature L1 error
                X = new_X
                if errors.max().item() < tolerance:
                    break

            torch.cuda.synchronize()

        benchmark.pedantic(runner, rounds=20, warmup_rounds=5)

    # =========================================================================
    # 2a. PBR Engine — fixed iterations, no convergence check
    #     (apples-to-apples vs test_benchmark_torch_ppr)
    # =========================================================================
    @mark.parametrize('block_size', [2, 4])
    def test_benchmark_pbr_fixed_iters(self, transition_matrices, source_nodes, system_size, features, alpha, max_iterations, block_size, benchmark):
        A_norm, _ = transition_matrices
        pbr_mat_obj = graph_link.csr_to_pbr(A_norm, block_rows=block_size, block_cols=block_size, min_nnz_per_block=1)
        pbr_device = pbr_mat_obj.to('cuda')
        source_tensor = torch.tensor(source_nodes, dtype=torch.int64, device='cuda')
        feat_idx = torch.arange(features, device='cuda')

        @torch.no_grad()
        def runner():
            X = torch.zeros((system_size, features), dtype=torch.float32, device='cuda')
            X[source_tensor, feat_idx] = 1.0

            for _ in range(max_iterations):
                Y = graph_link.pbr_matmul(pbr_device, X)
                X = alpha * Y
                X[source_tensor, feat_idx] += (1.0 - alpha)

            torch.cuda.synchronize()

        benchmark.pedantic(runner, rounds=20, warmup_rounds=5)

    # =========================================================================
    # 2b. PBR Engine — full PPR with per-iteration convergence check
    #     (apples-to-apples vs test_benchmark_torch_ppr_with_convergence)
    # =========================================================================
    @mark.parametrize('block_size', [2, 4])
    def test_benchmark_pbr_engine(self, transition_matrices, source_nodes, system_size, features, alpha, max_iterations, tolerance, block_size, benchmark):
        A_norm, _ = transition_matrices

        pbr_mat_obj = graph_link.csr_to_pbr(A_norm, block_rows=block_size, block_cols=block_size, min_nnz_per_block=1)
        pbr_device = pbr_mat_obj.to('cuda')
        source_tensor = torch.tensor(source_nodes, dtype=torch.int32, device='cuda')

        @torch.no_grad()
        def runner():
            graph_link.run_personalized_pagerank(
                pbr_device,
                source_tensor,
                alpha=alpha,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
            torch.cuda.synchronize()

        benchmark.pedantic(runner, rounds=20, warmup_rounds=5)
