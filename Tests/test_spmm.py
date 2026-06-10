import numpy as np
import pytest
import scipy.sparse as sp
import torch

import graph_link


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_csr(N, density, dtype=np.float32, seed=42):
    return sp.random(N, N, density=density, format='csr', dtype=dtype,
                     random_state=np.random.default_rng(seed).integers(2**31))


def _random_x(N, num_features, seed=0):
    return np.random.default_rng(seed).random((N, num_features)).astype(np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def device():
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    return torch.device('cuda')


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('N',            [64, 128, 1024])
@pytest.mark.parametrize('num_features', [8, 64, 256])
@pytest.mark.parametrize('density',      [0.05, 0.15])
@pytest.mark.parametrize('block_size',   [2, 4, 8])
@pytest.mark.parametrize('min_nnz',      [1, 4])
def test_spmm_accuracy(N, num_features, density, block_size, min_nnz, device):
    """PBR SpMM matches scipy reference across all parameter combinations."""
    A_csr = _random_csr(N, density)
    X = _random_x(N, num_features)
    Y_ref = A_csr.dot(X)

    pbr = graph_link.csr_to_pbr(
        A_csr,
        block_rows=block_size,
        block_cols=block_size,
        min_nnz_per_block=min_nnz,
    ).to(device)
    Y_cuda = graph_link.pbr_matmul(pbr, torch.from_numpy(X).to(device)).cpu().numpy()

    assert np.allclose(Y_cuda, Y_ref, atol=1e-4), (
        f"Max diff: {np.max(np.abs(Y_cuda - Y_ref)):.2e}"
    )


@pytest.mark.parametrize('num_features', [3, 5, 6, 7, 9, 12])
@pytest.mark.parametrize('block_size',   [2, 4, 8])
@pytest.mark.parametrize('min_nnz',      [1, 4])
def test_spmm_non_pow2_features(num_features, block_size, min_nnz, device):
    """Feature counts that are NOT a power-of-2 multiple of the kernel's
    shared-memory partition p exercise the short tail chunk of the feature
    loop (PPR hits this whenever the seed count is 3, 5, 6, ...).

    12 is a multiple of the float4 lane width, so it takes the vectorized
    kernel and exercises its tail (vp floors to 2 -> p=8, tail of 4); the
    others take the scalar kernel.
    """
    N = 128
    A_csr = _random_csr(N, density=0.1)
    X = _random_x(N, num_features)
    Y_ref = A_csr.dot(X)

    pbr = graph_link.csr_to_pbr(
        A_csr,
        block_rows=block_size,
        block_cols=block_size,
        min_nnz_per_block=min_nnz,
    ).to(device)
    Y_cuda = graph_link.pbr_matmul(pbr, torch.from_numpy(X).to(device)).cpu().numpy()

    assert np.allclose(Y_cuda, Y_ref, atol=1e-4), (
        f"Max diff: {np.max(np.abs(Y_cuda - Y_ref)):.2e}"
    )
