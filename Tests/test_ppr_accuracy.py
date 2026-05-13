import numpy as np
import networkx as nx
import pytest
import scipy.sparse as sp
import torch

import graph_link


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_graph_and_csr(N: int, p: float, seed: int = 42):
    """
    Directed Erdős-Rényi graph → (G, column-stochastic CSR).

    NetworkX gives A[i,j] = edge i→j (row = source).
    PPR needs column-stochastic: A_ppr[i,j] = prob of reaching i from j,
    so A_ppr = normalize_cols(A_nx.T).
    """
    G = nx.erdos_renyi_graph(N, p, directed=True, seed=seed)
    A_nx = nx.to_scipy_sparse_array(G, nodelist=range(N), format='csr', dtype=np.float32)
    A_col = sp.csr_matrix(A_nx.T)
    return G, graph_link.normalize_transition_matrix(A_col)


def _nx_ppr(G, N, src, alpha, tol):
    """NetworkX PPR reference with dangling mass redistributed uniformly (matches our A_norm)."""
    pers = {n: (1.0 if n == src else 0.0) for n in range(N)}
    dangling_uniform = {n: 1.0 / N for n in range(N)}
    scores = nx.pagerank(
        G,
        alpha=alpha,
        personalization=pers,
        dangling=dangling_uniform,
        tol=tol,
        max_iter=500,
    )
    return np.array([scores[n] for n in range(N)], dtype=np.float32)


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

@pytest.mark.parametrize('N,p',        [(50, 0.10), (100, 0.05)])
@pytest.mark.parametrize('alpha',      [0.85, 0.90])
@pytest.mark.parametrize('block_size', [2, 4, 8])
@pytest.mark.parametrize('min_nnz',    [1, 4])
def test_ppr_accuracy(N, p, alpha, block_size, min_nnz, device):
    """CUDA PPR matches NetworkX reference for multiple source nodes."""
    G, A_norm = _build_graph_and_csr(N, p)
    sources = [0, N // 4, N // 2, N - 1]

    pbr = graph_link.csr_to_pbr(
        A_norm,
        block_rows=block_size,
        block_cols=block_size,
        min_nnz_per_block=min_nnz,
    ).to(device)

    X, _, converged = graph_link.run_personalized_pagerank(
        pbr,
        torch.tensor(sources, dtype=torch.int32, device=device),
        alpha=alpha,
        max_iterations=500,
        tolerance=1e-6,
    )
    assert converged, f"Did not converge in 500 iterations (N={N}, p={p}, alpha={alpha})"

    X_cpu = X.cpu().numpy()

    for k, src in enumerate(sources):
        cuda_col = X_cpu[:, k]
        nx_col   = _nx_ppr(G, N, src, alpha, tol=1e-6)

        assert np.isclose(cuda_col.sum(), 1.0, atol=1e-4), (
            f"Mass not conserved for src={src}: sum={cuda_col.sum():.6f}"
        )
        l1 = np.abs(cuda_col - nx_col).sum()
        assert np.allclose(cuda_col, nx_col, atol=1e-3), (
            f"PPR mismatch src={src} (N={N}, alpha={alpha}, block={block_size}, "
            f"min_nnz={min_nnz}): L1={l1:.4f}"
        )
