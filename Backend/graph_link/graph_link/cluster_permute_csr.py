import numpy as np
from scipy import sparse as sp
from torch_amg_native import aggregate_max_cluster_size

from integration.conversions import scipy_csr_to_torch_csr, torch_csr_to_scipy_csr


def cluster_permute_csr(sparse_mat):
    orig_device = sparse_mat.device
    sparse_mat = torch_csr_to_scipy_csr(sparse_mat.cpu())
    clusters, cluster_sizes = aggregate_max_cluster_size(sparse_mat)
    clusters_perm = np.argsort(clusters)

    P = sp.eye(sparse_mat.shape[0], format='csr', dtype=sparse_mat.dtype)
    P = P[clusters_perm, :]

    # Convert to torch
    # Implies moving to GPU because of TestTorchNoGrad infra
    # All operations past these lines will be on GPU
    sparse_mat = scipy_csr_to_torch_csr(sparse_mat)
    P = scipy_csr_to_torch_csr(P)

    sparse_mat = P.t() @ sparse_mat @ P

    sparse_mat = sparse_mat.to(orig_device)
    P = P.to(orig_device)

    return sparse_mat, P
