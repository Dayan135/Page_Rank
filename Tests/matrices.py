import numpy as np
import scipy.sparse as sp


def generate_scattered_block_matrix(system_size: int, block_size: int, profiles: list):
    """
    Populates an N×N matrix with randomly-placed block tiles.

    profiles: list of (n_r, n_c, density) tuples.
      n_r, n_c  – active rows/cols inside each block_size×block_size tile
      density   – fraction of the full grid to fill with this tile type
    """
    grid_dim = system_size // block_size
    total_blocks = grid_dim * grid_dim
    all_rows, all_cols, all_data = [], [], []

    for (n_r, n_c, density) in profiles:
        num_blocks = int(total_blocks * density)
        if num_blocks == 0:
            continue

        block_idx = np.random.randint(0, total_blocks, size=num_blocks, dtype=np.int64)
        grid_r, grid_c = block_idx // grid_dim, block_idx % grid_dim

        local_rows = np.argsort(np.random.rand(num_blocks, block_size), axis=1)[:, :n_r]
        local_cols = np.argsort(np.random.rand(num_blocks, block_size), axis=1)[:, :n_c]

        global_r = (grid_r[:, None] * block_size) + local_rows
        global_c = (grid_c[:, None] * block_size) + local_cols

        r_mesh = np.broadcast_to(global_r[:, :, None], (num_blocks, n_r, n_c)).flatten()
        c_mesh = np.broadcast_to(global_c[:, None, :], (num_blocks, n_r, n_c)).flatten()
        data   = np.random.randn(len(r_mesh)).astype(np.float32)

        all_rows.append(r_mesh)
        all_cols.append(c_mesh)
        all_data.append(data)

    if not all_rows:
        return sp.csr_matrix((system_size, system_size), dtype=np.float32)

    return sp.csr_matrix(
        (np.concatenate(all_data),
         (np.concatenate(all_rows), np.concatenate(all_cols))),
        shape=(system_size, system_size),
    )


def generate_random_sparse(system_size: int, density: float):
    """Memory-efficient uniform random sparse matrix (avoids N² alloc)."""
    nnz  = int(system_size * system_size * density)
    rows = np.random.randint(0, system_size, size=nnz, dtype=np.int32)
    cols = np.random.randint(0, system_size, size=nnz, dtype=np.int32)
    data = np.random.randn(nnz).astype(np.float32)
    return sp.coo_matrix((data, (rows, cols)), shape=(system_size, system_size)).tocsr()


def generate_diag_blocks_plus_noise(
    system_size: int,
    block_size: int = 64,
    diag_density: float = 0.5,
    off_diag_density: float = 0.00005,
):
    """Block-diagonal dense core + random global noise (simulates clustered graphs)."""
    B = block_size
    if system_size % B != 0:
        raise ValueError(f"system_size ({system_size}) must be divisible by block_size ({B})")

    temp = sp.random(system_size, B, density=diag_density, format='coo', dtype=np.float32)
    global_rows_diag = temp.row
    global_cols_diag = (temp.row // B) * B + temp.col
    data_diag = temp.data

    nnz_noise        = int(system_size * system_size * off_diag_density)
    global_rows_noise = np.random.randint(0, system_size, size=nnz_noise, dtype=np.int32)
    global_cols_noise = np.random.randint(0, system_size, size=nnz_noise, dtype=np.int32)
    data_noise        = np.random.randn(nnz_noise).astype(np.float32)

    all_rows = np.concatenate([global_rows_diag, global_rows_noise])
    all_cols = np.concatenate([global_cols_diag, global_cols_noise])
    all_data = np.concatenate([data_diag, data_noise])

    return sp.csr_matrix((all_data, (all_rows, all_cols)), shape=(system_size, system_size))
