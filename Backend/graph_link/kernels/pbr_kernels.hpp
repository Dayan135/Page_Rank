#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "ppr_kernels.hpp"

template <typename index_t, typename scalar_t>
void launch_pbr_spmm(
    const int num_pbr_blocks,
    const int features,
    const int batch_size,
    const int cols,
    const int rows,
    const int BLOCK_ROWS,
    const int BLOCK_COLS,
    const uint64_t* block_codes,
    const index_t* block_coords,
    const index_t* block_offsets,
    const scalar_t* block_data,
    const scalar_t* X,
    scalar_t* Y,
    cudaStream_t stream = 0
);

template <typename scalar_t>
void launch_coo_spmm(
    int nnz,
    int features,
    int batch_size,
    int cols,
    int rows,
    const int32_t* coo_rows,
    const int32_t* coo_cols,
    const scalar_t* coo_vals,
    const scalar_t* X,
    scalar_t* Y,
    cudaStream_t stream
);
