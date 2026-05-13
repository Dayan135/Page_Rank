#pragma once
#include <cstdint>
#include <cuda_runtime.h>

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

template <typename scalar_t>
__global__ void batched_missing_mass_kernel(
    const scalar_t* __restrict__ Y,
    scalar_t* __restrict__ col_sums,
    int N, 
    int features
);

template <typename scalar_t>
void launch_ppr_update_normalized(
    const scalar_t* Y,
    scalar_t* X,
    const int* source_nodes,
    scalar_t alpha,
    int N,
    int features,
    scalar_t* errors
);

template <typename scalar_t>
void launch_init_ppr(scalar_t* X, const int* source_nodes, int N, int features);

template <typename scalar_t>
void launch_missing_mass(const scalar_t* Y, scalar_t* col_sums, int N, int features);

template <typename scalar_t>
void launch_ppr_update(
    const scalar_t* Y,
    const scalar_t* curr_X,
    scalar_t* next_X,
    const int* source_nodes,
    const scalar_t* col_sums,
    scalar_t damping,
    int N,
    int features,
    scalar_t* d_errors
);

template <typename scalar_t>
__global__ void init_ppr_sources_kernel(
    scalar_t* __restrict__ X, 
    const int* __restrict__ source_nodes, 
    int N, 
    int features
);

template <typename scalar_t>
__global__ void batched_ppr_update_kernel(
    const scalar_t* __restrict__ Y,
    const scalar_t* __restrict__ curr_X,
    scalar_t* __restrict__ next_X,
    const int* __restrict__ source_nodes,
    const scalar_t* __restrict__ col_sums,
    scalar_t damping,
    int N, 
    int features,
    scalar_t* __restrict__ d_errors
);
