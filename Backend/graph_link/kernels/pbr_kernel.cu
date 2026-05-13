// //ver4
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdint>

#define TOTAL_SHARED 512
#define THREADS_PER_BLOCK 128
#define FEAT_BLOCK_SIZE 512

template <typename index_t, typename scalar_t, int BLOCK_ROWS, int BLOCK_COLS>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 16) pbr_spmm_zero_idle_kernel(
    const uint32_t num_pbr_blocks,
    const uint32_t features,
    const uint32_t batch_size,
    const uint32_t cols,
    const uint32_t rows,
    const uint64_t* __restrict__ block_codes,
    const index_t* __restrict__ block_coords,
    const index_t* __restrict__ block_offsets,
    const scalar_t* __restrict__ block_data,
    const scalar_t* __restrict__ X,
    scalar_t* __restrict__ Y
) {
    // FIX 1: Reverted to Static Shared Memory. Guarantees allocation without needing host params!
    __shared__ scalar_t smem[TOTAL_SHARED];

    const uint32_t b_idx = blockIdx.x;
    const uint32_t batch_idx = blockIdx.y;
    const uint32_t feat_block_idx = blockIdx.z;
    const uint32_t tx = threadIdx.x;

    if (b_idx >= num_pbr_blocks || batch_idx >= batch_size) return;

    // Load Metadata to Registers
    const uint64_t code = block_codes[b_idx];
    const uint32_t row_origin = block_coords[b_idx * 2];
    const uint32_t col_origin = block_coords[b_idx * 2 + 1];
    const uint32_t data_offset = block_offsets[b_idx];

    // --- 1. SPARSITY SCAN & REGISTER PACKING ---
    uint32_t n_r = 0;
    uint64_t col_mask = 0;
    const uint64_t row_bits = (1ULL << BLOCK_COLS) - 1;

    uint32_t packed_active_rows = 0;
    uint64_t packed_row_offsets = 0;
    uint32_t current_relative_offset = 0;

    #pragma unroll
    for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
        const uint64_t row_mask = (code >> (r * BLOCK_COLS)) & row_bits;
        if (row_mask > 0) {
            packed_active_rows |= (r << (n_r * 3));
            packed_row_offsets |= ((uint64_t)current_relative_offset << (n_r * 7));

            n_r++;
            col_mask |= row_mask;
            current_relative_offset += __popcll(row_mask);
        }
    }

    const uint32_t n_c = __popcll(col_mask);
    if (n_r == 0 || n_c == 0) return;

    uint32_t packed_cols_physical = 0;
    uint32_t packed_cols_logical = 0;
    uint32_t pc_idx = 0;

    #pragma unroll
    for (uint32_t c = 0; c < BLOCK_COLS; ++c) {
        if ((col_mask >> c) & 1) {
            packed_cols_logical |= (pc_idx << (c * 3));
            packed_cols_physical |= (c << (pc_idx * 3));
            pc_idx++;
        }
    }

    // --- 2. DYNAMIC P CALCULATION ---
    uint32_t p = TOTAL_SHARED / (n_r + n_c);
    if (p > features) p = features;

    const uint32_t feat_start = feat_block_idx * FEAT_BLOCK_SIZE;
    const uint32_t feat_end = min(feat_start + FEAT_BLOCK_SIZE, features);

    // --- 3. GRID-STRIDE LOOP OVER FEATURES ---
    for (uint32_t base_f = feat_start; base_f < feat_end; base_f += p) {

        const uint32_t current_p = min(p, feat_end - base_f);

        // FIX 2: Dynamically partition shared memory inside the loop based on current_p
        scalar_t* smem_X = smem;
        scalar_t* smem_Y = smem + (n_c * current_p);

        // STEP A: Cooperative Load
        const uint32_t total_x_elements = n_c * current_p;
        for (uint32_t i = tx; i < total_x_elements; i += THREADS_PER_BLOCK) {
            const uint32_t c_idx = i / current_p;
            const uint32_t f_offset = i % current_p;

            const uint32_t local_c = (packed_cols_physical >> (c_idx * 3)) & 0x7;
            smem_X[i] = X[(batch_idx * cols * features) + ((col_origin + local_c) * features) + base_f + f_offset];
        }

        __syncthreads();

        // STEP B: FLATTENED WORKER POOL
        const uint32_t total_work = n_r * current_p;

        for (uint32_t work_idx = tx; work_idx < total_work; work_idx += THREADS_PER_BLOCK) {
            const uint32_t pr = work_idx / current_p;
            const uint32_t f_in_s = work_idx % current_p;

            const uint32_t actual_r = (packed_active_rows >> (pr * 3)) & 0x7;
            uint32_t local_data_ptr = data_offset + ((packed_row_offsets >> (pr * 7)) & 0x7F);

            const uint64_t row_code = (code >> (actual_r * BLOCK_COLS)) & row_bits;

            scalar_t sum = 0;

            #pragma unroll
            for (uint32_t c = 0; c < BLOCK_COLS; ++c) {
                if ((row_code >> c) & 1) {
                    const uint32_t pc = (packed_cols_logical >> (c * 3)) & 0x7;
                    // FIX 3: Replaced 'p' with 'current_p' to match Step A perfectly!
                    sum += block_data[local_data_ptr] * smem_X[pc * current_p + f_in_s];
                    local_data_ptr++;
                }
            }

            smem_Y[work_idx] = sum;
        }

        __syncthreads();

        // STEP C: Cooperative Atomic Commit
        for (uint32_t work_idx = tx; work_idx < total_work; work_idx += THREADS_PER_BLOCK) {
            const uint32_t pr = work_idx / current_p;
            const uint32_t f_in_s = work_idx % current_p;

            const uint32_t actual_r = (packed_active_rows >> (pr * 3)) & 0x7;
            const uint32_t global_r = row_origin + actual_r;

            atomicAdd(&Y[(batch_idx * rows * features) + (global_r * features) + base_f + f_in_s], smem_Y[work_idx]);
        }

        __syncthreads();
    }
}


template <typename scalar_t>
__global__ void coo_spmm_kernel(
    int nnz,
    int features,
    int batch_size,
    int cols,
    int rows,
    const int32_t* __restrict__ coo_rows,
    const int32_t* __restrict__ coo_cols,
    const scalar_t* __restrict__ coo_vals,
    const scalar_t* __restrict__ X,
    scalar_t* __restrict__ Y
) {
    const int nnz_idx   = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int feat      = blockIdx.z * blockDim.x + threadIdx.x;
    if (feat >= features) return;

    const int32_t row  = coo_rows[nnz_idx];
    const int32_t col  = coo_cols[nnz_idx];
    const scalar_t val = coo_vals[nnz_idx];

    atomicAdd(
        &Y[(batch_idx * rows * features) + (row * features) + feat],
        val * X[(batch_idx * cols * features) + (col * features) + feat]
    );
}


template <typename index_t, typename scalar_t>
void launch_pbr_spmm(
    const int num_pbr_blocks,
    const int features,
    const int batch_size,
    const int cols,
    const int rows,
    const int runtime_block_rows,
    const int runtime_block_cols,
    const uint64_t* block_codes,
    const index_t* block_coords,
    const index_t* block_offsets,
    const scalar_t* block_data,
    const scalar_t* X,
    scalar_t* Y,
    cudaStream_t stream
) {
    // Fast exit for perfectly empty matrices
    if (num_pbr_blocks == 0) {
        return; // Y is already allocated with zeros, just return it!
    }

    // Z-dimension: How many blocks of 512 features do we need?
    int feat_blocks = (features + FEAT_BLOCK_SIZE - 1) / FEAT_BLOCK_SIZE;

    // X = Number of PBR blocks
    // Y = Batch size
    // Z = Feature chunking
    dim3 grid(num_pbr_blocks, batch_size, feat_blocks);
    dim3 block(THREADS_PER_BLOCK);

    // 2. Dispatcher: Route to the statically unrolled template based on runtime vars
    // (Note: No dynamic shared memory size parameter is passed in the <<< >>> because
    // the kernel uses a statically sized __shared__ scalar_t smem[TOTAL_SHARED])
    if (runtime_block_rows == 2 && runtime_block_cols == 2) {
        pbr_spmm_zero_idle_kernel<index_t, scalar_t, 2, 2><<<grid, block, 0, stream>>>(
            num_pbr_blocks, features, batch_size, cols, rows,
            block_codes, block_coords, block_offsets, block_data, X, Y
        );
    } else if (runtime_block_rows == 4 && runtime_block_cols == 4) {
        pbr_spmm_zero_idle_kernel<index_t, scalar_t, 4, 4><<<grid, block, 0, stream>>>(
            num_pbr_blocks, features, batch_size, cols, rows,
            block_codes, block_coords, block_offsets, block_data, X, Y
        );
    } else if (runtime_block_rows == 8 && runtime_block_cols == 8) {
        pbr_spmm_zero_idle_kernel<index_t, scalar_t, 8, 8><<<grid, block, 0, stream>>>(
            num_pbr_blocks, features, batch_size, cols, rows,
            block_codes, block_coords, block_offsets, block_data, X, Y
        );
    } else {
        throw std::invalid_argument("Unsupported block size for PBR SpMM. Only 2x2, 4x4, and 8x8 are supported.");
    }
}


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
) {
    if (nnz == 0) return;
    const int threads = THREADS_PER_BLOCK;
    dim3 grid(nnz, batch_size, (features + threads - 1) / threads);
    coo_spmm_kernel<scalar_t><<<grid, threads, 0, stream>>>(
        nnz, features, batch_size, cols, rows, coo_rows, coo_cols, coo_vals, X, Y
    );
}


// Explicit instantiations
template void launch_pbr_spmm<int32_t, float>(const int, const int, const int, const int, const int, const int, const int, const uint64_t*, const int32_t*, const int32_t*, const float*,  const float*,  float*,  cudaStream_t);
template void launch_pbr_spmm<int64_t, float>(const int, const int, const int, const int, const int, const int, const int, const uint64_t*, const int64_t*, const int64_t*, const float*,  const float*,  float*,  cudaStream_t);
template void launch_pbr_spmm<int32_t, double>(const int, const int, const int, const int, const int, const int, const int, const uint64_t*, const int32_t*, const int32_t*, const double*, const double*, double*, cudaStream_t);
template void launch_pbr_spmm<int64_t, double>(const int, const int, const int, const int, const int, const int, const int, const uint64_t*, const int64_t*, const int64_t*, const double*, const double*, double*, cudaStream_t);

template void launch_coo_spmm<float> (int, int, int, int, int, const int32_t*, const int32_t*, const float*,  const float*,  float*,  cudaStream_t);
template void launch_coo_spmm<double>(int, int, int, int, int, const int32_t*, const int32_t*, const double*, const double*, double*, cudaStream_t);
