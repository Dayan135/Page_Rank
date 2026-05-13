// //ver4
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdint>

#define THREADS_PER_BLOCK_VER3 128

// HYPERPARAMETER: How many blocks to launch in the Z-dimension (features)
// Tune this to find the sweet spot for your GPU's SM scheduler!
#define FEAT_BLOCK_SIZE 512

template <typename index_t, typename scalar_t, int BLOCK_ROWS, int BLOCK_COLS>
__global__ void pbr_spmm_shared_reg_kernel(
    const int num_pbr_blocks,
    const int features,
    const int batch_size,
    const int cols,
    const int rows,
    const uint64_t* __restrict__ block_codes,
    const index_t* __restrict__ block_coords,
    const index_t* __restrict__ block_offsets,
    const scalar_t* __restrict__ block_data,
    const scalar_t* __restrict__ X,
    scalar_t* __restrict__ Y
) {
    __shared__ scalar_t smem_X[BLOCK_COLS * THREADS_PER_BLOCK_VER3];

    int b_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tx = threadIdx.x; 
    
    if (b_idx >= num_pbr_blocks || batch_idx >= batch_size) return;

    // Load block metadata ONCE for the entire block
    uint64_t code = block_codes[b_idx];
    index_t row_origin = block_coords[b_idx * 2];
    index_t col_origin = block_coords[b_idx * 2 + 1];
    index_t data_offset = block_offsets[b_idx];
    uint64_t row_mask = (1ULL << BLOCK_COLS) - 1;

    // The Grid-Stride Stride: How far the block jumps after each iteration
    // (Total blocks in Z) * (Threads per block)
    int stride = gridDim.z * THREADS_PER_BLOCK_VER3;

    // GRID-STRIDE LOOP: The block loops until all features are processed
    for (int base_feat = blockIdx.z * THREADS_PER_BLOCK_VER3; base_feat < features; base_feat += stride) {
        
        int feat = base_feat + tx;
        bool valid_feat = (feat < features);

        // 1. Load the X Chunk into Shared Memory perfectly coalesced
        if (valid_feat) {
            #pragma unroll
            for (int c = 0; c < BLOCK_COLS; ++c) {
                smem_X[c * THREADS_PER_BLOCK_VER3 + tx] = X[(batch_idx * cols * features) + ((col_origin + c) * features) + feat];
            }
        }
        
        __syncthreads();

        // 2. Fast Register Accumulation
        if (valid_feat) {
            index_t row_data_offset = data_offset;
            #pragma unroll
            for (int r = 0; r < BLOCK_ROWS; ++r) {
                uint64_t row_code = (code >> (r * BLOCK_COLS)) & row_mask;
                if (row_code == 0) continue;

                index_t current_data_offset = row_data_offset;
                uint64_t temp_code = row_code;
                
                scalar_t accum_y = 0.0;

                while (temp_code > 0) {
                    int bit_idx = __ffsll(temp_code) - 1;
                    scalar_t val = block_data[current_data_offset];
                    
                    accum_y += val * smem_X[bit_idx * THREADS_PER_BLOCK_VER3 + tx];

                    current_data_offset++;
                    temp_code &= ~(1ULL << bit_idx);
                }
                
                // 3. Direct Atomic Commit
                int global_row = row_origin + r;
                atomicAdd(&Y[(batch_idx * rows * features) + (global_row * features) + feat], accum_y);
                
                // Advance the data offset for the next row
                row_data_offset += __popcll(row_code);
            }
        }
        
        // CRITICAL BARRIER: Prevent fast threads from looping around and overwriting 
        // smem_X while slow threads are still doing math!
        __syncthreads(); 
    }
}

// Host Launcher Function
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
    scalar_t* Y
) {
    int threads_per_block = THREADS_PER_BLOCK_VER3; 
    
    // We now use your macro directly instead of calculating it based on the features variable
    dim3 grid(num_pbr_blocks, batch_size, (features + threads_per_block - 1) / threads_per_block);
    dim3 block(threads_per_block);

    if (runtime_block_rows == 2 && runtime_block_cols == 2) {
        pbr_spmm_shared_reg_kernel<index_t, scalar_t, 2, 2><<<grid, block>>>(
            num_pbr_blocks, features, batch_size, cols, rows, 
            block_codes, block_coords, block_offsets, block_data, X, Y
        );
    } else if (runtime_block_rows == 4 && runtime_block_cols == 4) {
        pbr_spmm_shared_reg_kernel<index_t, scalar_t, 4, 4><<<grid, block>>>(
            num_pbr_blocks, features, batch_size, cols, rows, 
            block_codes, block_coords, block_offsets, block_data, X, Y
        );
    } else if (runtime_block_rows == 8 && runtime_block_cols == 8) {
        pbr_spmm_shared_reg_kernel<index_t, scalar_t, 8, 8><<<grid, block>>>(
            num_pbr_blocks, features, batch_size, cols, rows, 
            block_codes, block_coords, block_offsets, block_data, X, Y
        );
    } else {
        throw std::invalid_argument("Unsupported block size.");
    }
}


// 1. Batched Missing Mass Kernel (Shared Memory Reduction)
template <typename scalar_t>
__global__ void batched_missing_mass_kernel(
    const scalar_t* __restrict__ Y,
    scalar_t* __restrict__ col_sums,
    int N, 
    int features
) {
    // Dynamic shared memory allocated at launch
    extern __shared__ char smem[];
    scalar_t* s_col_sums = reinterpret_cast<scalar_t*>(smem);

    // 1. Initialize shared memory for this block to 0
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        s_col_sums[i] = static_cast<scalar_t>(0);
    }
    __syncthreads();

    // 2. Grid-Stride Loop: Accumulate locally in fast shared memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N * features; i += stride) {
        int feat = i % features; 
        atomicAdd(&s_col_sums[feat], Y[i]); 
    }
    __syncthreads();

    // 3. Commit the block's partial sums to Global Memory ONLY ONCE
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        if (s_col_sums[i] > static_cast<scalar_t>(0)) {
            atomicAdd(&col_sums[i], s_col_sums[i]);
        }
    }
}

// 2. The PPR Update Kernel (Shared Memory Reduction for Convergence Errors)
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
) {
    extern __shared__ char smem[];
    scalar_t* s_errors = reinterpret_cast<scalar_t*>(smem);

    // 1. Initialize shared memory for this block's error tracking
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        s_errors[i] = static_cast<scalar_t>(0);
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N * features; i += stride) {
        int node = i / features;
        int feat = i % features;

        // Calculate teleportation math
        scalar_t missing_mass = static_cast<scalar_t>(1.0) - col_sums[feat];
        scalar_t teleport_mass = (damping * missing_mass) + (static_cast<scalar_t>(1.0) - damping);
        scalar_t new_val = damping * Y[i];
        
        if (node == source_nodes[feat]) {
            new_val += teleport_mass;
        }

        scalar_t old_val = curr_X[i];
        next_X[i] = new_val;

        // Calculate absolute error and add to FAST Shared Memory
        scalar_t diff = (new_val > old_val) ? (new_val - old_val) : (old_val - new_val);
        atomicAdd(&s_errors[feat], diff);
    }
    __syncthreads();

    // 2. Commit the block's total error to Global Memory ONLY ONCE
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        if (s_errors[i] > static_cast<scalar_t>(0)) {
            atomicAdd(&d_errors[i], s_errors[i]);
        }
    }
}

// --- LAUNCHERS ---

template <typename scalar_t>
void launch_missing_mass(const scalar_t* Y, scalar_t* col_sums, int N, int features) {
    int total_threads = N * features;
    int threads_per_block = 256;
    // Cap blocks at 1024 to force the stride loop to do heavy lifting, saving block scheduling overhead
    int blocks = std::min(1024, (total_threads + threads_per_block - 1) / threads_per_block);
    
    // Calculate required shared memory dynamically based on the batch size
    int shared_mem_bytes = features * sizeof(scalar_t);

    cudaMemset(col_sums, 0, features * sizeof(scalar_t));
    batched_missing_mass_kernel<scalar_t><<<blocks, threads_per_block, shared_mem_bytes>>>(
        Y, col_sums, N, features
    );
}

template <typename scalar_t>
void launch_ppr_update(const scalar_t* Y, const scalar_t* curr_X, scalar_t* next_X, const int* source_nodes, const scalar_t* col_sums, scalar_t damping, int N, int features, scalar_t* d_errors) {
    int total_threads = N * features;
    int threads_per_block = 256;
    int blocks = std::min(1024, (total_threads + threads_per_block - 1) / threads_per_block);
    
    // Shared memory size for error tracking
    int shared_mem_bytes = features * sizeof(scalar_t);

    batched_ppr_update_kernel<scalar_t><<<blocks, threads_per_block, shared_mem_bytes>>>(
        Y, curr_X, next_X, source_nodes, col_sums, damping, N, features, d_errors
    );
}

// Kernel: Sets the source node for each feature to 1.0, everything else is 0.0 (done via cudaMemset)
template <typename scalar_t>
__global__ void init_ppr_sources_kernel(
    scalar_t* __restrict__ X, 
    const int* __restrict__ source_nodes, 
    int N, 
    int features
) {
    int feat = blockIdx.x * blockDim.x + threadIdx.x;
    if (feat >= features) return;

    int source_node = source_nodes[feat];
    // Memory layout: [N][features]
    X[source_node * features + feat] = static_cast<scalar_t>(1.0);
}

// Launcher
template <typename scalar_t>
void launch_init_ppr(scalar_t* X, const int* source_nodes, int N, int features) {
    int threads = 256;
    int blocks = (features + threads - 1) / threads;
    init_ppr_sources_kernel<scalar_t><<<blocks, threads>>>(X, source_nodes, N, features);
}

// Add Explicit Instantiations at the bottom:
template void launch_init_ppr<float>(float*, const int*, int, int);
template void launch_init_ppr<double>(double*, const int*, int, int);

// And add these explicit instantiations for the LAUNCHERS:
template void launch_ppr_update<float>(const float*, const float*, float*, const int*, const float*, float, int, int, float*);
template void launch_ppr_update<double>(const double*, const double*, double*, const int*, const double*, double, int, int, double*);

template void launch_missing_mass<float>(const float*, float*, int, int);
template void launch_missing_mass<double>(const double*, double*, int, int);


// Explicit instantiations
template void launch_pbr_spmm<int32_t, float>(const int, const int, const int, const int, const int, const int, const int, const uint64_t*, const int32_t*, const int32_t*, const float*, const float*, float*);
template void launch_pbr_spmm<int64_t, float>(const int, const int, const int, const int, const int, const int, const int, const uint64_t*, const int64_t*, const int64_t*, const float*, const float*, float*);
template void launch_pbr_spmm<int32_t, double>(const int, const int, const int, const int, const int, const int, const int, const uint64_t*, const int32_t*, const int32_t*, const double*, const double*, double*);
template void launch_pbr_spmm<int64_t, double>(const int, const int, const int, const int, const int, const int, const int, const uint64_t*, const int64_t*, const int64_t*, const double*, const double*, double*);

// Missing Mass Kernel
template __global__ void batched_missing_mass_kernel<float>(const float*, float*, int, int);
template __global__ void batched_missing_mass_kernel<double>(const double*, double*, int, int);

// PPR Update Kernel
template __global__ void batched_ppr_update_kernel<float>(const float*, const float*, float*, const int*, const float*, float, int, int, float*);
template __global__ void batched_ppr_update_kernel<double>(const double*, const double*, double*, const int*, const double*, double, int, int, double*);