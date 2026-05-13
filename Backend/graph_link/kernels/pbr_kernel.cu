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
        pbr_spmm_zero_idle_kernel<index_t, scalar_t, 2, 2><<<grid, block>>>(
            num_pbr_blocks, features, batch_size, cols, rows, 
            block_codes, block_coords, block_offsets, block_data, X, Y
        );
    } else if (runtime_block_rows == 4 && runtime_block_cols == 4) {
        pbr_spmm_zero_idle_kernel<index_t, scalar_t, 4, 4><<<grid, block>>>(
            num_pbr_blocks, features, batch_size, cols, rows, 
            block_codes, block_coords, block_offsets, block_data, X, Y
        );
    } else if (runtime_block_rows == 8 && runtime_block_cols == 8) {
        pbr_spmm_zero_idle_kernel<index_t, scalar_t, 8, 8><<<grid, block>>>(
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
    const int threads = THREADS_PER_BLOCK_VER3;
    dim3 grid(nnz, batch_size, (features + threads - 1) / threads);
    coo_spmm_kernel<scalar_t><<<grid, threads, 0, stream>>>(
        nnz, features, batch_size, cols, rows, coo_rows, coo_cols, coo_vals, X, Y
    );
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

// PPR update for column-stochastic A: X = alpha*Y + (1-alpha)*e_s  (in-place, no col_sums needed)
// Also accumulates per-feature L1 convergence error in shared memory.
template <typename scalar_t>
__global__ void ppr_update_normalized_kernel(
    const scalar_t* __restrict__ Y,
    scalar_t*       __restrict__ X,          // updated in-place
    const int*      __restrict__ source_nodes,
    scalar_t alpha,
    int N,
    int features,
    scalar_t* __restrict__ errors
) {
    extern __shared__ char smem[];
    scalar_t* s_errors = reinterpret_cast<scalar_t*>(smem);

    for (int f = threadIdx.x; f < features; f += blockDim.x)
        s_errors[f] = static_cast<scalar_t>(0);
    __syncthreads();

    const scalar_t complement = static_cast<scalar_t>(1) - alpha;
    const int stride = blockDim.x * gridDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N * features; i += stride) {
        const int node = i / features;
        const int feat = i % features;

        scalar_t new_val = alpha * Y[i];
        if (node == source_nodes[feat])
            new_val += complement;

        scalar_t diff = new_val - X[i];
        diff = diff < static_cast<scalar_t>(0) ? -diff : diff;
        atomicAdd(&s_errors[feat], diff);
        X[i] = new_val;
    }
    __syncthreads();

    for (int f = threadIdx.x; f < features; f += blockDim.x)
        if (s_errors[f] > static_cast<scalar_t>(0))
            atomicAdd(&errors[f], s_errors[f]);
}

template <typename scalar_t>
void launch_ppr_update_normalized(
    const scalar_t* Y, scalar_t* X, const int* source_nodes,
    scalar_t alpha, int N, int features, scalar_t* errors
) {
    const int threads = 256;
    const int blocks  = std::min(1024, (N * features + threads - 1) / threads);
    const int smem    = features * sizeof(scalar_t);
    cudaMemsetAsync(errors, 0, features * sizeof(scalar_t));
    ppr_update_normalized_kernel<scalar_t><<<blocks, threads, smem>>>(
        Y, X, source_nodes, alpha, N, features, errors
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
template void launch_pbr_spmm<int32_t, float>(const int, const int, const int, const int, const int, const int, const int, const uint64_t*, const int32_t*, const int32_t*, const float*,  const float*,  float*,  cudaStream_t);
template void launch_pbr_spmm<int64_t, float>(const int, const int, const int, const int, const int, const int, const int, const uint64_t*, const int64_t*, const int64_t*, const float*,  const float*,  float*,  cudaStream_t);
template void launch_pbr_spmm<int32_t, double>(const int, const int, const int, const int, const int, const int, const int, const uint64_t*, const int32_t*, const int32_t*, const double*, const double*, double*, cudaStream_t);
template void launch_pbr_spmm<int64_t, double>(const int, const int, const int, const int, const int, const int, const int, const uint64_t*, const int64_t*, const int64_t*, const double*, const double*, double*, cudaStream_t);

template void launch_coo_spmm<float> (int, int, int, int, int, const int32_t*, const int32_t*, const float*,  const float*,  float*,  cudaStream_t);
template void launch_coo_spmm<double>(int, int, int, int, int, const int32_t*, const int32_t*, const double*, const double*, double*, cudaStream_t);

template void launch_ppr_update_normalized<float> (const float*,  float*,  const int*, float,  int, int, float*);
template void launch_ppr_update_normalized<double>(const double*, double*, const int*, double, int, int, double*);

// Missing Mass Kernel
template __global__ void batched_missing_mass_kernel<float>(const float*, float*, int, int);
template __global__ void batched_missing_mass_kernel<double>(const double*, double*, int, int);

// PPR Update Kernel
template __global__ void batched_ppr_update_kernel<float>(const float*, const float*, float*, const int*, const float*, float, int, int, float*);
template __global__ void batched_ppr_update_kernel<double>(const double*, const double*, double*, const int*, const double*, double, int, int, double*);