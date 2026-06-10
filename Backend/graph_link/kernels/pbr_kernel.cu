#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

// 16-byte vector type per scalar: float4 (4 lanes) / double2 (2 lanes). The
// vectorized kernel processes `lanes` contiguous features per thread so the
// hot shared-memory loads/stores become a single LDS.128 / STS.128.
template <typename scalar_t> struct PbrVec;
template <> struct PbrVec<float>  { using type = float4;  static constexpr int lanes = 4; };
template <> struct PbrVec<double> { using type = double2; static constexpr int lanes = 2; };

template <typename index_t, typename scalar_t,
          int BLOCK_ROWS, int BLOCK_COLS,
          int TOTAL_SHARED, int THREADS_PER_BLOCK, int FEAT_BLOCK_SIZE>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 16) pbr_spmm_zero_idle_kernel(
    const uint32_t num_pbr_blocks,
    const uint32_t features,
    const uint32_t batch_size,
    const uint32_t cols,
    const uint32_t rows,
    const int64_t* __restrict__ block_codes,
    const index_t* __restrict__ block_coords,
    const index_t* __restrict__ block_offsets,
    const scalar_t* __restrict__ block_data,
    const scalar_t* __restrict__ X,
    scalar_t* __restrict__ Y
) {
    __shared__ scalar_t smem[TOTAL_SHARED];

    const uint32_t b_idx = blockIdx.x;
    const uint32_t batch_idx = blockIdx.y;
    const uint32_t feat_block_idx = blockIdx.z;
    const uint32_t tx = threadIdx.x;

    if (b_idx >= num_pbr_blocks || batch_idx >= batch_size) return;

    // Load Metadata to Registers
    const uint64_t code = static_cast<uint64_t>(block_codes[b_idx]);
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
    // p is floored to a power of 2 so inner-loop /p and %p become bit-shift
    // and mask ops. The shared buffers always use stride p; in the tail chunk
    // (features not a multiple of p) lanes past valid_p load as 0 and are
    // skipped at commit, so any feature count is handled.
    const uint32_t p_log2 = 31u - __clz(min(TOTAL_SHARED / (n_r + n_c), features));
    const uint32_t p      = 1u << p_log2;
    const uint32_t p_mask = p - 1u;

    const uint32_t feat_start = feat_block_idx * FEAT_BLOCK_SIZE;
    const uint32_t feat_end = min(feat_start + FEAT_BLOCK_SIZE, features);

    // --- 3. GRID-STRIDE LOOP OVER FEATURES ---
    for (uint32_t base_f = feat_start; base_f < feat_end; base_f += p) {

        const uint32_t valid_p = min(p, feat_end - base_f);

        scalar_t* smem_X = smem;
        scalar_t* smem_Y = smem + (n_c * p);

        // STEP A: Cooperative Load (lanes beyond valid_p are zero-filled)
        const uint32_t total_x_elements = n_c * p;
        for (uint32_t i = tx; i < total_x_elements; i += THREADS_PER_BLOCK) {
            const uint32_t c_idx   = i >> p_log2;
            const uint32_t f_offset = i & p_mask;

            const uint32_t local_c = (packed_cols_physical >> (c_idx * 3)) & 0x7;
            smem_X[i] = (f_offset < valid_p)
                ? X[(batch_idx * cols * features) + ((col_origin + local_c) * features) + base_f + f_offset]
                : scalar_t(0);
        }

        __syncthreads();

        // STEP B: FLATTENED WORKER POOL
        const uint32_t total_work = n_r * p;

        for (uint32_t work_idx = tx; work_idx < total_work; work_idx += THREADS_PER_BLOCK) {
            const uint32_t pr    = work_idx >> p_log2;
            const uint32_t f_in_s = work_idx & p_mask;

            const uint32_t actual_r = (packed_active_rows >> (pr * 3)) & 0x7;
            uint32_t local_data_ptr = data_offset + ((packed_row_offsets >> (pr * 7)) & 0x7F);

            const uint64_t row_code = (code >> (actual_r * BLOCK_COLS)) & row_bits;

            scalar_t sum = 0;

            #pragma unroll
            for (uint32_t c = 0; c < BLOCK_COLS; ++c) {
                if ((row_code >> c) & 1) {
                    const uint32_t pc = (packed_cols_logical >> (c * 3)) & 0x7;
                    sum += __ldg(&block_data[local_data_ptr]) * smem_X[pc * p + f_in_s];
                    local_data_ptr++;
                }
            }

            smem_Y[work_idx] = sum;
        }

        __syncthreads();

        // STEP C: Cooperative Atomic Commit (skip zero-padded lanes)
        for (uint32_t work_idx = tx; work_idx < total_work; work_idx += THREADS_PER_BLOCK) {
            const uint32_t pr    = work_idx >> p_log2;
            const uint32_t f_in_s = work_idx & p_mask;
            if (f_in_s >= valid_p) continue;

            const uint32_t actual_r = (packed_active_rows >> (pr * 3)) & 0x7;
            const uint32_t global_r = row_origin + actual_r;

            atomicAdd(&Y[(batch_idx * rows * features) + (global_r * features) + base_f + f_in_s], smem_Y[work_idx]);
        }

        __syncthreads();
    }
}


// Vectorized variant: each thread owns `L` (= PbrVec lanes) contiguous features.
// Requires features % L == 0 so the per-column feature run is L-aligned in both
// global X and shared memory; the launcher guarantees this before dispatching.
template <typename index_t, typename scalar_t,
          int BLOCK_ROWS, int BLOCK_COLS,
          int TOTAL_SHARED, int THREADS_PER_BLOCK, int FEAT_BLOCK_SIZE>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 16) pbr_spmm_vec_kernel(
    const uint32_t num_pbr_blocks,
    const uint32_t features,
    const uint32_t batch_size,
    const uint32_t cols,
    const uint32_t rows,
    const int64_t* __restrict__ block_codes,
    const index_t* __restrict__ block_coords,
    const index_t* __restrict__ block_offsets,
    const scalar_t* __restrict__ block_data,
    const scalar_t* __restrict__ X,
    scalar_t* __restrict__ Y
) {
    using vec_t = typename PbrVec<scalar_t>::type;
    constexpr int L = PbrVec<scalar_t>::lanes;
    union lane_u { vec_t v; scalar_t s[L]; };

    __shared__ __align__(16) scalar_t smem[TOTAL_SHARED];

    const uint32_t b_idx = blockIdx.x;
    const uint32_t batch_idx = blockIdx.y;
    const uint32_t feat_block_idx = blockIdx.z;
    const uint32_t tx = threadIdx.x;

    if (b_idx >= num_pbr_blocks || batch_idx >= batch_size) return;

    const uint64_t code = static_cast<uint64_t>(block_codes[b_idx]);
    const uint32_t row_origin = block_coords[b_idx * 2];
    const uint32_t col_origin = block_coords[b_idx * 2 + 1];
    const uint32_t data_offset = block_offsets[b_idx];

    // --- 1. SPARSITY SCAN & REGISTER PACKING (identical to scalar kernel) ---
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

    // --- 2. DYNAMIC P CALCULATION (vp floored to power of 2 → bit-op division) ---
    // vp_log2 / vp_mask replace /vp and %vp in inner loops. The shared buffers
    // always use stride p; in the tail chunk (features not a multiple of p)
    // vectors past valid_vp load as 0 and are skipped at commit. Vectors are
    // never partially valid: features and FEAT_BLOCK_SIZE are multiples of L.
    const uint32_t vp_log2 = 31u - __clz(min(TOTAL_SHARED / (n_r + n_c), features) / L);
    const uint32_t vp_mask = (1u << vp_log2) - 1u;
    const uint32_t vp      = 1u << vp_log2;       // vectors per column (full chunk)
    const uint32_t p       = vp * L;

    const uint32_t feat_start = feat_block_idx * FEAT_BLOCK_SIZE;
    const uint32_t feat_end = min(feat_start + FEAT_BLOCK_SIZE, features);

    // --- 3. GRID-STRIDE LOOP OVER FEATURES (in units of L) ---
    for (uint32_t base_f = feat_start; base_f < feat_end; base_f += p) {

        const uint32_t valid_vp = min(p, feat_end - base_f) / L;  // valid vectors

        scalar_t* smem_X = smem;
        scalar_t* smem_Y = smem + (n_c * p);

        // STEP A: Cooperative vectorized load of X -> smem_X (zero-fill past valid_vp)
        const uint32_t total_xv = n_c * vp;
        for (uint32_t i = tx; i < total_xv; i += THREADS_PER_BLOCK) {
            const uint32_t c_idx   = i >> vp_log2;
            const uint32_t v_idx   = i & vp_mask;
            const uint32_t f_off   = v_idx * L;
            const uint32_t local_c = (packed_cols_physical >> (c_idx * 3)) & 0x7;

            vec_t xv;
            if (v_idx < valid_vp) {
                xv = *reinterpret_cast<const vec_t*>(
                    &X[(batch_idx * cols * features) + ((col_origin + local_c) * features) + base_f + f_off]);
            } else {
                lane_u zero;
                #pragma unroll
                for (int l = 0; l < L; ++l) zero.s[l] = 0;
                xv = zero.v;
            }
            *reinterpret_cast<vec_t*>(&smem_X[c_idx * p + f_off]) = xv;
        }

        __syncthreads();

        // STEP B: FLATTENED WORKER POOL (one float4/double2 per thread)
        const uint32_t total_vwork = n_r * vp;
        for (uint32_t vwork = tx; vwork < total_vwork; vwork += THREADS_PER_BLOCK) {
            const uint32_t pr    = vwork >> vp_log2;
            const uint32_t f_off = (vwork & vp_mask) * L;

            const uint32_t actual_r = (packed_active_rows >> (pr * 3)) & 0x7;
            uint32_t local_data_ptr = data_offset + ((packed_row_offsets >> (pr * 7)) & 0x7F);
            const uint64_t row_code = (code >> (actual_r * BLOCK_COLS)) & row_bits;

            lane_u acc;
            #pragma unroll
            for (int l = 0; l < L; ++l) acc.s[l] = 0;

            #pragma unroll
            for (uint32_t c = 0; c < BLOCK_COLS; ++c) {
                if ((row_code >> c) & 1) {
                    const uint32_t pc = (packed_cols_logical >> (c * 3)) & 0x7;
                    const scalar_t coef = __ldg(&block_data[local_data_ptr]);

                    lane_u xu;
                    xu.v = *reinterpret_cast<const vec_t*>(&smem_X[pc * p + f_off]);
                    #pragma unroll
                    for (int l = 0; l < L; ++l) acc.s[l] += coef * xu.s[l];

                    local_data_ptr++;
                }
            }

            *reinterpret_cast<vec_t*>(&smem_Y[pr * p + f_off]) = acc.v;
        }

        __syncthreads();

        // STEP C: Cooperative atomic commit (vector load from smem, L scalar atomics;
        // zero-padded vectors are skipped)
        for (uint32_t vwork = tx; vwork < total_vwork; vwork += THREADS_PER_BLOCK) {
            const uint32_t pr    = vwork >> vp_log2;
            const uint32_t v_idx = vwork & vp_mask;
            if (v_idx >= valid_vp) continue;
            const uint32_t f_off = v_idx * L;

            const uint32_t actual_r = (packed_active_rows >> (pr * 3)) & 0x7;
            const uint32_t global_r = row_origin + actual_r;

            lane_u yu;
            yu.v = *reinterpret_cast<const vec_t*>(&smem_Y[pr * p + f_off]);
            #pragma unroll
            for (int l = 0; l < L; ++l) {
                atomicAdd(&Y[(batch_idx * rows * features) + (global_r * features) + base_f + f_off + l], yu.s[l]);
            }
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
    const int64_t* block_codes,
    const index_t* block_coords,
    const index_t* block_offsets,
    const scalar_t* block_data,
    const scalar_t* X,
    scalar_t* Y,
    cudaStream_t stream
) {
    if (num_pbr_blocks == 0) {
        return;
    }

    constexpr int TOTAL_SHARED     = 2048;
    constexpr int THREADS_PER_BLOCK = 128;
    constexpr int FEAT_BLOCK_SIZE  = 512;
    constexpr int L = PbrVec<scalar_t>::lanes;

    const int feat_blocks = (features + FEAT_BLOCK_SIZE - 1) / FEAT_BLOCK_SIZE;
    const dim3 grid(num_pbr_blocks, batch_size, feat_blocks);
    const dim3 block(THREADS_PER_BLOCK);

    // Use the vectorized kernel when the feature count divides the vector width
    // (it always does for the tested 8/32/512); otherwise fall back to scalar.
    const bool use_vec = (features % L == 0);

    #define PBR_LAUNCH(BR, BC)                                                                      \
        do {                                                                                       \
            if (use_vec)                                                                           \
                pbr_spmm_vec_kernel<index_t, scalar_t, BR, BC, TOTAL_SHARED, THREADS_PER_BLOCK,     \
                                    FEAT_BLOCK_SIZE><<<grid, block, 0, stream>>>(                   \
                    num_pbr_blocks, features, batch_size, cols, rows,                               \
                    block_codes, block_coords, block_offsets, block_data, X, Y);                    \
            else                                                                                   \
                pbr_spmm_zero_idle_kernel<index_t, scalar_t, BR, BC, TOTAL_SHARED,                  \
                                          THREADS_PER_BLOCK, FEAT_BLOCK_SIZE><<<grid, block, 0,      \
                                          stream>>>(                                                \
                    num_pbr_blocks, features, batch_size, cols, rows,                               \
                    block_codes, block_coords, block_offsets, block_data, X, Y);                    \
        } while (0)

    if (runtime_block_rows == 2 && runtime_block_cols == 2) {
        PBR_LAUNCH(2, 2);
    } else if (runtime_block_rows == 4 && runtime_block_cols == 4) {
        PBR_LAUNCH(4, 4);
    } else if (runtime_block_rows == 8 && runtime_block_cols == 8) {
        PBR_LAUNCH(8, 8);
    } else {
        throw std::invalid_argument("Unsupported block size for PBR SpMM. Only 2x2, 4x4, and 8x8 are supported.");
    }

    #undef PBR_LAUNCH
}

template void launch_pbr_spmm<int32_t, float>(const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int32_t*, const int32_t*, const float*, const float*, float*, cudaStream_t);
template void launch_pbr_spmm<int64_t, float>(const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int64_t*, const int64_t*, const float*, const float*, float*, cudaStream_t);
template void launch_pbr_spmm<int32_t, double>(const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int32_t*, const int32_t*, const double*, const double*, double*, cudaStream_t);
template void launch_pbr_spmm<int64_t, double>(const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int64_t*, const int64_t*, const double*, const double*, double*, cudaStream_t);


// --- CSR remainder kernel (warp-per-row / CSR-Vector) ---
//
// One warp = 32 lanes owns one output row for ALL features.
// The sparse structure (csr_vals, col_ind) is loaded once per row and reused
// across all feature iterations; the scalar kernel would reload it once per
// 32-feature tile, giving ceil(features/32) redundant passes through L2/DRAM
// for those arrays. At features=512 that is a 16× saving.
//
// X reads remain coalesced: all 32 lanes share the same col_ind[j] but read
// consecutive features (f, f+1, …, f+31) → one 128-byte transaction per nnz.
//
// atomicAdd is still required: a row can receive contributions from the block
// kernel running concurrently on another stream. Within this kernel each row
// belongs to exactly one warp, so there is no intra-kernel contention.
//
// Grid: (ceil(rows / WARPS_PER_BLOCK), batch_size), Block: (WARPS_PER_BLOCK * 32).

constexpr int CSR_WARP_SIZE      = 32;
constexpr int CSR_WARPS_PER_BLOCK = 8;   // 256 threads/block; 6 blocks/SM = full occ on RTX 3090

template <typename index_t, typename scalar_t>
__global__ void csr_spmm_kernel(
    const uint32_t rows,
    const uint32_t features,
    const uint32_t batch_size,
    const uint32_t cols,
    const index_t* __restrict__ indptr,
    const index_t* __restrict__ col_ind,
    const scalar_t* __restrict__ csr_vals,
    const scalar_t* __restrict__ X,
    scalar_t* __restrict__ Y
) {
    const uint32_t warp_id = threadIdx.x / CSR_WARP_SIZE;
    const uint32_t lane    = threadIdx.x % CSR_WARP_SIZE;
    const uint32_t row     = blockIdx.x * CSR_WARPS_PER_BLOCK + warp_id;
    const uint32_t batch   = blockIdx.y;

    if (row >= rows || batch >= batch_size) return;

    const index_t rs = indptr[row];
    const index_t re = indptr[row + 1];
    if (rs == re) return;

    const scalar_t* x_ptr = X + batch * cols * features;
    scalar_t*       y_ptr = Y + batch * rows * features + row * features;

    // Each lane owns a distinct feature offset.  The +CSR_WARP_SIZE stride
    // covers the entire feature dimension in one pass per row.
    for (uint32_t f = lane; f < features; f += CSR_WARP_SIZE) {
        scalar_t sum = 0;
        for (index_t j = rs; j < re; ++j)
            sum += csr_vals[j] * x_ptr[col_ind[j] * features + f];
        atomicAdd(&y_ptr[f], sum);
    }
}

template <typename index_t, typename scalar_t>
void launch_csr_spmm(
    const int rows,
    const int features,
    const int batch_size,
    const int cols,
    const index_t* indptr,
    const index_t* col_ind,
    const scalar_t* csr_vals,
    const scalar_t* X,
    scalar_t* Y,
    cudaStream_t stream
) {
    if (rows == 0 || features == 0 || batch_size == 0) return;
    const dim3 grid((rows + CSR_WARPS_PER_BLOCK - 1) / CSR_WARPS_PER_BLOCK, batch_size);
    const dim3 block(CSR_WARPS_PER_BLOCK * CSR_WARP_SIZE);
    csr_spmm_kernel<index_t, scalar_t><<<grid, block, 0, stream>>>(
        rows, features, batch_size, cols, indptr, col_ind, csr_vals, X, Y);
}

template void launch_csr_spmm<int32_t, float>(const int, const int, const int, const int, const int32_t*, const int32_t*, const float*, const float*, float*, cudaStream_t);
template void launch_csr_spmm<int64_t, float>(const int, const int, const int, const int, const int64_t*, const int64_t*, const float*, const float*, float*, cudaStream_t);
template void launch_csr_spmm<int32_t, double>(const int, const int, const int, const int, const int32_t*, const int32_t*, const double*, const double*, double*, cudaStream_t);
template void launch_csr_spmm<int64_t, double>(const int, const int, const int, const int, const int64_t*, const int64_t*, const double*, const double*, double*, cudaStream_t);
