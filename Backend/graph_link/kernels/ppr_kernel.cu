#include "ppr_kernels.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>

// ---------------------------------------------------------------------------
// batched_missing_mass_kernel
//
// Accumulates Y into col_sums using per-block shared memory to reduce
// global atomic pressure.  Each block maintains a local partial sum in
// shared memory and commits once at the end.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void batched_missing_mass_kernel(
    const scalar_t* __restrict__ Y,
    scalar_t*       __restrict__ col_sums,
    int N,
    int features
) {
    extern __shared__ char smem[];
    scalar_t* s_col_sums = reinterpret_cast<scalar_t*>(smem);

    for (int i = threadIdx.x; i < features; i += blockDim.x)
        s_col_sums[i] = static_cast<scalar_t>(0);
    __syncthreads();

    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N * features; i += stride)
        atomicAdd(&s_col_sums[i % features], Y[i]);
    __syncthreads();

    for (int i = threadIdx.x; i < features; i += blockDim.x)
        if (s_col_sums[i] > static_cast<scalar_t>(0))
            atomicAdd(&col_sums[i], s_col_sums[i]);
}

template <typename scalar_t>
void launch_missing_mass(const scalar_t* Y, scalar_t* col_sums, int N, int features) {
    const int threads = 256;
    const int blocks  = std::min(1024, (N * features + threads - 1) / threads);
    const int smem    = features * sizeof(scalar_t);
    cudaMemset(col_sums, 0, features * sizeof(scalar_t));
    batched_missing_mass_kernel<scalar_t><<<blocks, threads, smem>>>(Y, col_sums, N, features);
}

template void launch_missing_mass<float> (const float*,  float*,  int, int);
template void launch_missing_mass<double>(const double*, double*, int, int);


// ---------------------------------------------------------------------------
// batched_ppr_update_kernel
//
// PPR step with missing-mass correction.  Teleports the missing probability
// mass back to the source node for each query.  L1 errors are staged in
// shared memory before a single commit to d_errors per block.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void batched_ppr_update_kernel(
    const scalar_t* __restrict__ Y,
    const scalar_t* __restrict__ curr_X,
    scalar_t*       __restrict__ next_X,
    const int*      __restrict__ source_nodes,
    const scalar_t* __restrict__ col_sums,
    scalar_t damping,
    int N,
    int features,
    scalar_t* __restrict__ d_errors
) {
    extern __shared__ char smem[];
    scalar_t* s_errors = reinterpret_cast<scalar_t*>(smem);

    for (int i = threadIdx.x; i < features; i += blockDim.x)
        s_errors[i] = static_cast<scalar_t>(0);
    __syncthreads();

    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N * features; i += stride) {
        int node = i / features;
        int feat = i % features;

        scalar_t missing_mass  = static_cast<scalar_t>(1.0) - col_sums[feat];
        scalar_t teleport_mass = damping * missing_mass + (static_cast<scalar_t>(1.0) - damping);
        scalar_t new_val       = damping * Y[i];
        if (node == source_nodes[feat])
            new_val += teleport_mass;

        scalar_t diff = new_val - curr_X[i];
        diff = diff < static_cast<scalar_t>(0) ? -diff : diff;
        atomicAdd(&s_errors[feat], diff);
        next_X[i] = new_val;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < features; i += blockDim.x)
        if (s_errors[i] > static_cast<scalar_t>(0))
            atomicAdd(&d_errors[i], s_errors[i]);
}

template <typename scalar_t>
void launch_ppr_update(
    const scalar_t* Y,
    const scalar_t* curr_X,
    scalar_t*       next_X,
    const int*      source_nodes,
    const scalar_t* col_sums,
    scalar_t        damping,
    int             N,
    int             features,
    scalar_t*       d_errors
) {
    const int threads = 256;
    const int blocks  = std::min(1024, (N * features + threads - 1) / threads);
    const int smem    = features * sizeof(scalar_t);
    batched_ppr_update_kernel<scalar_t><<<blocks, threads, smem>>>(
        Y, curr_X, next_X, source_nodes, col_sums, damping, N, features, d_errors
    );
}

template void launch_ppr_update<float> (const float*,  const float*,  float*,  const int*, const float*,  float,  int, int, float*);
template void launch_ppr_update<double>(const double*, const double*, double*, const int*, const double*, double, int, int, double*);


// ---------------------------------------------------------------------------
// ppr_update_normalized_kernel
//
// In-place PPR step for a column-stochastic transition matrix.
// No missing-mass correction needed.  L1 errors are staged in shared memory.
// errors is zeroed by the launcher via cudaMemsetAsync before the kernel runs.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void ppr_update_normalized_kernel(
    const scalar_t* __restrict__ Y,
    scalar_t*       __restrict__ X,
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
    const scalar_t* Y,
    scalar_t*       X,
    const int*      source_nodes,
    scalar_t        alpha,
    int             N,
    int             features,
    scalar_t*       errors
) {
    const int threads = 256;
    const int blocks  = std::min(1024, (N * features + threads - 1) / threads);
    const int smem    = features * sizeof(scalar_t);
    cudaMemsetAsync(errors, 0, features * sizeof(scalar_t));
    ppr_update_normalized_kernel<scalar_t><<<blocks, threads, smem>>>(
        Y, X, source_nodes, alpha, N, features, errors
    );
}

template void launch_ppr_update_normalized<float> (const float*,  float*,  const int*, float,  int, int, float*);
template void launch_ppr_update_normalized<double>(const double*, double*, const int*, double, int, int, double*);


// ---------------------------------------------------------------------------
// init_ppr_sources_kernel
//
// One thread per feature.  Sets the source-node entry to 1.0.
// X must already be zeroed before this kernel runs.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void init_ppr_sources_kernel(
    scalar_t*  __restrict__ X,
    const int* __restrict__ source_nodes,
    int N,
    int features
) {
    int feat = blockIdx.x * blockDim.x + threadIdx.x;
    if (feat >= features) return;
    X[source_nodes[feat] * features + feat] = static_cast<scalar_t>(1.0);
}

template <typename scalar_t>
void launch_init_ppr(scalar_t* X, const int* source_nodes, int N, int features) {
    const int threads = 256;
    const int blocks  = (features + threads - 1) / threads;
    init_ppr_sources_kernel<scalar_t><<<blocks, threads>>>(X, source_nodes, N, features);
}

template void launch_init_ppr<float> (float*,  const int*, int, int);
template void launch_init_ppr<double>(double*, const int*, int, int);
