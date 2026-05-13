#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/SparseCsrTensorUtils.h>

#include "../kernels/pbr_kernels.hpp"

// Combined SpMM wrapper: launches the block kernel and the COO remainder kernel
// on two independent CUDA streams so they can overlap on the GPU.
template <typename index_t, typename scalar_t>
void pbr_full_spmm_cuda_wrapper(
    int num_pbr_blocks, int features, int batch_size, int cols, int rows,
    int BLOCK_ROWS, int BLOCK_COLS,
    at::Tensor block_codes,
    at::Tensor block_coords,
    at::Tensor block_offsets,
    at::Tensor block_data,
    int rem_nnz,
    at::Tensor rem_rows,
    at::Tensor rem_cols,
    at::Tensor rem_vals,
    at::Tensor X,
    at::Tensor Y
) {
    cudaStream_t s_blocks, s_coo;
    cudaStreamCreate(&s_blocks);
    cudaStreamCreate(&s_coo);

    // PyTorch stores block_codes as int64; reinterpret as uint64 for bitwise ops.
    const uint64_t* codes_ptr = reinterpret_cast<const uint64_t*>(block_codes.data_ptr<int64_t>());

    launch_pbr_spmm<index_t, scalar_t>(
        num_pbr_blocks, features, batch_size, cols, rows, BLOCK_ROWS, BLOCK_COLS,
        codes_ptr,
        block_coords.data_ptr<index_t>(),
        block_offsets.data_ptr<index_t>(),
        block_data.data_ptr<scalar_t>(),
        X.data_ptr<scalar_t>(),
        Y.data_ptr<scalar_t>(),
        s_blocks
    );

    if (rem_nnz > 0) {
        launch_coo_spmm<scalar_t>(
            rem_nnz, features, batch_size, cols, rows,
            rem_rows.data_ptr<int32_t>(),
            rem_cols.data_ptr<int32_t>(),
            rem_vals.data_ptr<scalar_t>(),
            X.data_ptr<scalar_t>(),
            Y.data_ptr<scalar_t>(),
            s_coo
        );
    }

    cudaStreamSynchronize(s_blocks);
    if (rem_nnz > 0) cudaStreamSynchronize(s_coo);

    cudaStreamDestroy(s_blocks);
    cudaStreamDestroy(s_coo);
}

// PPR update for column-stochastic A (in-place, no col_sums)
template <typename scalar_t>
void ppr_update_normalized_wrapper(
    at::Tensor Y, at::Tensor X, at::Tensor source_nodes,
    double alpha, int N, int features, at::Tensor errors
) {
    launch_ppr_update_normalized<scalar_t>(
        Y.data_ptr<scalar_t>(),
        X.data_ptr<scalar_t>(),
        source_nodes.data_ptr<int32_t>(),
        static_cast<scalar_t>(alpha),
        N, features,
        errors.data_ptr<scalar_t>()
    );
}

// 1. Init PPR Wrapper
template <typename scalar_t>
void init_ppr_cuda_wrapper(at::Tensor X, at::Tensor source_nodes, int N, int features) {
    launch_init_ppr<scalar_t>(
        X.data_ptr<scalar_t>(),
        source_nodes.data_ptr<int32_t>(), // PyTorch int32 corresponds to standard C++ int
        N, features
    );
}

// 2. Missing Mass Wrapper
template <typename scalar_t>
void missing_mass_cuda_wrapper(at::Tensor Y, at::Tensor col_sums, int N, int features) {
    launch_missing_mass<scalar_t>(
        Y.data_ptr<scalar_t>(),
        col_sums.data_ptr<scalar_t>(),
        N, features
    );
}

// 3. PPR Update Wrapper
template <typename scalar_t>
void ppr_update_cuda_wrapper(
    at::Tensor Y, at::Tensor curr_X, at::Tensor next_X,
    at::Tensor source_nodes, at::Tensor col_sums,
    double damping, int N, int features, at::Tensor errors
) {
    launch_ppr_update<scalar_t>(
        Y.data_ptr<scalar_t>(),
        curr_X.data_ptr<scalar_t>(),
        next_X.data_ptr<scalar_t>(),
        source_nodes.data_ptr<int32_t>(),
        col_sums.data_ptr<scalar_t>(),
        static_cast<scalar_t>(damping),
        N, features,
        errors.data_ptr<scalar_t>()
    );
}

void bind_cuda_functions(py::module_& m) {
    m.def("pbr_full_spmm_cuda_int32_float",  &pbr_full_spmm_cuda_wrapper<int32_t, float>);
    m.def("pbr_full_spmm_cuda_int64_float",  &pbr_full_spmm_cuda_wrapper<int64_t, float>);
    m.def("pbr_full_spmm_cuda_int32_double", &pbr_full_spmm_cuda_wrapper<int32_t, double>);
    m.def("pbr_full_spmm_cuda_int64_double", &pbr_full_spmm_cuda_wrapper<int64_t, double>);
    
    // Init PPR
    m.def("init_ppr_cuda_float", &init_ppr_cuda_wrapper<float>);
    m.def("init_ppr_cuda_double", &init_ppr_cuda_wrapper<double>);
    
    // Missing Mass
    m.def("missing_mass_cuda_float", &missing_mass_cuda_wrapper<float>);
    m.def("missing_mass_cuda_double", &missing_mass_cuda_wrapper<double>);
    
    // PPR Update (legacy: handles non-normalized A via col_sums)
    m.def("ppr_update_cuda_float",  &ppr_update_cuda_wrapper<float>);
    m.def("ppr_update_cuda_double", &ppr_update_cuda_wrapper<double>);

    // PPR Update (normalized A: no col_sums, in-place)
    m.def("ppr_update_normalized_cuda_float",  &ppr_update_normalized_wrapper<float>);
    m.def("ppr_update_normalized_cuda_double", &ppr_update_normalized_wrapper<double>);
}
