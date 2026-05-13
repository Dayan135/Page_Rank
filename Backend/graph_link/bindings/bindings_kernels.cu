#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/SparseCsrTensorUtils.h>

#include "../kernels/pbr_kernels.hpp"

// The Bridge: Converts PyTorch ATen GPU tensors into raw C++ pointers for your CUDA kernel
template <typename index_t, typename scalar_t>
void pbr_spmm_cuda_wrapper(
    int num_pbr_blocks, int features, int batch_size, int cols, int rows,
    int BLOCK_ROWS, int BLOCK_COLS,
    at::Tensor block_codes,
    at::Tensor block_coords,
    at::Tensor block_offsets,
    at::Tensor block_data,
    at::Tensor X,
    at::Tensor Y
) {
    // PyTorch doesn't have an explicit uint64 tensor type, so we use int64 in Python
    // and safely cast the raw memory pointer to uint64_t* for the bitwise CUDA math.
    const uint64_t* codes_ptr = reinterpret_cast<const uint64_t*>(block_codes.data_ptr<int64_t>());
    launch_pbr_spmm<index_t, scalar_t>(
        num_pbr_blocks, features, batch_size, cols, rows, BLOCK_ROWS, BLOCK_COLS,
        codes_ptr,
        block_coords.data_ptr<index_t>(),
        block_offsets.data_ptr<index_t>(),
        block_data.data_ptr<scalar_t>(),
        X.data_ptr<scalar_t>(),
        Y.data_ptr<scalar_t>()
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
    m.def("pbr_spmm_cuda_int32_float", &pbr_spmm_cuda_wrapper<int32_t, float>);
    m.def("pbr_spmm_cuda_int64_float", &pbr_spmm_cuda_wrapper<int64_t, float>);
    m.def("pbr_spmm_cuda_int32_double", &pbr_spmm_cuda_wrapper<int32_t, double>);
    m.def("pbr_spmm_cuda_int64_double", &pbr_spmm_cuda_wrapper<int64_t, double>);
    
    // Init PPR
    m.def("init_ppr_cuda_float", &init_ppr_cuda_wrapper<float>);
    m.def("init_ppr_cuda_double", &init_ppr_cuda_wrapper<double>);
    
    // Missing Mass
    m.def("missing_mass_cuda_float", &missing_mass_cuda_wrapper<float>);
    m.def("missing_mass_cuda_double", &missing_mass_cuda_wrapper<double>);
    
    // PPR Update
    m.def("ppr_update_cuda_float", &ppr_update_cuda_wrapper<float>);
    m.def("ppr_update_cuda_double", &ppr_update_cuda_wrapper<double>);
}
