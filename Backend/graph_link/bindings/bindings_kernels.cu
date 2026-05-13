#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/SparseCsrTensorUtils.h>

#include "../kernels/pbr_kernels.hpp"

// ---------------------------------------------------------------------------
// SpMM wrapper: zeroes Y asynchronously on s_blocks before launching kernels.
// Block SpMM runs on s_blocks; COO SpMM runs on s_coo concurrently.
// Both streams are created here and destroyed after synchronization.
// ---------------------------------------------------------------------------
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

    const uint64_t* codes_ptr = reinterpret_cast<const uint64_t*>(block_codes.data_ptr<int64_t>());

    // Async-zero Y on s_blocks before block SpMM writes to it.
    // Callers no longer need to call Y.zero_() from Python.
    const size_t y_bytes = (size_t)rows * features * batch_size * sizeof(scalar_t);
    cudaMemsetAsync(Y.data_ptr<scalar_t>(), 0, y_bytes, s_blocks);

    // s_coo must also wait for Y to be zeroed before COO writes to it.
    // We record an event after the memset and make s_coo wait on it.
    cudaEvent_t e_y_cleared;
    cudaEventCreateWithFlags(&e_y_cleared, cudaEventDisableTiming);
    cudaEventRecord(e_y_cleared, s_blocks);
    cudaStreamWaitEvent(s_coo, e_y_cleared);

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

    cudaEventDestroy(e_y_cleared);
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
        errors.data_ptr<scalar_t>(),
        /*stream=*/0
    );
}

// ---------------------------------------------------------------------------
// Full GPU PPR loop — runs all power-iteration steps entirely in C++.
//
// Eliminates per-iteration Python overhead (3 GPU–CPU sync points × 100 iters).
// Uses three persistent streams + two events per PPR run:
//   s_spmm  : memset(Y) → block_SpMM → coo_SpMM
//   s_update: waits for s_spmm via event → memset(errors) → update_kernel
//
// check_interval: check convergence every N iterations (0 = no check, fixed iters).
// Returns (iterations_run, converged) as a Python tuple.
// ---------------------------------------------------------------------------
template <typename index_t, typename scalar_t>
py::tuple run_ppr_cuda_loop_wrapper(
    int num_pbr_blocks, int features, int cols, int rows,
    int BLOCK_ROWS, int BLOCK_COLS,
    at::Tensor block_codes,
    at::Tensor block_coords,
    at::Tensor block_offsets,
    at::Tensor block_data,
    int rem_nnz,
    at::Tensor rem_rows,
    at::Tensor rem_cols,
    at::Tensor rem_vals,
    at::Tensor source_nodes,
    at::Tensor X,
    at::Tensor Y,
    at::Tensor errors,
    double alpha,
    int max_iterations,
    double tolerance,
    int check_interval
) {
    const uint64_t* codes_ptr = reinterpret_cast<const uint64_t*>(block_codes.data_ptr<int64_t>());
    const size_t y_bytes = (size_t)rows * features * sizeof(scalar_t);

    cudaStream_t s_spmm, s_update;
    cudaStreamCreate(&s_spmm);
    cudaStreamCreate(&s_update);

    // Event signalling that s_spmm has finished writing Y (block + COO SpMM done).
    // cudaEventDisableTiming avoids the ~5 µs timing overhead of the default event.
    cudaEvent_t e_spmm_done;
    cudaEventCreateWithFlags(&e_spmm_done, cudaEventDisableTiming);

    int  iters_run = max_iterations;
    bool converged = false;

    for (int t = 0; t < max_iterations; ++t) {

        // 1. Async-zero Y, then launch SpMM kernels — all on s_spmm (ordered).
        cudaMemsetAsync(Y.data_ptr<scalar_t>(), 0, y_bytes, s_spmm);

        launch_pbr_spmm<index_t, scalar_t>(
            num_pbr_blocks, features, /*batch=*/1, cols, rows, BLOCK_ROWS, BLOCK_COLS,
            codes_ptr,
            block_coords.data_ptr<index_t>(),
            block_offsets.data_ptr<index_t>(),
            block_data.data_ptr<scalar_t>(),
            X.data_ptr<scalar_t>(),
            Y.data_ptr<scalar_t>(),
            s_spmm
        );

        if (rem_nnz > 0) {
            launch_coo_spmm<scalar_t>(
                rem_nnz, features, /*batch=*/1, cols, rows,
                rem_rows.data_ptr<int32_t>(),
                rem_cols.data_ptr<int32_t>(),
                rem_vals.data_ptr<scalar_t>(),
                X.data_ptr<scalar_t>(),
                Y.data_ptr<scalar_t>(),
                s_spmm  // same stream: COO starts only after block SpMM & memset
            );
        }

        // 2. Record event after SpMM; s_update waits for it before touching Y.
        cudaEventRecord(e_spmm_done, s_spmm);
        cudaStreamWaitEvent(s_update, e_spmm_done);

        // 3. Update step on s_update (async w.r.t. s_spmm after the wait event).
        launch_ppr_update_normalized<scalar_t>(
            Y.data_ptr<scalar_t>(),
            X.data_ptr<scalar_t>(),
            source_nodes.data_ptr<int32_t>(),
            static_cast<scalar_t>(alpha),
            rows, features,
            errors.data_ptr<scalar_t>(),
            s_update
        );

        // 4. Convergence check — sync s_update, then read errors to CPU.
        if (check_interval > 0 && (t + 1) % check_interval == 0) {
            cudaStreamSynchronize(s_update);
            auto max_err = errors.max().item<scalar_t>();
            if (max_err < static_cast<scalar_t>(tolerance)) {
                iters_run = t + 1;
                converged = true;
                break;
            }
        }
    }

    // Final sync — ensures X is fully written before returning to Python.
    cudaStreamSynchronize(s_update);

    if (!converged && check_interval > 0) {
        // Final convergence check in case max_iterations wasn't a multiple of check_interval.
        auto max_err = errors.max().item<scalar_t>();
        converged = (max_err < static_cast<scalar_t>(tolerance));
    }

    cudaEventDestroy(e_spmm_done);
    cudaStreamDestroy(s_spmm);
    cudaStreamDestroy(s_update);

    return py::make_tuple(iters_run, converged);
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
    m.def("init_ppr_cuda_float",  &init_ppr_cuda_wrapper<float>);
    m.def("init_ppr_cuda_double", &init_ppr_cuda_wrapper<double>);

    // Missing Mass
    m.def("missing_mass_cuda_float",  &missing_mass_cuda_wrapper<float>);
    m.def("missing_mass_cuda_double", &missing_mass_cuda_wrapper<double>);

    // PPR Update (legacy: non-normalized A via col_sums)
    m.def("ppr_update_cuda_float",  &ppr_update_cuda_wrapper<float>);
    m.def("ppr_update_cuda_double", &ppr_update_cuda_wrapper<double>);

    // PPR Update (normalized A: in-place, no col_sums)
    m.def("ppr_update_normalized_cuda_float",  &ppr_update_normalized_wrapper<float>);
    m.def("ppr_update_normalized_cuda_double", &ppr_update_normalized_wrapper<double>);

    // Full GPU PPR loop — entire power iteration in C++, zero Python overhead per iter.
    m.def("run_ppr_cuda_loop_int32_float",  &run_ppr_cuda_loop_wrapper<int32_t, float>);
    m.def("run_ppr_cuda_loop_int64_float",  &run_ppr_cuda_loop_wrapper<int64_t, float>);
    m.def("run_ppr_cuda_loop_int32_double", &run_ppr_cuda_loop_wrapper<int32_t, double>);
    m.def("run_ppr_cuda_loop_int64_double", &run_ppr_cuda_loop_wrapper<int64_t, double>);
}
