#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "pbr_matrix/pbr_matrix.hpp"
#include "kernels/pbr_kernel.hpp" // Ensure all your launchers are declared here

// A struct to hold the batched results
template <typename scalar_t>
struct BatchedPPRResult {
    std::vector<scalar_t> scores; // Flattened 2D array [N * features]
    int iterations_run;
    bool converged;
};

template <typename index_t, typename scalar_t>
BatchedPPRResult<scalar_t> run_batched_personalized_pagerank(
    const pbr_device_matrix_t<index_t, scalar_t>& d_mat,
    const std::vector<int>& host_source_nodes,
    scalar_t damping_factor = 0.85,
    int max_iterations = 100,
    scalar_t tolerance = 1e-6
) {
    int N = d_mat.rows;
    int features = host_source_nodes.size();
    int batch_size = 1; // Keeping batching simple for the sparse matrix dimension

    // 1. Allocate GPU Memory for the execution state
    scalar_t *d_X_curr, *d_X_next, *d_Y;
    scalar_t *d_col_sums, *d_errors;
    int *d_source_nodes;

    cudaMalloc(&d_X_curr, N * features * sizeof(scalar_t));
    cudaMalloc(&d_X_next, N * features * sizeof(scalar_t));
    cudaMalloc(&d_Y, N * features * sizeof(scalar_t));
    cudaMalloc(&d_col_sums, features * sizeof(scalar_t));
    cudaMalloc(&d_errors, features * sizeof(scalar_t));
    cudaMalloc(&d_source_nodes, features * sizeof(int));

    // 2. Initialize Data
    cudaMemcpy(d_source_nodes, host_source_nodes.data(), features * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMemset(d_X_curr, 0, N * features * sizeof(scalar_t));
    // Launch our tiny init kernel to place 1.0s
    launch_init_ppr(d_X_curr, d_source_nodes, N, features);

    std::vector<scalar_t> host_errors(features);
    int iter = 0;
    bool all_converged = false;

    // 3. THE MASTER LOOP
    for (; iter < max_iterations; ++iter) {
        
        // Safety: Zero out ALL intermediate buffers every iteration
        cudaMemset(d_Y, 0, N * features * sizeof(scalar_t));
        cudaMemset(d_col_sums, 0, features * sizeof(scalar_t));
        cudaMemset(d_errors, 0, features * sizeof(scalar_t));

        // Step A: SpMM
        launch_pbr_spmm<index_t, scalar_t>(
            d_mat.num_blocks, features, batch_size, 
            d_mat.cols, d_mat.rows, d_mat.block_rows, d_mat.block_cols,
            d_mat.d_block_codes, d_mat.d_block_coords, d_mat.d_block_offsets, d_mat.d_block_data,
            d_X_curr, d_Y
        );

        // Step B: Find Sinkhole Mass
        // We pass the explicit pointer and shared memory requirements
        launch_missing_mass(d_Y, d_col_sums, N, features);

        // Step C: Apply Damping & Teleport 
        launch_ppr_update(
            d_Y, d_X_curr, d_X_next, d_source_nodes, d_col_sums, 
            damping_factor, N, features, d_errors
        );

        // Step D: Convergence Check
        // During debugging, check every iteration to see where it stalls
        cudaMemcpy(host_errors.data(), d_errors, features * sizeof(scalar_t), cudaMemcpyDeviceToHost);
        
        all_converged = true;
        for (int f = 0; f < features; ++f) {
            // DEBUG: 
            std::cout << "Iter " << iter << " Feat " << f << " Err: " << host_errors[f] << std::endl;
            if (host_errors[f] > tolerance) {
                all_converged = false;
            }
        }

        if (all_converged) {
            iter++; 
            break;
        }

        std::swap(d_X_curr, d_X_next);
    }

    // 4. Return the Results
    BatchedPPRResult<scalar_t> result;
    result.scores.resize(N * features);
    result.iterations_run = iter;
    result.converged = all_converged;

    // Copy the final vectors back to host RAM
    // If all_converged is true, the final answers are in d_X_next (which just got swapped to d_X_curr)
    cudaMemcpy(result.scores.data(), d_X_curr, N * features * sizeof(scalar_t), cudaMemcpyDeviceToHost);

    return result;
}