#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Personalized PageRank (PPR) CUDA Kernel Launchers
//
// All launchers operate on a batched PPR workload: N nodes × F simultaneous
// source queries (the "features" dimension).  Memory layout for X and Y is
// row-major [N][F].
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// launch_init_ppr
//
// One-hot initialisation of the PPR score matrix X for a batch of source
// queries.  Assumes X has already been zeroed (e.g. via cudaMemset). Sets
//   X[source_nodes[f] * features + f] = 1.0
// for every feature f, making each column a delta distribution centred on
// its source node.
//
// Parameters:
//   X            – device buffer [N * features], must be pre-zeroed
//   source_nodes – device array [features], one source node index per query
//   N            – number of graph nodes
//   features     – number of simultaneous PPR queries (batch width)
// ---------------------------------------------------------------------------
template <typename scalar_t>
void launch_init_ppr(scalar_t* X, const int* source_nodes, int N, int features);


// ---------------------------------------------------------------------------
// launch_ppr_update_normalized
//
// Single PPR power-iteration step for a column-stochastic transition matrix.
// Updates X in-place:
//   X[node * F + f] = alpha * Y[node * F + f]
//                   + (1 - alpha)   [only at node == source_nodes[f]]
//
// Also accumulates the per-feature L1 convergence error |new - old| into
// `errors`, which is zeroed internally via cudaMemsetAsync before the
// kernel runs.  Convergence is reached when all errors[f] < tolerance.
//
// Use this variant when A is already column-stochastic (no dangling nodes).
// No missing-mass correction is needed.
//
// Parameters:
//   Y            – SpMM result  Y = A @ X_prev, device [N * features]
//   X            – current PPR scores, updated in-place, device [N * features]
//   source_nodes – device array [features], source node per query
//   alpha        – damping factor (typically 0.85)
//   N            – number of graph nodes
//   features     – batch width
//   errors       – device array [features], receives per-feature L1 error
// ---------------------------------------------------------------------------
template <typename scalar_t>
void launch_ppr_update_normalized(
    const scalar_t* Y,
    scalar_t*       X,
    const int*      source_nodes,
    scalar_t        alpha,
    int             N,
    int             features,
    scalar_t*       errors
);


// ---------------------------------------------------------------------------
// launch_missing_mass
//
// Computes the column sums of Y (the SpMM result) into col_sums[f]:
//   col_sums[f] = sum_{i=0}^{N-1} Y[i * features + f]
//
// For a perfectly column-stochastic matrix every col_sums[f] == 1.
// Values below 1 indicate dangling nodes whose outgoing probability is
// missing and must be redistributed by launch_ppr_update.
//
// col_sums is zeroed internally (synchronous cudaMemset) before accumulation.
//
// Parameters:
//   Y        – SpMM result, device [N * features]
//   col_sums – output device array [features]
//   N        – number of graph nodes
//   features – batch width
// ---------------------------------------------------------------------------
template <typename scalar_t>
void launch_missing_mass(const scalar_t* Y, scalar_t* col_sums, int N, int features);


// ---------------------------------------------------------------------------
// launch_ppr_update
//
// Single PPR power-iteration step with explicit missing-mass correction.
// For each element (node, f):
//   missing_mass  = 1 - col_sums[f]
//   teleport_mass = damping * missing_mass + (1 - damping)
//   next_X = damping * Y[node, f] + teleport_mass  (if node == source_nodes[f])
//          = damping * Y[node, f]                   (otherwise)
//
// The per-feature L1 convergence error |next_X - curr_X| is accumulated
// into d_errors (caller is responsible for zeroing before use).
//
// Use this variant when the transition matrix may have dangling nodes.
// Call launch_missing_mass first to obtain col_sums.
//
// Parameters:
//   Y            – SpMM result, device [N * features]
//   curr_X       – PPR scores from the previous iteration, device [N * features]
//   next_X       – output PPR scores for this iteration, device [N * features]
//   source_nodes – device array [features]
//   col_sums     – column sums of Y from launch_missing_mass, device [features]
//   damping      – damping factor (typically 0.85)
//   N            – number of graph nodes
//   features     – batch width
//   d_errors     – device array [features], receives per-feature L1 error
// ---------------------------------------------------------------------------
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
);
