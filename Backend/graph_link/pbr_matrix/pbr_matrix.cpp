#include "pbr_matrix.hpp"

#include <iostream>
#include <cmath>
#include <algorithm>

using std::pair, std::make_pair;
using std::make_move_iterator, std::move;
using std::sqrt;
using std::unordered_map;
using std::vector;

//bitmap
// Helper to count bits set in the mask (Replaces bitset.count())
inline int count_set_bits(uint64_t n) {
    return __builtin_popcountll(n);
}
//bitmap
// Helper to check a specific bit (Replaces bitset.test())
inline bool is_bit_set(uint64_t mask, int bit) {
    return (mask >> bit) & 1ULL;
}


template <typename index_t, typename scalar_t>
void pbr_batched_matmul_cpu(const pbr_matrix_t<index_t, scalar_t>& pbr_mat,
                            const py::array_t<scalar_t, py::array::c_style> x,
                            py::array_t<scalar_t, py::array::c_style> y,
                            const int batch_size,
                            const int width) {
    
    // Extract pointers and strides from py::array_t
    const scalar_t* x_ptr = x.data();
    scalar_t* y_ptr = y.mutable_data();

    const int x_batch_stride = pbr_mat.cols * width;
    const int y_batch_stride = pbr_mat.rows * width;

    // Initialize output tensor Y to zero
    std::fill(y_ptr, y_ptr + (batch_size * y_batch_stride), static_cast<scalar_t>(0));

    for (int b = 0; b < batch_size; ++b) {
        const scalar_t* curr_x = x_ptr + (b * x_batch_stride);
        scalar_t* curr_y = y_ptr + (b * y_batch_stride);

        // --- Process Compressed Blocks ---
        index_t data_idx = 0;
        for (size_t block_idx = 0; block_idx < pbr_mat.block_codes.size(); ++block_idx) {
            const block_code_t code = pbr_mat.block_codes[block_idx];
            const auto& coord = pbr_mat.block_coords[block_idx];

            for (index_t r = 0; r < pbr_mat.block_rows; ++r) {
                const index_t global_row = coord.row + r;
                if (global_row >= pbr_mat.rows) continue;

                for (index_t c = 0; c < pbr_mat.block_cols; ++c) {
                    const index_t global_col = coord.col + c;
                    if (global_col >= pbr_mat.cols) continue;

                    if (is_bit_set(code, r * pbr_mat.block_cols + c)) {
                        const scalar_t val = pbr_mat.block_data[data_idx++];
                        
                        for (int w = 0; w < width; ++w) {
                            curr_y[global_row * width + w] += val * curr_x[global_col * width + w];
                        }
                    }
                }
            }
        }

        // --- Process Remainder COO ---
        for (const auto& elem : pbr_mat.remainder_coo) {
            for (int w = 0; w < width; ++w) {
                curr_y[elem.row * width + w] += elem.val * curr_x[elem.col * width + w];
            }
        }
    }
}


template <typename index_t, typename scalar_t>
void analyze_csr_blocks(const index_t block_rows, const index_t block_cols,
                        const index_t rows, const index_t cols,
                        const index_t*  indptr, const index_t*  indices,
                        pbr_stats_t<index_t, scalar_t>& pbr_stats) {
    const index_t num_col_blocks = (cols + block_cols - 1) / block_cols;

    // A stripe is a set of rows that are processed together,
    // essentially a row of blocks.
    index_t last_stripe_index = 0;
    typename pbr_stats_t<index_t, scalar_t>::block_code_map stripe_block_codes;
    vector<index_t> stripe_block_indices;
    for (index_t row = 0; row < rows; ++row) {
        const index_t block_row_offset = row / block_rows;
        const index_t row_offset = row % block_rows;

        for (index_t nnz_idx = indptr[row]; nnz_idx < indptr[row + 1]; ++nnz_idx) {
            const index_t col = indices[nnz_idx];
            const index_t block_col_offset = col / block_cols;
            const index_t col_offset = col % block_cols;
            const index_t block_index = block_row_offset * num_col_blocks + block_col_offset;
            const index_t bit_index = row_offset * block_cols + col_offset;

            const auto& code = stripe_block_codes.find(block_index);
            if (code == stripe_block_codes.end()) {
                const index_t bit_index = row_offset * block_cols + col_offset;
                stripe_block_indices.push_back(block_index);
                //bitmap
                stripe_block_codes[block_index] = (1ULL << bit_index); // Bitwise set
                // stripe_block_codes[block_index].set(bit_index);
            } else {
                //bitmap
                code->second |= (1ULL << bit_index); // Bitwise OR
                // code->second.set(bit_index);
            }
        }

        auto& block_stats = pbr_stats.block_stats;

        // Process the stripe if we are about to move to a new stripe
        const index_t next_stripe_index = (row + 1) / block_rows;
        if (next_stripe_index != last_stripe_index) {
            for (const auto& block_index : stripe_block_indices) {
                const index_t block_coord_row = block_index / num_col_blocks * block_rows;
                const index_t block_coord_col = block_index % num_col_blocks * block_cols;
                
                //bitmap
                const block_code_t block_code = stripe_block_codes[block_index];
                // const auto& block_code = stripe_block_codes.find(block_index)->second.to_ulong();

                // Record this block code as having been seen once more
                // and store the coordinates of the block
                block_stats[block_code].occurrence++;
                block_stats[block_code].block_coords.push_back({block_coord_row, block_coord_col});
            }

            stripe_block_codes.clear();
            stripe_block_indices.clear();

            last_stripe_index = next_stripe_index;
        }
    }
}

template <typename index_t, typename scalar_t>
void process_block_stats(const index_t nnz, const index_t cols, const index_t min_nnz_coverage,
                         pbr_stats_t<index_t, scalar_t>& pbr_stats) {
    index_t remainder_nnz = 0;
    index_t accounted_blocks = 0;
    index_t accounted_patterns = 0;

    for (auto& [block_code, block_desc] : pbr_stats.block_stats) {
        const index_t occurrence = block_desc.occurrence;
        const int code_bits = __builtin_popcountll(block_code);
        const index_t nnz_coverage = occurrence * code_bits;

        if (nnz_coverage < min_nnz_coverage) {
            remainder_nnz += nnz_coverage;
            block_desc.compress = false;
        } else {
            accounted_blocks += occurrence;
            accounted_patterns++;
            block_desc.compress = true;
        }
    }

    pbr_stats.accounted_nnz = nnz - remainder_nnz;
    pbr_stats.remainder_nnz = remainder_nnz;
    pbr_stats.accounted_patterns = accounted_patterns;
    pbr_stats.accounted_blocks = accounted_blocks;
    pbr_stats.remainder_coo_overhead = (float) remainder_nnz * ((2.0 * sizeof(index_t)) + sizeof(scalar_t));
    pbr_stats.csr_index_size = (float) (nnz + cols + 1) * sizeof(index_t);
    pbr_stats.pbr_index_size = (float) (2.0 * accounted_blocks + pbr_stats.remainder_coo_overhead) * sizeof(index_t);
}

//template <typename index_t, typename scalar_t>
//void pbr_build(const index_t block_rows, const index_t block_cols,
//               const index_t rows, const index_t cols, const index_t nnz,
//               const index_t* __restrict__ indptr, const index_t* __restrict__ indices, const scalar_t* __restrict__ data,
//               pbr_stats_t<index_t>& pbr_stats) {
//    const index_t accounted_nnz = pbr_stats.accounted_nnz;
//    const index_t accounted_patterns = pbr_stats.accounted_patterns;
//    const block_stats_map<index_t>& block_stats = pbr_stats.block_stats;
//
//    const index_t row_blocks = (rows + block_rows - 1) / block_rows;
//    const index_t col_blocks = (cols + block_cols - 1) / block_cols;
//
//    // Allocate memory for the fields of the PBR data structure
//    block_code_t<index_t>* block_codes = new block_code_t[accounted_patterns];
//    uint32_t* freq = new uint32_t[accounted_patterns];
//    coord_t<index_t>* block_coords = new coord_t[accounted_blocks];
//    scalar_t* pbr_data = new scalar_t[accounted_nnz];
//
//    index_t pattern_index = 0;
//    for (const auto& block_stat : block_stats) {
//        const auto& block_desc = it->second;
//
//         if (block_desc.compress) {
//            const auto& code = it->first;
//            const uint32_t& occurrence = block_desc.occurrence;
//
//            block_codes[pattern_index] = code;
//            freq[pattern_index] = occurrence;
//            ++pattern_index;
//        }
//    }
//
//    // Allocate memory for remainder COO matrix
//    index_t* remainder_rows = new index_t[remainder_nnz];
//    index_t* remainder_cols = new index_t[remainder_nnz];
//    scalar_t* remainder_data = new scalar_t[remainder_nnz];
//
//    index_t last_stripe_index = 0;
//    block_code_map<index_t> stripe_block_codes;
//    unordered_map<index_t,  vector<coo_elem_t<index_t, scalar_t>>> stripe_nnzs;
//    for (index_t row = 0; row < rows; ++row) {
//        const index_t block_row_offset = row / block_rows;
//        const index_t row_offset = row % block_rows;
//
//        for (index_t nnz_idx = indptr[row]; nnz_idx < indptr[row + 1]; ++nnz_idx) {
//            const index_t col = indices[nnz_idx];
//            const index_t block_col_offset = col / col_size;
//            const index_t col_offset = col % col_size;
//            const index_t block_index = block_row_offset * col_blocks + block_col_offset;
//            const index_t bit_index = row_offset * block_cols + col_offset;
//
//            stripe_block_codes[block_index].set(bit_index);
//
//            // If the block has not been seen before, allocate memory for it
//            const auto& block_nnzs = stripe_nnzs.find(block_index);
//            if (block_nnzs == stripe_nnzs.end())
//                stripe_nnzs[block_index].reserve(BITS_PER_CODE);
//
//            // Assign the current nnz to the block
//            const scalar_t val = data[nnz_idx];
//            stripe_nnzs[block_index][bit_index] = { row, col, val };
//        }
//
//        // Process the stripe if we are about to move to a new stripe
//        const index_t next_stripe_index = (row + 1) / block_rows;
//        if (next_stripe_index != last_stripe_index) {
//            pbr_build_process_stripe(stripe_block_codes, stripe_nnzs, block_rows, block_cols, col_blocks, block_stats,
//                                     remainder_index, remainder_rows, remainder_cols, remainder_data,
//                                     block_coords, pbr_data);
//
//            stripe_block_codes.clear();
//            stripe_nnzs.clear();
//
//            last_stripe_index = next_stripe_index;
//        }
//    }
//}

//template <typename index_t, typename scalar_t>
//void pbr_build_process_stripe(const block_code_map<index_t>& stripe_block_codes,
//                              const unordered_map<index_t, vector<coo_elem_t<index_t, scalar_t>>>& stripe_nnzs,
//                              const index_t block_rows, const index_t block_cols,
//                              const index_t num_col_blocks,
//                              const block_stats_map<index_t>& block_stats,
//                              index_t& remainder_index,
//                              index_t* __restrict__ remainder_rows, index_t* __restrict__ remainder_cols, scalar_t* __restrict__ remainder_data,
//                              coord_t<index_t>* __restrict__ block_coords, scalar_t* __restrict__ pbr_data) {
//    for (const auto& stripe_block_code : stripe_block_codes) {
//        [const auto block_index, const auto block_code] = stripe_block_code;
//        const auto& block_desc = block_stats.find(block_code.to_ulong())->second;
//
//        // Does this block go to the remainder matrix or to the PBR matrix?
//        const auto& block_nnzs = stripe_nnzs[block_index];
//        if (!block_desc.valid) {
//            // Add the nnzs to the remainder matrix and forget about this block
//            for (auto& block_nnz : block_nnzs) {
//                remainder_rows[remainder_index] = block_nnz.row;
//                remainder_cols[remainder_index] = block_nnz.col;
//                remainder_data[remainder_index] = block_nnz.val;
//
//                ++remainder_index;
//            }
//        } else {
//            // Add the block to the PBR data structure
//            block_coords[trDesc.destIndexPntr].first = top_left_row;
//            block_coords[trDesc.destIndexPntr].second = top_left_col;
//            trDesc.destIndexPntr++;
//
//            for (auto& block_nnz : block_nnzs) {
//                // Add the nnz to the PBR data structure
//                pbr_data[trDesc.destArrayPntr] = block_nnz.val;
//                ++trDesc.destArrayPntr;
//            }
//        }
//    }
//}

template <typename index_t, typename scalar_t>
pbr_stats_t<index_t, scalar_t> pbr_analyze_csr(const py::array_t<index_t, py::array::c_style> indptr,
                                               const py::array_t<index_t, py::array::c_style> indices,
                                               const index_t rows, const index_t cols, const index_t nnz,
                                               const index_t block_rows, const index_t block_cols,
                                               const index_t min_nnz_coverage) {
    pbr_stats_t<index_t, scalar_t> pbr_stats(rows, cols, nnz, block_rows, block_cols, min_nnz_coverage);

    analyze_csr_blocks(block_rows, block_cols, rows, cols, indptr.data(), indices.data(), pbr_stats);
    process_block_stats(nnz, cols, min_nnz_coverage, pbr_stats);

    return pbr_stats;
}

template <typename index_t, typename scalar_t>
void pbr_to_csr(const pbr_matrix_t<index_t, scalar_t>& pbr_mat,
                py::array_t<index_t, py::array::c_style> indptr,
                py::array_t<index_t, py::array::c_style> indices,
                py::array_t<scalar_t, py::array::c_style> data) {
    
    auto indptr_ptr = indptr.mutable_data();
    auto indices_ptr = indices.mutable_data();
    auto data_ptr = data.mutable_data();
    // Intermediate storage to group and sort columns per row
    std::vector<std::vector<std::pair<index_t, scalar_t>>> row_bins(pbr_mat.rows);
    // 1. Unpack compressed blocks
    index_t data_idx = 0;
    for (size_t b = 0; b < pbr_mat.block_codes.size(); ++b) {
        const auto& code = pbr_mat.block_codes[b];
        const auto& coord = pbr_mat.block_coords[b];
        for (index_t r = 0; r < pbr_mat.block_rows; ++r) {
            // Safety check: ensure we don't write past the last row of the matrix
            if (coord.row + r >= pbr_mat.rows) continue;
            for (index_t c = 0; c < pbr_mat.block_cols; ++c) {
                // Safety check: ensure we don't write past the last column
                if (coord.col + c >= pbr_mat.cols) continue;
                
                //bitmap
                if(is_bit_set(code, r * pbr_mat.block_cols + c)){
                //if (code.test(r * pbr_mat.block_cols + c)) {
                    row_bins[coord.row + r].push_back({coord.col + c, pbr_mat.block_data[data_idx++]});
                }
            }
        }
    }
    // 2. Unpack remainder COO
    for (const auto& elem : pbr_mat.remainder_coo) {
        row_bins[elem.row].push_back({elem.col, elem.val});
    }
    // 3. Write back to CSR format
    index_t current_nnz = 0;
    indptr_ptr[0] = 0;
    for (index_t r = 0; r < pbr_mat.rows; ++r) {
        // CSR requires column indices to be sorted within a row
        std::sort(row_bins[r].begin(), row_bins[r].end());
        for (const auto& entry : row_bins[r]) {
            indices_ptr[current_nnz] = entry.first;
            data_ptr[current_nnz] = entry.second;
            current_nnz++;
        }
        indptr_ptr[r + 1] = current_nnz;
    }
}

template <typename index_t, typename scalar_t>
pbr_matrix_t<index_t, scalar_t> csr_to_pbr(const py::array_t<index_t, py::array::c_style> indptr,
                                           const py::array_t<index_t, py::array::c_style> indices,
                                           const py::array_t<scalar_t, py::array::c_style> data,
                                           const index_t rows, const index_t cols,
                                           const index_t block_rows, const index_t block_cols,
                                           const uint8_t min_nnz_per_block) {
    const index_t total_nnz = data.size();
    const index_t* indptr_data = indptr.data();
    const index_t* indices_data = indices.data();
    const scalar_t* data_data = data.data();

    const uint64_t num_row_blocks = (rows + block_rows - 1) / block_rows;
    const uint64_t num_col_blocks = (cols + block_cols - 1) / block_cols;

    // PBR data structures
    vector<block_code_t> block_codes;
    block_codes.reserve(sqrt(num_row_blocks * num_col_blocks));

    vector<coord_t<index_t>> block_coords;
    block_coords.reserve(2 * sqrt(num_row_blocks * num_col_blocks));

    vector<scalar_t> block_data;
    block_data.reserve(total_nnz);

    vector<coo_elem_t<index_t,scalar_t>> remainder_coo;

    // NEW: Offsets vector to track data start positions
    vector<index_t> block_offsets;
    index_t current_offset = 0;
    block_offsets.push_back(current_offset); // Start of first block

    // Data structures for the currently processed stripe
    unordered_map<index_t, block_code_t> stripe_block_codes(sqrt(num_col_blocks));
    unordered_map<index_t, coord_t<index_t>> stripe_block_coords(sqrt(num_col_blocks));
    unordered_map<index_t, vector<scalar_t>> stripe_block_data(sqrt(num_col_blocks));

    for (uint64_t row = 0; row < rows; ++row) {
        const index_t block_row = row / block_rows;
        const index_t row_offset_in_block = row % block_rows;

        for (uint64_t nnz_idx = indptr_data[row]; nnz_idx < indptr_data[row + 1]; ++nnz_idx) {
            const uint64_t col = indices_data[nnz_idx];
            const uint64_t block_col = col / block_cols;
            const uint64_t col_offset_in_block = col % block_cols;
            const uint64_t block_index = block_row * num_col_blocks + block_col;
            const uint64_t bit_index = row_offset_in_block * block_cols + col_offset_in_block;

            const auto& code_found = stripe_block_codes.emplace(block_index, 0);
            auto& code = code_found.first->second;
            if (code_found.second) {
                //stripe_block_coords.emplace(block_index, coord_t<index_t>(row, col));
                //update, think that bug was here!
                //saved the first nnz as the start of the block. changed to the real start
                stripe_block_coords.emplace(block_index, coord_t<index_t>(block_row * block_rows, block_col * block_cols));
            }

            //bitmap
            code |= (1ULL << bit_index);
            // code.set(bit_index);
            stripe_block_data[block_index].emplace_back(data_data[nnz_idx]);
        }

        // Are we done processing a stripe (row of blocks)?
        // If so, decide for each block whether it should be stored in the PBR matrix or in the remainder COO matrix.
        if ((row + 1) / block_rows > block_row) {
            for (const auto& [block_index, code] : stripe_block_codes) {
                int nnz_in_block = count_set_bits(code);
                // If the block has more than the minimum number of nonzeros, store it in the PBR matrix.
                // Otherwise, store the nonzeros in the remainder COO matrix.
                //bitmap
                if (count_set_bits(code) >= min_nnz_per_block) {
                // if (code.count() >= min_nnz_per_block) {
                    block_codes.emplace_back(code);
                    block_coords.emplace_back(std::move(stripe_block_coords[block_index]));
                    block_data.insert(block_data.end(), make_move_iterator(stripe_block_data[block_index].begin()),
                                                        make_move_iterator(stripe_block_data[block_index].end()));

                    // NEW: Update offset tracking
                    current_offset += nnz_in_block;
                    block_offsets.push_back(current_offset);
                } else {
                    index_t block_nnz_index = 0;
                    for (index_t row_offset = 0; row_offset < block_rows; ++row_offset) {
                        for (index_t col_offset = 0; col_offset < block_cols; ++col_offset) {
                            //bitmap
                            if (is_bit_set(code, row_offset * block_cols + col_offset)) {
                            // if (code.test(row_offset * block_cols + col_offset)) {
                                const auto [block_row, block_col] = stripe_block_coords[block_index];
                                remainder_coo.emplace_back(block_row + row_offset, block_col + col_offset,
                                                           stripe_block_data[block_index][block_nnz_index]);
                                ++block_nnz_index;
                            }
                        }
                    }
                }
            }

            stripe_block_codes.clear();
            stripe_block_coords.clear();
            stripe_block_data.clear();
        }
    }

    return pbr_matrix_t<index_t, scalar_t>(rows, cols, block_rows, block_cols, total_nnz,
                                           std::move(block_codes), std::move(block_coords),
                                           std::move(block_offsets),
                                           data_order_t::BLOCK_ROW_MAJOR, std::move(block_data),
                                           std::move(remainder_coo));
}

template void pbr_batched_matmul_cpu<int64_t, float>(const pbr_matrix_t<int64_t, float>& pbr_mat, 
                                                    const py::array_t<float, py::array::c_style> x, 
                                                    py::array_t<float, py::array::c_style> y, 
                                                    const int batch_size, 
                                                    const int width);

template void pbr_batched_matmul_cpu<int32_t, float>(const pbr_matrix_t<int32_t, float>& pbr_mat, 
                                                    const py::array_t<float, py::array::c_style> x, 
                                                    py::array_t<float, py::array::c_style> y, 
                                                    const int batch_size, 
                                                    const int width);

template void pbr_batched_matmul_cpu<int64_t, double>(const pbr_matrix_t<int64_t, double>& pbr_mat, 
                                                     const py::array_t<double, py::array::c_style> x, 
                                                     py::array_t<double, py::array::c_style> y, 
                                                     const int batch_size, 
                                                     const int width);

template void pbr_batched_matmul_cpu<int32_t, double>(const pbr_matrix_t<int32_t, double>& pbr_mat, 
                                                     const py::array_t<double, py::array::c_style> x, 
                                                     py::array_t<double, py::array::c_style> y, 
                                                     const int batch_size, 
                                                     const int width);


template pbr_stats_t<uint64_t, float> pbr_analyze_csr(const py::array_t<uint64_t, py::array::c_style> indptr,
                                                      const py::array_t<uint64_t, py::array::c_style> indices,
                                                      const uint64_t rows, const uint64_t cols, const uint64_t nnz,
                                                      const uint64_t block_rows, const uint64_t block_cols,
                                                      const uint64_t min_nnz_coverage);
template pbr_stats_t<uint32_t, float> pbr_analyze_csr(const py::array_t<uint32_t, py::array::c_style> indptr,
                                                      const py::array_t<uint32_t, py::array::c_style> indices,
                                                      const uint32_t rows, const uint32_t cols, const uint32_t nnz,
                                                      const uint32_t block_rows, const uint32_t block_cols,
                                                      const uint32_t min_nnz_coverage);
template pbr_stats_t<uint64_t, double> pbr_analyze_csr(const py::array_t<uint64_t, py::array::c_style> indptr,
                                                       const py::array_t<uint64_t, py::array::c_style> indices,
                                                       const uint64_t rows, const uint64_t cols, const uint64_t nnz,
                                                       const uint64_t block_rows, const uint64_t block_cols,
                                                       const uint64_t min_nnz_coverage);
template pbr_stats_t<uint32_t, double> pbr_analyze_csr(const py::array_t<uint32_t, py::array::c_style> indptr,
                                                       const py::array_t<uint32_t, py::array::c_style> indices,
                                                       const uint32_t rows, const uint32_t cols, const uint32_t nnz,
                                                       const uint32_t block_rows, const uint32_t block_cols,
                                                       const uint32_t min_nnz_coverage);


template pbr_matrix_t<int64_t, float> csr_to_pbr(const py::array_t<int64_t, py::array::c_style> indptr,
                                                 const py::array_t<int64_t, py::array::c_style> indices,
                                                 const py::array_t<float, py::array::c_style> data,
                                                 const int64_t rows, const int64_t cols,
                                                 const int64_t block_rows, const int64_t block_cols,
                                                 const uint8_t min_nnz_per_block);
template pbr_matrix_t<int32_t, float> csr_to_pbr(const py::array_t<int32_t, py::array::c_style> indptr,
                                                 const py::array_t<int32_t, py::array::c_style> indices,
                                                 const py::array_t<float, py::array::c_style> data,
                                                 const int32_t rows, const int32_t cols,
                                                 const int32_t block_rows, const int32_t block_cols,
                                                 const uint8_t min_nnz_per_block);
template pbr_matrix_t<int64_t, double> csr_to_pbr(const py::array_t<int64_t, py::array::c_style> indptr,
                                                  const py::array_t<int64_t, py::array::c_style> indices,
                                                  const py::array_t<double, py::array::c_style> data,
                                                  const int64_t rows, const int64_t cols,
                                                  const int64_t block_rows, const int64_t block_cols,
                                                  const uint8_t min_nnz_per_block);
template pbr_matrix_t<int32_t, double> csr_to_pbr(const py::array_t<int32_t, py::array::c_style> indptr,
                                                  const py::array_t<int32_t, py::array::c_style> indices,
                                                  const py::array_t<double, py::array::c_style> data,
                                                  const int32_t rows, const int32_t cols,
                                                  const int32_t block_rows, const int32_t block_cols,
                                                  const uint8_t min_nnz_per_block);


template void pbr_to_csr<int64_t, float>(const pbr_matrix_t<int64_t, float>&, 
                                        const py::array_t<int64_t, py::array::c_style>, 
                                        const py::array_t<int64_t, py::array::c_style>, 
                                        const py::array_t<float, py::array::c_style>);
template void pbr_to_csr<int32_t, float>(const pbr_matrix_t<int32_t, float>&, 
                                        const py::array_t<int32_t, py::array::c_style>, 
                                        const py::array_t<int32_t, py::array::c_style>, 
                                        const py::array_t<float, py::array::c_style>);
template void pbr_to_csr<int64_t, double>(const pbr_matrix_t<int64_t, double>&, 
                                        const py::array_t<int64_t, py::array::c_style>, 
                                        const py::array_t<int64_t, py::array::c_style>, 
                                        const py::array_t<double, py::array::c_style>);
template void pbr_to_csr<int32_t, double>(const pbr_matrix_t<int32_t, double>&, 
                                        const py::array_t<int32_t, py::array::c_style>, 
                                        const py::array_t<int32_t, py::array::c_style>, 
                                        const py::array_t<double, py::array::c_style>);