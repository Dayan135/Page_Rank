#include <ATen/SparseCsrTensorUtils.h>

#include "../pbr_matrix/pbr_matrix.hpp"

#if CUDA_ENABLED == 1
void bind_cuda_functions(py::module_& m);
#endif

#define BIND_PBR_MATRIX_CLASS(T, S, py_name) \
        py::class_<pbr_matrix_t<T,S>>(m, (py_name)) \
            .def_readonly("rows", &pbr_matrix_t<T, S>::rows) \
            .def_readonly("cols", &pbr_matrix_t<T, S>::cols) \
            .def_readonly("block_rows", &pbr_matrix_t<T, S>::block_rows) \
            .def_readonly("block_cols", &pbr_matrix_t<T, S>::block_cols) \
            .def_readonly("total_nnz", &pbr_matrix_t<T, S>::total_nnz) \
            .def_readonly("block_codes", &pbr_matrix_t<T, S>::block_codes) \
            .def_readonly("block_coords", &pbr_matrix_t<T, S>::block_coords) \
            .def_readonly("data_order", &pbr_matrix_t<T, S>::data_order) \
            .def_readonly("block_data", &pbr_matrix_t<T, S>::block_data) \
            .def_readonly("remainder_coo", &pbr_matrix_t<T, S>::remainder_coo) \
            .def_readonly("block_offsets", &pbr_matrix_t<T, S>::block_offsets) \
            .def("__copy__", [](const pbr_matrix_t<T, S> &self) { return pbr_matrix_t<T, S>(self); }) \
            .def("to_dict", &pbr_matrix_t<T, S>::to_dict) \
            .def("accounted_blocks", &pbr_matrix_t<T, S>::accounted_blocks) \
            .def("compressed_nnz", &pbr_matrix_t<T, S>::compressed_nnz) \
            .def("remainder_nnz", &pbr_matrix_t<T, S>::remainder_nnz);// \
//            .def("to_csr", &pbr_matrix_t<T, S>::to_csr);

PYBIND11_MODULE(graph_link_core, m) {
    py::enum_<data_order_t>(m, "DataOrder")
        .value("BLOCK_ROW_MAJOR", data_order_t::BLOCK_ROW_MAJOR)
        .value("BLOCK_COLUMN_MAJOR", data_order_t::BLOCK_COLUMN_MAJOR)
        .value("INTERLEAVED_ROW_MAJOR", data_order_t::INTERLEAVED_ROW_MAJOR)
        .value("INTERLEAVED_COLUMN_MAJOR", data_order_t::INTERLEAVED_COLUMN_MAJOR);

    BIND_PBR_MATRIX_CLASS(int64_t, float, "PBRMatrixInt64Float");
    BIND_PBR_MATRIX_CLASS(int32_t, float, "PBRMatrixInt32Float");
    BIND_PBR_MATRIX_CLASS(int64_t, double, "PBRMatrixInt64Double");
    BIND_PBR_MATRIX_CLASS(int32_t, double, "PBRMatrixInt32Double");

    // Batched MatMul (A * X = Y)
    m.def("pbr_batched_matmul_cpu", &pbr_batched_matmul_cpu<int64_t, float>);
    m.def("pbr_batched_matmul_cpu", &pbr_batched_matmul_cpu<int32_t, float>);
    m.def("pbr_batched_matmul_cpu", &pbr_batched_matmul_cpu<int64_t, double>);
    m.def("pbr_batched_matmul_cpu", &pbr_batched_matmul_cpu<int32_t, double>);

    m.def("csr_to_pbr", &csr_to_pbr<int64_t, float>);
    m.def("csr_to_pbr", &csr_to_pbr<int32_t, float>);
    m.def("csr_to_pbr", &csr_to_pbr<int64_t, double>);
    m.def("csr_to_pbr", &csr_to_pbr<int32_t, double>);

    m.def("pbr_to_csr", &pbr_to_csr<int64_t, float>);
    m.def("pbr_to_csr", &pbr_to_csr<int32_t, float>);
    m.def("pbr_to_csr", &pbr_to_csr<int64_t, double>);
    m.def("pbr_to_csr", &pbr_to_csr<int32_t, double>);

    m.def("pbr_analyze_csr", &pbr_analyze_csr<uint64_t, float>);
    m.def("pbr_analyze_csr", &pbr_analyze_csr<uint32_t, float>);
    m.def("pbr_analyze_csr", &pbr_analyze_csr<uint64_t, double>);
    m.def("pbr_analyze_csr", &pbr_analyze_csr<uint32_t, double>);

    #if CUDA_ENABLED == 1
        bind_cuda_functions(m);
    #endif
}