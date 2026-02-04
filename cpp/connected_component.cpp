#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cstring>   // memcpy
#include <limits>

namespace py = pybind11;

struct Run {
    int32_t col;     // column index
    int32_t row0;    // first row (inclusive)
    int32_t len;     // number of pixels in run
    int64_t pos0;    // first CSC position (in [0..nnz))
    int32_t id;      // DSU id for this run
};

struct DSU {
    std::vector<int32_t> parent;
    std::vector<int32_t> rank;

    int32_t make_set() {
        int32_t id = (int32_t)parent.size();
        parent.push_back(id);
        rank.push_back(0);
        return id;
    }

    int32_t find(int32_t a) {
        // iterative path compression (safer than recursion for deep trees)
        int32_t r = a;
        while (parent[r] != r) r = parent[r];
        while (parent[a] != a) {
            int32_t p = parent[a];
            parent[a] = r;
            a = p;
        }
        return r;
    }

    void unite(int32_t a, int32_t b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (rank[a] < rank[b]) std::swap(a, b);
        parent[b] = a;
        if (rank[a] == rank[b]) rank[a]++;
    }
};

struct Result {
    int32_t n_components = 0;
    std::vector<double> stats;          // size = K*3 (mean_row, mean_col, sum_intensity)
    std::vector<int32_t> labels_nnz;    // size = nnz, labels per CSC position (0 background, else 1..K)
    std::vector<int32_t> col_of_pos;    // size = nnz, column per CSC position
    std::vector<int64_t> comp_indptr;   // size = K+1
    std::vector<int64_t> comp_indices;  // size = nnz_fg (CSC positions grouped by component)
};

template <typename IndPtrT, typename IndicesT, typename DataT>
static Result ccl_csc8_core(
    const IndPtrT* indptr, int64_t n_indptr,
    const IndicesT* indices, int64_t nnz,
    const DataT* data,
    int64_t n_rows
) {
    if (n_indptr < 2) throw std::runtime_error("indptr must have length >= 2");
    if (n_rows <= 0) throw std::runtime_error("n_rows must be > 0");

    const int64_t n_cols = n_indptr - 1;

    // Basic indptr sanity: monotone, within [0..nnz], ends at nnz
    if ((int64_t)indptr[0] != 0) {
        throw std::runtime_error("Precondition failed: indptr[0] must be 0");
    }
    if ((int64_t)indptr[n_cols] != nnz) {
        throw std::runtime_error("Precondition failed: indptr[last] must equal nnz");
    }
    for (int64_t c = 0; c < n_cols; c++) {
        int64_t p0 = (int64_t)indptr[c];
        int64_t p1 = (int64_t)indptr[c + 1];
        if (p0 < 0 || p1 < p0 || p1 > nnz) {
            throw std::runtime_error("Precondition failed: indptr must be monotone and within [0..nnz]");
        }
    }

    Result out;
    out.labels_nnz.assign((size_t)nnz, 0);
    out.col_of_pos.assign((size_t)nnz, 0);

    // Map CSC position -> run id (or -1 for background)
    std::vector<int32_t> run_of_pos((size_t)nnz, -1);

    std::vector<Run> runs;
    runs.reserve((size_t)std::max<int64_t>(1, nnz / 8));

    std::vector<int64_t> run_col_ptr((size_t)(n_cols + 1), 0); // runs for col c: [ptr[c], ptr[c+1])
    DSU dsu;

    // -------- Build runs (foreground = data > 0) and assert sorted indices --------
    for (int64_t c = 0; c < n_cols; c++) {
        run_col_ptr[(size_t)c] = (int64_t)runs.size();

        const int64_t p0 = (int64_t)indptr[c];
        const int64_t p1 = (int64_t)indptr[c + 1];

        // Assert strictly increasing row indices in this column (no sorting done)
        int64_t prev_r64 = std::numeric_limits<int64_t>::min();

        int64_t run_pos0 = -1;
        int32_t run_row0 = 0;
        int32_t run_len  = 0;
        int32_t prev_row = std::numeric_limits<int32_t>::min();
        int32_t cur_run_id = -1;

        for (int64_t p = p0; p < p1; p++) {
            out.col_of_pos[(size_t)p] = (int32_t)c;

            const int64_t r64 = (int64_t)indices[p];
            if (r64 <= prev_r64) {
                throw std::runtime_error(
                    "Precondition failed: CSC indices must be strictly increasing within each column "
                    "(call csc.sort_indices() upstream; this function will not sort)."
                );
            }
            prev_r64 = r64;

            if (r64 < 0 || r64 >= n_rows) {
                throw std::runtime_error("Row index out of bounds");
            }
            const int32_t r = (int32_t)r64;

            const DataT v = data[p];
            const bool fg = (v > (DataT)0); // match hot.py positive-mass convention

            if (!fg) {
                // Close any active run
                if (run_len > 0) {
                    runs.push_back(Run{(int32_t)c, run_row0, run_len, run_pos0, cur_run_id});
                    run_len = 0;
                    cur_run_id = -1;
                }
                out.labels_nnz[(size_t)p] = 0;
                run_of_pos[(size_t)p] = -1;
                continue;
            }

            // Foreground pixel
            if (run_len == 0) {
                cur_run_id = dsu.make_set();
                run_pos0   = p;
                run_row0   = r;
                run_len    = 1;
                prev_row   = r;
            } else if (r == prev_row + 1) {
                run_len++;
                prev_row = r;
            } else {
                // Close previous run, start new
                runs.push_back(Run{(int32_t)c, run_row0, run_len, run_pos0, cur_run_id});
                cur_run_id = dsu.make_set();
                run_pos0   = p;
                run_row0   = r;
                run_len    = 1;
                prev_row   = r;
            }

            run_of_pos[(size_t)p] = cur_run_id;
        }

        if (run_len > 0) {
            runs.push_back(Run{(int32_t)c, run_row0, run_len, run_pos0, cur_run_id});
        }

        run_col_ptr[(size_t)(c + 1)] = (int64_t)runs.size();
    }

    const int64_t R = (int64_t)runs.size();
    if (R == 0) {
        // No foreground at all
        out.n_components = 0;
        out.stats.clear();
        out.comp_indptr = {0};
        out.comp_indices.clear();
        return out;
    }

    // -------- Union runs between adjacent columns using 8-connectivity --------
    // For 8-connectivity between col c-1 and c: connect if expanded intervals overlap:
    // run A rows [a0..a1] expands to [a0-1..a1+1], same for B.
    for (int64_t c = 1; c < n_cols; c++) {
        const int64_t a0i = run_col_ptr[(size_t)(c - 1)];
        const int64_t a1i = run_col_ptr[(size_t)(c)];
        const int64_t b0i = run_col_ptr[(size_t)(c)];
        const int64_t b1i = run_col_ptr[(size_t)(c + 1)];

        int64_t i = a0i;
        int64_t j = b0i;

        while (i < a1i && j < b1i) {
            const Run& A = runs[(size_t)i];
            const Run& B = runs[(size_t)j];

            const int64_t A0 = (int64_t)A.row0;
            const int64_t A1 = (int64_t)A.row0 + (int64_t)A.len - 1;
            const int64_t B0 = (int64_t)B.row0;
            const int64_t B1 = (int64_t)B.row0 + (int64_t)B.len - 1;

            // Expanded intervals
            const int64_t Ae0 = A0 - 1;
            const int64_t Ae1 = A1 + 1;
            const int64_t Be0 = B0 - 1;
            const int64_t Be1 = B1 + 1;

            if (Ae1 < Be0) {
                i++;
                continue;
            }
            if (Be1 < Ae0) {
                j++;
                continue;
            }

            // Overlap => union
            dsu.unite(A.id, B.id);

            // Advance the interval that ends first (in expanded space)
            if (Ae1 < Be1) i++;
            else j++;
        }
    }

    // -------- Assign compact component ids (1..K) deterministically --------
    std::vector<int32_t> comp_id_for_root((size_t)R, 0);
    int32_t K = 0;
    for (int64_t rid = 0; rid < R; rid++) {
        // note: run ids are 0..R-1 in creation order because make_set() called once per run
        int32_t root = dsu.find((int32_t)rid);
        if (comp_id_for_root[(size_t)root] == 0) {
            comp_id_for_root[(size_t)root] = ++K;
        }
    }
    out.n_components = K;

    // -------- Build labels per CSC position --------
    int64_t nnz_fg = 0;
    for (int64_t p = 0; p < nnz; p++) {
        int32_t run_id = run_of_pos[(size_t)p];
        if (run_id < 0) {
            out.labels_nnz[(size_t)p] = 0;
            continue;
        }
        int32_t root = dsu.find(run_id);
        int32_t cid  = comp_id_for_root[(size_t)root]; // 1..K
        out.labels_nnz[(size_t)p] = cid;
        nnz_fg++;
    }

    // -------- Build comp_indices grouped by component (CSC positions) --------
    out.comp_indptr.assign((size_t)(K + 1), 0);
    std::vector<int64_t> counts((size_t)K, 0);

    for (int64_t p = 0; p < nnz; p++) {
        int32_t lab = out.labels_nnz[(size_t)p];
        if (lab > 0) counts[(size_t)(lab - 1)]++;
    }

    out.comp_indptr[0] = 0;
    for (int32_t k = 0; k < K; k++) {
        out.comp_indptr[(size_t)(k + 1)] = out.comp_indptr[(size_t)k] + counts[(size_t)k];
    }

    out.comp_indices.assign((size_t)nnz_fg, 0);
    std::vector<int64_t> write_ptr = out.comp_indptr; // copy

    for (int64_t p = 0; p < nnz; p++) {
        int32_t lab = out.labels_nnz[(size_t)p];
        if (lab <= 0) continue;
        const int32_t k = lab - 1;
        const int64_t dst = write_ptr[(size_t)k]++;
        out.comp_indices[(size_t)dst] = p; // CSC position
    }

    // -------- Weighted stats: mean_row, mean_col weighted by intensity --------
    // mean_row = sum(row * w) / sum(w)
    // mean_col = sum(col * w) / sum(w)
    // sum_intensity = sum(w)
    std::vector<double> sum_w((size_t)K, 0.0);
    std::vector<double> sum_row_w((size_t)K, 0.0);
    std::vector<double> sum_col_w((size_t)K, 0.0);

    // fallback (only used if sum_w becomes 0 due to weird inputs)
    std::vector<double> sum_row((size_t)K, 0.0);
    std::vector<double> sum_col((size_t)K, 0.0);
    std::vector<int64_t> cnt((size_t)K, 0);

    for (int64_t p = 0; p < nnz; p++) {
        int32_t lab = out.labels_nnz[(size_t)p];
        if (lab <= 0) continue;
        const int32_t k = lab - 1;

        const double w = (double)data[p]; // > 0 by construction
        const double r = (double)(int64_t)indices[p];
        const double c = (double)out.col_of_pos[(size_t)p];

        sum_w[(size_t)k]     += w;
        sum_row_w[(size_t)k] += r * w;
        sum_col_w[(size_t)k] += c * w;

        sum_row[(size_t)k] += r;
        sum_col[(size_t)k] += c;
        cnt[(size_t)k] += 1;
    }

    out.stats.assign((size_t)K * 3, 0.0);
    for (int32_t k = 0; k < K; k++) {
        const double sw = sum_w[(size_t)k];
        double mean_r, mean_c;

        if (sw != 0.0) {
            mean_r = sum_row_w[(size_t)k] / sw;
            mean_c = sum_col_w[(size_t)k] / sw;
        } else {
            // Defensive fallback (should not happen if weights are strictly positive and don't underflow)
            const double denom = (cnt[(size_t)k] > 0) ? (double)cnt[(size_t)k] : 1.0;
            mean_r = sum_row[(size_t)k] / denom;
            mean_c = sum_col[(size_t)k] / denom;
        }

        out.stats[(size_t)k * 3 + 0] = mean_r;
        out.stats[(size_t)k * 3 + 1] = mean_c;
        out.stats[(size_t)k * 3 + 2] = sw;
    }

    return out;
}

static py::dict ccl_csc8(py::array indptr_arr,
                         py::array indices_arr,
                         py::array data_arr,
                         int64_t n_rows) {
    // Ensure contiguous 1D arrays (may copy; does not sort)
    indptr_arr  = py::array::ensure(indptr_arr,  py::array::c_style);
    indices_arr = py::array::ensure(indices_arr, py::array::c_style);
    data_arr    = py::array::ensure(data_arr,    py::array::c_style);

    if (!indptr_arr || !indices_arr || !data_arr) {
        throw std::runtime_error("Failed to ensure contiguous arrays");
    }
    if (indptr_arr.ndim() != 1 || indices_arr.ndim() != 1 || data_arr.ndim() != 1) {
        throw std::runtime_error("indptr, indices, data must be 1D arrays");
    }

    const int64_t n_indptr = (int64_t)indptr_arr.shape(0);
    const int64_t nnz      = (int64_t)indices_arr.shape(0);

    if ((int64_t)data_arr.shape(0) != nnz) {
        throw std::runtime_error("indices and data must have same length (nnz)");
    }

    // dtype dispatch
    const py::dtype dt_indptr  = py::dtype(indptr_arr.dtype());
    const py::dtype dt_indices = py::dtype(indices_arr.dtype());
    const py::dtype dt_data    = py::dtype(data_arr.dtype());

    auto is_i32 = [](const py::dtype& d){ return d.is(py::dtype::of<int32_t>()); };
    auto is_i64 = [](const py::dtype& d){ return d.is(py::dtype::of<int64_t>()); };
    auto is_f32 = [](const py::dtype& d){ return d.is(py::dtype::of<float>()); };
    auto is_f64 = [](const py::dtype& d){ return d.is(py::dtype::of<double>()); };

    if (!(is_i32(dt_indptr) || is_i64(dt_indptr)))   throw std::runtime_error("indptr must be int32 or int64");
    if (!(is_i32(dt_indices) || is_i64(dt_indices))) throw std::runtime_error("indices must be int32 or int64");
    if (!(is_f32(dt_data) || is_f64(dt_data)))       throw std::runtime_error("data must be float32 or float64");

    // Get raw pointers (safe: arrays are contiguous)
    const void* indptr_ptr  = indptr_arr.data();
    const void* indices_ptr = indices_arr.data();
    const void* data_ptr    = data_arr.data();

    Result res;

    // Heavy compute can run without GIL because we only touch raw pointers + std::vector
    {
        py::gil_scoped_release release;

        // 8 combos
        if (is_i64(dt_indptr) && is_i64(dt_indices) && is_f32(dt_data)) {
            res = ccl_csc8_core((const int64_t*)indptr_ptr, n_indptr,
                                (const int64_t*)indices_ptr, nnz,
                                (const float*)data_ptr,
                                n_rows);
        } else if (is_i64(dt_indptr) && is_i64(dt_indices) && is_f64(dt_data)) {
            res = ccl_csc8_core((const int64_t*)indptr_ptr, n_indptr,
                                (const int64_t*)indices_ptr, nnz,
                                (const double*)data_ptr,
                                n_rows);
        } else if (is_i64(dt_indptr) && is_i32(dt_indices) && is_f32(dt_data)) {
            res = ccl_csc8_core((const int64_t*)indptr_ptr, n_indptr,
                                (const int32_t*)indices_ptr, nnz,
                                (const float*)data_ptr,
                                n_rows);
        } else if (is_i64(dt_indptr) && is_i32(dt_indices) && is_f64(dt_data)) {
            res = ccl_csc8_core((const int64_t*)indptr_ptr, n_indptr,
                                (const int32_t*)indices_ptr, nnz,
                                (const double*)data_ptr,
                                n_rows);
        } else if (is_i32(dt_indptr) && is_i64(dt_indices) && is_f32(dt_data)) {
            res = ccl_csc8_core((const int32_t*)indptr_ptr, n_indptr,
                                (const int64_t*)indices_ptr, nnz,
                                (const float*)data_ptr,
                                n_rows);
        } else if (is_i32(dt_indptr) && is_i64(dt_indices) && is_f64(dt_data)) {
            res = ccl_csc8_core((const int32_t*)indptr_ptr, n_indptr,
                                (const int64_t*)indices_ptr, nnz,
                                (const double*)data_ptr,
                                n_rows);
        } else if (is_i32(dt_indptr) && is_i32(dt_indices) && is_f32(dt_data)) {
            res = ccl_csc8_core((const int32_t*)indptr_ptr, n_indptr,
                                (const int32_t*)indices_ptr, nnz,
                                (const float*)data_ptr,
                                n_rows);
        } else {
            // i32, i32, f64
            res = ccl_csc8_core((const int32_t*)indptr_ptr, n_indptr,
                                (const int32_t*)indices_ptr, nnz,
                                (const double*)data_ptr,
                                n_rows);
        }
    } // GIL automatically reacquired here

    // Convert Result -> Python objects
    const int32_t K = res.n_components;

    py::array_t<double> stats_arr({K, 3});
    if ((size_t)K * 3 != res.stats.size()) throw std::runtime_error("Internal error: stats size mismatch");
    std::memcpy(stats_arr.mutable_data(), res.stats.data(), res.stats.size() * sizeof(double));

    py::array_t<int32_t> labels_arr((py::ssize_t)res.labels_nnz.size());
    std::memcpy(labels_arr.mutable_data(), res.labels_nnz.data(), res.labels_nnz.size() * sizeof(int32_t));

    py::array_t<int32_t> col_of_pos_arr((py::ssize_t)res.col_of_pos.size());
    std::memcpy(col_of_pos_arr.mutable_data(), res.col_of_pos.data(), res.col_of_pos.size() * sizeof(int32_t));

    py::array_t<int64_t> comp_indptr_arr((py::ssize_t)res.comp_indptr.size());
    std::memcpy(comp_indptr_arr.mutable_data(), res.comp_indptr.data(), res.comp_indptr.size() * sizeof(int64_t));

    py::array_t<int64_t> comp_indices_arr((py::ssize_t)res.comp_indices.size());
    std::memcpy(comp_indices_arr.mutable_data(), res.comp_indices.data(), res.comp_indices.size() * sizeof(int64_t));

    py::dict out;
    out["n_components"] = K;
    out["stats"] = stats_arr;
    out["labels_nnz"] = labels_arr;
    out["col_of_pos"] = col_of_pos_arr;
    out["comp_indptr"] = comp_indptr_arr;
    out["comp_indices"] = comp_indices_arr;
    return out;
}

PYBIND11_MODULE(connected_component, m) {
    m.doc() = "Sparse 2D connected components for CSC (8-connectivity) + weighted stats";
    m.def(
        "ccl_csc8",
        &ccl_csc8,
        py::arg("indptr"),
        py::arg("indices"),
        py::arg("data"),
        py::arg("n_rows"),
        R"pbdoc(
Compute 8-connected components on a 2D CSC sparse matrix foreground (data > 0).

Preconditions (asserted, not fixed):
- indptr[0] == 0
- indptr[last] == nnz
- indptr is monotone
- within each column, indices are strictly increasing (sorted CSC). This function does NOT sort.

Returns a dict with:
- n_components: int
- stats: (K,3) float64 = [weighted_mean_row, weighted_mean_col, sum_intensity] per component
- labels_nnz: (nnz,) int32 labels aligned with CSC positions (0=background, else 1..K)
- col_of_pos: (nnz,) int32 mapping CSC position -> column index
- comp_indptr: (K+1,) int64 pointer into comp_indices
- comp_indices: (nnz_fg,) int64 CSC positions grouped by component id (1..K)
)pbdoc"
    );
}
