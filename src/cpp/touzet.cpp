// pybind11 bindings for Touzet's iterative refinement algorithm.
// Both functions work in the deficit domain: delta[l][c] = M_l - theta[l][c].

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include "Matrix.h"

namespace py = pybind11;

static std::string fmt_seconds(double t) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.3g", t);
    return std::string(buf);
}

static void check_abort(const Matrix& m) {
    if (m.abortReason == 1) {
        throw py::value_error(
            "Touzet algorithm aborted: DP dictionary exceeded max_dict_size ("
            + std::to_string(m.maxDictSize) + " entries). "
            "Try increasing tol, or raising max_dict_size.");
    }
    if (m.abortReason == 2) {
        throw py::value_error(
            "Touzet algorithm aborted: wall-clock time exceeded max_time ("
            + fmt_seconds(m.maxTime) + "s). "
            "Try increasing tol, or raising max_time.");
    }
}

static std::vector<std::vector<double>>
numpy_to_vec(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    auto buf = arr.unchecked<2>();
    int L = (int)buf.shape(0);
    int C = (int)buf.shape(1);
    std::vector<std::vector<double>> v(L, std::vector<double>(C));
    for (int l = 0; l < L; l++)
        for (int c = 0; c < C; c++)
            v[l][c] = buf(l, c);
    return v;
}

struct ThetaInfo {
    std::vector<std::vector<double>> delta; // excitation costs, L x C
    double fmax;
    int L, C;
};

static ThetaInfo
preprocess_theta(py::array_t<double> theta_arr) {
    auto theta = numpy_to_vec(theta_arr);
    ThetaInfo info;
    info.L = (int)theta.size();
    info.C = (int)theta[0].size();
    info.fmax = 0.0;
    info.delta.resize(info.L, std::vector<double>(info.C));
    for (int l = 0; l < info.L; l++) {
        double M_l = theta[l][0];
        for (int c = 1; c < info.C; c++) {
            if (theta[l][c] > M_l) M_l = theta[l][c];
        }
        info.fmax += M_l;
        for (int c = 0; c < info.C; c++) {
            info.delta[l][c] = M_l - theta[l][c];
        }
    }
    return info;
}

static std::pair<double, double>
compute_tail_count(py::array_t<double> theta_arr, double F,
                   double tol = 0.0,
                   double initialGranularity = 0.1,
                   double maxGranularity = 1e-10,
                   double refinementFactor = 10.0,
                   double maxTime = 10.0,
                   size_t maxDictSize = 1000000) {
    auto info = preprocess_theta(theta_arr);
    double eps_max = info.fmax - F;
    if (eps_max < 0) return {0.0, 0.0};

    double CtoL = pow((double)info.C, (double)info.L);

    Matrix m;
    m.initFromTheta(info.delta);
    m.maxTime = maxTime;
    m.maxDictSize = maxDictSize;
    if (maxTime > 0)
        m.deadline = std::chrono::steady_clock::now()
                   + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                         std::chrono::duration<double>(maxTime));

    double count_hi = CtoL;
    double count_lo = 0.0;

    for (double granularity = initialGranularity;
         granularity >= maxGranularity;
         granularity /= refinementFactor) {
        m.computesIntegerMatrix(granularity, true);

        qlonglong budget = (qlonglong)(eps_max * m.granularity + m.offset);
        qlonglong budgetHi = budget + (qlonglong)ceil(m.errorMax) + 1;

        auto nbocc = m.calcDistribWithMapMinMax(0, budgetHi);
        check_abort(m);

        double prob_hi = 0.0;
        double prob_lo = 0.0;
        for (auto& kv : nbocc[m.length - 1]) {
            if (kv.first <= budget) {
                prob_hi += kv.second;
            }
            if (kv.first <= budget - (qlonglong)ceil(m.errorMax)) {
                prob_lo += kv.second;
            }
        }

        count_hi = prob_hi * CtoL;
        count_lo = prob_lo * CtoL;

        if (fabs(count_hi - count_lo) < 0.5) {
            return {count_hi, count_hi};
        }

        if (tol > 0.0 && count_hi > 0.0) {
            double frac_err = (count_hi - count_lo) / count_hi;
            if (frac_err <= tol) {
                return {count_lo, count_hi};
            }
        }

        if (granularity / refinementFactor < maxGranularity) {
            return {count_lo, count_hi};
        }
    }

    return {count_lo, count_hi};
}

static std::pair<std::vector<double>, std::vector<double>>
compute_deficit_spectrum(py::array_t<double> theta_arr, double eps_max,
                         double tol = 0.0,
                         double initialGranularity = 0.1,
                         double maxGranularity = 1e-10,
                         double refinementFactor = 10.0,
                         double maxTime = 10.0,
                         size_t maxDictSize = 1000000) {
    auto info = preprocess_theta(theta_arr);
    double CtoL = pow((double)info.C, (double)info.L);

    Matrix m;
    m.initFromTheta(info.delta);
    m.maxTime = maxTime;
    m.maxDictSize = maxDictSize;
    if (maxTime > 0)
        m.deadline = std::chrono::steady_clock::now()
                   + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                         std::chrono::duration<double>(maxTime));

    std::vector<double> result_deficits;
    std::vector<double> result_counts;

    for (double granularity = initialGranularity;
         granularity >= maxGranularity;
         granularity /= refinementFactor) {
        m.computesIntegerMatrix(granularity, true);

        qlonglong budget = (qlonglong)(eps_max * m.granularity + m.offset + m.errorMax + 1);

        auto nbocc = m.calcDistribWithMapMinMax(0, budget);
        check_abort(m);

        double E = m.errorMax / m.granularity;

        bool certified = true;
        double total_count = 0.0;
        double ambiguous_count = 0.0;
        result_deficits.clear();
        result_counts.clear();

        for (auto& kv : nbocc[m.length - 1]) {
            double realDeficit = (kv.first - m.offset) / m.granularity;
            if (realDeficit < -1e-12) continue;
            if (realDeficit > eps_max + 1e-12) continue;

            double cnt = kv.second * CtoL;
            result_deficits.push_back(realDeficit);
            result_counts.push_back(cnt);
            total_count += cnt;

            if (realDeficit > eps_max - E && realDeficit <= eps_max + 1e-12) {
                certified = false;
                ambiguous_count += cnt;
            }
        }

        if (certified) break;

        if (tol > 0.0 && total_count > 0.0) {
            double frac_err = ambiguous_count / total_count;
            if (frac_err <= tol) break;
        }

        if (granularity / refinementFactor < maxGranularity) break;
    }

    return {result_deficits, result_counts};
}

PYBIND11_MODULE(_touzet_cpp, mod) {
    mod.doc() = "Touzet iterative refinement algorithm for exact tail counts";

    mod.def("score_to_pvalue", &compute_tail_count,
            "Compute N_geq(F) bounds: (lower, upper) count of sequences with fitness >= F.",
            py::arg("theta"), py::arg("score"),
            py::arg("tol") = 0.0,
            py::arg("initial_granularity") = 0.1,
            py::arg("max_granularity") = 1e-10,
            py::arg("refinement_factor") = 10.0,
            py::arg("max_time") = 10.0,
            py::arg("max_dict_size") = (size_t)1000000);

    mod.def("deficit_spectrum", &compute_deficit_spectrum,
            "Compute deficit spectrum: all (deficit, count) pairs for deficit <= eps_max.",
            py::arg("theta"), py::arg("eps_max"),
            py::arg("tol") = 0.0,
            py::arg("initial_granularity") = 0.1,
            py::arg("max_granularity") = 1e-10,
            py::arg("refinement_factor") = 10.0,
            py::arg("max_time") = 10.0,
            py::arg("max_dict_size") = (size_t)1000000);
}
