// pybind11 (and Python.h) must precede any standard header: Python.h sets
// feature-test macros that break <ctime> with the conda gcc toolchain otherwise.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "boussinesq_kernel.hpp"
#include "cluster_tree.hpp"
#include "contact_solver.hpp"
#include "fft_operator.hpp"
#include "fourier_precond.hpp"
#include "h2_operator.hpp"
#include "hmatrix.hpp"
#include "nested_solve.hpp"

#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace py = pybind11;

namespace {

Eigen::VectorXd to_vector(const py::array_t<double, py::array::c_style |
                                                        py::array::forcecast>& a,
                          int expected) {
    if (a.size() != expected)
        throw std::invalid_argument("array has " + std::to_string(a.size()) +
                                    " entries, expected " +
                                    std::to_string(expected));
    Eigen::VectorXd v(expected);
    std::memcpy(v.data(), a.data(), sizeof(double) * expected);
    return v;
}

py::array_t<double> as_grid(const Eigen::VectorXd& v, int Ns) {
    py::array_t<double> a({Ns, Ns});
    std::memcpy(a.mutable_data(), v.data(), sizeof(double) * v.size());
    return a;
}

struct PyResult {
    hmc::ContactResult r;
    int Ns = 0;
};

class PyContactSolver {
public:
    PyContactSolver(int grid_size, double domain_size, double E_star, double eta,
                    double aca_tol, int leaf_size, bool use_hmatrix,
                    bool use_acagp, double central_fraction, double inline_svd_tol,
                    std::string backend, int q, int near_radius, int h2_leaf_side)
        : kernel_(grid_size, domain_size, E_star) {
        if (backend.empty()) backend = use_hmatrix ? "hmatrix" : "dense";
        backend_ = backend;
        if (backend_ == "h2") {
            h2_ = std::make_unique<hmc::H2Operator>(
                kernel_, hmc::H2Params{h2_leaf_side, q, near_radius});
            h2_->build();
        } else if (backend_ == "fft") {
            fft_ = std::make_unique<hmc::FFTOperator>(kernel_);
            fft_->build();
        } else if (backend_ == "hmatrix") {
            tree_ = std::make_unique<hmc::ClusterTree>(grid_size, leaf_size);
            hmat_ = std::make_unique<hmc::HMatrix>(
                kernel_, *tree_, eta, aca_tol, use_acagp, central_fraction,
                inline_svd_tol);
        } else if (backend_ == "dense") {
            dense_ = kernel_.assemble_dense();
        } else {
            throw std::invalid_argument("unknown backend: " + backend_ +
                                        " (expected dense, hmatrix, h2, or fft)");
        }
    }

    Eigen::VectorXd apply(const Eigen::VectorXd& p) const {
        if (backend_ == "h2") return h2_->matvec(p);
        if (backend_ == "fft") return fft_->matvec(p);
        if (backend_ == "hmatrix") return hmat_->matvec(p);
        return dense_ * p;
    }

    py::array_t<double>
    matvec(const py::array_t<double, py::array::c_style | py::array::forcecast>& p)
        const {
        Eigen::VectorXd v = to_vector(p, kernel_.size());
        Eigen::VectorXd u;
        {
            py::gil_scoped_release release;
            u = apply(v);
        }
        py::array_t<double> out(kernel_.size());
        std::memcpy(out.mutable_data(), u.data(), sizeof(double) * u.size());
        return out;
    }

    PyResult solve(const py::array_t<double, py::array::c_style |
                                                 py::array::forcecast>& gap,
                   double p_nominal, double tol, int max_iter,
                   bool use_pr, const std::string& precond,
                   const py::object& p_init) const {
        Eigen::VectorXd g0 = to_vector(gap, kernel_.size());

        hmc::Precond pc;
        if (precond == "fourier") {
            auto fp = std::make_shared<hmc::FourierPreconditioner>(
                kernel_.grid_size());
            pc = [fp](const Eigen::VectorXd& g,
                      const std::vector<std::uint8_t>& contact) {
                return fp->apply(g, contact);
            };
        } else if (precond != "none" && !precond.empty()) {
            throw std::invalid_argument("precond must be 'none' or 'fourier'");
        }

        Eigen::VectorXd p0;
        const Eigen::VectorXd* p0ptr = nullptr;
        if (!p_init.is_none()) {
            p0 = to_vector(p_init.cast<py::array_t<double, py::array::c_style |
                                                          py::array::forcecast>>(),
                           kernel_.size());
            p0ptr = &p0;
        }

        PyResult out;
        out.Ns = kernel_.grid_size();
        {
            py::gil_scoped_release release;
            auto op = [this](const Eigen::VectorXd& v) { return apply(v); };
            out.r = hmc::solve_contact(op, g0, p_nominal, tol, max_iter, use_pr,
                                       pc, p0ptr);
        }
        return out;
    }

    // Returns (N_blocks, 6) array:
    //   [row_begin, row_size, col_begin, col_size, is_dense, rank]
    // rank = U.cols() for low-rank blocks, 0 for dense blocks.
    // All indices are in permuted (cluster) index space.
    py::array_t<double> block_layout() const {
        if (backend_ != "hmatrix")
            return py::array_t<double>(std::vector<py::ssize_t>{0, 6});
        const auto& blks = hmat_->blocks();
        const int nb = static_cast<int>(blks.size());
        py::array_t<double> out({nb, 6});
        auto r = out.mutable_unchecked<2>();
        for (int i = 0; i < nb; ++i) {
            r(i, 0) = blks[i].row_begin;
            r(i, 1) = blks[i].row_size;
            r(i, 2) = blks[i].col_begin;
            r(i, 3) = blks[i].col_size;
            r(i, 4) = blks[i].dense ? 1.0 : 0.0;
            r(i, 5) = blks[i].dense ? 0.0 : static_cast<double>(blks[i].U.cols());
        }
        return out;
    }

    void recompress(double svd_tol) {
        if (backend_ == "hmatrix") hmat_->recompress(svd_tol);
    }

    py::dict hmatrix_info() const {
        py::dict d;
        if (backend_ == "h2") {
            h2_->print_statistics();
            const auto s = h2_->info();
            d["backend"] = "h2";
            d["n"] = s.N;
            d["q"] = s.q;
            d["r"] = s.r;
            d["leaf_side"] = s.leaf_side;
            d["n_far_interactions"] = static_cast<long long>(s.n_far_interactions);
            d["n_near_interactions"] = static_cast<long long>(s.n_near_interactions);
            d["n_unique_couplings"] = s.n_unique_couplings;
            d["n_near_stencils"] = s.n_near_stencils;
            d["n_boxes"] = s.n_boxes;
            d["n_leaves"] = s.n_leaves;
            d["nlevels"] = s.nlevels;
            d["bytes"] = s.bytes_total;
            d["bytes_coupling"] = static_cast<long long>(s.bytes_coupling);
            d["bytes_near"] = static_cast<long long>(s.bytes_near);
            d["bytes_buffers"] = static_cast<long long>(s.bytes_buffers);
            d["compression"] =
                double(s.bytes_total) / (8.0 * double(s.N) * double(s.N));
            return d;
        }
        if (backend_ == "fft") {
            fft_->print_statistics();
            const auto s = fft_->info();
            d["backend"] = "fft";
            d["n"] = s.N;
            d["padded_side"] = s.M;
            d["bytes"] = s.bytes_total;
            d["bytes_spectrum"] = static_cast<long long>(s.bytes_spectrum);
            d["bytes_scratch"] = static_cast<long long>(s.bytes_scratch);
            d["compression"] =
                double(s.bytes_total) / (8.0 * double(s.N) * double(s.N));
            return d;
        }
        if (backend_ != "hmatrix") {
            d["dense"] = true;
            d["bytes"] = 8LL * kernel_.size() * kernel_.size();
            py::print("dense influence matrix,", kernel_.size(), "x",
                      kernel_.size());
            return d;
        }
        const auto s = hmat_->info();
        d["n"] = s.n;
        d["n_dense_blocks"] = s.n_dense;
        d["n_lowrank_blocks"] = s.n_lowrank;
        d["max_rank"] = s.max_rank;
        d["avg_rank"] = s.avg_rank;
        d["bytes"] = s.bytes;
        d["compression"] = s.compression;
        py::print(s.to_string());
        return d;
    }

private:
    hmc::BoussinesqKernel kernel_;
    std::string backend_;
    std::unique_ptr<hmc::ClusterTree> tree_;
    std::unique_ptr<hmc::HMatrix> hmat_;
    std::unique_ptr<hmc::H2Operator> h2_;
    std::unique_ptr<hmc::FFTOperator> fft_;
    Eigen::MatrixXd dense_;
};

// Single-entry nested-grid (cascadic/FMG) solve: builds the coarse->fine
// hierarchy and H2 operators internally, no Python orchestration.
PyResult py_solve_nested(
    int grid_size,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& gap,
    double p_nominal, double domain_size, double E_star, int coarsest, int q,
    int leaf_side, bool precond, double tol, double coarse_tol, int max_iter,
    bool use_pr, bool single_precision, bool light_result,
    const std::string& backend, bool record_error_history) {
    Eigen::VectorXd g0 = to_vector(gap, grid_size * grid_size);
    hmc::NestedParams np{coarsest, q, leaf_side, precond, coarse_tol,
                         single_precision, light_result, backend,
                         record_error_history};
    PyResult out;
    out.Ns = grid_size;
    {
        py::gil_scoped_release release;
        out.r = hmc::solve_contact_nested(grid_size, domain_size, E_star,
                                          std::move(g0), p_nominal, tol,
                                          max_iter, use_pr, np);
    }
    return out;
}

} // namespace

PYBIND11_MODULE(aspher, m) {
    m.doc() = "ASPHER - Accelerated SPectral and HiERarchical contact solver. "
              "BEM normal contact on an elastic half-space (Boussinesq kernel, "
              "Love element integration, H-matrix/H2-FMM operators, "
              "FFTW |q| preconditioner, Polonsky-Keer CG)";

    py::class_<PyResult>(m, "ContactResult")
        .def_property_readonly(
            "pressure", [](const PyResult& s) { return as_grid(s.r.pressure, s.Ns); })
        .def_property_readonly(
            "displacement", [](const PyResult& s) -> py::object {
                if (s.r.displacement.size() == 0) return py::none(); // light_result
                return as_grid(s.r.displacement, s.Ns);
            })
        .def_property_readonly(
            "gap", [](const PyResult& s) -> py::object {
                if (s.r.gap.size() == 0) return py::none(); // light_result
                return as_grid(s.r.gap, s.Ns);
            })
        .def_property_readonly("objective",
                               [](const PyResult& s) { return s.r.objective; })
        .def_property_readonly(
            "contact_area", [](const PyResult& s) { return s.r.contact_fraction; })
        .def_property_readonly(
            "mean_pressure", [](const PyResult& s) { return s.r.mean_pressure; })
        .def_property_readonly("approach",
                               [](const PyResult& s) { return s.r.approach; })
        .def_property_readonly("iterations",
                               [](const PyResult& s) { return s.r.iterations; })
        .def_property_readonly("error", [](const PyResult& s) { return s.r.error; })
        .def_property_readonly(
            "error_history",
            [](const PyResult& s) {
                return py::array_t<double>(s.r.error_history.size(),
                                           s.r.error_history.data());
            })
        .def_property_readonly("converged",
                               [](const PyResult& s) { return s.r.converged; })
        .def("__repr__", [](const PyResult& s) {
            std::ostringstream os;
            os << "<ContactResult: " << (s.r.converged ? "converged" : "NOT converged")
               << " in " << s.r.iterations << " iters, error " << s.r.error
               << ", contact area " << s.r.contact_fraction << ", mean p "
               << s.r.mean_pressure << ">";
            return os.str();
        });

    py::class_<PyContactSolver>(m, "ContactSolver")
        .def(py::init<int, double, double, double, double, int, bool, bool, double,
                      double, std::string, int, int, int>(),
             py::arg("grid_size"), py::arg("domain_size") = 1.0,
             py::arg("E_star") = 1.0, py::arg("eta") = 2.0,
             py::arg("aca_tol") = 1e-6, py::arg("leaf_size") = 64,
             py::arg("use_hmatrix") = true,
             py::arg("use_acagp") = false,
             py::arg("central_fraction") = 0.3,
             py::arg("inline_svd_tol") = 0.0,
             py::arg("backend") = "", py::arg("q") = 4,
             py::arg("near_radius") = 1, py::arg("h2_leaf_side") = 8)
        .def("matvec", &PyContactSolver::matvec, py::arg("p"),
             "Influence-matrix product u = S p; accepts shape (N,) or (Ns, Ns)")
        .def("solve", &PyContactSolver::solve, py::arg("gap"),
             py::arg("p_nominal"), py::arg("tol") = 1e-8,
             py::arg("max_iter") = 5000, py::arg("use_pr") = true,
             py::arg("precond") = "none", py::arg("p_init") = py::none(),
             "Solve the normal contact problem. use_pr=True (default) uses "
             "Polak-Ribiere+ beta. precond='fourier' enables the |q| spectral "
             "preconditioner; p_init is an optional warm-start pressure field.")
        .def("block_layout", &PyContactSolver::block_layout,
             "Return (N_blocks, 5) array [row_begin, row_size, col_begin, col_size, is_dense]")
        .def("recompress", &PyContactSolver::recompress, py::arg("svd_tol"),
             "Truncated SVD recompression: drop singular values below svd_tol * sigma_max")
        .def("hmatrix_info", &PyContactSolver::hmatrix_info,
             "Print and return H-matrix block/compression statistics");

    m.def("solve_nested", &py_solve_nested, py::arg("grid_size"),
          py::arg("gap"), py::arg("p_nominal"), py::arg("domain_size") = 1.0,
          py::arg("E_star") = 1.0, py::arg("coarsest") = 64, py::arg("q") = 6,
          py::arg("leaf_side") = 8, py::arg("precond") = true,
          py::arg("tol") = 1e-8, py::arg("coarse_tol") = 1e-4,
          py::arg("max_iter") = 20000, py::arg("use_pr") = true,
          py::arg("single_precision") = false, py::arg("light_result") = false,
          py::arg("backend") = "h2", py::arg("record_error_history") = false,
          "Single-entry nested-grid (cascadic/FMG) contact solve: builds the "
          "coarse->fine hierarchy and H2 operators internally and warm-starts "
          "each level with the prolonged coarse pressure. grid_size must equal "
          "coarsest * 2^k. Returns a ContactResult. backend='h2' (O(N) memory) "
          "or 'fft' (exact convolution, fastest at Ns<=8192). "
          "record_error_history=True fills .error_history with the finest "
          "level's per-iteration complementarity error (off by default).");
}
