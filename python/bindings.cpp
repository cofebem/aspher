// pybind11 (and Python.h) must precede any standard header: Python.h sets
// feature-test macros that break <ctime> with the conda gcc toolchain otherwise.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "boussinesq_kernel.hpp"
#include "cluster_tree.hpp"
#include "contact_solver.hpp"
#include "fft_operator.hpp"
#include "fourier_precond.hpp"
#include "friction_model.hpp"
#include "h2_operator.hpp"
#include "hmatrix.hpp"
#include "nested_solve.hpp"

#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

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

// numpy (N,) or (Ns,Ns) float64 -> flat Eigen vector of length `expected`.
// (Eigen::Index-sized sibling of to_vector, for the friction path's 2N
// vectors — reuse to_vector where an int length is already in hand.)
Eigen::VectorXd to_flat(const py::array_t<double, py::array::c_style |
                                                     py::array::forcecast>& a,
                       Eigen::Index expected) {
    if (a.size() != expected)
        throw std::invalid_argument("array has " + std::to_string(a.size()) +
                                    " entries, expected " +
                                    std::to_string(expected));
    Eigen::VectorXd v(expected);
    std::memcpy(v.data(), a.data(), sizeof(double) * expected);
    return v;
}

py::array_t<double> as_flat(const Eigen::VectorXd& v) {
    py::array_t<double> a(v.size());
    std::memcpy(a.mutable_data(), v.data(), sizeof(double) * v.size());
    return a;
}

// UserFriction: a CallbackModel whose C++ threshold functor calls back into a
// Python callable fn(p, v, T) -> s. The functor RE-ACQUIRES the GIL because
// FrictionSolver.step() releases it around the (long) C++ solve; this nested
// acquire is the standard pybind11 pattern. A Python exception raised inside
// fn surfaces as py::error_already_set, propagates through the transactional
// driver (state left intact), and is restored by pybind11 at the step()
// boundary where the GIL is held again. Ownership: the py::function is held
// by value inside the std::function, so the callable outlives the model.
std::shared_ptr<hmc::CallbackModel>
make_user_friction(py::function fn, bool velocity_dependent) {
    hmc::CallbackModel::Fn cfn =
        [fn](const Eigen::VectorXd& p, const Eigen::VectorXd& v,
             const Eigen::VectorXd& T, Eigen::VectorXd& s) {
            py::gil_scoped_acquire gil;
            py::object out = fn(as_flat(p), as_flat(v), as_flat(T));
            s = to_flat(py::cast<py::array_t<double, py::array::c_style |
                                                         py::array::forcecast>>(
                            out),
                        p.size());
        };
    return std::make_shared<hmc::CallbackModel>(std::move(cfn),
                                                velocity_dependent);
}

// Shared threshold(p, v, T) helper for the model .def below.
py::array_t<double> model_threshold(const hmc::FrictionModel& m,
                                    const py::array_t<double, py::array::c_style |
                                                                 py::array::forcecast>& p,
                                    const py::array_t<double, py::array::c_style |
                                                                 py::array::forcecast>& v,
                                    const py::array_t<double, py::array::c_style |
                                                                 py::array::forcecast>& T) {
    const Eigen::Index N = p.size();
    Eigen::VectorXd pv = to_flat(p, N), vv = to_flat(v, N), Tv = to_flat(T, N), s;
    m.threshold(pv, vv, Tv, s);
    return as_flat(s);
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
    const std::string& backend, bool record_error_history, bool active_set,
    double active_delta, int active_halo, int active_max_rounds) {
    const py::ssize_t expected =
        static_cast<py::ssize_t>(grid_size) * grid_size;
    if (gap.size() != expected)
        throw std::invalid_argument("array has " + std::to_string(gap.size()) +
                                    " entries, expected " +
                                    std::to_string(expected));
    // zero-copy view of the numpy buffer: the gap is never copied C++-side
    // (2.1 GiB at Ns=16384 double). The py::array parameter keeps the buffer
    // alive across the GIL-released call; forcecast only materialises a
    // temporary when the input is not already a C-contiguous float64 array.
    Eigen::Map<const Eigen::VectorXd> g0(gap.data(), expected);
    hmc::NestedParams np{coarsest, q, leaf_side, precond, coarse_tol,
                         single_precision, light_result, backend,
                         record_error_history};
    np.active_set = active_set;
    np.active_delta = active_delta;
    np.active_halo = active_halo;
    np.active_max_rounds = active_max_rounds;
    PyResult out;
    out.Ns = grid_size;
    {
        py::gil_scoped_release release;
        out.r = hmc::solve_contact_nested(grid_size, domain_size, E_star, g0,
                                          p_nominal, tol, max_iter, use_pr, np);
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
        .def_property_readonly(
            "active_rounds", [](const PyResult& s) { return s.r.active_rounds; })
        .def_property_readonly(
            "active_fallback",
            [](const PyResult& s) { return s.r.active_fallback; })
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
          py::arg("active_set") = false, py::arg("active_delta") = 0.05,
          py::arg("active_halo") = 2, py::arg("active_max_rounds") = 5,
          "Single-entry nested-grid (cascadic/FMG) contact solve: builds the "
          "coarse->fine hierarchy and H2 operators internally and warm-starts "
          "each level with the prolonged coarse pressure. grid_size must equal "
          "coarsest * 2^k. Returns a ContactResult. backend='h2' (O(N) memory) "
          "or 'fft' (exact convolution, fastest at Ns<=8192). "
          "record_error_history=True fills .error_history with the finest "
          "level's per-iteration complementarity error (off by default). "
          "active_set=True (h2 only) solves the finest level restricted to a "
          "candidate set (dilated coarse contact + coarse gap < "
          "active_delta*scale) through the masked H2 matvec, with per-round "
          "full-grid verification and a full-solve fallback after "
          "active_max_rounds (see .active_rounds/.active_fallback).");

    py::class_<hmc::FrictionModel, std::shared_ptr<hmc::FrictionModel>>(
        m, "FrictionModel",
        "Abstract friction threshold model s = tau_c(p, |v|, T). Use "
        "TrescaFriction, CoulombFriction, or UserFriction.")
        .def("threshold", &model_threshold, py::arg("p"), py::arg("v"),
             py::arg("T"),
             "Evaluate the threshold field s (flat (N,)) from pressure p, slip "
             "speed v, temperature T (each (N,) or (Ns,Ns)). s = 0 where p<=0.")
        .def_property_readonly("velocity_dependent",
                               &hmc::FrictionModel::velocity_dependent);

    py::class_<hmc::TrescaModel, hmc::FrictionModel,
               std::shared_ptr<hmc::TrescaModel>>(
        m, "TrescaFriction",
        "Tresca friction: pressure-independent threshold s = tau_c inside "
        "contact (0 outside).")
        .def(py::init<double>(), py::arg("tau_c"));

    py::class_<hmc::CoulombModel, hmc::FrictionModel,
               std::shared_ptr<hmc::CoulombModel>>(
        m, "CoulombFriction",
        "Coulomb friction: s = mu * p (p clamped at 0).")
        .def(py::init<double>(), py::arg("mu"));

    py::class_<hmc::CallbackModel, hmc::FrictionModel,
               std::shared_ptr<hmc::CallbackModel>>(
        m, "UserFriction",
        "User friction law: fn(p, v, T) -> s, operating on flat float64 "
        "arrays (each (N,)); the returned s is sanitized (clamped >= 0, "
        "zeroed where p<=0, non-finite -> 0). Set velocity_dependent=True if "
        "s depends on the slip speed v (enables the driver's threshold "
        "fixed-point loop). Derivatives are NOT required (the solvers are "
        "derivative-free).")
        .def(py::init(&make_user_friction), py::arg("fn"),
             py::arg("velocity_dependent") = false);
}
