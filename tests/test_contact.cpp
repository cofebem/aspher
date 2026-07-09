#include "boussinesq_kernel.hpp"
#include "cluster_tree.hpp"
#include "contact_solver.hpp"
#include "hmatrix.hpp"

#include <cmath>
#include <cstdio>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAILED: %s (line %d)\n", #cond, __LINE__);        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

int main() {
    // Hertz: rigid sphere (parabolic gap) on elastic half-space.
    const int Ns = 64;
    const double L = 1.0, E = 1.0, R = 0.5, p_bar = 0.0123;
    const double h = L / Ns;
    const int N = Ns * Ns;

    hmc::BoussinesqKernel K(Ns, L, E);
    Eigen::VectorXd g0(N);
    for (int iy = 0; iy < Ns; ++iy)
        for (int ix = 0; ix < Ns; ++ix) {
            const double x = (ix + 0.5) * h - 0.5 * L;
            const double y = (iy + 0.5) * h - 0.5 * L;
            g0(iy * Ns + ix) = (x * x + y * y) / (2.0 * R);
        }

    const Eigen::MatrixXd S = K.assemble_dense();
    auto dense_op = [&](const Eigen::VectorXd& v) -> Eigen::VectorXd { return S * v; };
    hmc::ClusterTree tree(Ns, 32);
    hmc::HMatrix H(K, tree, 2.0, 1e-7);
    auto h_op = [&](const Eigen::VectorXd& v) { return H.matvec(v); };

    const auto rd = hmc::solve_contact(dense_op, g0, p_bar, 1e-10, 4000);
    const auto rh = hmc::solve_contact(h_op, g0, p_bar, 1e-10, 4000);
    std::printf("dense: %d iters, err %.2e | hmat: %d iters, err %.2e\n",
                rd.iterations, rd.error, rh.iterations, rh.error);
    CHECK(rd.converged && rh.converged);

    // Hertz analytics: F = p_bar L^2, a = (3FR/(4E*))^(1/3), p0 = 3F/(2 pi a^2)
    const double F = p_bar * L * L;
    const double a = std::cbrt(3.0 * F * R / (4.0 * E));
    const double p0 = 3.0 * F / (2.0 * M_PI * a * a);

    for (const auto* r : {&rd, &rh}) {
        const double area = r->contact_fraction * L * L;
        const double a_num = std::sqrt(area / M_PI);
        const double p_max = r->pressure.maxCoeff();
        std::printf("  a_num/a = %.4f, p_max/p0 = %.4f\n", a_num / a, p_max / p0);
        CHECK(std::abs(a_num / a - 1.0) < 0.05);
        CHECK(std::abs(p_max / p0 - 1.0) < 0.05);
        CHECK(std::abs(r->mean_pressure / p_bar - 1.0) < 1e-12);
        CHECK(r->gap.minCoeff() > -1e-5 * g0.maxCoeff()); // no penetration
        CHECK(r->pressure.minCoeff() >= 0.0);
    }

    // dense and H-matrix solutions must agree
    const double pdiff = (rd.pressure - rh.pressure).norm() / rd.pressure.norm();
    std::printf("  dense vs H pressure rel L2 diff: %.2e\n", pdiff);
    CHECK(pdiff < 1e-3);
    CHECK(std::abs(rd.objective / rh.objective - 1.0) < 1e-6);

    // opt-in convergence history: off by default (no cost, no data)
    CHECK(rd.error_history.empty());

    // with record_history=true, the trace is non-empty and its last entry
    // matches the reported final error (see the design spec for why this
    // is checked instead of size() == iterations)
    const auto rd_hist = hmc::solve_contact(dense_op, g0, p_bar, 1e-10, 4000,
                                            true, hmc::Precond{}, nullptr,
                                            false, true);
    CHECK(!rd_hist.error_history.empty());
    CHECK(std::abs(rd_hist.error_history.back() - rd_hist.error) < 1e-12);
    std::printf("  error_history: %zu entries, last=%.3e\n",
                rd_hist.error_history.size(), rd_hist.error_history.back());

    std::printf("test_contact: all checks passed\n");
    return 0;
}
