#include "boussinesq_kernel.hpp"
#include "contact_solver.hpp"
#include "fourier_precond.hpp"

#include <cstdio>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAILED: %s (line %d)\n", #cond, __LINE__);        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

int main() {
    const int Ns = 64;
    const double L = 1.0, E_star = 1.0, R = 0.5, p_bar = 0.02;
    hmc::BoussinesqKernel kernel(Ns, L, E_star);
    const Eigen::MatrixXd S = kernel.assemble_dense();
    auto op = [&S](const Eigen::VectorXd& v) -> Eigen::VectorXd { return S * v; };

    // paraboloid gap (Hertz), zero at the centre
    const double h = L / Ns;
    Eigen::VectorXd g0(Ns * Ns);
    for (int iy = 0; iy < Ns; ++iy)
        for (int ix = 0; ix < Ns; ++ix) {
            const double x = (ix + 0.5) * h - 0.5 * L;
            const double y = (iy + 0.5) * h - 0.5 * L;
            g0(iy * Ns + ix) = (x * x + y * y) / (2.0 * R);
        }

    hmc::FourierPreconditioner fp(Ns);
    hmc::Precond pc = [&fp](const Eigen::VectorXd& g,
                           const std::vector<std::uint8_t>& contact) {
        return fp.apply(g, contact);
    };

    auto r0 = hmc::solve_contact(op, g0, p_bar, 1e-10, 5000, true);     // none
    auto r1 = hmc::solve_contact(op, g0, p_bar, 1e-10, 5000, true, pc); // fourier
    CHECK(r0.converged);
    CHECK(r1.converged);

    // same solution
    const double relp = (r1.pressure - r0.pressure).norm() / r0.pressure.norm();
    std::printf("precond: none %d it, fourier %d it, relp %.2e, dArea %.2e\n",
                r0.iterations, r1.iterations, relp,
                std::abs(r1.contact_fraction - r0.contact_fraction));
    CHECK(relp < 1e-5);
    CHECK(std::abs(r1.contact_fraction - r0.contact_fraction) < 1e-3);
    // preconditioner must not increase the iteration count
    CHECK(r1.iterations <= r0.iterations);

    // warm start from the converged pressure -> converges almost immediately
    auto r2 = hmc::solve_contact(op, g0, p_bar, 1e-10, 5000, true, pc,
                                 &r0.pressure);
    CHECK(r2.converged);
    CHECK(r2.iterations < r1.iterations);
    std::printf("warm-start from solution: %d it\n", r2.iterations);

    // single-precision path (solve_contact_impl<float>): same solution to ~float
    // accuracy at a reachable tolerance.
    hmc::MatVecIntoT<float> opf = [&S](const Eigen::VectorXf& v,
                                       Eigen::VectorXf& out) {
        out = (S * v.cast<double>()).cast<float>();
    };
    Eigen::VectorXf g0f = g0.cast<float>();
    auto r3 = hmc::solve_contact_impl<float>(opf, g0f, float(p_bar), 5e-6f, 5000,
                                             true, hmc::PrecondIntoT<float>{},
                                             nullptr);
    CHECK(r3.converged);
    const double relpf = (r3.pressure - r0.pressure).norm() / r0.pressure.norm();
    std::printf("single precision: %d it, relp vs double %.2e\n",
                r3.iterations, relpf);
    CHECK(relpf < 1e-3);
    CHECK(std::abs(r3.contact_fraction - r0.contact_fraction) < 2e-3);

    return 0;
}
