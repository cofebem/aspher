#include "boussinesq_kernel.hpp"
#include "contact_solver.hpp"
#include "fourier_precond.hpp"
#include "stencil_precond.hpp"

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
    hmc::SolveOptions fopt;
    fopt.tol = 5e-6;
    fopt.max_iter = 5000;
    auto r3 = hmc::solve_contact_impl<float>(opf, g0f, float(p_bar), fopt,
                                             hmc::PrecondIntoT<float>{},
                                             nullptr);
    CHECK(r3.converged);
    const double relpf = (r3.pressure - r0.pressure).norm() / r0.pressure.norm();
    std::printf("single precision: %d it, relp vs double %.2e\n",
                r3.iterations, relpf);
    CHECK(relpf < 1e-3);
    CHECK(std::abs(r3.contact_fraction - r0.contact_fraction) < 2e-3);

    // ── S4/S5: the stencil preconditioner solves the same problem ─────────
    // Same operator, different implementation: the answer must agree to
    // roundoff and the iteration count must stay in a documented band.
    {
        hmc::StencilPreconditioner sp(Ns, 2);
        hmc::Precond pcs = [&sp](const Eigen::VectorXd& g,
                                 const std::vector<std::uint8_t>& contact) {
            Eigen::VectorXd z;
            sp.apply_into(g, contact, z);
            return z;
        };
        auto r4 = hmc::solve_contact(op, g0, p_bar, 1e-10, 5000, true, pcs);
        CHECK(r4.converged);
        const double rel = (r4.pressure - r1.pressure).norm() / r1.pressure.norm();
        std::printf("stencil: %d it (fft %d), pressure rel %.2e, dArea %.2e\n",
                    r4.iterations, r1.iterations, rel,
                    std::abs(r4.contact_fraction - r1.contact_fraction));
        CHECK(rel < 1e-6);                                        // S4
        CHECK(r4.contact_fraction == r1.contact_fraction);        // S4
        CHECK(r4.iterations <= r1.iterations * 6 / 5 + 1);        // S5: +20%
    }

    // ── the cost gate fires on a deliberately expensive preconditioner ────
    {
        hmc::FourierPreconditioner slow_fp(Ns);
        hmc::PrecondIntoT<double> slow =
            [&slow_fp](const Eigen::VectorXd& g,
                       const std::vector<std::uint8_t>& contact,
                       Eigen::VectorXd& z) {
                // same operator, applied 40 times: correct, just costly
                for (int rep = 0; rep < 40; ++rep)
                    slow_fp.apply_into(g, contact, z);
            };
        hmc::MatVecIntoT<double> opi = [&S](const Eigen::VectorXd& v,
                                            Eigen::VectorXd& out) {
            out = S * v;
        };
        hmc::SolveOptions go;
        go.tol = 1e-10;
        go.max_iter = 5000;
        auto rg = hmc::solve_contact_impl<double>(opi, g0, p_bar, go, slow,
                                                  nullptr);
        CHECK(rg.converged);
        CHECK(rg.precond_dropped);
        std::printf("cost gate: dropped=%d, %d it, area matches %d\n",
                    int(rg.precond_dropped), rg.iterations,
                    int(rg.contact_fraction == r0.contact_fraction));
        CHECK(std::abs(rg.contact_fraction - r0.contact_fraction) < 1e-3);

        // and does NOT fire when disabled
        go.precond_cost_gate = 0.0;
        auto rn = hmc::solve_contact_impl<double>(opi, g0, p_bar, go, slow,
                                                  nullptr);
        CHECK(rn.converged);
        CHECK(!rn.precond_dropped);
    }

    return 0;
}
