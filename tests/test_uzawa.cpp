#include "boussinesq_kernel.hpp"
#include "fft_operator.hpp"
#include "cerruti_kernel.hpp"
#include "tangential_operator.hpp"
#include "contact_solver.hpp"
#include "friction_solve.hpp"
#include "bipotential.hpp"

#include <cmath>
#include <cstdio>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAILED: %s (line %d)\n", #cond, __LINE__);        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

// The analytic cone projection: verify the three sectors and the projection
// property (idempotence + distance minimization checked against a dense
// sampled search on the cone boundary).
static int test_cone_projection() {
    const double mu = 0.4;
    // separation sector: mu|t_t| < -t_n -> zero
    {
        double tx = 0.1, ty = 0.0, tn = -0.5;
        hmc::project_coulomb_cone(mu, tx, ty, tn);
        CHECK(tx == 0.0 && ty == 0.0 && tn == 0.0);
    }
    // stick sector: |t_t| <= mu t_n -> unchanged
    {
        double tx = 0.1, ty = 0.05, tn = 1.0;
        const double tx0 = tx, ty0 = ty, tn0 = tn;
        hmc::project_coulomb_cone(mu, tx, ty, tn);
        CHECK(tx == tx0 && ty == ty0 && tn == tn0);
    }
    // slip sector: radial return lands ON the cone, is idempotent, and is
    // closer to tau than any sampled cone-boundary point
    {
        const double tx0 = 1.0, ty0 = 0.3, tn0 = 0.8; // |t_t| > mu tn
        double tx = tx0, ty = ty0, tn = tn0;
        hmc::project_coulomb_cone(mu, tx, ty, tn);
        const double qt = std::hypot(tx, ty);
        CHECK(std::abs(qt - mu * tn) <= 1e-14 * std::max(1.0, qt));
        double tx2 = tx, ty2 = ty, tn2 = tn;
        hmc::project_coulomb_cone(mu, tx2, ty2, tn2);
        CHECK(tx2 == tx && ty2 == ty && tn2 == tn);
        const double d2 = (tx - tx0) * (tx - tx0) + (ty - ty0) * (ty - ty0) +
                          (tn - tn0) * (tn - tn0);
        for (int k = 0; k <= 200; ++k) { // sample the boundary generatrix
            const double pn = 0.01 * k;
            // boundary point along the tangential direction of tau
            const double tt = std::hypot(tx0, ty0);
            const double bx = mu * pn * tx0 / tt, by = mu * pn * ty0 / tt;
            const double dd = (bx - tx0) * (bx - tx0) +
                              (by - ty0) * (by - ty0) +
                              (pn - tn0) * (pn - tn0);
            CHECK(d2 <= dd + 1e-12);
        }
    }
    return 0;
}

// helper: Hertz-like analytic threshold cap (mirrors test_friction's)
static Eigen::VectorXd hertz_gap(int Ns, double L, double R) {
    // parabolic indenter gap x^2/(2R), centered
    Eigen::VectorXd g0(Ns * Ns);
    const double h = L / Ns;
    for (int iy = 0; iy < Ns; ++iy)
        for (int ix = 0; ix < Ns; ++ix) {
            const double x = (ix + 0.5) * h - 0.5 * L,
                         y = (iy + 0.5) * h - 0.5 * L;
            g0(iy * Ns + ix) = (x * x + y * y) / (2.0 * R);
        }
    return g0;
}

// Cross-check: production staggered path (normal PK at p_bar -> alpha*, p*;
// tangential displacement-control at delta_t -> q*) vs the bipotential
// Uzawa at the SAME imposed (alpha*, delta_t). Independent solvers, same
// fixed point (uncoupled elasticity).
static int test_cross_check_coulomb() {
    const int Ns = 32, N = Ns * Ns;
    const double L = 1.0, E = 1.0, nu = 0.3, mu = 0.4, p_bar = 3e-3;
    hmc::BoussinesqKernel BK(Ns, L, E);
    hmc::FFTOperator S(BK);
    S.build();
    hmc::CerrutiKernel CK(Ns, L, E, nu);
    hmc::TangentialFFTOperator C(CK);
    C.build();
    const Eigen::VectorXd g0 = hertz_gap(Ns, L, 2.0);

    // production normal
    hmc::MatVec Sop = [&S](const Eigen::VectorXd& x) { return S.matvec(x); };
    hmc::ContactResult nr = hmc::solve_contact(Sop, g0, p_bar, 1e-12, 20000);
    CHECK(nr.converged);
    const double alpha = nr.approach;

    // production tangential (displacement control, partial slip)
    hmc::TanMatVecInto Cop = [&C](const Eigen::VectorXd& x,
                                  Eigen::VectorXd& y) { C.matvec_into(x, y); };
    const Eigen::VectorXd s = mu * nr.pressure;
    // pick delta_t ~ half the gross-slip shift so both stick and slip exist
    const Eigen::Vector2d dt(6e-4, 2e-4);
    hmc::TangentialResult tr =
        hmc::solve_tangential(Cop, s, false, dt, 1e-6, 200000);
    CHECK(tr.converged);
    CHECK(tr.n_stick > 0 && tr.n_slip > 0);

    // bipotential at the same (alpha, dt)
    auto Sinto = [&S](const Eigen::VectorXd& x, Eigen::VectorXd& y) {
        y = S.matvec(x);
    };
    auto Cinto = [&C](const Eigen::VectorXd& x, Eigen::VectorXd& y) {
        C.matvec_into(x, y);
    };
    hmc::BipotentialResult br = hmc::solve_bipotential(
        Sinto, Cinto, g0, mu, alpha, dt, 1e-9, 400000);
    std::printf("uzawa coulomb: it %d err %.3e states %d/%d/%d "
                "(prod stick/slip %d/%d)\n",
                br.iterations, br.error, br.n_open, br.n_stick, br.n_slip,
                tr.n_stick, tr.n_slip);
    CHECK(br.converged);

    const double relp = (br.p - nr.pressure).norm() / nr.pressure.norm();
    const double relq = (br.q - tr.q).norm() / tr.q.norm();
    std::printf("uzawa coulomb: rel p %.3e  rel q %.3e\n", relp, relq);
    CHECK(relp < 5e-3);
    CHECK(relq < 5e-3);
    const int n_contact = N - br.n_open;
    CHECK(std::abs(br.n_stick - tr.n_stick) <= std::max(4, n_contact / 50));
    return 0;
}

// Tresca variant through the frozen-threshold cylinder.
static int test_cross_check_tresca() {
    const int Ns = 32;
    const double L = 1.0, E = 1.0, nu = 0.3, p_bar = 3e-3, tau_c = 6e-4;
    hmc::BoussinesqKernel BK(Ns, L, E);
    hmc::FFTOperator S(BK);
    S.build();
    hmc::CerrutiKernel CK(Ns, L, E, nu);
    hmc::TangentialFFTOperator C(CK);
    C.build();
    const Eigen::VectorXd g0 = hertz_gap(Ns, L, 2.0);
    hmc::MatVec Sop = [&S](const Eigen::VectorXd& x) { return S.matvec(x); };
    hmc::ContactResult nr = hmc::solve_contact(Sop, g0, p_bar, 1e-12, 20000);
    CHECK(nr.converged);
    Eigen::VectorXd s(Ns * Ns);
    for (int i = 0; i < Ns * Ns; ++i)
        s(i) = (nr.pressure(i) > 0.0) ? tau_c : 0.0;
    hmc::TanMatVecInto Cop = [&C](const Eigen::VectorXd& x,
                                  Eigen::VectorXd& y) { C.matvec_into(x, y); };
    const Eigen::Vector2d dt(6e-4, 0.0);
    hmc::TangentialResult tr =
        hmc::solve_tangential(Cop, s, false, dt, 1e-4, 200000);
    CHECK(tr.converged);
    auto Sinto = [&S](const Eigen::VectorXd& x, Eigen::VectorXd& y) {
        y = S.matvec(x);
    };
    auto Cinto = [&C](const Eigen::VectorXd& x, Eigen::VectorXd& y) {
        C.matvec_into(x, y);
    };
    hmc::BipotentialResult br = hmc::solve_bipotential(
        Sinto, Cinto, g0, /*mu=*/0.0, nr.approach, dt, 1e-8, 400000, 1.0, &s);
    const double relp = (br.p - nr.pressure).norm() / nr.pressure.norm();
    const double relq = (br.q - tr.q).norm() / tr.q.norm();
    std::printf("uzawa tresca: it %d rel p %.3e rel q %.3e\n", br.iterations,
                relp, relq);
    CHECK(br.converged);
    CHECK(relp < 5e-3);
    CHECK(relq < 1e-2);
    return 0;
}

int main() {
    if (int rc = test_cone_projection()) return rc;
    if (int rc = test_cross_check_coulomb()) return rc;
    if (int rc = test_cross_check_tresca()) return rc;
    std::printf("test_uzawa: all checks passed\n");
    return 0;
}
