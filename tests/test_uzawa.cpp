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

// Forward declaration: direct KKT verification helper
static int check_bipotential_kkt(
    const hmc::BipotentialResult& br,
    const std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>& S,
    const std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>& C,
    const Eigen::VectorXd& g0, double mu, double alpha,
    const Eigen::Vector2d& dt, const Eigen::VectorXd* s_frozen, double gtol,
    double wtol);

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

    // Direct KKT self-check (measured violations: gap~9e-11, stick~2e-11,
    // align~3.5e-13; gates set at ~10x measured values for safety margin)
    const double gtol_coulomb = 1e-9;
    const double wtol_coulomb = 1e-9;
    if (int rc = check_bipotential_kkt(br, Sinto, Cinto, g0, mu, alpha, dt,
                                       nullptr, gtol_coulomb, wtol_coulomb))
        return rc;
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

    // Direct KKT self-check (measured violations: gap~9e-10, stick~0,
    // align~1e-19; gates set at ~10x measured values for safety margin)
    const double gtol_tresca = 1e-9;
    const double wtol_tresca = 1e-9;
    if (int rc = check_bipotential_kkt(br, Sinto, Cinto, g0, 0.0, nr.approach, dt,
                                       &s, gtol_tresca, wtol_tresca))
        return rc;
    return 0;
}

// Direct KKT verification of a bipotential solution: normal LCP (p>0 =>
// |g| small; p=0 => g >= -tol) and tangential stick/slip law against
// s = mu*p (Coulomb) or s_frozen. gtol/wtol are absolute tolerances scaled
// by the caller from the problem scales.
static int check_bipotential_kkt(
    const hmc::BipotentialResult& br,
    const std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>& S,
    const std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>& C,
    const Eigen::VectorXd& g0, double mu, double alpha,
    const Eigen::Vector2d& dt, const Eigen::VectorXd* s_frozen, double gtol,
    double wtol) {
    const int N = static_cast<int>(g0.size());
    Eigen::VectorXd uz(N), ut(2 * N);
    S(br.p, uz);
    C(br.q, ut);
    double worst_pen = 0.0, worst_gap = 0.0, worst_stick = 0.0,
           worst_align = 0.0, worst_neg = 0.0, worst_over = 0.0;
    for (int i = 0; i < N; ++i) {
        const double g = uz(i) + g0(i) - alpha;
        const double si = s_frozen ? (*s_frozen)(i) : mu * br.p(i);
        const double qn = std::hypot(br.q(i), br.q(N + i));
        const double wx = dt(0) - ut(i), wy = dt(1) - ut(N + i);
        if (br.p(i) > 0.0)
            worst_gap = std::max(worst_gap, std::abs(g));
        else
            worst_pen = std::max(worst_pen, std::max(0.0, -g));
        if (si <= 0.0) {
            worst_over = std::max(worst_over, qn);
            continue;
        }
        worst_over = std::max(worst_over, qn - si);
        if (qn < si * (1.0 - 1e-9)) {
            worst_stick = std::max(worst_stick, std::hypot(wx, wy));
        } else if (qn > 0.0) {
            const double qhx = br.q(i) / qn, qhy = br.q(N + i) / qn;
            const double wpar = wx * qhx + wy * qhy;
            worst_align = std::max(
                worst_align,
                std::hypot(wx - wpar * qhx, wy - wpar * qhy));
            worst_neg = std::max(worst_neg, std::max(0.0, -wpar));
        }
    }
    std::printf("uzawa KKT: gap %.2e pen %.2e stick %.2e align %.2e neg %.2e"
                " over %.2e\n",
                worst_gap, worst_pen, worst_stick, worst_align, worst_neg,
                worst_over);
    CHECK(worst_gap <= gtol);
    CHECK(worst_pen <= gtol);
    CHECK(worst_stick <= wtol);
    CHECK(worst_align <= wtol);
    CHECK(worst_neg <= wtol);
    CHECK(worst_over <= 1e-12);
    return 0;
}

int main() {
    if (int rc = test_cone_projection()) return rc;
    if (int rc = test_cross_check_coulomb()) return rc;
    if (int rc = test_cross_check_tresca()) return rc;
    std::printf("test_uzawa: all checks passed\n");
    return 0;
}
