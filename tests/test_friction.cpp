#include "fourier_precond.hpp"
#include "friction_solve.hpp"
#include "tangential_operator.hpp"
#include "cerruti_kernel.hpp"
#include "contact_solver.hpp"
#include "fft_operator.hpp"
#include "boussinesq_kernel.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAILED: %s (line %d)\n", #cond, __LINE__);        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

// The tangential preconditioner's symbol is the analytic inverse of the
// Cerruti symbol up to an irrelevant overall scale:
//   M⁻¹(k) = |k_int| (I + γ k̂k̂ᵀ),  γ = ν/(1−ν)
// (integer wavenumbers; the r2c+c2r round-trip scale is folded in). On a
// full mask a pure Fourier mode is an exact eigenfunction (the grid FFT is
// exact): longitudinal polarization (v ∥ k) responds with |k|(1+γ),
// transverse (v ⊥ k) with |k|.
static int test_precond_symbol() {
    const int Ns = 64, N = Ns * Ns;
    const double nu = 0.3, gamma = nu / (1.0 - nu);
    hmc::TangentialFourierPreconditioner M(Ns, nu);
    const int mx = 3, my = 5;
    const double knorm = std::hypot(double(mx), double(my));
    const double kx = mx / knorm, ky = my / knorm; // unit k̂
    std::vector<std::uint8_t> mask(N, 1);

    auto run = [&](double vx, double vy, double lambda) {
        Eigen::VectorXd g(2 * N), z;
        for (int iy = 0; iy < Ns; ++iy)
            for (int ix = 0; ix < Ns; ++ix) {
                const double ph =
                    2.0 * M_PI * (mx * ix + my * iy) / double(Ns);
                g(iy * Ns + ix) = std::cos(ph) * vx;
                g(N + iy * Ns + ix) = std::cos(ph) * vy;
            }
        M.apply_into(g, mask, false, z);
        const double err = (z - lambda * g).lpNorm<Eigen::Infinity>() /
                           (lambda * g.lpNorm<Eigen::Infinity>());
        std::printf("precond mode response err %.3e (lambda %.6f)\n", err,
                    lambda);
        CHECK(err < 1e-10);
        return 0;
    };
    if (run(kx, ky, knorm * (1.0 + gamma))) return 1;  // longitudinal
    if (run(-ky, kx, knorm)) return 1;                 // transverse

    // Nyquist-touching mode: the odd cross term is zeroed there (self-
    // conjugate mirror), so the response is diagonal: |k|(1 + gamma*k̂a²)
    // per Cartesian polarization.
    {
        const int mxn = 3, myn = Ns / 2;
        const double kn = std::hypot(double(mxn), double(myn));
        const double kxh = mxn / kn;
        Eigen::VectorXd g(2 * N), z;
        for (int iy = 0; iy < Ns; ++iy)
            for (int ix = 0; ix < Ns; ++ix) {
                const double ph =
                    2.0 * M_PI * (mxn * ix + myn * iy) / double(Ns);
                g(iy * Ns + ix) = std::cos(ph);      // x-polarized
                g(N + iy * Ns + ix) = 0.0;
            }
        M.apply_into(g, mask, false, z);
        const double lam = kn * (1.0 + gamma * kxh * kxh);
        double err = 0.0;
        for (int i = 0; i < N; ++i) {
            err = std::max(err, std::abs(z(i) - lam * g(i)));
            err = std::max(err, std::abs(z(N + i))); // y stays zero
        }
        err /= lam * g.lpNorm<Eigen::Infinity>();
        std::printf("precond nyquist mode err %.3e\n", err);
        CHECK(err < 1e-10);
    }
    return 0;
}

// mask and mean-removal behavior
static int test_precond_mask_mean() {
    const int Ns = 32, N = Ns * Ns;
    hmc::TangentialFourierPreconditioner M(Ns, 0.3);
    std::vector<std::uint8_t> mask(N, 0);
    for (int i = 0; i < N; ++i) mask[i] = (i % 3 == 0) ? 1 : 0;
    Eigen::VectorXd g = Eigen::VectorXd::Random(2 * N), z;
    M.apply_into(g, mask, true, z);
    double mx = 0.0, my = 0.0;
    int nm = 0;
    for (int i = 0; i < N; ++i) {
        if (!mask[i]) {
            CHECK(z(i) == 0.0 && z(N + i) == 0.0); // strictly zero off-mask
        } else {
            mx += z(i);
            my += z(N + i);
            ++nm;
        }
    }
    CHECK(std::abs(mx / nm) < 1e-12 && std::abs(my / nm) < 1e-12);
    return 0;
}

// Synthetic smooth threshold field: Hertz-like cap, s = mu * p_analytic.
// Returns s (size N) with s = 0 outside the cap (open points).
static Eigen::VectorXd hertz_threshold(int Ns, double L, double a, double mu,
                                       double p0) {
    const double h = L / Ns, xc = 0.5 * L, yc = 0.5 * L;
    Eigen::VectorXd s(Ns * Ns);
    for (int iy = 0; iy < Ns; ++iy)
        for (int ix = 0; ix < Ns; ++ix) {
            const double x = (ix + 0.5) * h - xc, y = (iy + 0.5) * h - yc;
            const double r2 = (x * x + y * y) / (a * a);
            s(iy * Ns + ix) = (r2 < 1.0) ? mu * p0 * std::sqrt(1.0 - r2) : 0.0;
        }
    return s;
}

// Verify the discrete KKT conditions of the QP directly, independent of the
// solver's own error metric:
//   s == 0            -> q == 0
//   interior (stick)  -> |w| small           (w = delta_t - u)
//   bound (slip)      -> |q| == s, w aligned with q̂ (w·q̂ >= 0, |w_perp| small)
// Returns 0 on success. wtol is relative to wref (the residual scale).
static int check_kkt(const hmc::TangentialResult& res, const Eigen::VectorXd& s,
                     const Eigen::VectorXd& u, const Eigen::Vector2d& dt,
                     double wref, double wtol) {
    const int N = static_cast<int>(s.size());
    double worst_stick = 0.0, worst_align = 0.0, worst_neg = 0.0;
    for (int i = 0; i < N; ++i) {
        const double qx = res.q(i), qy = res.q(N + i);
        const double qn = std::hypot(qx, qy);
        const double wx = dt(0) - u(i), wy = dt(1) - u(N + i);
        if (s(i) == 0.0) {
            CHECK(qx == 0.0 && qy == 0.0);
            continue;
        }
        if (qn < s(i) * (1.0 - 1e-9)) { // stick: slip must vanish
            worst_stick = std::max(worst_stick, std::hypot(wx, wy));
        } else { // bound: |q| pinned at s, slip along +q̂
            CHECK(std::abs(qn - s(i)) <= 1e-9 * s(i));
            const double qhx = qx / qn, qhy = qy / qn;
            const double wpar = wx * qhx + wy * qhy;
            const double wperp =
                std::hypot(wx - wpar * qhx, wy - wpar * qhy);
            worst_align = std::max(worst_align, wperp);
            worst_neg = std::max(worst_neg, std::max(0.0, -wpar));
        }
    }
    std::printf("KKT: stick |w| %.3e  slip |w_perp| %.3e  slip (-w·q̂)+ %.3e"
                "  (wref %.3e)\n",
                worst_stick, worst_align, worst_neg, wref);
    CHECK(worst_stick <= wtol * wref);
    CHECK(worst_align <= wtol * wref);
    CHECK(worst_neg <= wtol * wref);
    return 0;
}

static int test_kkt_displacement() {
    const int Ns = 32;
    const double L = 1.0, E = 1.0, nu = 0.3;
    hmc::CerrutiKernel K(Ns, L, E, nu);
    hmc::TangentialFFTOperator C(K);
    C.build();
    hmc::TanMatVecInto Cop = [&C](const Eigen::VectorXd& x,
                                  Eigen::VectorXd& y) { C.matvec_into(x, y); };
    const Eigen::VectorXd s = hertz_threshold(Ns, L, 0.3, 0.4, 1.0);
    // delta_t large enough that an outer annulus slips, small enough that a
    // stick core survives (partial slip; both KKT branches exercised)
    const Eigen::Vector2d dt(3e-2, 1e-2);
    hmc::TangentialResult res = hmc::solve_tangential(
        Cop, s, /*force_control=*/false, dt, 1e-5, 20000);
    std::printf("disp-control: it %d err %.3e stick %d slip %d open %d\n",
                res.iterations, res.error, res.n_stick, res.n_slip,
                res.n_open);
    CHECK(res.converged);
    CHECK(res.n_stick > 0 && res.n_slip > 0); // genuinely partial slip
    Eigen::VectorXd u;
    C.matvec_into(res.q, u);
    if (int rc = check_kkt(res, s, u, dt, dt.norm(), 1e-4)) return rc;
    // reported delta_t echoes the imposed one under displacement control
    CHECK((res.delta_t - dt).norm() == 0.0);
    return 0;
}

static int test_kkt_force() {
    const int Ns = 32, N = Ns * Ns;
    const double L = 1.0, E = 1.0, nu = 0.3;
    hmc::CerrutiKernel K(Ns, L, E, nu);
    hmc::TangentialFFTOperator C(K);
    C.build();
    hmc::TanMatVecInto Cop = [&C](const Eigen::VectorXd& x,
                                  Eigen::VectorXd& y) { C.matvec_into(x, y); };
    const Eigen::VectorXd s = hertz_threshold(Ns, L, 0.3, 0.4, 1.0);
    // impose ~55% of the fully-sliding load in x (partial slip guaranteed)
    const double Sbar = s.sum() / N;
    const Eigen::Vector2d qbar(0.55 * Sbar, 0.1 * Sbar);
    hmc::TangentialResult res = hmc::solve_tangential(
        Cop, s, /*force_control=*/true, qbar, 1e-5, 20000);
    std::printf("force-control: it %d err %.3e stick %d slip %d\n",
                res.iterations, res.error, res.n_stick, res.n_slip);
    CHECK(res.converged);
    CHECK(res.n_stick > 0 && res.n_slip > 0);
    // load met to solver tolerance
    const double load_err = (res.q_mean - qbar).norm() / qbar.norm();
    std::printf("force-control load constraint (rel): %.3e\n", load_err);
    CHECK(load_err <= 1e-8);
    Eigen::VectorXd u;
    C.matvec_into(res.q, u);
    // High-slip vector problems (40% slip here) have a shallower inner
    // metric floor than the displacement test's 1e-4 gate: slip DIRECTIONS
    // keep rotating and every clamp perturbs the CG geometry. Measured floor
    // on this problem: stick 1.2e-3 / align 4.9e-3 relative. Gate at 1e-2;
    // the sharp force-control physics gates live in the Cattaneo-Mindlin and
    // Ciavarella-Jager tests (nu=0: directions do not rotate, floor deep).
    if (int rc = check_kkt(res, s, u, res.delta_t, res.delta_t.norm(), 1e-2))
        return rc;
    return 0;
}

// Preconditioner A/B: identical solution, fewer (or equal) iterations.
static int test_precond_ab() {
    const int Ns = 64;
    const double L = 1.0, E = 1.0, nu = 0.3;
    hmc::CerrutiKernel K(Ns, L, E, nu);
    hmc::TangentialFFTOperator C(K);
    C.build();
    hmc::TanMatVecInto Cop = [&C](const Eigen::VectorXd& x,
                                  Eigen::VectorXd& y) { C.matvec_into(x, y); };
    const Eigen::VectorXd s = hertz_threshold(Ns, L, 0.3, 0.4, 1.0);
    const Eigen::Vector2d dt(3e-2, 1e-2);
    hmc::TangentialResult r0 =
        hmc::solve_tangential(Cop, s, false, dt, 1e-5, 20000);
    hmc::TangentialFourierPreconditioner M(Ns, nu);
    hmc::TanPrecondInto Mop =
        [&M](const Eigen::VectorXd& g, const std::vector<std::uint8_t>& mask,
             bool rm, Eigen::VectorXd& z) { M.apply_into(g, mask, rm, z); };
    hmc::TangentialResult r1 =
        hmc::solve_tangential(Cop, s, false, dt, 1e-5, 20000, true, Mop);
    const double dq = (r1.q - r0.q).norm() / r0.q.norm();
    std::printf("precond A/B: it %d -> %d, dq %.3e\n", r0.iterations,
                r1.iterations, dq);
    CHECK(r0.converged && r1.converged);
    // Both runs stop at their own metric floor (~1e-5), so the iterates
    // agree to floor level, not to machine precision (measured dq ~8e-4).
    CHECK(dq < 5e-3);
    CHECK(r1.iterations <= r0.iterations);  // never worse on this problem
    // both must be valid KKT points of the SAME QP (the real equivalence)
    Eigen::VectorXd u0, u1;
    C.matvec_into(r0.q, u0);
    C.matvec_into(r1.q, u1);
    // Ns=64 metric floor is shallower than Ns=32's (measured align ratio
    // ~1.1e-3 unpreconditioned, with run-to-run OpenMP jitter): gate 3e-3.
    if (int rc = check_kkt(r0, s, u0, dt, dt.norm(), 3e-3)) return rc;
    if (int rc = check_kkt(r1, s, u1, dt, dt.norm(), 3e-3)) return rc;
    return 0;
}

// Full stick: rigid circular region (radius a) dragged by delta_x on an
// otherwise traction-free surface = tangential flat punch. Mindlin:
//   Q_x = 8 G a delta_x / (2 - nu),  G = E*(1-nu)/2.
// Discretization error is O(h) at the singular edge (normal-path Hertz
// analog measured 1.6% at comparable a/h) — gate at 4%, record measured.
static int test_full_stick_stiffness() {
    const int Ns = 128, N = Ns * Ns;
    const double L = 1.0, E = 1.0, nu = 0.3, a = 0.25, h = L / Ns;
    hmc::CerrutiKernel K(Ns, L, E, nu);
    hmc::TangentialFFTOperator C(K);
    C.build();
    hmc::TanMatVecInto Cop = [&C](const Eigen::VectorXd& x,
                                  Eigen::VectorXd& y) { C.matvec_into(x, y); };
    Eigen::VectorXd s(N);
    for (int iy = 0; iy < Ns; ++iy)
        for (int ix = 0; ix < Ns; ++ix) {
            const double x = (ix + 0.5) * h - 0.5 * L,
                         y = (iy + 0.5) * h - 0.5 * L;
            s(iy * Ns + ix) = (x * x + y * y < a * a) ? 1e12 : 0.0;
        }
    const Eigen::Vector2d dt(1e-3, 0.0);
    hmc::TangentialResult res =
        hmc::solve_tangential(Cop, s, false, dt, 1e-5, 20000);
    CHECK(res.converged);
    CHECK(res.n_slip == 0); // threshold huge: nothing slips
    const double G = 0.5 * E * (1.0 - nu);
    const double Qx_num = res.q_mean(0) * L * L; // mean(q)·area = total force
    const double Qx_ana = 8.0 * G * a * dt(0) / (2.0 - nu);
    const double ratio = Qx_num / Qx_ana;
    std::printf("full-stick stiffness: Qx num %.6e ana %.6e ratio %.4f\n",
                Qx_num, Qx_ana, ratio);
    CHECK(std::abs(ratio - 1.0) < 0.04);
    // y-force vanishes by symmetry (xy coupling is odd; gate loose for
    // roundoff accumulation)
    CHECK(std::abs(res.q_mean(1)) < 1e-8 * std::abs(res.q_mean(0)));
    return 0;
}

// Cattaneo–Mindlin partial slip at nu = 0 (where the tangential kernel is
// isotropic and proportional to the normal one, so the classical solution
// is exact for the continuum problem): imposed Hertz threshold s = mu*p,
// force control Q_x = (1 - (c/a)^3) mu P with c/a = 0.5^(1/3) for Q/(mu P)
// = 0.5. Gates: stick radius within 1.5 cells; q_x profile rel-L2 < 4%
// (continuum-vs-grid discretization); q_y == 0 exactly at nu = 0 with a
// y-symmetric problem (gate at roundoff).
static int test_cattaneo_mindlin() {
    const int Ns = 128, N = Ns * Ns;
    const double L = 1.0, E = 1.0, nu = 0.0, a = 0.2, mu = 0.5, p0 = 1.0;
    const double h = L / Ns;
    hmc::CerrutiKernel K(Ns, L, E, nu);
    hmc::TangentialFFTOperator C(K);
    C.build();
    hmc::TanMatVecInto Cop = [&C](const Eigen::VectorXd& x,
                                  Eigen::VectorXd& y) { C.matvec_into(x, y); };
    const Eigen::VectorXd s = hertz_threshold(Ns, L, a, mu, p0);
    // mean of s = mu*p over the grid; total sliding load muP = muP_mean·N·h²,
    // so imposing q̄_x = 0.5·muP_mean means Q_x = 0.5·muP.
    const double muP_mean = s.sum() / N;
    const double QoverMuP = 0.5;
    const Eigen::Vector2d qbar(QoverMuP * muP_mean, 0.0);
    hmc::TangentialResult res =
        hmc::solve_tangential(Cop, s, true, qbar, 1e-5, 40000);
    CHECK(res.converged);

    const double ca = std::cbrt(1.0 - QoverMuP); // stick radius ratio
    // stick radius from the state map: largest slip-point radius inside c,
    // smallest stick-point radius outside — bracket must straddle c*a
    double r_stick_max = 0.0;
    for (int iy = 0; iy < Ns; ++iy)
        for (int ix = 0; ix < Ns; ++ix) {
            const int i = iy * Ns + ix;
            if (res.state[i] != 1) continue;
            const double x = (ix + 0.5) * h - 0.5 * L,
                         y = (iy + 0.5) * h - 0.5 * L;
            r_stick_max = std::max(r_stick_max, std::hypot(x, y));
        }
    std::printf("cattaneo-mindlin: c/a num %.4f ana %.4f (1.5-cell tol %.4f)\n",
                r_stick_max / a, ca, 1.5 * h / a);
    CHECK(std::abs(r_stick_max - ca * a) < 1.5 * h);

    // traction profile: q_x(r) = mu p0 [sqrt(1-r²/a²) − (c/a) sqrt(1-r²/c²)]
    double num2 = 0.0, den2 = 0.0, qymax = 0.0, qxmax = 0.0;
    for (int iy = 0; iy < Ns; ++iy)
        for (int ix = 0; ix < Ns; ++ix) {
            const int i = iy * Ns + ix;
            const double x = (ix + 0.5) * h - 0.5 * L,
                         y = (iy + 0.5) * h - 0.5 * L;
            const double rr = std::hypot(x, y);
            double qa = 0.0;
            if (rr < a) qa = mu * p0 * std::sqrt(1.0 - rr * rr / (a * a));
            if (rr < ca * a)
                qa -= mu * p0 * ca *
                      std::sqrt(1.0 - rr * rr / (ca * ca * a * a));
            num2 += (res.q(i) - qa) * (res.q(i) - qa);
            den2 += qa * qa;
            qxmax = std::max(qxmax, std::abs(res.q(i)));
            qymax = std::max(qymax, std::abs(res.q(N + i)));
        }
    const double relL2 = std::sqrt(num2 / den2);
    std::printf("cattaneo-mindlin: qx rel-L2 %.3e  qy/qx %.3e\n", relL2,
                qymax / qxmax);
    CHECK(relL2 < 0.04);
    CHECK(qymax <= 1e-10 * qxmax); // exact decoupling at nu = 0
    return 0;
}

// Deterministic smooth "rough" surface: a few incommensurate cosines.
static Eigen::VectorXd wavy_gap(int Ns, double L) {
    Eigen::VectorXd g0(Ns * Ns);
    const double h = L / Ns;
    const int mj[6][2] = {{1, 2}, {3, 1}, {2, 4}, {5, 3}, {4, 6}, {7, 2}};
    const double Aj[6] = {0.8, 0.45, 0.3, 0.18, 0.12, 0.08};
    const double ph[6] = {0.3, 1.1, 2.0, 4.2, 5.1, 0.7};
    for (int iy = 0; iy < Ns; ++iy)
        for (int ix = 0; ix < Ns; ++ix) {
            const double x = (ix + 0.5) * h / L, y = (iy + 0.5) * h / L;
            double z = 0.0;
            for (int j = 0; j < 6; ++j)
                z += Aj[j] * std::cos(2.0 * M_PI * (mj[j][0] * x +
                                                    mj[j][1] * y) + ph[j]);
            g0(iy * Ns + ix) = 0.02 * z; // heights ~2% of L
        }
    g0.array() -= g0.minCoeff(); // gap >= 0 somewhere-in-contact convention
    return g0;
}

static int test_ciavarella_jager() {
    const int Ns = 64, N = Ns * Ns;
    const double L = 1.0, E = 1.0, nu = 0.0, mu = 0.4, p_bar = 0.01;
    hmc::BoussinesqKernel BK(Ns, L, E);
    hmc::FFTOperator S(BK);
    S.build();
    hmc::MatVec Sop = [&S](const Eigen::VectorXd& x) { return S.matvec(x); };
    const Eigen::VectorXd g0 = wavy_gap(Ns, L);

    hmc::ContactResult full = hmc::solve_contact(Sop, g0, p_bar, 1e-12, 20000);
    CHECK(full.converged);
    CHECK(full.contact_fraction > 0.05 && full.contact_fraction < 0.95);

    const double qx_bar = 0.5 * mu * p_bar; // Q = 0.5 muP
    hmc::ContactResult red =
        hmc::solve_contact(Sop, g0, p_bar - qx_bar / mu, 1e-12, 20000);
    CHECK(red.converged);

    // tangential solve: threshold mu*p, imposed mean traction qx_bar
    hmc::CerrutiKernel K(Ns, L, E, nu);
    hmc::TangentialFFTOperator C(K);
    C.build();
    hmc::TanMatVecInto Cop = [&C](const Eigen::VectorXd& x,
                                  Eigen::VectorXd& y) { C.matvec_into(x, y); };
    const Eigen::VectorXd s = mu * full.pressure;
    hmc::TangentialResult res = hmc::solve_tangential(
        Cop, s, true, Eigen::Vector2d(qx_bar, 0.0), 1e-6, 40000);
    CHECK(res.converged);

    // discrete-exact superposition: q_x = mu (p - p*), q_y = 0
    const Eigen::VectorXd q_ref = mu * (full.pressure - red.pressure);
    const double rel =
        (res.q.head(N) - q_ref).norm() / q_ref.norm();
    const double qy_rel = res.q.tail(N).norm() / q_ref.norm();
    // stick fraction (within contact) matches the reduced contact area
    const int n_contact = static_cast<int>(
        (full.pressure.array() > 0.0).count());
    const int n_red = static_cast<int>((red.pressure.array() > 0.0).count());
    std::printf("ciavarella-jager: qx rel %.3e  qy rel %.3e  stick %d vs "
                "reduced-contact %d\n",
                rel, qy_rel, res.n_stick, n_red);
    CHECK(rel < 1e-4);
    CHECK(qy_rel < 1e-8);
    CHECK(std::abs(res.n_stick - n_red) <= std::max(4, n_contact / 50));
    return 0;
}

int main() {
    if (int rc = test_precond_symbol()) return rc;
    if (int rc = test_precond_mask_mean()) return rc;
    if (int rc = test_kkt_displacement()) return rc;
    if (int rc = test_kkt_force()) return rc;
    if (int rc = test_precond_ab()) return rc;
    if (int rc = test_full_stick_stiffness()) return rc;
    if (int rc = test_cattaneo_mindlin()) return rc;
    if (int rc = test_ciavarella_jager()) return rc;
    std::printf("test_friction: all checks passed\n");
    return 0;
}
