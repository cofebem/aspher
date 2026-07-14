#include "fourier_precond.hpp"
#include "friction_solve.hpp"
#include "tangential_operator.hpp"
#include "cerruti_kernel.hpp"

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

int main() {
    if (int rc = test_precond_symbol()) return rc;
    if (int rc = test_precond_mask_mean()) return rc;
    if (int rc = test_kkt_displacement()) return rc;
    std::printf("test_friction: all checks passed\n");
    return 0;
}
