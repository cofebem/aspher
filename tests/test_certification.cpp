// Certification, status and activation regressions (spec A01/A02, tests
// T01-T05, T31 of doc/plans/2026-09-08-accuracy-efficiency-validation.md).
//
// Every acceptance quantity is recomputed by tests/contact_oracle.hpp, which
// shares no code with the solver.

#include "boussinesq_kernel.hpp"
#include "contact_oracle.hpp"
#include "contact_solver.hpp"
#include "fourier_precond.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <random>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAILED: %s (line %d)\n", #cond, __LINE__);        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

namespace {

hmc::MatVec dense_op(const Eigen::MatrixXd& S) {
    return [&S](const Eigen::VectorXd& v) -> Eigen::VectorXd { return S * v; };
}

double g_range(const Eigen::VectorXd& g0) {
    const double r = g0.maxCoeff() - g0.minCoeff();
    return r > 0.0 ? r : 1.0;
}

// Compare the solver's reported diagnostics with the independent ones.
bool diagnostics_agree(const hmc::ContactResult& r, const oracle::Diagnostics& d,
                       double atol) {
    const double s = std::max(1.0, std::abs(d.fw_error));
    return std::abs(r.fw_error - d.fw_error) <= atol * s &&
           std::abs(r.penetration_error - d.penetration) <= atol &&
           std::abs(r.load_error - d.load_error) <= atol &&
           std::abs(r.approach - d.approach) <=
               atol * std::max(1.0, std::abs(d.approach)) &&
           std::abs(r.pk_error - d.pk_error) <= atol * std::max(1.0, d.pk_error);
}

// Parabolic (Hertz-like) gap on an Ns x Ns unit grid.
Eigen::VectorXd parabolic_gap(int Ns) {
    Eigen::VectorXd g0(Ns * Ns);
    for (int iy = 0; iy < Ns; ++iy)
        for (int ix = 0; ix < Ns; ++ix) {
            const double x = (ix + 0.5) / Ns - 0.5;
            const double y = (iy + 0.5) / Ns - 0.5;
            g0(iy * Ns + ix) = x * x + y * y;
        }
    return g0;
}

Eigen::VectorXd wavy_gap(int Ns) {
    Eigen::VectorXd g(Ns * Ns);
    for (int iy = 0; iy < Ns; ++iy)
        for (int ix = 0; ix < Ns; ++ix) {
            const double x = (ix + 0.5) / Ns, y = (iy + 0.5) / Ns;
            g(iy * Ns + ix) =
                0.02 * (std::cos(2 * M_PI * (x + 2 * y) + 0.3) +
                        0.45 * std::cos(2 * M_PI * (3 * x + y) + 1.1) +
                        0.3 * std::cos(2 * M_PI * (2 * x + 4 * y) + 2.0));
        }
    return g;
}

} // namespace

// ── T01: certificate and known solution ────────────────────────────────────
int t01_manufactured() {
    const int n = 10;
    const Eigen::MatrixXd S = oracle::spd_kernel(n);

    struct Case {
        const char* name;
        Eigen::VectorXd p_star, g_star;
    };
    std::vector<Case> cases;
    {   // singleton support
        Eigen::VectorXd p = Eigen::VectorXd::Zero(n), g(n);
        p(4) = 1.0;
        for (int i = 0; i < n; ++i) g(i) = (i == 4) ? 0.0 : 0.5 + 0.1 * i;
        cases.push_back({"singleton", p, g});
    }
    {   // full contact
        Eigen::VectorXd p(n), g = Eigen::VectorXd::Zero(n);
        for (int i = 0; i < n; ++i) p(i) = 0.3 + 0.05 * i;
        cases.push_back({"full", p, g});
    }
    {   // two disconnected islands, with one degenerate p*=g*=0 point
        Eigen::VectorXd p = Eigen::VectorXd::Zero(n), g = Eigen::VectorXd::Zero(n);
        p(0) = 0.7; p(1) = 0.4; p(7) = 0.9; p(8) = 0.2;
        for (int i = 0; i < n; ++i)
            if (p(i) == 0.0) g(i) = (i == 5) ? 0.0 : 0.3 + 0.2 * i; // i=5 degenerate
        cases.push_back({"islands+degenerate", p, g});
    }

    for (const auto& c : cases) {
        const auto f = oracle::manufacture(S, c.p_star, c.g_star, 0.25);
        const double p_bar = f.P_total / n;
        const double gref = g_range(f.g0);

        hmc::SolveOptions opt;
        opt.tol = 1e-12;
        opt.max_iter = 20000;
        opt.scales.g_ref = gref;

        // the KKT construction proves p_star is optimal; the enumeration
        // oracle must independently find the same objective
        const auto en = oracle::enumerate_qp(S, f.g0, f.P_total);
        CHECK(en.found);
        const double obj_star = 0.5 * c.p_star.dot(S * c.p_star) + c.p_star.dot(f.g0);
        CHECK(std::abs(en.objective - obj_star) < 1e-9 * (1.0 + std::abs(obj_star)));

        for (int warm = 0; warm < 3; ++warm) {
            Eigen::VectorXd p0;
            if (warm == 1) p0 = c.p_star;                       // valid warm start
            if (warm == 2) {                                    // invalid warm start
                p0 = Eigen::VectorXd::Zero(n);
                p0(n - 1) = f.P_total;
            }
            const auto r = hmc::solve_contact_opt(dense_op(S), f.g0, p_bar, opt, {},
                                                  (warm == 0) ? nullptr : &p0);
            const auto d = oracle::diagnose(S, f.g0, r.pressure, f.P_total, gref);
            const double relp =
                (r.pressure - c.p_star).norm() / c.p_star.norm();
            std::printf("T01 %-20s warm=%d status=%-14s it=%3d fw=%.2e pen=%.2e "
                        "relp=%.2e\n",
                        c.name, warm, hmc::to_string(r.status), r.iterations,
                        r.fw_error, r.penetration_error, relp);
            CHECK(r.converged);
            CHECK(r.status == hmc::SolveStatus::converged);
            CHECK(d.min_pressure >= 0.0);
            CHECK(d.fw_error <= 1e-12);
            CHECK(d.penetration <= 1e-12);
            CHECK(d.load_error <= 1e-12);
            CHECK(diagnostics_agree(r, d, 1e-9));
            CHECK(relp < 1e-7);
            // an already valid warm start finishes immediately; an invalid one
            // must not
            if (warm == 1) CHECK(r.iterations == 0);
            if (warm == 2) CHECK(r.iterations > 0);
        }
    }
    return 0;
}

// ── T02: singleton warm-start counterexample (review §2) ───────────────────
int t02_singleton_warm_start() {
    const int Ns = 8, N = Ns * Ns;
    hmc::BoussinesqKernel K(Ns, 1.0, 1.0);
    const Eigen::MatrixXd S = K.assemble_dense();
    const Eigen::VectorXd g0 = parabolic_gap(Ns);
    const double p_bar = 0.01, P = p_bar * N, gref = g_range(g0);

    hmc::FourierPreconditioner fp(Ns);
    hmc::Precond pc = [&fp](const Eigen::VectorXd& g,
                            const std::vector<std::uint8_t>& m) {
        return fp.apply(g, m);
    };

    hmc::SolveOptions opt;
    opt.tol = 1e-10;
    opt.max_iter = 20000;
    opt.scales.g_ref = gref;

    const auto cold = hmc::solve_contact_opt(dense_op(S), g0, p_bar, opt);
    CHECK(cold.converged);

    // the historical "successful" answer: all load on one corner. The
    // independent checker must reject it.
    Eigen::VectorXd bad = Eigen::VectorXd::Zero(N);
    bad(0) = P;
    const auto dbad = oracle::diagnose(S, g0, bad, P, gref);
    std::printf("T02 historical iterate: fw=%.3e pen=%.3e (must be rejected)\n",
                dbad.fw_error, dbad.penetration);
    CHECK(dbad.fw_error > 1e-3);
    CHECK(dbad.penetration > 1e-3);

    for (int variant = 0; variant < 3; ++variant) {
        Eigen::VectorXd p0 = Eigen::VectorXd::Zero(N);
        if (variant == 0) p0(0) = P;                       // corner
        if (variant == 1) p0(Ns / 2 * Ns + Ns / 2) = P;    // centre
        if (variant == 2) p0(Ns * Ns - 1) = P;             // opposite corner
        for (int usepc = 0; usepc < 2; ++usepc) {
            Eigen::VectorXd start = p0;
            const auto r = hmc::solve_contact_opt(dense_op(S), g0, p_bar, opt,
                                                  usepc ? pc : hmc::Precond{},
                                                  &start);
            const auto d = oracle::diagnose(S, g0, r.pressure, P, gref);
            const double relp =
                (r.pressure - cold.pressure).norm() / cold.pressure.norm();
            std::printf("T02 variant=%d precond=%d status=%-10s it=%3d fw=%.2e "
                        "pen=%.2e relp=%.2e id=%d\n",
                        variant, usepc, hmc::to_string(r.status), r.iterations,
                        d.fw_error, d.penetration, relp, r.identification_steps);
            CHECK(r.converged);
            CHECK(r.iterations > 0);
            CHECK(d.fw_error <= 1e-10);
            CHECK(d.penetration <= 1e-10);
            CHECK(d.min_pressure >= 0.0);
            CHECK(relp < 1e-6);
        }
    }
    return 0;
}

// ── T03: termination and result consistency ────────────────────────────────
int t03_termination() {
    const int Ns = 16, N = Ns * Ns;
    hmc::BoussinesqKernel K(Ns, 1.0, 1.0);
    const Eigen::MatrixXd S = K.assemble_dense();
    const Eigen::VectorXd g0 = parabolic_gap(Ns);
    const double p_bar = 0.02, P = p_bar * N, gref = g_range(g0);

    // max_iter = 0 and 1: honest non-convergence, diagnostics on the returned
    // iterate
    for (int mi : {0, 1}) {
        hmc::SolveOptions opt;
        opt.tol = 1e-10;
        opt.max_iter = mi;
        opt.scales.g_ref = gref;
        const auto r = hmc::solve_contact_opt(dense_op(S), g0, p_bar, opt);
        const auto d = oracle::diagnose(S, g0, r.pressure, P, gref);
        std::printf("T03 max_iter=%d status=%s fw=%.3e\n", mi,
                    hmc::to_string(r.status), r.fw_error);
        CHECK(!r.converged);
        CHECK(r.status == hmc::SolveStatus::max_iterations);
        CHECK(diagnostics_agree(r, d, 1e-9));
    }

    // unattainable tolerance: must report failure, never success
    {
        hmc::SolveOptions opt;
        opt.tol = 1e-25;
        opt.max_iter = 400;
        opt.stall_limit = 20;
        opt.scales.g_ref = gref;
        const auto r = hmc::solve_contact_opt(dense_op(S), g0, p_bar, opt);
        std::printf("T03 tol=1e-25 status=%s reason=%s it=%d fw=%.3e "
                    "returned_best=%d\n",
                    hmc::to_string(r.status), r.status_reason.c_str(),
                    r.iterations, r.fw_error, int(r.returned_best));
        const auto d = oracle::diagnose(S, g0, r.pressure, P, gref);
        // The plan warns against a flaky "must fail at 1e-25" expectation: on
        // a small grid the certificate can legitimately underflow to exactly
        // zero, which satisfies any tolerance. Accept success only when the
        // INDEPENDENT check confirms an exactly zero certificate; otherwise
        // require an honest failure status.
        if (r.converged) {
            CHECK(d.fw_gap == 0.0);
            CHECK(d.min_gap >= 0.0);
        } else {
            CHECK(r.status == hmc::SolveStatus::stagnated ||
                  r.status == hmc::SolveStatus::max_iterations);
        }
        CHECK(diagnostics_agree(r, d, 1e-8));
    }

    // non-finite operator output -> `nonfinite`
    {
        int calls = 0;
        hmc::MatVec bad = [&](const Eigen::VectorXd& v) -> Eigen::VectorXd {
            Eigen::VectorXd y = S * v;
            if (++calls > 3) y(0) = std::numeric_limits<double>::quiet_NaN();
            return y;
        };
        hmc::SolveOptions opt;
        opt.tol = 1e-10;
        opt.max_iter = 100;
        opt.scales.g_ref = gref;
        const auto r = hmc::solve_contact_opt(bad, g0, p_bar, opt);
        std::printf("T03 nonfinite operator: status=%s\n",
                    hmc::to_string(r.status));
        CHECK(!r.converged);
        CHECK(r.status == hmc::SolveStatus::nonfinite);
    }

    // non-finite input is rejected before the operator is ever called
    {
        Eigen::VectorXd g_bad = g0;
        g_bad(3) = std::numeric_limits<double>::infinity();
        int calls = 0;
        hmc::MatVec counting = [&](const Eigen::VectorXd& v) -> Eigen::VectorXd {
            ++calls;
            return S * v;
        };
        bool threw = false;
        try {
            hmc::solve_contact(counting, g_bad, p_bar, 1e-10, 10);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        std::printf("T03 nonfinite input: threw=%d operator_calls=%d\n",
                    int(threw), calls);
        CHECK(threw);
        CHECK(calls == 0);
    }

    // invalid parameters raise, they are not numerical termination states
    {
        bool a = false, b = false, c = false;
        try { hmc::solve_contact(dense_op(S), g0, -1.0, 1e-10, 10); }
        catch (const std::invalid_argument&) { a = true; }
        try { hmc::solve_contact(dense_op(S), g0, p_bar, 0.0, 10); }
        catch (const std::invalid_argument&) { b = true; }
        try { hmc::solve_contact(dense_op(S), g0, p_bar, 1e-10, -3); }
        catch (const std::invalid_argument&) { c = true; }
        CHECK(a && b && c);
    }
    return 0;
}

// ── T04: preconditioner scale invariance ───────────────────────────────────
int t04_precond_scale() {
    const int Ns = 32, N = Ns * Ns;
    hmc::BoussinesqKernel K(Ns, 1.0, 1.0);
    const Eigen::MatrixXd S = K.assemble_dense();
    const Eigen::VectorXd g0 = wavy_gap(Ns);
    const double p_bar = 0.005, P = p_bar * N, gref = g_range(g0);

    hmc::SolveOptions opt;
    opt.tol = 1e-10;
    opt.max_iter = 5000;
    opt.scales.g_ref = gref;

    hmc::FourierPreconditioner fp(Ns);
    Eigen::VectorXd reference;
    for (double c : {0.0, 1e-6, 1e-3, 1.0, 1e3, 1e6}) {
        hmc::Precond pc;
        if (c > 0.0)
            pc = [&fp, c](const Eigen::VectorXd& g,
                          const std::vector<std::uint8_t>& m) -> Eigen::VectorXd {
                return c * fp.apply(g, m);
            };
        const auto r = hmc::solve_contact_opt(dense_op(S), g0, p_bar, opt, pc);
        const auto d = oracle::diagnose(S, g0, r.pressure, P, gref);
        if (c == 1.0) reference = r.pressure;
        std::printf("T04 c=%-8.0e status=%-10s it=%3d fw=%.2e pen=%.2e\n", c,
                    hmc::to_string(r.status), r.iterations, d.fw_error,
                    d.penetration);
        CHECK(r.converged);
        CHECK(d.fw_error <= 1e-10);
        CHECK(d.penetration <= 1e-10);
        CHECK(d.min_pressure >= 0.0);
    }
    // every scaled solve must agree with c=1 to solver accuracy
    for (double c : {1e-6, 1e-3, 1e3, 1e6}) {
        hmc::Precond pc = [&fp, c](const Eigen::VectorXd& g,
                                   const std::vector<std::uint8_t>& m)
            -> Eigen::VectorXd { return c * fp.apply(g, m); };
        const auto r = hmc::solve_contact_opt(dense_op(S), g0, p_bar, opt, pc);
        const double rel = (r.pressure - reference).norm() / reference.norm();
        std::printf("T04 c=%-8.0e pressure rel vs c=1: %.3e\n", c, rel);
        CHECK(rel < 1e-6);
    }

    // On a fixed support the feasible step must be invariant under M -> cM.
    // Check the zero-sum property of the CG direction indirectly: the solution
    // and its load are identical, and the load residual stays at roundoff.
    return 0;
}

// ── T31: objective and error-bound mathematics ─────────────────────────────
int t31_bounds() {
    const int n = 8;
    const Eigen::MatrixXd S = oracle::spd_kernel(n, 0.7);
    Eigen::VectorXd p_star = Eigen::VectorXd::Zero(n), g_star(n);
    p_star(1) = 0.5; p_star(2) = 1.5; p_star(6) = 0.75;
    for (int i = 0; i < n; ++i) g_star(i) = (p_star(i) > 0) ? 0.0 : 0.2 + 0.05 * i;
    const auto f = oracle::manufacture(S, p_star, g_star, 0.1);
    const double P = f.P_total, gref = g_range(f.g0);
    const double f_star = 0.5 * p_star.dot(S * p_star) + p_star.dot(f.g0);

    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
    const double lam_min = es.eigenvalues()(0);
    CHECK(lam_min > 0.0);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double worst = 0.0;
    for (int trial = 0; trial < 400; ++trial) {
        Eigen::VectorXd y(n);
        for (int i = 0; i < n; ++i) y(i) = U(rng);
        const Eigen::VectorXd p = oracle::sorting_projection(y, P);
        const auto d = oracle::diagnose(S, f.g0, p, P, gref);
        const double gap = d.objective - f_star;
        CHECK(gap >= -1e-12);                       // p* is optimal
        CHECK(gap <= d.fw_gap + 1e-12);             // 0 <= f(p)-f(p*) <= G
        const Eigen::VectorXd e = p - p_star;
        const double eS = e.dot(S * e);
        CHECK(eS <= 2.0 * d.fw_gap + 1e-12);        // ||p-p*||_S^2 <= 2G
        worst = std::max(worst, gap / std::max(d.fw_gap, 1e-300));
    }
    // G = 0 exactly at the manufactured optimum
    const auto dopt = oracle::diagnose(S, f.g0, p_star, P, gref);
    std::printf("T31 worst gap/G = %.3f, G(p*) = %.3e\n", worst, dopt.fw_gap);
    CHECK(std::abs(dopt.fw_gap) < 1e-12);

    // operator perturbation: G_S(p) <= G_{S+E}(p) + 2 P ||E p||_inf
    Eigen::MatrixXd E = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j) {
            const double e = 1e-3 * std::cos(3.0 * i + 1.7 * j);
            E(i, j) = e;
            E(j, i) = e;
        }
    const Eigen::MatrixXd St = S + E;
    for (int trial = 0; trial < 100; ++trial) {
        Eigen::VectorXd y(n);
        for (int i = 0; i < n; ++i) y(i) = U(rng);
        const Eigen::VectorXd p = oracle::sorting_projection(y, P);
        const double GS = oracle::diagnose(S, f.g0, p, P, gref).fw_gap;
        const double Gt = oracle::diagnose(St, f.g0, p, P, gref).fw_gap;
        const double eps = (E * p).cwiseAbs().maxCoeff();
        CHECK(GS <= Gt + 2.0 * P * eps + 1e-12);
    }

    // A non-feasible pressure gets no unqualified bound: the certificate
    // identity itself relies on 1'p = P.
    Eigen::VectorXd pinf = p_star;
    pinf *= 0.5; // load-infeasible
    const auto dinf = oracle::diagnose(S, f.g0, pinf, P, gref);
    const double gap_inf = dinf.objective - f_star;
    std::printf("T31 infeasible iterate: load_error=%.3e, f-f* = %.3e, G=%.3e\n",
                dinf.load_error, gap_inf, dinf.fw_gap);
    CHECK(dinf.load_error > 1e-3);
    return 0;
}

// ── T21 (partial): production projector vs the sorting oracle ──────────────
int t21_projector() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (int trial = 0; trial < 200; ++trial) {
        const int n = 1 + (trial % 40);
        const double scale = std::pow(10.0, (trial % 13) - 6);
        Eigen::VectorXd y(n);
        for (int i = 0; i < n; ++i) y(i) = scale * U(rng);
        if (trial % 5 == 0) y.setConstant(scale); // repeated values
        const double P = 0.37 * scale * n;
        Eigen::VectorXd prod = y;
        hmc::project_load_simplex<double>(prod, P);
        const Eigen::VectorXd ref = oracle::sorting_projection(y, P);
        const double rel = (prod - ref).norm() / std::max(ref.norm(), 1e-300);
        CHECK(prod.minCoeff() >= 0.0);
        CHECK(std::abs(prod.sum() - P) <= 1e-12 * P * std::max(1, n));
        CHECK(rel < 1e-12);
        // idempotence
        Eigen::VectorXd again = prod;
        hmc::project_load_simplex<double>(again, P);
        CHECK((again - prod).norm() <= 1e-12 * std::max(prod.norm(), 1e-300));
    }
    std::printf("T21 projector matches the sorting oracle on 200 cases\n");
    return 0;
}

int main() {
    if (t01_manufactured()) return 1;
    if (t02_singleton_warm_start()) return 1;
    if (t03_termination()) return 1;
    if (t04_precond_scale()) return 1;
    if (t31_bounds()) return 1;
    if (t21_projector()) return 1;
    std::printf("test_certification: all checks passed\n");
    return 0;
}
