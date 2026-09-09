// Gap datum, units and reduction regressions (spec A04/A09, tests T07-T09).

#include "boussinesq_kernel.hpp"
#include "contact_oracle.hpp"
#include "contact_solver.hpp"

#include <cmath>
#include <cstdio>
#include <random>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAILED: %s (line %d)\n", #cond, __LINE__);        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

namespace {

Eigen::VectorXd rough_gap(int Ns, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    Eigen::VectorXd h = Eigen::VectorXd::Zero(Ns * Ns);
    // band-limited: a few smooth modes, deterministic per seed
    for (int m = 1; m <= 6; ++m)
        for (int n = 1; n <= 6; ++n) {
            const double a = nd(rng) / (m * m + n * n);
            const double ph = nd(rng);
            for (int iy = 0; iy < Ns; ++iy)
                for (int ix = 0; ix < Ns; ++ix) {
                    const double x = (ix + 0.5) / Ns, y = (iy + 0.5) / Ns;
                    h(iy * Ns + ix) +=
                        a * std::cos(2 * M_PI * (m * x + n * y) + ph);
                }
        }
    h.array() -= h.minCoeff();
    return h;
}

} // namespace

// ── T07: gap datum invariance ───────────────────────────────────────────────
int t07_datum() {
    const int Ns = 32, N = Ns * Ns;
    hmc::BoussinesqKernel K(Ns, 1.0, 1.0);
    const Eigen::MatrixXd S = K.assemble_dense();
    const Eigen::VectorXd g0 = rough_gap(Ns, 11) * 0.02;
    const double p_bar = 0.01, P = p_bar * N;
    const double gref = g0.maxCoeff() - g0.minCoeff();

    hmc::MatVecIntoT<double> op = [&S](const Eigen::VectorXd& x,
                                       Eigen::VectorXd& y) { y = S * x; };
    hmc::MatVecIntoT<float> opf = [&S](const Eigen::VectorXf& x,
                                       Eigen::VectorXf& y) {
        y = (S * x.cast<double>()).cast<float>();
    };

    hmc::SolveOptions base;
    base.max_iter = 20000;
    base.scales.g_ref = gref;

    // double reference at the natural datum
    hmc::SolveOptions dopt = base;
    dopt.tol = 1e-12;
    Eigen::VectorXd gcopy = g0;
    const auto ref =
        hmc::solve_contact_impl<double>(op, gcopy, p_bar, dopt, {}, nullptr);
    CHECK(ref.converged);

    for (double c : {0.0, 1e3, -1e3, 1e6, -1e6}) {
        Eigen::VectorXd gs = g0.array() + c;
        // double: the solver removes the datum on the fly
        hmc::SolveOptions o = dopt;
        const auto rd =
            hmc::solve_contact_impl<double>(op, gs, p_bar, o, {}, nullptr);
        const double relp = (rd.pressure - ref.pressure).norm() /
                            ref.pressure.norm();
        const double dapp = rd.approach - ref.approach - c;
        const double dobj = rd.objective - ref.objective - c * P;
        const double dgap = (rd.gap - ref.gap).cwiseAbs().maxCoeff();
        const double ddisp =
            (rd.displacement - ref.displacement).cwiseAbs().maxCoeff();
        std::printf("T07 double c=%-9.0e relp=%.2e dapproach=%.2e dobj=%.2e "
                    "dgap=%.2e ddisp=%.2e\n",
                    c, relp, dapp, dobj, dgap, ddisp);
        // Adding c to a double array already rounds it: the smallest
        // representable variation beside the offset is c*eps, i.e. a relative
        // geometry uncertainty of c*eps/g_ref. Centering cannot recover
        // information the caller's array no longer holds, so the acceptance
        // bound is that uncertainty (with margin), not exact equality.
        const double geo_unc =
            20.0 * std::abs(c) * 2.220446049250313e-16 / gref;
        CHECK(rd.converged);
        CHECK(relp < std::max(1e-8, geo_unc));
        CHECK(std::abs(dapp) < 1e-9 * std::max(1.0, std::abs(c)));
        CHECK(std::abs(dobj) < 1e-9 * std::max(1.0, std::abs(c * P)));
        CHECK(dgap < std::max(1e-9, geo_unc) * gref);
        CHECK(ddisp < std::max(1e-9, geo_unc) * gref);

        // float: the datum must be removed BEFORE the cast. Rounding the
        // offset field first is exactly what destroyed the float solution
        // (review §4), so this fused cast is the contract under test.
        Eigen::VectorXf gf = (gs.array() - c).cast<float>();
        hmc::SolveOptions fo = base;
        fo.tol = 2e-6;
        fo.scales.datum_mode = hmc::SolveScales::DatumMode::caller;
        fo.scales.datum = c;
        const auto rf =
            hmc::solve_contact_impl<float>(opf, gf, float(p_bar), fo, {}, nullptr);
        const double relf =
            (rf.pressure - ref.pressure).norm() / ref.pressure.norm();
        std::printf("T07 float  c=%-9.0e status=%-10s relp=%.2e approach=%.6f\n",
                    c, hmc::to_string(rf.status), relf, rf.approach);
        CHECK(rf.converged);
        CHECK(relf < std::max(1e-4, geo_unc));
        CHECK(std::abs(rf.approach - ref.approach - c) <
              1e-5 * std::max(1.0, std::abs(c)));
    }

    // A flat gap has zero range: g_ref must fall back to something positive
    // and the solve must stay finite.
    {
        Eigen::VectorXd flat = Eigen::VectorXd::Constant(N, 3.0);
        hmc::SolveOptions o;
        o.tol = 1e-10;
        o.max_iter = 500;
        const auto r =
            hmc::solve_contact_impl<double>(op, flat, p_bar, o, {}, nullptr);
        std::printf("T07 flat gap: status=%s g_ref=%.3e approach=%.6f\n",
                    hmc::to_string(r.status), r.g_ref, r.approach);
        CHECK(r.g_ref > 0.0);
        CHECK(std::isfinite(r.approach));
        CHECK(r.pressure.minCoeff() >= 0.0);
    }
    return 0;
}

// ── T08: units and geometry/material scaling ───────────────────────────────
int t08_units() {
    const int Ns = 24, N = Ns * Ns;
    const Eigen::VectorXd h = rough_gap(Ns, 5) * 0.02;
    const double p_bar = 0.01;

    hmc::BoussinesqKernel K0(Ns, 1.0, 1.0);
    const Eigen::MatrixXd S0 = K0.assemble_dense();
    hmc::MatVecIntoT<double> op0 = [&S0](const Eigen::VectorXd& x,
                                         Eigen::VectorXd& y) { y = S0 * x; };
    hmc::SolveOptions o;
    o.tol = 1e-12;
    o.max_iter = 20000;
    Eigen::VectorXd g0 = h;
    const auto r0 = hmc::solve_contact_impl<double>(op0, g0, p_bar, o, {}, nullptr);
    CHECK(r0.converged);
    const Eigen::VectorXd base_contact =
        (r0.pressure.array() > 0.0).cast<double>();

    for (double a : {1e-3, 1.0, 1e3}) {
        for (double b : {1e-3, 1.0, 1e3}) {
            hmc::BoussinesqKernel K(Ns, a * 1.0, b * 1.0);
            const Eigen::MatrixXd S = K.assemble_dense();
            hmc::MatVecIntoT<double> op = [&S](const Eigen::VectorXd& x,
                                               Eigen::VectorXd& y) { y = S * x; };
            Eigen::VectorXd g = a * h;
            const auto r =
                hmc::solve_contact_impl<double>(op, g, b * p_bar, o, {}, nullptr);
            const double relp =
                (r.pressure / b - r0.pressure).norm() / r0.pressure.norm();
            const double relu = (r.displacement / a - r0.displacement).norm() /
                                std::max(r0.displacement.norm(), 1e-300);
            const Eigen::VectorXd cont = (r.pressure.array() > 0.0).cast<double>();
            const double dtopo = (cont - base_contact).cwiseAbs().sum();
            std::printf("T08 a=%-7.0e b=%-7.0e status=%-10s relp=%.2e relu=%.2e "
                        "topology_diff=%.0f\n",
                        a, b, hmc::to_string(r.status), relp, relu, dtopo);
            CHECK(r.converged);
            CHECK(relp < 1e-8);
            CHECK(relu < 1e-8);
            CHECK(std::abs(r.approach / a - r0.approach) <
                  1e-8 * std::abs(r0.approach));
            CHECK(dtopo <= 2.0); // at most a couple of boundary cells
            (void)N;
        }
    }
    return 0;
}

// ── T09: reductions over large dynamic ranges ──────────────────────────────
int t09_reductions() {
    // The load renormalisation and the certificate reductions accumulate in
    // double for both working precisions. Compare with a long-double
    // reference over a vector with a wide dynamic range.
    const int n = 4000000;
    Eigen::VectorXf y(n);
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    long double exact = 0.0L;
    for (int i = 0; i < n; ++i) {
        const double v = U(rng) * std::pow(10.0, -6.0 + 8.0 * U(rng));
        y(i) = static_cast<float>(v);
        exact += static_cast<long double>(y(i));
    }
    const double P = static_cast<double>(exact) * 0.25;
    hmc::project_load_simplex<float>(y, P);
    long double got = 0.0L;
    for (int i = 0; i < n; ++i) got += static_cast<long double>(y(i));
    const double rel = std::abs(static_cast<double>(got - (long double)P)) / P;
    std::printf("T09 float simplex projection over %d entries: load rel err %.3e\n",
                n, rel);
    CHECK(y.minCoeff() >= 0.0f);
    CHECK(rel < 1e-5); // float storage, double accumulation
    return 0;
}

int main() {
    if (t07_datum()) return 1;
    if (t08_units()) return 1;
    if (t09_reductions()) return 1;
    std::printf("test_precision: all checks passed\n");
    return 0;
}
