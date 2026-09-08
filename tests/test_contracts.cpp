// Input validation, checked sizes and build/apply lifecycle (spec A19,
// tests T33/T34 of doc/plans/2026-09-08-accuracy-efficiency-validation.md).

#include "boussinesq_kernel.hpp"
#include "contact_solver.hpp"
#include "fft_operator.hpp"
#include "h2_operator.hpp"
#include "nested_solve.hpp"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <limits>
#include <stdexcept>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAILED: %s (line %d)\n", #cond, __LINE__);        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

// Runs `fn` and reports whether it threw std::invalid_argument.
template <class F>
static bool rejects(F&& fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

// ── T33: bad inputs are rejected before allocation or iteration ────────────
static int t33_bad_inputs() {
    const int Ns = 32;
    Eigen::VectorXd g0 = Eigen::VectorXd::Zero(Ns * Ns);
    for (int i = 0; i < Ns * Ns; ++i) g0(i) = 1e-3 * (i % 17);

    struct Case { const char* name; hmc::NestedParams np; int ns; double L, E, p, tol; int it; };
    hmc::NestedParams ok;
    ok.coarsest = 16;
    ok.q = 4;
    ok.leaf_side = 8;

    // the reference call must work
    {
        auto r = hmc::solve_contact_nested(Ns, 1.0, 1.0, g0, 0.01, 1e-8, 5000,
                                           true, ok);
        CHECK(r.pressure.size() == Ns * Ns);
    }

    std::vector<Case> bad;
    { auto np = ok; np.coarsest = 0;  bad.push_back({"coarsest=0", np, Ns, 1, 1, 0.01, 1e-8, 5000}); }
    { auto np = ok; np.coarsest = -8; bad.push_back({"coarsest<0", np, Ns, 1, 1, 0.01, 1e-8, 5000}); }
    { auto np = ok; np.coarsest = 64; bad.push_back({"coarsest>Ns", np, Ns, 1, 1, 0.01, 1e-8, 5000}); }
    { auto np = ok; np.coarsest = 12; bad.push_back({"Ns != coarsest*2^k", np, Ns, 1, 1, 0.01, 1e-8, 5000}); }
    { auto np = ok; np.q = 1;         bad.push_back({"q<2", np, Ns, 1, 1, 0.01, 1e-8, 5000}); }
    { auto np = ok; np.leaf_side = 6; bad.push_back({"leaf_side not 2^k", np, Ns, 1, 1, 0.01, 1e-8, 5000}); }
    { auto np = ok; np.leaf_side = 0; bad.push_back({"leaf_side=0", np, Ns, 1, 1, 0.01, 1e-8, 5000}); }
    { auto np = ok; np.coarse_tol = 0.0; bad.push_back({"coarse_tol=0", np, Ns, 1, 1, 0.01, 1e-8, 5000}); }
    { auto np = ok; np.backend = "nope"; bad.push_back({"unknown backend", np, Ns, 1, 1, 0.01, 1e-8, 5000}); }
    { auto np = ok; np.active_set = true; np.backend = "fft";
      bad.push_back({"active_set with fft", np, Ns, 1, 1, 0.01, 1e-8, 5000}); }
    { auto np = ok; np.active_set = true; np.active_max_rounds = 0;
      bad.push_back({"active_max_rounds=0", np, Ns, 1, 1, 0.01, 1e-8, 5000}); }
    bad.push_back({"Ns=0", ok, 0, 1, 1, 0.01, 1e-8, 5000});
    bad.push_back({"Ns<0", ok, -32, 1, 1, 0.01, 1e-8, 5000});
    bad.push_back({"L=0", ok, Ns, 0, 1, 0.01, 1e-8, 5000});
    bad.push_back({"L<0", ok, Ns, -1, 1, 0.01, 1e-8, 5000});
    bad.push_back({"E*=0", ok, Ns, 1, 0, 0.01, 1e-8, 5000});
    bad.push_back({"p_bar=0", ok, Ns, 1, 1, 0.0, 1e-8, 5000});
    bad.push_back({"p_bar<0", ok, Ns, 1, 1, -0.01, 1e-8, 5000});
    bad.push_back({"tol=0", ok, Ns, 1, 1, 0.01, 0.0, 5000});
    bad.push_back({"tol<0", ok, Ns, 1, 1, 0.01, -1e-8, 5000});
    bad.push_back({"max_iter<0", ok, Ns, 1, 1, 0.01, 1e-8, -1});

    for (const auto& c : bad) {
        const bool r = rejects([&] {
            hmc::solve_contact_nested(c.ns, c.L, c.E, g0, c.p, c.tol, c.it,
                                      true, c.np);
        });
        std::printf("T33 %-24s rejected=%d\n", c.name, int(r));
        CHECK(r);
    }

    // wrong gap size and non-finite gap
    {
        Eigen::VectorXd wrong = Eigen::VectorXd::Zero(Ns * Ns - 1);
        CHECK(rejects([&] {
            hmc::solve_contact_nested(Ns, 1.0, 1.0, wrong, 0.01, 1e-8, 5000, true, ok);
        }));
        Eigen::VectorXd nf = g0;
        nf(5) = std::numeric_limits<double>::quiet_NaN();
        CHECK(rejects([&] {
            hmc::solve_contact_nested(Ns, 1.0, 1.0, nf, 0.01, 1e-8, 5000, true, ok);
        }));
    }

    // H2 operator parameters, checked WITHOUT building anything of that size
    hmc::BoussinesqKernel K(Ns, 1.0, 1.0);
    CHECK(rejects([&] { hmc::H2Operator o(K, {8, 4, 0}); }));   // near_radius 0
    CHECK(rejects([&] { hmc::H2Operator o(K, {6, 4, 1}); }));   // leaf not 2^k
    CHECK(rejects([&] { hmc::H2Operator o(K, {8, 1, 1}); }));   // q < 2
    CHECK(rejects([&] { hmc::H2Operator o(K, {64, 4, 1}); }));  // leaf > Ns
    // Ns=32768 index arithmetic: Ns*Ns = 2^30 fits int, (2Ns)^2 does not fit
    // int and must be caught by the checked path — without allocating it.
    CHECK(rejects([&] {
        Eigen::VectorXd tiny(1);
        hmc::solve_contact_nested(32768, 1.0, 1.0, tiny, 0.01, 1e-8, 10, true, ok);
    }));
    std::printf("T33 all input cases rejected\n");
    return 0;
}

// ── T34: build/apply lifecycle ─────────────────────────────────────────────
static int t34_lifecycle() {
    const int Ns = 32, N = Ns * Ns;
    hmc::BoussinesqKernel K(Ns, 1.0, 1.0);
    hmc::H2Operator A(K, {8, 4, 1});
    CHECK(!A.is_built());
    A.build();
    CHECK(A.is_built());
    const auto info1 = A.info();

    Eigen::VectorXd x = Eigen::VectorXd::Random(N), y1(N), y2(N);
    A.matvec_into(x, y1);

    // build() is idempotent: a second call must not append interactions,
    // duplicate couplings or leave a stale float cache behind
    A.build();
    const auto info2 = A.info();
    CHECK(info2.n_far_interactions == info1.n_far_interactions);
    CHECK(info2.n_near_interactions == info1.n_near_interactions);
    CHECK(info2.n_unique_couplings == info1.n_unique_couplings);
    CHECK(info2.n_leaves == info1.n_leaves);
    CHECK(info2.bytes_total == info1.bytes_total);
    A.matvec_into(x, y2);
    std::printf("T34 build twice: interactions %lld==%lld, matvec diff %.3e\n",
                (long long)info1.n_far_interactions,
                (long long)info2.n_far_interactions, (y1 - y2).norm());
    CHECK((y1 - y2).norm() == 0.0); // bit-for-bit

    // float caches before and after double use, in both orders
    hmc::H2Operator B(K, {8, 4, 1});
    B.build();
    B.build_single_caches();
    Eigen::VectorXf xf = x.cast<float>(), yf1(N), yf2(N);
    B.matvec_single_into(xf, yf1);
    B.build_single_caches(); // repeat must be a no-op
    B.matvec_single_into(xf, yf2);
    CHECK((yf1 - yf2).norm() == 0.0f);
    Eigen::VectorXd yb(N);
    B.matvec_into(x, yb);
    CHECK((yb - y1).norm() == 0.0);

    // FFT operator: same idempotence contract
    hmc::FFTOperator F(K);
    F.build();
    Eigen::VectorXd f1(N), f2(N);
    F.matvec_into(x, f1);
    F.build();
    F.matvec_into(x, f2);
    CHECK((f1 - f2).norm() == 0.0);
    std::printf("T34 lifecycle: all contracts hold\n");
    return 0;
}

// ── T13: compact near-offset cache equals the full-table operator ─────────
static int t13_compact_cache() {
    for (int Ns : {8, 16, 32, 64}) {
        for (int ls : {4, 8, 16}) {
            if (ls > Ns) continue;
            for (int rad : {1, 2, 3}) {
                for (int q : {4, 6}) {
                    hmc::H2Params par{ls, q, rad};
                    hmc::BoussinesqKernel K(Ns, 1.0, 1.0);
                    hmc::H2Operator full(K, par);
                    full.build();
                    auto compact = hmc::make_boussinesq_h2(Ns, 1.0, 1.0, par);
                    compact->build();

                    // every required near entry must match the full table
                    const int b = hmc::near_table_extent(Ns, ls, rad);
                    CHECK(compact->info().near_table_extent == b);
                    CHECK(b <= Ns);
                    CHECK(b >= std::min(Ns, (rad + 1) * ls));

                    const int N = Ns * Ns;
                    Eigen::VectorXd x = Eigen::VectorXd::Random(N), y1(N), y2(N);
                    full.matvec_into(x, y1);
                    compact->matvec_into(x, y2);
                    if ((y1 - y2).cwiseAbs().maxCoeff() != 0.0) {
                        std::printf("FAILED compact Ns=%d ls=%d rad=%d q=%d "
                                    "diff=%.3e\n",
                                    Ns, ls, rad, q,
                                    (y1 - y2).cwiseAbs().maxCoeff());
                        return 1;
                    }
                    // and in float
                    full.build_single_caches();
                    compact->build_single_caches();
                    Eigen::VectorXf xf = x.cast<float>(), f1(N), f2(N);
                    full.matvec_single_into(xf, f1);
                    compact->matvec_single_into(xf, f2);
                    CHECK((f1 - f2).cwiseAbs().maxCoeff() == 0.0f);

                    // the compact operator OWNS its coefficients and reports
                    // them; the full-table one borrows the caller's kernel
                    CHECK(compact->info().bytes_kernel ==
                          static_cast<std::int64_t>(b) * b * 8);
                    CHECK(full.info().bytes_kernel == 0);
                }
            }
        }
    }
    // the Ns = leaf_side case (b clipped to Ns)
    {
        hmc::H2Params par{8, 4, 1};
        auto op = hmc::make_boussinesq_h2(8, 1.0, 1.0, par);
        op->build();
        CHECK(op->info().near_table_extent == 8);
    }
    // storage: at Ns=1024, leaf 8, radius 1 the compact table is 2 KiB where
    // the full one is 8 MiB
    {
        auto op = hmc::make_boussinesq_h2(1024, 1.0, 1.0, {8, 4, 1});
        op->build();
        const std::int64_t full_bytes = 8LL * 1024 * 1024;
        std::printf("T13 Ns=1024: compact table %lld B vs full %lld B (%.0fx)\n",
                    (long long)op->info().bytes_kernel, (long long)full_bytes,
                    double(full_bytes) / double(op->info().bytes_kernel));
        CHECK(op->info().bytes_kernel == 16 * 16 * 8);
    }
    std::printf("T13 compact near cache: bit-for-bit over 48 configurations\n");
    return 0;
}

int main() {
    if (t13_compact_cache()) return 1;
    if (t33_bad_inputs()) return 1;
    if (t34_lifecycle()) return 1;
    std::printf("test_contracts: all checks passed\n");
    return 0;
}
