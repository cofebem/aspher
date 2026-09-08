// Input validation, checked sizes and build/apply lifecycle (spec A19,
// tests T33/T34 of doc/plans/2026-09-08-accuracy-efficiency-validation.md).

#include "boussinesq_kernel.hpp"
#include "contact_solver.hpp"
#include "fft_operator.hpp"
#include "h2_operator.hpp"
#include "nested_solve.hpp"

#include <cmath>
#include <cstdio>
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

int main() {
    if (t33_bad_inputs()) return 1;
    if (t34_lifecycle()) return 1;
    std::printf("test_contracts: all checks passed\n");
    return 0;
}
