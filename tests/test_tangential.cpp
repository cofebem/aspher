#include "tangential_operator.hpp"

#include "cerruti_kernel.hpp"

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

static Eigen::VectorXd random_q(int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    Eigen::VectorXd q(n);
    for (int i = 0; i < n; ++i) q(i) = d(rng);
    return q;
}

// Exactness gate: the FFT tangential matvec must reproduce the dense
// CerrutiKernel block matrix to roundoff (no interpolation, no truncation).
static int test_fft_vs_dense() {
    for (int Ns : {8, 16}) {
        hmc::CerrutiKernel C(Ns, 1.0, 1.7, 0.3);
        hmc::TangentialFFTOperator F(C);
        F.build();
        const Eigen::MatrixXd D = C.assemble_dense();
        const Eigen::VectorXd q = random_q(2 * Ns * Ns, 7 + Ns);
        const Eigen::VectorXd ref = D * q;
        const Eigen::VectorXd u = F.matvec(q);
        const double rel = (u - ref).norm() / ref.norm();
        std::printf("tangential fft vs dense Ns=%d: rel %.3e\n", Ns, rel);
        CHECK(rel < 1e-13);
    }
    return 0;
}

// H2 gate: far-field Chebyshev interpolation error only (near field exact).
// Mirrors test_h2's Love gates; the xy component's odd parity is exercised
// through the signed-offset coupling/stencil caches.
static int test_h2_vs_dense() {
    const int Ns = 32;
    hmc::CerrutiKernel C(Ns, 1.0, 1.7, 0.3);
    const Eigen::MatrixXd D = C.assemble_dense();
    const Eigen::VectorXd q = random_q(2 * Ns * Ns, 11);
    const Eigen::VectorXd ref = D * q;
    auto relerr = [&](int qo) {
        hmc::TangentialH2Operator A(C, {8, qo, 1});
        A.build();
        return (A.matvec(q) - ref).norm() / ref.norm();
    };
    const double e4 = relerr(4), e6 = relerr(6);
    std::printf("tangential H2 rel err: q=4 %.3e  q=6 %.3e\n", e4, e6);
    CHECK(e4 < 5e-3);
    CHECK(e6 < e4);
    CHECK(e6 < 1e-4);
    return 0;
}

int main() {
    if (int rc = test_fft_vs_dense()) return rc;
    if (int rc = test_h2_vs_dense()) return rc;
    std::printf("test_tangential: all checks passed\n");
    return 0;
}
