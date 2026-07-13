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

int main() {
    if (int rc = test_fft_vs_dense()) return rc;
    std::printf("test_tangential: all checks passed\n");
    return 0;
}
