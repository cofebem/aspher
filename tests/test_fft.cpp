#include "boussinesq_kernel.hpp"
#include "contact_solver.hpp"
#include "fft_operator.hpp"

#include <cmath>
#include <cstdio>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAILED: %s (line %d)\n", #cond, __LINE__);        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

// The FFT backend is exact: it must match the dense matvec to roundoff
// (double) — strictly better than H2's interpolation error. Ns=24 guards the
// wrap-around indexing on a non-power-of-two grid (the operator itself has no
// power-of-two constraint).
static int test_matvec_exact() {
    for (int Ns : {24, 32, 64}) {
        hmc::BoussinesqKernel k(Ns, 1.0, 1.0);
        const Eigen::MatrixXd S = k.assemble_dense();
        hmc::FFTOperator A(k);
        A.build();
        A.print_statistics();
        Eigen::VectorXd x = Eigen::VectorXd::Random(Ns * Ns);
        const Eigen::VectorXd ref = S * x;
        const Eigen::VectorXd y = A.matvec(x);
        const double err = (y - ref).norm() / ref.norm();
        const Eigen::VectorXf yf = A.matvec_single(x.cast<float>());
        const double errf = (yf.cast<double>() - ref).norm() / ref.norm();
        std::printf("FFT matvec rel err (Ns=%d): double %.3e  float %.3e\n",
                    Ns, err, errf);
        CHECK(err < 1e-12);
        CHECK(errf < 1e-5);
    }
    return 0;
}

// matvec_into must reuse the object-owned scratch across calls and give the
// same answer on the second call (stale-padding guard: the previous inverse
// transform wrote the whole padded grid).
static int test_repeat_calls() {
    const int Ns = 32;
    hmc::BoussinesqKernel k(Ns, 1.0, 1.0);
    hmc::FFTOperator A(k);
    A.build();
    Eigen::VectorXd x = Eigen::VectorXd::Random(Ns * Ns);
    Eigen::VectorXd y1, y2;
    A.matvec_into(x, y1);
    Eigen::VectorXd other = Eigen::VectorXd::Random(Ns * Ns);
    A.matvec_into(other, y2); // scribble the scratch with a different field
    A.matvec_into(x, y2);
    CHECK((y1 - y2).norm() == 0.0);
    return 0;
}

// Validation gate (spec §4): the FFT-backend Hertz solve must reproduce the
// dense-backend solve — same iteration count, pressure to roundoff.
static int test_hertz_solve() {
    const int Ns = 64;
    const double R = 0.5, p_bar = 0.05, h = 1.0 / Ns;
    hmc::BoussinesqKernel k(Ns, 1.0, 1.0);
    Eigen::VectorXd g0(Ns * Ns);
    for (int iy = 0; iy < Ns; ++iy)
        for (int ix = 0; ix < Ns; ++ix) {
            const double x = (ix + 0.5) * h - 0.5, y = (iy + 0.5) * h - 0.5;
            g0(iy * Ns + ix) = (x * x + y * y) / (2.0 * R);
        }

    const Eigen::MatrixXd S = k.assemble_dense();
    auto mv_dense = [&S](const Eigen::VectorXd& v) -> Eigen::VectorXd {
        return S * v;
    };
    hmc::FFTOperator A(k);
    A.build();
    auto mv_fft = [&A](const Eigen::VectorXd& v) { return A.matvec(v); };

    const auto rd = hmc::solve_contact(mv_dense, g0, p_bar, 1e-8, 2000);
    const auto rf = hmc::solve_contact(mv_fft, g0, p_bar, 1e-8, 2000);
    CHECK(rd.converged);
    CHECK(rf.converged);
    std::printf("Hertz dense: %d it, Ac/A %.6f | fft: %d it, Ac/A %.6f\n",
                rd.iterations, rd.contact_fraction, rf.iterations,
                rf.contact_fraction);
    CHECK(rd.iterations == rf.iterations);
    CHECK(rd.contact_fraction == rf.contact_fraction);
    // PCG stops when the complementarity error crosses tol, so two operators
    // identical to machine roundoff still yield stopping iterates that agree
    // only to O(tol) (measured ~6e-8 at tol=1e-8; dp scales linearly with tol,
    // and even two dense-only matvec variants differ by more). 1e-6 still
    // discriminates: any *approximate* operator (H2 q=6 ~3e-6, dcfft Gibbs
    // %-level) fails it. [Calibrated 2026-07-09 with user approval.]
    const double dp = (rf.pressure - rd.pressure).norm() / rd.pressure.norm();
    std::printf("Hertz pressure rel diff (fft vs dense): %.3e\n", dp);
    CHECK(dp < 1e-6);
    return 0;
}

int main() {
    if (int rc = test_matvec_exact()) return rc;
    if (int rc = test_repeat_calls()) return rc;
    if (int rc = test_hertz_solve()) return rc;
    std::printf("test_fft: all passed\n");
    return 0;
}
