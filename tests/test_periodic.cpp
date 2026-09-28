#include "contact_solver.hpp"
#include "periodic_fft_operator.hpp"

#include <cmath>
#include <cstdio>
#include <utility>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAILED: %s (line %d)\n", #cond, __LINE__);       \
            return 1;                                                      \
        }                                                                  \
    } while (0)

static int test_modes() {
    const int n = 32;
    const double L = 3.0, Estar = 2.5;
    hmc::PeriodicFFTOperator op(n, L, Estar);
    const double pi = std::acos(-1.0);
    for (auto mode : {std::pair<int, int>{1, 0}, {0, 3}, {2, 5}}) {
        Eigen::VectorXd p(n * n), ref(n * n);
        const double q = 2.0 * pi * std::hypot(mode.first, mode.second) / L;
        const double factor = 2.0 / (Estar * q);
        for (int iy = 0; iy < n; ++iy)
            for (int ix = 0; ix < n; ++ix) {
                const double phase = 2.0 * pi *
                    (mode.first * ix + mode.second * iy) / n;
                p(iy * n + ix) = std::cos(phase);
                ref(iy * n + ix) = factor * std::cos(phase);
            }
        const Eigen::VectorXd u = op.matvec(p);
        CHECK((u - ref).norm() / ref.norm() < 2e-13);
    }
    return 0;
}

static int test_dc_and_repeat() {
    const int n = 32;
    hmc::PeriodicFFTOperator op(n, 1.0, 1.0);
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(n * n);
    CHECK(op.matvec(ones).norm() < 1e-13);
    Eigen::VectorXd x = Eigen::VectorXd::Random(n * n), y1, y2;
    op.matvec_into(x, y1);
    op.matvec_into(Eigen::VectorXd::Random(n * n), y2);
    op.matvec_into(x, y2);
    CHECK((y1 - y2).norm() == 0.0);
    const Eigen::VectorXf yf = op.matvec_single(x.cast<float>());
    CHECK((yf.cast<double>() - y1).norm() / y1.norm() < 2e-6);
    return 0;
}

// Odd grids (no power-of-two constraint), aliasing x == y, and the
// assembled matrix: symmetric, positive semi-definite, translation invariant
// with wrap-around (a circulant), null space = constants.
static int test_matrix_properties() {
    for (int n : {7, 12, 16}) {
        hmc::PeriodicFFTOperator op(n, 1.7, 0.8);
        const int N = n * n;
        Eigen::MatrixXd S(N, N);
        for (int j = 0; j < N; ++j) {
            Eigen::VectorXd e = Eigen::VectorXd::Zero(N);
            e(j) = 1.0;
            S.col(j) = op.matvec(e);
        }
        CHECK((S - S.transpose()).norm() / S.norm() < 1e-14);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
        CHECK(es.eigenvalues()(0) > -1e-13 * es.eigenvalues()(N - 1));
        CHECK(es.eigenvalues()(1) > 1e-6 * es.eigenvalues()(N - 1));
        // circulant: column of source (sx,sy) is column 0 shifted
        const int sx = 3 % n, sy = 5 % n;
        for (int iy = 0; iy < n; ++iy)
            for (int ix = 0; ix < n; ++ix) {
                const int a = iy * n + ix;
                const int b = ((iy - sy + n) % n) * n + (ix - sx + n) % n;
                CHECK(std::abs(S(a, sy * n + sx) - S(b, 0)) <
                      1e-14 * S(0, 0));
            }
        Eigen::VectorXd x = Eigen::VectorXd::Random(N), y = x;
        const Eigen::VectorXd ref = op.matvec(x);
        op.matvec_into(y, y); // aliased
        CHECK((y - ref).norm() <= 1e-15 * ref.norm());
        Eigen::VectorXf xf = x.cast<float>();
        op.matvec_single_into(xf, xf); // aliased, float
        CHECK((xf.cast<double>() - ref).norm() < 2e-6 * ref.norm());
    }
    return 0;
}

// Westergaard (1939) wavy contact, h(x) = D cos(2 pi x / L): full contact
// for p_bar >= p* = pi E* D / L with p = p_bar + p* cos(2 pi x / L); the
// cosine is an eigenvector of the discrete operator, so the discrete solution
// is exact. Partial contact has half-width a = (L/pi) asin(sqrt(p_bar/p*)).
static int test_westergaard() {
    const int n = 256;
    const double L = 2.0, Es = 1.3, D = 0.01;
    const double pi = std::acos(-1.0), pstar = pi * Es * D / L;
    hmc::PeriodicFFTOperator op(n, L, Es);
    auto S = [&op](const Eigen::VectorXd& v) { return op.matvec(v); };
    Eigen::VectorXd g0(n * n), cosx(n * n);
    for (int iy = 0; iy < n; ++iy)
        for (int ix = 0; ix < n; ++ix) {
            cosx(iy * n + ix) = std::cos(2.0 * pi * ix / n);
            g0(iy * n + ix) = D * (1.0 - cosx(iy * n + ix));
        }
    {
        const double pbar = 1.5 * pstar;
        const hmc::ContactResult r = hmc::solve_contact(S, g0, pbar, 1e-12);
        CHECK(r.converged);
        const Eigen::VectorXd pex =
            Eigen::VectorXd::Constant(n * n, pbar) + pstar * cosx;
        std::printf("  Westergaard full contact: rel %.2e, %d it\n",
                    (r.pressure - pex).norm() / pex.norm(), r.iterations);
        CHECK((r.pressure - pex).norm() / pex.norm() < 1e-9);
    }
    {
        const double pbar = 0.3 * pstar;
        const hmc::ContactResult r = hmc::solve_contact(S, g0, pbar, 1e-10);
        CHECK(r.converged);
        int ncol = 0; // contact columns along x in row 0
        for (int ix = 0; ix < n; ++ix) ncol += r.pressure(ix) > 0.0;
        const double width = ncol * L / n;
        const double wex = 2.0 * (L / pi) * std::asin(std::sqrt(pbar / pstar));
        std::printf("  Westergaard partial: width %.5f vs %.5f (h = %.5f)\n",
                    width, wex, L / n);
        CHECK(std::abs(width - wex) <= 2.0 * L / n);
    }
    return 0;
}

int main() {
    if (int rc = test_modes()) return rc;
    if (int rc = test_dc_and_repeat()) return rc;
    if (int rc = test_matrix_properties()) return rc;
    if (int rc = test_westergaard()) return rc;
    std::printf("test_periodic: all passed\n");
    return 0;
}
