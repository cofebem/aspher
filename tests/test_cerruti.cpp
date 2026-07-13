#include "cerruti_kernel.hpp"

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

// Gauss-Legendre nodes/weights on [-1, 1] (Newton on the recurrence).
static void gauss_legendre(int n, std::vector<double>& xs,
                           std::vector<double>& ws) {
    xs.resize(n);
    ws.resize(n);
    for (int i = 0; i < n; ++i) {
        double x = std::cos(M_PI * (i + 0.75) / (n + 0.5));
        double p1 = 0.0, dp = 0.0;
        for (int it = 0; it < 100; ++it) {
            double p0 = 1.0;
            p1 = x;
            for (int k = 2; k <= n; ++k) {
                const double p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k;
                p0 = p1;
                p1 = p2;
            }
            dp = n * (x * p1 - p0) / (x * x - 1.0);
            const double step = p1 / dp;
            x -= step;
            if (std::abs(step) < 1e-15) break;
        }
        xs[i] = x;
        ws[i] = 2.0 / ((1.0 - x * x) * dp * dp);
    }
}

// Tensor GL quadrature of f(xi, eta) over [-a,a] x [-b,b].
template <class F>
static double quad2d(F f, double a, double b, int n) {
    std::vector<double> xs, ws;
    gauss_legendre(n, xs, ws);
    double s = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            s += ws[i] * ws[j] * f(a * xs[i], b * xs[j]);
    return s * a * b;
}

static int test_brackets_vs_quadrature() {
    const double h = 1.0 / 32, a = 0.5 * h, nu = 0.3;
    // point-kernel integrands (brackets, no elastic constants):
    //   uxx: (1-nu)/s + nu*(x-xi)^2/s^3       uxy: (x-xi)(y-eta)/s^3
    const int offsets[][2] = {{1, 0}, {0, 2}, {1, 1}, {2, 1}, {5, 3}, {9, 4}};
    for (auto& o : offsets) {
        const double x = o[0] * h, y = o[1] * h;
        auto fxx = [&](double xi, double eta) {
            const double s = std::hypot(x - xi, y - eta);
            const double dx2 = (x - xi) * (x - xi);
            return (1.0 - nu) / s + nu * dx2 / (s * s * s);
        };
        auto fxy = [&](double xi, double eta) {
            const double s = std::hypot(x - xi, y - eta);
            return (x - xi) * (y - eta) / (s * s * s);
        };
        const double qxx = quad2d(fxx, a, a, 64);
        const double qxy = quad2d(fxy, a, a, 64);
        CHECK(std::abs(hmc::cerruti_uxx(x, y, a, a, nu) / qxx - 1.0) < 1e-9);
        if (o[0] != 0 && o[1] != 0) // xy vanishes on the axes
            CHECK(std::abs(hmc::cerruti_uxy(x, y, a, a) / qxy - 1.0) < 1e-9);
    }
    return 0;
}

static int test_identities() {
    const double a = 0.011, b = 0.017; // deliberately anisotropic rectangle
    const double x = 0.05, y = -0.03, nu = 0.31;

    // nu = 0 collapses uxx to the Love bracket
    CHECK(std::abs(hmc::cerruti_uxx(x, y, a, b, 0.0) /
                       hmc::love_uz(x, y, a, b) - 1.0) < 1e-13);

    // parity: uxx even in x and y; uxy odd in x and odd in y
    CHECK(hmc::cerruti_uxx(x, y, a, b, nu) == hmc::cerruti_uxx(-x, y, a, b, nu));
    CHECK(hmc::cerruti_uxx(x, y, a, b, nu) == hmc::cerruti_uxx(x, -y, a, b, nu));
    CHECK(hmc::cerruti_uxy(-x, y, a, b) == -hmc::cerruti_uxy(x, y, a, b));
    CHECK(hmc::cerruti_uxy(x, -y, a, b) == -hmc::cerruti_uxy(x, y, a, b));

    // self terms on a square element: uxx = (2-nu) * 4a ln(1+sqrt(2));
    // uxy = 0 exactly (FP-exact cancellation)
    const double self = (2.0 - nu) * 4.0 * a * std::log(1.0 + std::sqrt(2.0));
    CHECK(std::abs(hmc::cerruti_uxx(0.0, 0.0, a, a, nu) / self - 1.0) < 1e-14);
    CHECK(hmc::cerruti_uxy(0.0, 0.0, a, a) == 0.0);
    return 0;
}

static int test_kernel_class() {
    const int Ns = 16;
    const double L = 1.0, E = 1.7, nu = 0.3, h = L / Ns;
    hmc::CerrutiKernel C(Ns, L, E, nu);
    CHECK(C.grid_size() == Ns && C.size() == Ns * Ns);
    CHECK(C.element_size() == h && C.E_star() == E && C.nu() == nu);

    // scaled self terms: xx = (2-nu)*4a*ln(1+sqrt(2)) / (pi E* (1-nu)); xy = 0
    const double a = 0.5 * h;
    const double self =
        (2.0 - nu) * 4.0 * a * std::log(1.0 + std::sqrt(2.0)) /
        (M_PI * E * (1.0 - nu));
    CHECK(std::abs(C.xx_offset(0, 0) / self - 1.0) < 1e-14);
    CHECK(C.xy_offset(0, 0) == 0.0);

    // x<->y swap serves yy; parity/sign handling of the signed offsets
    CHECK(C.yy_offset(3, 1) == C.xx_offset(1, 3));
    CHECK(C.xx_offset(-3, 1) == C.xx_offset(3, 1));
    CHECK(C.xy_offset(2, 5) == C.xy_offset(-2, -5));
    CHECK(C.xy_offset(-2, 5) == -C.xy_offset(2, 5));
    CHECK(C.xy_offset(2, -5) == -C.xy_offset(2, 5));
    CHECK(C.xy_offset(0, 4) == 0.0 && C.xy_offset(4, 0) == 0.0);

    // out-of-span offsets are zero (entry_offset convention)
    CHECK(C.xx_offset(Ns, 0) == 0.0 && C.xy_offset(0, -Ns) == 0.0);

    // far evaluators agree with the tables at integer offsets (exact:
    // same closed form, same arithmetic)
    CHECK(C.xx_far(4 * h, 2 * h) == C.xx_offset(4, 2));
    CHECK(C.yy_far(4 * h, 2 * h) == C.yy_offset(4, 2));
    CHECK(C.xy_far(4 * h, 2 * h) == C.xy_offset(4, 2));

    // dense block: symmetric and positive definite (elastic compliance)
    const Eigen::MatrixXd M = C.assemble_dense();
    CHECK(M.rows() == 2 * Ns * Ns);
    CHECK((M - M.transpose()).lpNorm<Eigen::Infinity>() == 0.0);
    Eigen::LLT<Eigen::MatrixXd> llt(M);
    CHECK(llt.info() == Eigen::Success);
    return 0;
}

static int test_symbol() {
    const int Ns = 128;
    const double L = 1.0, E = 1.7, nu = 0.3, h = L / Ns;
    hmc::CerrutiKernel C(Ns, L, E, nu);
    hmc::BoussinesqKernel B(Ns, L, E);

    // exact analytic properties at an arbitrary wavevector
    {
        const double kx = 7.3, ky = -2.1, kk = std::hypot(kx, ky);
        const Eigen::Matrix2d S = C.symbol(kx, ky);
        CHECK((S - S.transpose()).lpNorm<Eigen::Infinity>() < 1e-18);
        // longitudinal eigenvalue == the normal (Love) symbol 2/(E*|k|)
        Eigen::Vector2d el(kx / kk, ky / kk), et(-ky / kk, kx / kk);
        CHECK(std::abs(el.dot(S * el) / (2.0 / (E * kk)) - 1.0) < 1e-14);
        // transverse eigenvalue == 2/(E*(1-nu)|k|)
        CHECK(std::abs(et.dot(S * et) / (2.0 / (E * (1.0 - nu) * kk)) - 1.0) <
              1e-14);
        // isotropy: same eigenvalues at a rotated wavevector of equal norm
        const Eigen::Matrix2d S2 = C.symbol(kk, 0.0);
        CHECK(std::abs(S2(0, 0) - el.dot(S * el)) < 1e-13 * S2(0, 0));
        CHECK(std::abs(S2(1, 1) - et.dot(S * et)) < 1e-13 * S2(1, 1));
        CHECK(std::abs(S2(0, 1)) < 1e-18);
    }

    // truncated-lattice DFT of the offset tables vs the continuum symbol.
    // Same-machinery calibration: the Love table vs 2/(E*|k|) bounds the
    // truncation error of the method itself; the Cerruti components must
    // match their symbols at the same order.
    auto dft = [&](auto entry, double kx, double ky) {
        double s = 0.0;
        for (int dy = -(Ns - 1); dy <= Ns - 1; ++dy)
            for (int dx = -(Ns - 1); dx <= Ns - 1; ++dx)
                s += entry(dx, dy) * std::cos(kx * dx * h + ky * dy * h);
        return s;
    };
    const int modes[][2] = {{8, 0}, {0, 8}, {6, 6}, {12, 5}};
    for (auto& mo : modes) {
        const double kx = 2.0 * M_PI * mo[0] / L, ky = 2.0 * M_PI * mo[1] / L;
        const double kk = std::hypot(kx, ky);
        const double love_err =
            std::abs(dft([&](int dx, int dy) { return B.entry_offset(dx, dy); },
                         kx, ky) /
                         (2.0 / (E * kk)) -
                     1.0);
        const Eigen::Matrix2d S = C.symbol(kx, ky);
        const double exx =
            std::abs(dft([&](int dx, int dy) { return C.xx_offset(dx, dy); },
                         kx, ky) /
                         S(0, 0) -
                     1.0);
        const double eyy =
            std::abs(dft([&](int dx, int dy) { return C.yy_offset(dx, dy); },
                         kx, ky) /
                         S(1, 1) -
                     1.0);
        std::printf("mode (%2d,%2d): love %.2e  xx %.2e  yy %.2e\n", mo[0],
                    mo[1], love_err, exx, eyy);
        CHECK(love_err < 0.15);
        CHECK(exx < 0.18 && eyy < 0.18);
        if (mo[0] != 0 && mo[1] != 0) { // xy symbol vanishes on the axes
            const double exy =
                std::abs(dft([&](int dx, int dy) { return C.xy_offset(dx, dy); },
                             kx, ky) /
                             S(0, 1) -
                         1.0);
            std::printf("             xy %.2e\n", exy);
            CHECK(exy < 0.10);
        }
    }
    return 0;
}

int main() {
    if (int rc = test_brackets_vs_quadrature()) return rc;
    if (int rc = test_identities()) return rc;
    if (int rc = test_kernel_class()) return rc;
    if (int rc = test_symbol()) return rc;
    std::printf("test_cerruti: all checks passed\n");
    return 0;
}
