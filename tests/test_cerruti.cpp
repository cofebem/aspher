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

int main() {
    if (int rc = test_brackets_vs_quadrature()) return rc;
    if (int rc = test_identities()) return rc;
    std::printf("test_cerruti: all checks passed\n");
    return 0;
}
