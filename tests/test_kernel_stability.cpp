// Stable source-cell integration and parity (spec A05, tests T10-T12).
//
// The reference values come from tests/data/love_reference.txt, generated
// offline at 80 decimal digits by tests/generate_kernel_reference.py and
// cross-checked there against independent singularity-aware adaptive
// quadrature (agreement 1.2e-69). The fixture is READ ONLY: a failing test
// must never regenerate it, and mpmath is not needed to run this test.

#include "boussinesq_kernel.hpp"
#include "cerruti_kernel.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef ASPHER_TEST_DATA
#define ASPHER_TEST_DATA "tests/data"
#endif

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAILED: %s (line %d)\n", #cond, __LINE__);        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

namespace {

struct Ref { double x, y, a, b, value; };

std::vector<Ref> load_fixture(const std::string& path, bool* ok) {
    std::vector<Ref> out;
    std::ifstream in(path);
    *ok = static_cast<bool>(in);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        Ref r;
        if (ss >> r.x >> r.y >> r.a >> r.b >> r.value) out.push_back(r);
    }
    return out;
}

} // namespace

// ── T10/T11: compiled kernel vs the high-precision fixture ────────────────
static int t10_fixture() {
    bool ok = false;
    const std::string path = std::string(ASPHER_TEST_DATA) + "/love_reference.txt";
    const auto refs = load_fixture(path, &ok);
    if (!ok) {
        std::printf("FAILED: cannot read %s\n", path.c_str());
        return 1;
    }
    CHECK(refs.size() > 300);

    double worst = 0.0, worst_r = 0.0;
    int nfar = 0;
    for (const auto& r : refs) {
        const double got = hmc::love_uz(r.x, r.y, r.a, r.b);
        CHECK(std::isfinite(got));           // T11: never NaN, corners included
        CHECK(got > 0.0);                    // the integrand is positive
        const double rel = std::abs(got - r.value) / std::abs(r.value);
        const double rr = std::hypot(r.x, r.y) / std::max(r.a, r.b);
        if (rel > worst) { worst = rel; worst_r = rr; }
        if (rr > 1e4) ++nfar;
        if (rel > 1e-12) {
            std::printf("FAILED at (%.6g, %.6g) a=%g b=%g: got %.17g want %.17g "
                        "rel %.3e (r/m = %.3g)\n",
                        r.x, r.y, r.a, r.b, got, r.value, rel, rr);
            return 1;
        }
    }
    std::printf("T10/T11 %zu fixture points (incl. %d beyond r/m = 1e4): "
                "worst relative error %.3e at r/m = %.3g\n",
                refs.size(), nfar, worst, worst_r);
    CHECK(worst <= 1e-12);
    return 0;
}

// ── T11: self, edge, corner and branch continuity ─────────────────────────
static int t11_limits() {
    const double h = 1.0, a = 0.5 * h;
    // self term: 4h ln(1 + sqrt 2)
    const double self = hmc::love_uz(0.0, 0.0, a, a);
    const double want = 4.0 * h * std::log(1.0 + std::sqrt(2.0));
    std::printf("T11 self term: %.17g vs %.17g (rel %.3e)\n", self, want,
                std::abs(self - want) / want);
    CHECK(std::abs(self - want) <= 1e-15 * want);

    // every edge/corner is finite, and both sides of each agree with the
    // limit to the perturbation size
    for (double px : {0.0, a, -a}) {
        for (double py : {0.0, a, -a}) {
            const double v = hmc::love_uz(px, py, a, a);
            CHECK(std::isfinite(v) && v > 0.0);
            for (double d : {1e-13, 1e-10, 1e-7}) {
                const double lo = hmc::love_uz(px - d, py, a, a);
                const double hi = hmc::love_uz(px + d, py, a, a);
                CHECK(std::isfinite(lo) && std::isfinite(hi));
                // Continuity with the correct modulus: the cell-averaged
                // kernel is continuous everywhere, but its derivative has a
                // logarithmic singularity on the cell edges, so the honest
                // bound is C d (1 + |ln d|), not C d. Measured at the edge:
                // |dv| = 3.3e-6 at d = 1e-7, i.e. a ratio of 33 ~ 2|ln d|.
                const double mod = 10.0 * d * (1.0 + std::abs(std::log(d)));
                CHECK(std::abs(hi - v) <= mod + 1e-14);
                CHECK(std::abs(lo - v) <= mod + 1e-14);
            }
            // nextafter neighbours must not jump
            const double na = hmc::love_uz(std::nextafter(px, 1.0), py, a, a);
            CHECK(std::abs(na - v) <= 1e-13 * v);
        }
    }

    // continuity ACROSS the far-branch switch (r/m = 1000): both sides must
    // agree to the oracle tolerance
    for (double ang : {0.0, 0.3, 0.7853981633974483, 1.2}) {
        for (double f : {0.999, 0.9999, 1.0, 1.0001, 1.001}) {
            const double r = 1500.0 * a * f;
            const double x = r * std::cos(ang), y = r * std::sin(ang);
            const double v = hmc::love_uz(x, y, a, a);
            CHECK(std::isfinite(v) && v > 0.0);
        }
        // Both sides of the switch must agree AT THE SAME POINT. Comparing
        // two nearby points would mostly measure the kernel's own 1/r
        // variation, so evaluate the near (asinh) branch just inside the
        // switch and compare it with the far formula written out here.
        const double r0 = 1500.0 * a * (1.0 - 1e-9); // just inside
        const double x0 = r0 * std::cos(ang), y0 = r0 * std::sin(ang);
        const double near_side = hmc::love_uz(x0, y0, a, a);
        const double rr2 = x0 * x0 + y0 * y0, rr = std::sqrt(rr2);
        const double A = 4.0 * a * a;
        const double quad = (a * a * (3.0 * x0 * x0 - rr2) +
                             a * a * (3.0 * y0 * y0 - rr2)) / (6.0 * rr2 * rr2);
        const double far_side = A * (1.0 + quad) / rr;
        const double jump = std::abs(far_side - near_side) / near_side;
        std::printf("T11 far-branch switch at angle %.3f: branch disagreement "
                    "%.3e\n", ang, jump);
        CHECK(jump < 1e-12);
    }
    return 0;
}

// ── T12: parity, component swaps, homogeneity ─────────────────────────────
static int t12_parity() {
    const double a = 0.5, b = 0.5;
    double worst_swap = 0.0;
    for (double x : {0.0, 0.3, 1.0, 7.5, 1234.0, 5e5}) {
        for (double y : {0.0, 0.2, 2.0, 33.0, 9999.0, 3e5}) {
            const double v = hmc::love_uz(x, y, a, b);
            // even in x and in y, and under joint reversal — these are
            // EXACT, because the folding happens before any arithmetic
            CHECK(hmc::love_uz(-x, y, a, b) == v);
            CHECK(hmc::love_uz(x, -y, a, b) == v);
            CHECK(hmc::love_uz(-x, -y, a, b) == v);
            // x <-> y swap with a <-> b swapped too: mathematically identical
            // but a different summation order, so it holds to the kernel's
            // declared accuracy rather than bitwise
            const double sw = hmc::love_uz(y, x, b, a);
            worst_swap = std::max(worst_swap,
                                  std::abs(sw - v) / std::abs(v));
        }
    }
    std::printf("T12 worst xy-swap discrepancy: %.3e\n", worst_swap);
    CHECK(worst_swap <= 1e-12);
    // homogeneity: scaling ALL lengths by s multiplies the integrated
    // geometric kernel by s (area s^2 divided by distance s)
    for (double s : {1e-3, 0.1, 3.0, 1e3, 1e6}) {
        for (double x : {0.0, 0.7, 12.0, 4000.0}) {
            const double y = 0.31 * x + 0.05;
            const double v = hmc::love_uz(x, y, 0.5, 0.4);
            const double vs = hmc::love_uz(s * x, s * y, s * 0.5, s * 0.4);
            const double rel = std::abs(vs - s * v) / std::abs(s * v);
            if (rel > 1e-13) {
                std::printf("FAILED homogeneity s=%g x=%g: rel %.3e\n", s, x, rel);
                return 1;
            }
        }
    }
    // Scaling only the SEPARATION at fixed cell size is NOT 1/r scaling: the
    // cell correction h^2/(24 r^2) changes with the ratio, which is exactly
    // why a point kernel cannot replace the far Love kernel.
    {
        const double a2 = 0.5;
        const double v1 = hmc::love_uz(4.0, 0.0, a2, a2);
        const double v2 = hmc::love_uz(8.0, 0.0, a2, a2);
        const double point_ratio = 4.0 / 8.0;
        const double actual = v2 / v1;
        std::printf("T12 separation-only scaling: %.12f vs point kernel %.12f "
                    "(difference %.3e)\n",
                    actual, point_ratio, std::abs(actual - point_ratio));
        CHECK(std::abs(actual - point_ratio) > 1e-4);
    }
    // aspect ratios 1/16 .. 16
    for (double ar : {1.0 / 16, 1.0 / 4, 1.0, 4.0, 16.0}) {
        const double aa = 0.5 * std::sqrt(ar), bb = 0.5 / std::sqrt(ar);
        for (double x : {0.0, 1.0, 100.0, 1e4}) {
            const double v = hmc::love_uz(x, 0.0, aa, bb);
            CHECK(std::isfinite(v) && v > 0.0);
            CHECK(hmc::love_uz(-x, 0.0, aa, bb) == v);
        }
    }
    return 0;
}

// ── T10/T11 for the Cerruti brackets ──────────────────────────────────────
static int t10_cerruti_fixture() {
    struct CRef { double x, y, a, b, ylog, uxy; };
    std::vector<CRef> refs;
    {
        const std::string path =
            std::string(ASPHER_TEST_DATA) + "/cerruti_reference.txt";
        std::ifstream in(path);
        if (!in) {
            std::printf("FAILED: cannot read %s\n", path.c_str());
            return 1;
        }
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            CRef r;
            if (ss >> r.x >> r.y >> r.a >> r.b >> r.ylog >> r.uxy)
                refs.push_back(r);
        }
    }
    CHECK(refs.size() > 40);
    double worst_xx = 0.0, worst_xy = 0.0;
    for (const auto& r : refs) {
        // uxx = (1-nu) Love + nu ylog: recover ylog at nu = 1 and check it
        const double got_ylog = hmc::cerruti_uxx(r.x, r.y, r.a, r.b, 1.0);
        const double got_uxy = hmc::cerruti_uxy(r.x, r.y, r.a, r.b);
        CHECK(std::isfinite(got_ylog) && std::isfinite(got_uxy));
        // Both Cerruti brackets vanish where the normal kernel does not: uxy
        // is odd (identically zero on the axes) and ylog ~ A x^2/r^3 dies off
        // as x -> 0. Per the spec, such components are gated by ABSOLUTE
        // error against the local normal-kernel scale — that is the size they
        // actually contribute to u_x — never by dividing by a value the
        // symmetry forces to zero. Where the bracket IS of the kernel's own
        // size the two gates coincide.
        const double scale = hmc::love_uz(r.x, r.y, r.a, r.b);
        const double e1 = std::abs(got_ylog - r.ylog) /
                          std::max(std::abs(r.ylog), scale);
        worst_xx = std::max(worst_xx, e1);
        const double e2 = std::abs(got_uxy - r.uxy) /
                          std::max(std::abs(r.uxy), scale);
        worst_xy = std::max(worst_xy, e2);
        if (e1 > 1e-12 || e2 > 1e-12) {
            std::printf("FAILED Cerruti at (%.6g, %.6g): ylog err %.3e, "
                        "uxy err %.3e (love scale %.3e, ylog %.3e)\n",
                        r.x, r.y, e1, e2, scale, r.ylog);
            return 1;
        }
        // the assembled bracket that actually feeds u_x must meet the same
        // target against the normal-kernel scale
        const double uxx_ref = 0.7 * scale + 0.3 * r.ylog; // nu = 0.3
        const double uxx_got = hmc::cerruti_uxx(r.x, r.y, r.a, r.b, 0.3);
        CHECK(std::abs(uxx_got - uxx_ref) <= 1e-12 * scale);
    }
    std::printf("T10 Cerruti %zu fixture points: worst ylog %.3e, worst xy %.3e\n",
                refs.size(), worst_xx, worst_xy);
    // corners must be finite (the printed log form is NaN there)
    for (double px : {0.0, 0.5, -0.5})
        for (double py : {0.0, 0.5, -0.5}) {
            CHECK(std::isfinite(hmc::cerruti_uxx(px, py, 0.5, 0.5, 0.3)));
            CHECK(std::isfinite(hmc::cerruti_uxy(px, py, 0.5, 0.5)));
        }
    return 0;
}

// ── T12: Cerruti parities (unchanged conventions must be preserved) ───────
static int t12_cerruti() {
    const int Ns = 16;
    hmc::CerrutiKernel CK(Ns, 1.0, 1.0, 0.3);
    for (int dx = -6; dx <= 6; ++dx)
        for (int dy = -6; dy <= 6; ++dy) {
            // diagonals are even in each coordinate
            CHECK(CK.xx_offset(dx, dy) == CK.xx_offset(-dx, dy));
            CHECK(CK.xx_offset(dx, dy) == CK.xx_offset(dx, -dy));
            CHECK(CK.yy_offset(dx, dy) == CK.yy_offset(-dx, dy));
            CHECK(CK.yy_offset(dx, dy) == CK.yy_offset(dx, -dy));
            // xy is odd in each coordinate, even under joint reversal
            CHECK(CK.xy_offset(dx, dy) == -CK.xy_offset(-dx, dy));
            CHECK(CK.xy_offset(dx, dy) == -CK.xy_offset(dx, -dy));
            CHECK(CK.xy_offset(dx, dy) == CK.xy_offset(-dx, -dy));
            // xx and yy are coordinate swaps of each other
            CHECK(std::abs(CK.xx_offset(dx, dy) - CK.yy_offset(dy, dx)) <=
                  1e-15 * std::abs(CK.xx_offset(dx, dy) + 1e-300));
        }
    // nu = 0: the cross term vanishes identically
    hmc::CerrutiKernel CK0(Ns, 1.0, 1.0, 0.0);
    for (int dx = -5; dx <= 5; ++dx)
        for (int dy = -5; dy <= 5; ++dy) CHECK(CK0.xy_offset(dx, dy) == 0.0);
    std::printf("T12 Cerruti parities and nu=0 cross term: OK\n");
    return 0;
}

int main() {
    if (t10_fixture()) return 1;
    if (t11_limits()) return 1;
    if (t12_parity()) return 1;
    if (t10_cerruti_fixture()) return 1;
    if (t12_cerruti()) return 1;
    std::printf("test_kernel_stability: all checks passed\n");
    return 0;
}
