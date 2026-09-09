#include "stencil_precond.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAILED: %s (line %d)\n", #cond, __LINE__);        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

// Universal constants of w = F^-1|k| normalised by w(0,0), converged to six
// figures by Ns=512 (spec §2.1). This table is the ORACLE: the implementation
// derives the weights from an FFT and must reproduce these.
static double oracle(int dx, int dy) {
    const int a = std::abs(dx), b = std::abs(dy);
    const int lo = std::min(a, b), hi = std::max(a, b);
    if (hi == 0 && lo == 0) return 1.0;
    if (hi == 1 && lo == 0) return -0.180875;
    if (hi == 1 && lo == 1) return -0.032860;
    if (hi == 2 && lo == 0) return 0.021189;
    if (hi == 2 && lo == 1) return -0.003912;
    if (hi == 2 && lo == 2) return -0.003198;
    if (hi == 3 && lo == 0) return -0.015473;
    if (hi == 4 && lo == 0) return 0.006277;
    return std::numeric_limits<double>::quiet_NaN(); // not tabulated
}

int main() {
    // ── S1: weights match the oracle to six figures, at every Ns >= 512 ────
    for (int Ns : {512, 1024, 2048}) {
        hmc::StencilPreconditioner sp(Ns, 2);
        CHECK(sp.ntaps() == 13); // l2 disc of radius 2
        for (int t = 0; t < sp.ntaps(); ++t) {
            const double ref = oracle(sp.tap_dx()[t], sp.tap_dy()[t]);
            CHECK(ref == ref); // every radius-2 tap must be tabulated
            const double got = sp.tap_w()[t];
            if (std::abs(got - ref) > 5e-6) {
                std::printf("Ns=%d tap (%d,%d): got %.9f want %.9f\n", Ns,
                            sp.tap_dx()[t], sp.tap_dy()[t], got, ref);
                return 1;
            }
        }
        std::printf("S1 Ns=%5d: 13 taps match the oracle\n", Ns);
    }

    // the centre tap must be exactly 1 (the normalisation)
    {
        hmc::StencilPreconditioner sp(1024, 2);
        for (int t = 0; t < sp.ntaps(); ++t)
            if (sp.tap_dx()[t] == 0 && sp.tap_dy()[t] == 0)
                CHECK(sp.tap_w()[t] == 1.0);
    }

    // ── S2: the truncated symbol stays strictly positive off DC ────────────
    // Direct DFT of the tap set: what_R(k) = sum_t w_t cos(2pi k.d/Ns) (w is
    // real and even, so the sine part cancels). If this went negative the
    // preconditioner would not be SPD and the CG theory would not apply.
    for (int R : {1, 2, 4}) {
        const int Ns = 64;
        hmc::StencilPreconditioner sp(Ns, R);
        double mn = 1e300, mx = -1e300;
        for (int ky = 0; ky < Ns; ++ky)
            for (int kx = 0; kx < Ns; ++kx) {
                if (kx == 0 && ky == 0) continue; // DC is zeroed by design
                double s = 0.0;
                for (int t = 0; t < sp.ntaps(); ++t)
                    s += sp.tap_w()[t] *
                         std::cos(2.0 * M_PI *
                                  (double(kx) * sp.tap_dx()[t] +
                                   double(ky) * sp.tap_dy()[t]) / Ns);
                mn = std::min(mn, s);
                mx = std::max(mx, s);
            }
        std::printf("S2 R=%d: symbol in [%.5f, %.5f], min/max %.5f\n", R, mn,
                    mx, mn / mx);
        CHECK(mn > 0.0);
    }

    std::printf("test_stencil: OK\n");
    return 0;
}
