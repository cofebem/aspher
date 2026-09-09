#include "stencil_precond.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
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

    // ── S3: compressed apply == full-grid apply on the same data ──────────
    // Tap sums are bit-for-bit identical between the two paths (both route
    // through the shared tap_sum helper, gathering zeros for out-of-contact/
    // out-of-candidate-set neighbours rather than skipping them). But the
    // final z is NOT bit-for-bit: apply_full and apply_blocked reduce the
    // contact-set mean (zsum/nc, subtracted as zmean) over different
    // traversal orders -- per-block vs. row-major over the whole grid -- so
    // each entry's difference from the tap sum rounds differently depending
    // on |z(i)|. That makes the per-entry difference d = zc - zfull vary by
    // roughly one ULP of z's magnitude across entries, not a fixed constant,
    // and not exactly zero. So we assert what is actually true: d has a tiny
    // SPREAD (bounded by a few ULPs of the largest |z| on the contact set)
    // and a tiny MAGNITUDE, not exact equality. This is the same caveat
    // already documented on FourierPreconditioner::apply_into_indexed:
    // matches up to the reduction summation order, not bit-for-bit.
    {
        const int Ns = 64, ls = 8, R = 2;
        const int nl = Ns / ls, N = Ns * Ns, ls2 = ls * ls;
        hmc::StencilPreconditioner sp(Ns, R);

        // occupied blocks: a diagonal band plus a block on the domain edge,
        // so the periodic wrap is genuinely exercised
        hmc::StencilBlockLayout L;
        L.Ns = Ns; L.ls = ls;
        L.block_slot.assign(static_cast<std::size_t>(nl) * nl, -1);
        for (int by = 0; by < nl; ++by)
            for (int bx = 0; bx < nl; ++bx) {
                const bool keep = (bx == by) || (bx == 0 && by == nl - 1) ||
                                  (bx == nl - 1 && by == 0);
                if (!keep) continue;
                L.block_slot[static_cast<std::size_t>(by) * nl + bx] =
                    static_cast<int>(L.slot_bx.size());
                L.slot_bx.push_back(bx);
                L.slot_by.push_back(by);
            }
        L.nslots = static_cast<int>(L.slot_bx.size());
        CHECK(L.nslots > 2);

        const std::ptrdiff_t S = static_cast<std::ptrdiff_t>(L.nslots) * ls2;
        Eigen::VectorXd gc(S), zc;
        std::vector<std::uint8_t> contact_c(S, 0);
        Eigen::VectorXd gfull = Eigen::VectorXd::Zero(N), zfull;
        std::vector<std::uint8_t> contact(N, 0);

        // deterministic pseudo-random data; roughly half the candidate
        // entries are in contact
        std::uint64_t st = 0x9E3779B97F4A7C15ull;
        auto rnd = [&st]() {
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            return static_cast<double>(st >> 11) / 9007199254740992.0;
        };
        for (int s = 0; s < L.nslots; ++s)
            for (int ly = 0; ly < ls; ++ly)
                for (int lx = 0; lx < ls; ++lx) {
                    const std::ptrdiff_t k =
                        static_cast<std::ptrdiff_t>(s) * ls2 + ly * ls + lx;
                    const int gx = L.slot_bx[s] * ls + lx;
                    const int gy = L.slot_by[s] * ls + ly;
                    const std::ptrdiff_t i =
                        static_cast<std::ptrdiff_t>(gy) * Ns + gx;
                    const double v = rnd() - 0.5;
                    const std::uint8_t c = (rnd() < 0.5) ? 1 : 0;
                    gc(k) = v; contact_c[k] = c;
                    gfull(i) = v; contact[i] = c;
                }

        sp.apply_into(gfull, contact, zfull);
        sp.apply_into_blocked(gc, contact_c, L, zc);

        // gather the difference on the contact set, plus the largest |zfull|
        // (the scale against which the spread/magnitude bounds are stated)
        double dmin = 1e300, dmax = -1e300, maxabs_z = 0.0;
        long ncheck = 0;
        for (int s = 0; s < L.nslots; ++s)
            for (int ly = 0; ly < ls; ++ly)
                for (int lx = 0; lx < ls; ++lx) {
                    const std::ptrdiff_t k =
                        static_cast<std::ptrdiff_t>(s) * ls2 + ly * ls + lx;
                    if (!contact_c[k]) { CHECK(zc(k) == 0.0); continue; }
                    const int gx = L.slot_bx[s] * ls + lx;
                    const int gy = L.slot_by[s] * ls + ly;
                    const double zf =
                        zfull(static_cast<std::ptrdiff_t>(gy) * Ns + gx);
                    const double d = zc(k) - zf;
                    dmin = std::min(dmin, d); dmax = std::max(dmax, d);
                    maxabs_z = std::max(maxabs_z, std::abs(zf));
                    ++ncheck;
                }
        CHECK(ncheck > 100);
        const double spread_d = dmax - dmin;
        const double maxabs_d = std::max(std::abs(dmax), std::abs(dmin));
        std::printf("S3 double: %ld contact entries, spread of the "
                    "mean offset %.3e (max|z|=%.3e)\n",
                    ncheck, spread_d, maxabs_z);
        // Not bit-for-bit (see the comment above the block): bound the
        // spread by a few ULPs of the contact set's largest |z|, and the
        // magnitude by a generous multiple of that.
        CHECK(spread_d <=
              8.0 * std::numeric_limits<double>::epsilon() * maxabs_z);
        CHECK(maxabs_d <= 1e-12 * maxabs_z);

        // same in float
        Eigen::VectorXf gcf = gc.cast<float>(), zcf;
        Eigen::VectorXf gff = gfull.cast<float>(), zff;
        sp.apply_single_into(gff, contact, zff);
        sp.apply_single_into_blocked(gcf, contact_c, L, zcf);
        float fmin = 1e30f, fmax = -1e30f, maxabs_zf = 0.0f;
        for (int s = 0; s < L.nslots; ++s)
            for (int ly = 0; ly < ls; ++ly)
                for (int lx = 0; lx < ls; ++lx) {
                    const std::ptrdiff_t k =
                        static_cast<std::ptrdiff_t>(s) * ls2 + ly * ls + lx;
                    if (!contact_c[k]) continue;
                    const int gx = L.slot_bx[s] * ls + lx;
                    const int gy = L.slot_by[s] * ls + ly;
                    const float zf =
                        zff(static_cast<std::ptrdiff_t>(gy) * Ns + gx);
                    const float d = zcf(k) - zf;
                    fmin = std::min(fmin, d); fmax = std::max(fmax, d);
                    maxabs_zf = std::max(maxabs_zf, std::abs(zf));
                }
        const float spread_f = fmax - fmin;
        const float maxabs_f = std::max(std::abs(fmax), std::abs(fmin));
        std::printf("S3 float:  spread of the mean offset %.3e (max|z|=%.3e)\n",
                    double(spread_f), double(maxabs_zf));
        CHECK(spread_f <=
              8.0f * std::numeric_limits<float>::epsilon() * maxabs_zf);
        CHECK(maxabs_f <= 1e-4f * maxabs_zf);
    }

    std::printf("test_stencil: OK\n");
    return 0;
}
