// Active-set solver tests.
//
// M1 — masked H2 matvec exactness (h2_operator.hpp): for x supported on the
// src mask, matvec_masked_into equals the unmasked matvec BIT-FOR-BIT on
// tgt-occupied leaves — skipped terms are exact zeros and kept terms keep
// their summation order. Each masked call runs on a fresh H2Operator so the
// shared M/L scratch holds garbage, not correct values left by a previous
// unmasked apply: a pass that wrongly reads a skipped column fails loudly.
//
// M2 — active-set nested driver equivalence (nested_solve.hpp): on a rough
// multi-scale surface the active_set=true solve must reproduce the standard
// nested solve (equal area, tiny pressure difference) with a clean
// certificate, in both precisions; and a deliberately broken candidate set
// (delta=0, halo=0, one round) must trigger the full-solve fallback and
// still return the correct pressures rather than silently-wrong ones.
#include "boussinesq_kernel.hpp"
#include "h2_operator.hpp"
#include "nested_solve.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAILED: %s (line %d)\n", #cond, __LINE__);        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

namespace {

// Clustered random mask: a few tens of discs (islands) plus isolated pixels
// (orphan asperities), mimicking a real contact set.
std::vector<std::uint8_t> clustered_mask(int Ns, unsigned seed) {
    std::vector<std::uint8_t> m(static_cast<std::size_t>(Ns) * Ns, 0);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pos(0, Ns - 1), rad(2, 9);
    for (int d = 0; d < 24; ++d) {
        const int cx = pos(rng), cy = pos(rng), r = rad(rng);
        for (int iy = std::max(0, cy - r); iy <= std::min(Ns - 1, cy + r); ++iy)
            for (int ix = std::max(0, cx - r); ix <= std::min(Ns - 1, cx + r); ++ix)
                if ((ix - cx) * (ix - cx) + (iy - cy) * (iy - cy) <= r * r)
                    m[static_cast<std::size_t>(iy) * Ns + ix] = 1;
    }
    for (int d = 0; d < 40; ++d)
        m[static_cast<std::size_t>(pos(rng)) * Ns + pos(rng)] = 1;
    return m;
}

// Element (ix, iy) lies in a leaf box that intersects the grid mask.
std::vector<std::uint8_t> occupied_leaf_elements(
    const std::vector<std::uint8_t>& mask, int Ns, int ls) {
    std::vector<std::uint8_t> occ(mask.size(), 0);
    const int nls = Ns / ls;
    for (int by = 0; by < nls; ++by)
        for (int bx = 0; bx < nls; ++bx) {
            bool o = false;
            for (int ly = 0; ly < ls && !o; ++ly)
                for (int lx = 0; lx < ls; ++lx)
                    if (mask[static_cast<std::size_t>(by * ls + ly) * Ns +
                             bx * ls + lx]) {
                        o = true;
                        break;
                    }
            if (o)
                for (int ly = 0; ly < ls; ++ly)
                    for (int lx = 0; lx < ls; ++lx)
                        occ[static_cast<std::size_t>(by * ls + ly) * Ns +
                            bx * ls + lx] = 1;
        }
    return occ;
}

template <class Vec>
int count_mismatches(const Vec& a, const Vec& b,
                     const std::vector<std::uint8_t>& where) {
    int bad = 0;
    for (int i = 0; i < a.size(); ++i)
        if (where[i] && a(i) != b(i)) ++bad;
    return bad;
}

// Deterministic rough multi-scale gap: superposed cosines with random phases
// and directions, amplitudes k^-(1+H), wavenumbers up to the fine Nyquist so
// the coarse nested levels cannot see the smallest asperities (the orphan-
// island regime that makes the candidate-set gap threshold necessary).
Eigen::VectorXd rough_gap(int Ns, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uang(0.0, 2.0 * M_PI);
    Eigen::VectorXd h = Eigen::VectorXd::Zero(static_cast<long>(Ns) * Ns);
    const double H = 0.8;
    for (double kmag = 1.0; kmag <= Ns / 2; kmag *= 1.35) {
        for (int rep = 0; rep < 4; ++rep) {
            const double th = uang(rng), phi = uang(rng);
            const double kx = kmag * std::cos(th), ky = kmag * std::sin(th);
            const double amp = std::pow(kmag, -(1.0 + H));
            for (int iy = 0; iy < Ns; ++iy)
                for (int ix = 0; ix < Ns; ++ix)
                    h(static_cast<long>(iy) * Ns + ix) +=
                        amp * std::cos(2.0 * M_PI *
                                           (kx * (ix + 0.5) + ky * (iy + 0.5)) /
                                           Ns +
                                       phi);
        }
    }
    h.array() -= h.mean();
    h *= 0.02 / std::sqrt(h.squaredNorm() / h.size()); // rms 0.02
    return -h; // gap of a rigid flat against the surface
}

} // namespace

int main() {
    const int Ns = 256, ls = 8, q = 4;
    const std::size_t N = static_cast<std::size_t>(Ns) * Ns;
    hmc::BoussinesqKernel kernel(Ns, 1.0, 1.0);
    const hmc::H2Params hp{ls, q, 1};

    const auto mask = clustered_mask(Ns, 7);
    std::size_t nc = 0;
    for (auto v : mask) nc += v;
    std::printf("mask: N_c/N = %.4f\n", double(nc) / double(N));
    CHECK(nc > 0 && nc < N / 4);

    // x supported on the mask
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> val(-1.0, 1.0);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(N);
    for (std::size_t i = 0; i < N; ++i)
        if (mask[i]) x(i) = val(rng);

    // unmasked reference
    hmc::H2Operator ref_op(kernel, hp);
    ref_op.build();
    Eigen::VectorXd y_ref;
    ref_op.matvec_into(x, y_ref);

    const auto src = ref_op.build_mask(mask);
    // index-list overload must agree with the grid-mask overload
    {
        std::vector<int> idx;
        for (std::size_t i = 0; i < N; ++i)
            if (mask[i]) idx.push_back(static_cast<int>(i));
        const auto src2 = ref_op.build_mask(idx.data(), idx.size());
        CHECK(src.box_occ == src2.box_occ);
    }

    const std::vector<std::uint8_t> everywhere(N, 1);
    const auto on_tgt_leaves = occupied_leaf_elements(mask, Ns, ls);

    // A: masked source, all targets (verification mode) — bit-for-bit
    // everywhere, on a fresh operator (garbage scratch).
    {
        hmc::H2Operator op(kernel, hp);
        op.build();
        Eigen::VectorXd y;
        op.matvec_masked_into(x, y, src, nullptr);
        const int bad = count_mismatches(y, y_ref, everywhere);
        std::printf("double, tgt=all:  %d mismatches\n", bad);
        CHECK(bad == 0);
    }

    // B: masked source and target (restricted CG mode) — bit-for-bit on
    // tgt-occupied leaves; entries outside are unspecified.
    {
        hmc::H2Operator op(kernel, hp);
        op.build();
        Eigen::VectorXd y;
        op.matvec_masked_into(x, y, src, &src);
        const int bad = count_mismatches(y, y_ref, on_tgt_leaves);
        std::printf("double, tgt=src:  %d mismatches\n", bad);
        CHECK(bad == 0);
    }

    // C: full mask — the guards must not perturb the standard path.
    {
        hmc::H2Operator op(kernel, hp);
        op.build();
        const auto full = op.build_mask(everywhere);
        Eigen::VectorXd y;
        op.matvec_masked_into(x, y, full, &full);
        const int bad = count_mismatches(y, y_ref, everywhere);
        std::printf("double, full:     %d mismatches\n", bad);
        CHECK(bad == 0);
    }

    // float mirrors of A and B
    const Eigen::VectorXf xf = x.cast<float>();
    Eigen::VectorXf yf_ref;
    ref_op.matvec_single_into(xf, yf_ref);
    {
        hmc::H2Operator op(kernel, hp);
        op.build();
        Eigen::VectorXf y;
        op.matvec_masked_single_into(xf, y, src, nullptr);
        const int bad = count_mismatches(y, yf_ref, everywhere);
        std::printf("float,  tgt=all:  %d mismatches\n", bad);
        CHECK(bad == 0);
    }
    {
        hmc::H2Operator op(kernel, hp);
        op.build();
        Eigen::VectorXf y;
        op.matvec_masked_single_into(xf, y, src, &src);
        const int bad = count_mismatches(y, yf_ref, on_tgt_leaves);
        std::printf("float,  tgt=src:  %d mismatches\n", bad);
        CHECK(bad == 0);
    }

    std::printf("test_active (M1): all passed\n");

    // ── M2: active-set nested driver equivalence ─────────────────────────────
    {
        const int Nr = 256;
        const double pbar = 0.005, tol = 1e-8;
        const Eigen::VectorXd gap = rough_gap(Nr, 42);

        hmc::NestedParams np_std;
        np_std.coarsest = 64;
        auto r0 = hmc::solve_contact_nested(Nr, 1.0, 1.0, gap, pbar, tol,
                                            20000, true, np_std);
        CHECK(r0.converged);

        hmc::NestedParams np_act = np_std;
        np_act.active_set = true;
        auto r1 = hmc::solve_contact_nested(Nr, 1.0, 1.0, gap, pbar, tol,
                                            20000, true, np_act);
        CHECK(r1.converged);
        CHECK(!r1.active_fallback); // clean certificate
        CHECK(r1.active_rounds >= 1 && r1.active_rounds <= 2);
        const double rel =
            (r1.pressure - r0.pressure).norm() / r0.pressure.norm();
        std::printf("active f64: std %d it | active %d it, %d rounds, "
                    "relL2 %.2e, dArea %.2e\n",
                    r0.iterations, r1.iterations, r1.active_rounds, rel,
                    std::abs(r1.contact_fraction - r0.contact_fraction));
        CHECK(rel <= 1e-6);
        CHECK(std::abs(r1.contact_fraction - r0.contact_fraction) <= 1e-6);
        // full result fields present (not light) and consistent
        CHECK(r1.displacement.size() == r0.displacement.size());
        CHECK(r1.gap.size() == r0.gap.size());
        CHECK((r1.gap - r0.gap).norm() <= 1e-6 * (r0.gap.norm() + 1.0));

        // float precision: same solution to float accuracy, clean certificate
        hmc::NestedParams np_actf = np_act;
        np_actf.single_precision = true;
        auto r2 = hmc::solve_contact_nested(Nr, 1.0, 1.0, gap, pbar, tol,
                                            20000, true, np_actf);
        CHECK(r2.converged);
        CHECK(!r2.active_fallback);
        const double relf =
            (r2.pressure - r0.pressure).norm() / r0.pressure.norm();
        std::printf("active f32: %d it, %d rounds, relL2 vs f64 %.2e\n",
                    r2.iterations, r2.active_rounds, relf);
        CHECK(relf <= 1e-3);
        CHECK(std::abs(r2.contact_fraction - r0.contact_fraction) <= 2e-3);

        // delta-too-tight regression: a candidate set with no gap threshold
        // and no halo misses points; with a single round allowed the driver
        // must fall back to the full solve — and still return the right
        // pressures — rather than certify a wrong active set.
        hmc::NestedParams np_bad = np_act;
        np_bad.active_delta = 0.0;
        np_bad.active_halo = 0;
        np_bad.active_max_rounds = 1;
        auto r3 = hmc::solve_contact_nested(Nr, 1.0, 1.0, gap, pbar, tol,
                                            20000, true, np_bad);
        CHECK(r3.converged);
        CHECK(r3.active_fallback); // the broken candidate set was caught
        const double rel3 =
            (r3.pressure - r0.pressure).norm() / r0.pressure.norm();
        std::printf("active fallback: triggered, %d it total, relL2 %.2e, "
                    "dArea %.2e\n",
                    r3.iterations, rel3,
                    std::abs(r3.contact_fraction - r0.contact_fraction));
        // the fallback takes a different iterate path than the reference, so
        // boundary pixels scatter at the solver-tolerance scale (~1e-6, same
        // order as the validated none-vs-fourier path difference); the broken
        // pre-fix state returned 1.5e-2
        CHECK(rel3 <= 1e-4);
        CHECK(std::abs(r3.contact_fraction - r0.contact_fraction) <= 1e-4);
    }

    std::printf("test_active (M2): all passed\n");
    return 0;
}
