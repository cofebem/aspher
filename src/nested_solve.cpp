#include "nested_solve.hpp"

#include "boussinesq_kernel.hpp"
#include "fft_operator.hpp"
#include "fourier_precond.hpp"
#include "h2_operator.hpp"

#include <algorithm>
#include <atomic>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <cmath>
#include <limits>

namespace hmc {

// 2x2 block-average restriction: fine (Ns x Ns) -> coarse (Ns/2 x Ns/2).
static Eigen::VectorXd restrict_field(Eigen::Ref<const Eigen::VectorXd> f,
                                      int Ns) {
    const int c = Ns / 2;
    Eigen::VectorXd out(c * c);
#pragma omp parallel for schedule(static)
    for (int iy = 0; iy < c; ++iy)
        for (int ix = 0; ix < c; ++ix) {
            const int x0 = 2 * ix, y0 = 2 * iy;
            out(iy * c + ix) = 0.25 * (f((y0) * Ns + x0) + f((y0) * Ns + x0 + 1) +
                                       f((y0 + 1) * Ns + x0) + f((y0 + 1) * Ns + x0 + 1));
        }
    return out;
}

// Injection prolongation: coarse (Nc x Nc) -> fine (2Nc x 2Nc), replicate cells.
static Eigen::VectorXd prolong_field(const Eigen::VectorXd& pc, int Nc) {
    const int f = 2 * Nc;
    Eigen::VectorXd out(f * f);
#pragma omp parallel for schedule(static)
    for (int iy = 0; iy < Nc; ++iy)
        for (int ix = 0; ix < Nc; ++ix) {
            const double v = pc(iy * Nc + ix);
            const int x0 = 2 * ix, y0 = 2 * iy;
            out((y0) * f + x0) = v;
            out((y0) * f + x0 + 1) = v;
            out((y0 + 1) * f + x0) = v;
            out((y0 + 1) * f + x0 + 1) = v;
        }
    return out;
}

// Chebyshev (square) dilation of a grid mask by radius r: separable max
// filter, horizontal then vertical pass. Slightly more generous than the
// prototype's cross-shaped dilation — the safe direction for candidate sets.
static void dilate_mask(std::vector<std::uint8_t>& m, int Ns, int r) {
    if (r <= 0) return;
    std::vector<std::uint8_t> tmp(m.size());
#pragma omp parallel for schedule(static)
    for (int iy = 0; iy < Ns; ++iy) {
        const std::uint8_t* row = m.data() + static_cast<std::ptrdiff_t>(iy) * Ns;
        std::uint8_t* out = tmp.data() + static_cast<std::ptrdiff_t>(iy) * Ns;
        for (int ix = 0; ix < Ns; ++ix) {
            std::uint8_t v = 0;
            const int x1 = std::min(Ns - 1, ix + r);
            for (int x = std::max(0, ix - r); x <= x1 && !v; ++x) v = row[x];
            out[ix] = v;
        }
    }
#pragma omp parallel for schedule(static)
    for (int iy = 0; iy < Ns; ++iy) {
        std::uint8_t* out = m.data() + static_cast<std::ptrdiff_t>(iy) * Ns;
        const int y1 = std::min(Ns - 1, iy + r);
        for (int ix = 0; ix < Ns; ++ix) {
            std::uint8_t v = 0;
            for (int y = std::max(0, iy - r); y <= y1 && !v; ++y)
                v = tmp[static_cast<std::ptrdiff_t>(y) * Ns + ix];
            out[ix] = v;
        }
    }
}

// Finest-level active-set solve (plan doc/plans/2026-07-10-active-set-solver.md
// M2 driver + M3 O(N_c) state): candidate set from the coarse level,
// restricted Polonsky-Keer on COMPRESSED slot-blocked vectors through the
// compressed masked H2 matvec, streamed full-grid verification per round
// (per-leaf tiles, no N-sized u), violation-extension, and a full-solve
// fallback after active_max_rounds. p_init_d (the prolonged coarse pressure)
// is consumed. coarse_gap is the NEXT-TO-FINEST level's gap field ((Ns/2)²
// entries), sampled by injection — the prolonged fine-grid copy is never
// materialised (an N-sized array at Ns=16384 is 2.1 GiB).
template <class Real>
static ContactResult active_finest(const H2Operator& h2,
                                   const FourierPreconditioner* fp,
                                   Eigen::Ref<const VecT<Real>> g0,
                                   double p_bar, double lvl_tol, int max_iter,
                                   bool use_pr, const NestedParams& np, int Ns,
                                   Eigen::VectorXd& p_init_d,
                                   const Eigen::VectorXd& coarse_gap,
                                   bool record_history, bool light,
                                   double g_ref, double datum, double gsub) {
    using Vec = VecT<Real>;
    constexpr bool is_double = std::is_same_v<Real, double>;
    const int N = Ns * Ns;
    const int Nc2 = Ns / 2;

    const double gmax = static_cast<double>(g0.maxCoeff());
    const double gmin = static_cast<double>(g0.minCoeff());
    const double g_scale = (gmax > gmin) ? gmax - gmin : 1.0;
    const double delta = np.active_delta * g_scale;
    // violation threshold mirrors the prototype: negative gap outside C
    // beyond the solve tolerance × the gap scale
    const double vthresh = -lvl_tol * g_scale;

    // candidate set C = dilate(prolonged coarse contact) ∪ {coarse gap < δ}
    std::vector<std::uint8_t> cmask(N, 0);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) cmask[i] = (p_init_d(i) > 0.0) ? 1 : 0;
    dilate_mask(cmask, Ns, np.active_halo);
#pragma omp parallel for schedule(static)
    for (int iy = 0; iy < Ns; ++iy) {
        const double* gc =
            coarse_gap.data() + static_cast<std::ptrdiff_t>(iy / 2) * Nc2;
        std::uint8_t* cm = cmask.data() + static_cast<std::ptrdiff_t>(iy) * Ns;
        for (int ix = 0; ix < Ns; ++ix)
            if (gc[ix / 2] < delta) cm[ix] = 1;
    }

    // ── O(N_c) compressed state (M3): everything below works on slot-blocked
    // vectors of nslots·ls² entries; no N-sized Real vector is allocated on
    // the certified path (cmask/viol are N bytes, the FFT preconditioner
    // keeps its full grid — the documented trade-offs).
    const int ls2 = h2.info().leaf_side * h2.info().leaf_side;

    H2Mask mask = h2.build_mask(cmask);
    std::vector<int> gi = h2.slot_grid_indices(mask); // slot entry → grid index
    std::ptrdiff_t S = static_cast<std::ptrdiff_t>(gi.size());

    // compressed gap, candidate positions
    Vec g0c(S);
    std::vector<int> cidx;
    const Real gsub_r = static_cast<Real>(gsub);
    auto gather_level = [&] {
        g0c.resize(S);
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t k = 0; k < S; ++k) g0c(k) = g0(gi[k]) - gsub_r;
        cidx.clear();
        for (std::ptrdiff_t k = 0; k < S; ++k)
            if (cmask[gi[k]]) cidx.push_back(static_cast<int>(k));
    };
    gather_level();

    // compressed warm start; the driver's full double array is consumed here
    Vec p0(S);
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t k = 0; k < S; ++k)
        p0(k) = static_cast<Real>(p_init_d(gi[k]));
    p_init_d.resize(0);

    PrecondIntoT<Real> pc;
    if (fp) {
        pc = [fp, &gi](const Vec& g, const std::vector<std::uint8_t>& contact,
                       Vec& z) {
            if constexpr (is_double) fp->apply_into_indexed(g, contact, gi, z);
            else fp->apply_single_into_indexed(g, contact, gi, z);
        };
    }

    ContactResult res;
    std::vector<double> hist;
    std::vector<std::uint8_t> viol(N);
    int it_total = 0, rounds = 0;
    long long verify_mv = 0;
    bool certified = false;

    while (rounds < np.active_max_rounds) {
        ++rounds;
        MatVecIntoT<Real> mv = [&h2, &mask](const Vec& x, Vec& y) {
            if constexpr (is_double)
                h2.matvec_masked_compressed_into(x, y, mask);
            else
                h2.matvec_masked_compressed_single_into(x, y, mask);
        };
        SolveOptions aopt;
        // Split the level budget between the restricted certificate and the
        // outside-C penetration allowance so that their SUM meets the level
        // tolerance: with every outside gap >= -tol/2 * g_ref the global
        // certificate gains at most P*(tol/2)*g_ref, i.e. tol/2 in normalised
        // units, on top of the restricted certificate's tol/2 (spec A03).
        aopt.tol = 0.5 * lvl_tol;
        aopt.max_iter = max_iter;
        aopt.use_pr = use_pr;
        aopt.record_history = record_history;
        aopt.allow_tolerance_relaxation = true; // level policy, set by caller
        aopt.n_grid = N;
        aopt.scales.g_ref = g_ref;
        aopt.scales.datum_mode = SolveScales::DatumMode::caller;
        aopt.scales.datum = datum;
        RestrictedCertificate cert;
        res = solve_contact_active_impl<Real>(mv, g0c, static_cast<Real>(p_bar),
                                              aopt, pc, cidx, &p0, &cert);
        it_total += res.iterations;
        if (record_history)
            hist.insert(hist.end(), res.error_history.begin(),
                        res.error_history.end());

        // verification: streamed full-target matvec from the compressed
        // pressure — per-leaf ls² tiles, no N-sized u
        Vec pr; // working-precision pressure (res.pressure itself for double)
        const Vec* psrc;
        if constexpr (is_double) {
            psrc = &res.pressure;
        } else {
            pr = res.pressure.template cast<Real>();
            psrc = &pr;
        }
        const double a = res.approach - datum; // centred approach
        const int ls = h2.info().leaf_side;
        std::atomic<long> nviol{0};
        // The stream visits EVERY leaf, so this pass sees all physical
        // targets — candidates included — and yields the GLOBAL minimum
        // gradient, which is what turns the restricted certificate into a
        // global one (spec A03 step 3). The old code scanned only the points
        // outside C, so a violation that expansion had just pulled INTO C
        // could never be seen again.
        std::atomic<double> vmin_g{std::numeric_limits<double>::infinity()};
        auto sink = [&](int ix0, int iy0, const Real* tile) {
            long local = 0;
            double lmin = std::numeric_limits<double>::infinity();
            for (int ly = 0, t = 0; ly < ls; ++ly) {
                const std::ptrdiff_t row =
                    static_cast<std::ptrdiff_t>(iy0 + ly) * Ns + ix0;
                for (int lx = 0; lx < ls; ++lx, ++t) {
                    const std::ptrdiff_t i = row + lx;
                    const double v = static_cast<double>(tile[t]) +
                                     (static_cast<double>(g0(i)) - gsub);
                    if (v < lmin) lmin = v;
                    if (!cmask[i] && v - a < vthresh) {
                        viol[i] = 1;
                        ++local;
                    }
                }
            }
            if (local) nviol.fetch_add(local, std::memory_order_relaxed);
            double cur = vmin_g.load(std::memory_order_relaxed);
            while (lmin < cur &&
                   !vmin_g.compare_exchange_weak(cur, lmin,
                                                 std::memory_order_relaxed)) {
            }
        };
        std::fill(viol.begin(), viol.end(), 0);
        if constexpr (is_double) h2.matvec_masked_stream(*psrc, mask, sink);
        else h2.matvec_masked_stream_single(*psrc, mask, sink);
        ++verify_mv;

        // Global certificate from the restricted one and the streamed global
        // minimum. Written as G_local + P (vmin_local - vmin_global): both
        // terms are non-negative, so no large near-equal totals are
        // subtracted (spec §5.2).
        const double vmin_glob = vmin_g.load();
        const double G_glob =
            res.fw_gap + cert.P_total * (cert.vmin_local - vmin_glob);
        const double fw_glob = G_glob / (cert.P_total * cert.g_ref);
        const double pen_glob = std::max(0.0, -(vmin_glob - a)) / cert.g_ref;
        res.fw_gap = G_glob;
        res.fw_error = fw_glob;
        res.penetration_error = pen_glob;
        res.effective_tol = lvl_tol;

        // Accept only when the restricted solve met its own KKT conditions
        // AND the global check passes: an empty outside-violation list is not
        // by itself a certificate.
        if (res.status == SolveStatus::converged && nviol.load() == 0 &&
            fw_glob <= lvl_tol && pen_glob <= lvl_tol) {
            certified = true;
            break;
        }
        if (nviol.load() == 0) {
            // nothing to expand but still not certified (the restricted solve
            // itself did not converge, or the global gap exceeds the budget):
            // the candidate set cannot be improved, so go to the fallback.
            break;
        }
        // extend C with the dilated violations, rebuild the compressed
        // layout (old slots are a subset of the new ones), remap the warm
        // start slot-block-wise, and resume
        dilate_mask(viol, Ns, np.active_halo);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i)
            if (viol[i]) cmask[i] = 1;

        H2Mask old_mask = std::move(mask);
        mask = h2.build_mask(cmask);
        gi = h2.slot_grid_indices(mask);
        S = static_cast<std::ptrdiff_t>(gi.size());
        gather_level();

        Vec pnew = Vec::Zero(S);
        const int nold = old_mask.nslots();
#pragma omp parallel for schedule(static)
        for (int s = 0; s < nold; ++s) {
            const int sn = mask.leaf_slot[old_mask.slot_leaf[s]];
            pnew.segment(static_cast<std::ptrdiff_t>(sn) * ls2, ls2) =
                psrc->segment(static_cast<std::ptrdiff_t>(s) * ls2, ls2);
        }
        p0 = std::move(pnew);
    }

    if (!certified) {
        // safety net: the candidate set never certified — run the standard
        // FULL-GRID solve, warm-started from the last restricted iterate.
        // This path allocates the full CG state (it is the memory-disaster
        // escape hatch, not the normal route).
        MatVecIntoT<Real> mvf = [&h2](const Vec& x, Vec& y) {
            if constexpr (is_double) h2.matvec_into(x, y);
            else h2.matvec_single_into(x, y);
        };
        PrecondIntoT<Real> pcf;
        if (fp) {
            pcf = [fp](const Vec& g, const std::vector<std::uint8_t>& contact,
                       Vec& z) {
                if constexpr (is_double) fp->apply_into(g, contact, z);
                else fp->apply_single_into(g, contact, z);
            };
        }
        // scatter the last restricted solution (p0: the loop only exits
        // uncertified through the violations branch, which reassigns p0 on
        // the CURRENT slot layout) to the full grid.
        Vec pf = Vec::Zero(N);
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t k = 0; k < S; ++k) pf(gi[k]) = p0(k);
        p0.resize(0);
        // No violation seeding: the old code had to put pressure on the
        // penetrating points because the Σ p|g| metric was blind to
        // p = 0 ∧ g < 0 and would "converge" at iteration 0. The certified
        // stopping rule (A01) sees that penetration directly and the
        // identification step activates those points, so correctness no
        // longer depends on the seed.
        SolveOptions fopt;
        fopt.tol = lvl_tol;
        fopt.max_iter = max_iter;
        fopt.use_pr = use_pr;
        fopt.light = light;
        fopt.record_history = record_history;
        fopt.keep_best = !light;
        fopt.allow_tolerance_relaxation = true;
        fopt.scales.g_ref = g_ref;
        fopt.scales.datum_mode = (gsub != 0.0)
                                     ? SolveScales::DatumMode::solver
                                     : SolveScales::DatumMode::caller;
        fopt.scales.datum = datum;
        res = solve_contact_impl<Real>(mvf, g0, static_cast<Real>(p_bar), fopt,
                                       pcf, &pf);
        it_total += res.iterations;
        if (record_history)
            hist.insert(hist.end(), res.error_history.begin(),
                        res.error_history.end());
        res.active_fallback = true;
    } else {
        // certified: free the preconditioner's full-grid FFT scratch before
        // materialising any full field, then scatter the compressed pressure
        if (fp) fp->release_scratch();
        Eigen::VectorXd pfull = Eigen::VectorXd::Zero(N);
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t k = 0; k < S; ++k) pfull(gi[k]) = res.pressure(k);
        if (!light) {
            // full displacement/gap fields: one more streamed pass writing
            // the tiles into the result arrays
            Vec pr;
            const Vec* psrc;
            if constexpr (is_double) {
                psrc = &res.pressure;
            } else {
                pr = res.pressure.template cast<Real>();
                psrc = &pr;
            }
            Eigen::VectorXd disp(N), gp(N);
            const double a = res.approach - datum; // centred approach
            const int ls = h2.info().leaf_side;
            auto fill = [&](int ix0, int iy0, const Real* tile) {
                for (int ly = 0, t = 0; ly < ls; ++ly) {
                    const std::ptrdiff_t row =
                        static_cast<std::ptrdiff_t>(iy0 + ly) * Ns + ix0;
                    for (int lx = 0; lx < ls; ++lx, ++t) {
                        const std::ptrdiff_t i = row + lx;
                        disp(i) = static_cast<double>(tile[t]);
                        gp(i) =
                            disp(i) + (static_cast<double>(g0(i)) - gsub) - a;
                    }
                }
            };
            if constexpr (is_double) h2.matvec_masked_stream(*psrc, mask, fill);
            else h2.matvec_masked_stream_single(*psrc, mask, fill);
            res.displacement = std::move(disp);
            res.gap = std::move(gp);
        }
        res.pressure = std::move(pfull);
    }
    res.active_rounds = rounds;
    res.iterations = it_total;
    res.verification_matvec_count = verify_mv;
    // An uncertified restricted result must never be reported as success. The
    // full-solve fallback carries its own global certificate, so its status
    // stands as returned.
    if (!certified && !res.active_fallback &&
        res.status == SolveStatus::converged) {
        res.status = SolveStatus::verification_failed;
        res.status_reason = "global_check";
    }
    res.converged = (res.status == SolveStatus::converged);
    if (record_history) res.error_history = std::move(hist);
    return res;
}

ContactResult solve_contact_nested(int Ns, double L, double E_star,
                                   Eigen::Ref<const Eigen::VectorXd> g0,
                                   double p_bar,
                                   double tol, int max_iter, bool use_pr,
                                   const NestedParams& np) {
    // ── input validation before any allocation or iteration (spec A19) ─────
    auto pow2 = [](int v) { return v > 0 && (v & (v - 1)) == 0; };
    if (Ns <= 0)
        throw std::invalid_argument("solve_contact_nested: Ns must be positive");
    if (!(L > 0.0) || !std::isfinite(L))
        throw std::invalid_argument("solve_contact_nested: L must be positive");
    if (!(E_star > 0.0) || !std::isfinite(E_star))
        throw std::invalid_argument(
            "solve_contact_nested: E_star must be positive");
    // checked size arithmetic: N and the FFT/H2 padded products must fit the
    // index types before anything of that size is allocated
    const std::int64_t N64 = static_cast<std::int64_t>(Ns) * Ns;
    if (N64 > static_cast<std::int64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(
            "solve_contact_nested: Ns*Ns exceeds the int index range");
    if (4 * N64 > static_cast<std::int64_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        throw std::invalid_argument(
            "solve_contact_nested: the padded (2Ns)^2 grid exceeds ptrdiff_t");
    if (static_cast<std::int64_t>(g0.size()) != N64)
        throw std::invalid_argument("solve_contact_nested: g0 size != Ns*Ns");
    if (!g0.allFinite())
        throw std::invalid_argument(
            "solve_contact_nested: gap contains non-finite values");
    if (!(p_bar > 0.0) || !std::isfinite(p_bar))
        throw std::invalid_argument(
            "solve_contact_nested: p_nominal must be positive");
    if (!(tol > 0.0) || !std::isfinite(tol))
        throw std::invalid_argument("solve_contact_nested: tol must be positive");
    if (max_iter < 0)
        throw std::invalid_argument("solve_contact_nested: max_iter must be >= 0");
    // coarsest = 0 made the level loop non-progressing (n *= 2 never leaves 0)
    if (np.coarsest <= 0)
        throw std::invalid_argument(
            "solve_contact_nested: coarsest must be positive");
    if (np.coarsest > Ns)
        throw std::invalid_argument("solve_contact_nested: coarsest > Ns");
    if (np.q < 2)
        throw std::invalid_argument("solve_contact_nested: q must be >= 2");
    if (np.leaf_side <= 0 || !pow2(np.leaf_side))
        throw std::invalid_argument(
            "solve_contact_nested: leaf_side must be a positive power of two");
    if (!(np.coarse_tol > 0.0))
        throw std::invalid_argument(
            "solve_contact_nested: coarse_tol must be positive");
    if (np.active_set) {
        if (np.active_halo < 0)
            throw std::invalid_argument(
                "solve_contact_nested: active_halo must be >= 0");
        if (np.active_max_rounds < 1)
            throw std::invalid_argument(
                "solve_contact_nested: active_max_rounds must be >= 1");
        if (!(np.active_delta >= 0.0))
            throw std::invalid_argument(
                "solve_contact_nested: active_delta must be >= 0");
    }

    std::vector<int> levels;
    for (int n = np.coarsest; n <= Ns; n *= 2) levels.push_back(n);
    if (levels.empty() || levels.back() != Ns)
        throw std::invalid_argument(
            "solve_contact_nested: Ns must equal coarsest * 2^k");
    if (np.backend != "h2" && np.backend != "fft")
        throw std::invalid_argument(
            "solve_contact_nested: backend must be 'h2' or 'fft'");
    if (np.active_set) {
        if (np.backend != "h2")
            throw std::invalid_argument(
                "solve_contact_nested: active_set requires backend='h2' "
                "(masked matvec)");
        if (levels.size() < 2)
            throw std::invalid_argument(
                "solve_contact_nested: active_set needs a coarse level "
                "(Ns > coarsest)");
    }

    // restrict the fine gap down to the coarse levels only: the finest level
    // solves directly on the caller-owned g0 view, so no N-sized copy is made
    // (at Ns=16384 double that's a 2.1 GiB array; the coarse levels total ~N/3).
    const int ncoarse = static_cast<int>(levels.size()) - 1;
    std::vector<Eigen::VectorXd> gap(ncoarse);
    if (ncoarse > 0) {
        gap[ncoarse - 1] = restrict_field(g0, Ns);
        for (int i = ncoarse - 2; i >= 0; --i)
            gap[i] = restrict_field(gap[i + 1], levels[i + 1]);
    }

    // One consistent physical scale and gap datum for the whole hierarchy
    // (spec §3.1/A04): level diagnostics are only comparable when they use the
    // same normalisation, and every level's float cast must remove the same
    // additive datum. Both come from the finest gap, in double.
    const double g_min_f = g0.minCoeff(), g_max_f = g0.maxCoeff();
    const double datum = g_min_f + 0.5 * (g_max_f - g_min_f);
    const double g_ref = std::max(g_max_f - g_min_f, p_bar * L / E_star);

    Eigen::VectorXd p_init;
    Eigen::VectorXd coarse_gap; // next-to-finest gap field (active_set only)
    bool have_init = false;
    ContactResult res;

    for (std::size_t li = 0; li < levels.size(); ++li) {
        const int n = levels[li];
        // A06: the FFT backend needs the full Love table (it transforms it);
        // the H2 backend never touches it after build(), so it is built
        // through the compact near-offset path instead — 8*n^2 bytes saved at
        // every level (2 GiB at Ns=16384).
        std::unique_ptr<BoussinesqKernel> kernel;
        std::unique_ptr<H2Operator> h2;
        std::unique_ptr<FFTOperator> fop;
        MatVecIntoT<double> mv;
        if (np.backend == "fft") {
            kernel = std::make_unique<BoussinesqKernel>(n, L, E_star);
            fop = std::make_unique<FFTOperator>(*kernel);
            fop->build();
            mv = [&fop](const Eigen::VectorXd& v, Eigen::VectorXd& out) {
                fop->matvec_into(v, out);
            };
        } else {
            h2 = make_boussinesq_h2(n, L, E_star,
                                    H2Params{np.leaf_side, np.q, 1});
            h2->build();
            mv = [&h2](const Eigen::VectorXd& v, Eigen::VectorXd& out) {
                h2->matvec_into(v, out);
            };
        }

        FourierPreconditioner fp(n);
        PrecondIntoT<double> pc;
        if (np.precond)
            pc = [&fp](const Eigen::VectorXd& g,
                       const std::vector<std::uint8_t>& contact,
                       Eigen::VectorXd& z) { fp.apply_into(g, contact, z); };

        const bool finest = (li + 1 == levels.size());
        // finest level reads the caller-owned g0 directly; coarse levels own
        // their restricted copy (freed right after their solve).
        Eigen::Ref<const Eigen::VectorXd> glvl =
            finest ? Eigen::Ref<const Eigen::VectorXd>(g0)
                   : Eigen::Ref<const Eigen::VectorXd>(gap[li]);
        double lvl_tol = finest ? tol : np.coarse_tol;
        // float arithmetic cannot drive the complementarity error below ~1e-6,
        // so clamp the requested tolerance to a reachable floor in that mode.
        if (np.single_precision) lvl_tol = std::max(lvl_tol, 2e-6);
        // coarse levels only need the pressure (for prolongation), so drop
        // their displacement/gap; the finest honours light_result. Exception:
        // the active-set candidate set needs the next-to-finest gap field
        // ((Ns/2)²-sized, ~N/4 — negligible next to the finest solve).
        const bool keep_gap = np.active_set && (li + 2 == levels.size());
        const bool light = finest ? np.light_result : !keep_gap;
        // only the finest level's trace is meaningful (iterations is also
        // finest-level-only); coarse levels never record history.
        const bool record_history = finest && np.record_error_history;

        SolveOptions lopt;
        lopt.tol = lvl_tol;
        lopt.requested_tol = finest ? tol : np.coarse_tol;
        lopt.max_iter = max_iter;
        lopt.use_pr = use_pr;
        lopt.light = light;
        lopt.record_history = record_history;
        lopt.keep_best = !light;
        // the float floor and the coarse-level tolerances are deliberate
        // stage policies, not silent relaxations of the user's target: the
        // finest double stage carries the requested tolerance strictly.
        lopt.allow_tolerance_relaxation = !finest || np.single_precision;
        lopt.scales.g_ref = g_ref;
        lopt.scales.datum = datum;
        // double levels read their gap in place and subtract the datum on the
        // fly (no extra N-sized buffer); float levels get it removed inside
        // the cast, before any information can be rounded away.
        lopt.scales.datum_mode = np.single_precision
                                     ? SolveScales::DatumMode::caller
                                     : SolveScales::DatumMode::solver;

        if (finest && np.active_set) {
            // restricted (active-set) solve on the candidate set built from
            // the coarse contact + gap; p_init/coarse_gap are consumed
            const FourierPreconditioner* fpp = np.precond ? &fp : nullptr;
            if (np.single_precision) {
                h2->build_single_caches();
                // centre inside the cast: rounding a large offset first is
                // exactly what destroyed the float solution (review §4)
                Eigen::VectorXf g0f = (glvl.array() - datum).cast<float>();
                res = active_finest<float>(*h2, fpp, g0f, p_bar, lvl_tol,
                                           max_iter, use_pr, np, n, p_init,
                                           coarse_gap, record_history,
                                           np.light_result, g_ref, datum, 0.0);
            } else {
                res = active_finest<double>(*h2, fpp, glvl, p_bar, lvl_tol,
                                            max_iter, use_pr, np, n, p_init,
                                            coarse_gap, record_history,
                                            np.light_result, g_ref, datum,
                                            datum);
            }
            coarse_gap.resize(0);
        } else if (np.single_precision) {
            MatVecIntoT<float> mvf;
            if (fop) {
                fop->build_single_caches();
                mvf = [&fop](const Eigen::VectorXf& v, Eigen::VectorXf& out) {
                    fop->matvec_single_into(v, out);
                };
            } else {
                h2->build_single_caches();
                mvf = [&h2](const Eigen::VectorXf& v, Eigen::VectorXf& out) {
                    h2->matvec_single_into(v, out);
                };
            }
            PrecondIntoT<float> pcf;
            if (np.precond)
                pcf = [&fp](const Eigen::VectorXf& g,
                            const std::vector<std::uint8_t>& contact,
                            Eigen::VectorXf& z) {
                    fp.apply_single_into(g, contact, z);
                };
            Eigen::VectorXf g0f = (glvl.array() - datum).cast<float>();
            // free the coarse double gap once cast to float (the finest-level
            // double buffer is caller-owned and stays alive)
            if (!finest) gap[li].resize(0);
            Eigen::VectorXf p0f;
            if (have_init) {
                p0f = p_init.cast<float>();
                // free the double warm start once cast: it would otherwise sit
                // idle (2.1 GiB at Ns=16384) through the whole float solve.
                // p0f itself is consumed by solve_contact_impl (moved into the
                // pressure iterate), so neither copy outlives initialization.
                p_init.resize(0);
            }
            res = solve_contact_impl<float>(mvf, g0f, static_cast<float>(p_bar),
                                            lopt, pcf,
                                            have_init ? &p0f : nullptr);
        } else {
            res = solve_contact_impl<double>(mv, glvl, p_bar, lopt, pc,
                                             have_init ? &p_init : nullptr);
            if (!finest) gap[li].resize(0);
        }

        if (!finest) {
            p_init = prolong_field(res.pressure, n);
            have_init = true;
            if (keep_gap) coarse_gap = std::move(res.gap);
        }
    }
    return res;
}

} // namespace hmc
