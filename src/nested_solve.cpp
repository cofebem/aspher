#include "nested_solve.hpp"

#include "boussinesq_kernel.hpp"
#include "fft_operator.hpp"
#include "fourier_precond.hpp"
#include "h2_operator.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef __GLIBC__
#include <malloc.h>
#endif

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

ContactResult solve_contact_nested(int Ns, double L, double E_star,
                                   Eigen::Ref<const Eigen::VectorXd> g0,
                                   double p_bar,
                                   double tol, int max_iter, bool use_pr,
                                   const NestedParams& np) {
    if (static_cast<int>(g0.size()) != Ns * Ns)
        throw std::invalid_argument("solve_contact_nested: g0 size != Ns*Ns");

#ifdef __GLIBC__
    // The matvec and preconditioner allocate large temporaries every iteration
    // (M/L buffers, the y vector, the FFT arrays). Force allocations above
    // 128 KB to use mmap so free() returns them to the OS immediately (munmap),
    // instead of accumulating in glibc's arena — otherwise peak RSS climbs
    // steadily over the iterations and OOMs at large Ns.
    mallopt(M_MMAP_THRESHOLD, 128 * 1024);
    mallopt(M_TRIM_THRESHOLD, 128 * 1024);
#endif

    std::vector<int> levels;
    for (int n = np.coarsest; n <= Ns; n *= 2) levels.push_back(n);
    if (levels.empty() || levels.back() != Ns)
        throw std::invalid_argument(
            "solve_contact_nested: Ns must equal coarsest * 2^k");
    if (np.backend != "h2" && np.backend != "fft")
        throw std::invalid_argument(
            "solve_contact_nested: backend must be 'h2' or 'fft'");

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

    Eigen::VectorXd p_init;
    bool have_init = false;
    ContactResult res;

    for (std::size_t li = 0; li < levels.size(); ++li) {
        const int n = levels[li];
        BoussinesqKernel kernel(n, L, E_star);
        std::unique_ptr<H2Operator> h2;
        std::unique_ptr<FFTOperator> fop;
        MatVecIntoT<double> mv;
        if (np.backend == "fft") {
            fop = std::make_unique<FFTOperator>(kernel);
            fop->build();
            mv = [&fop](const Eigen::VectorXd& v, Eigen::VectorXd& out) {
                fop->matvec_into(v, out);
            };
        } else {
            h2 = std::make_unique<H2Operator>(kernel,
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
        // coarse levels only need the pressure (for prolongation), so drop their
        // displacement/gap unconditionally; the finest honours light_result.
        const bool light = finest ? np.light_result : true;
        // only the finest level's trace is meaningful (iterations is also
        // finest-level-only); coarse levels never record history.
        const bool record_history = finest && np.record_error_history;

        if (np.single_precision) {
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
            Eigen::VectorXf g0f = glvl.cast<float>();
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
            res = solve_contact_impl<float>(
                mvf, g0f, static_cast<float>(p_bar), static_cast<float>(lvl_tol),
                max_iter, use_pr, pcf, have_init ? &p0f : nullptr, light,
                record_history);
        } else {
            res = solve_contact_impl<double>(
                mv, glvl, p_bar, lvl_tol, max_iter, use_pr, pc,
                have_init ? &p_init : nullptr, light, record_history);
            if (!finest) gap[li].resize(0);
        }

        if (!finest) {
            p_init = prolong_field(res.pressure, n);
            have_init = true;
        }
    }
    return res;
}

} // namespace hmc
