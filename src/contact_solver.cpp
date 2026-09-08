#include "contact_solver.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace hmc {

const char* to_string(SolveStatus s) {
    switch (s) {
        case SolveStatus::converged: return "converged";
        case SolveStatus::stagnated: return "stagnated";
        case SolveStatus::max_iterations: return "max_iterations";
        case SolveStatus::nonpositive_curvature: return "nonpositive_curvature";
        case SolveStatus::nonfinite: return "nonfinite";
        case SolveStatus::verification_failed: return "verification_failed";
        case SolveStatus::resource_limit: return "resource_limit";
    }
    return "unknown";
}

// ── Projection onto the load simplex {p >= 0, sum p = P} ────────────────────
// p = max(y - θ, 0) with θ fixed by the load. Michelot's threshold iteration:
// start with the full index set, take θ as the mean of the retained entries
// minus P/|A|, drop the entries that fall below it, repeat. Each pass strictly
// shrinks A unless the answer is already consistent, so it terminates; in
// practice a handful of O(N) passes. A sorting projector is the test oracle.
template <class Real>
double project_load_simplex(VecT<Real>& y, double P_total) {
    const std::ptrdiff_t N = y.size();
    if (N == 0) return 0.0;
    double ssum = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : ssum)
    for (std::ptrdiff_t i = 0; i < N; ++i) ssum += static_cast<double>(y(i));
    std::ptrdiff_t nact = N;
    double theta = (ssum - P_total) / static_cast<double>(nact);
    for (int pass = 0; pass < 64; ++pass) {
        double s2 = 0.0;
        std::ptrdiff_t n2 = 0;
#pragma omp parallel for schedule(static) reduction(+ : s2, n2)
        for (std::ptrdiff_t i = 0; i < N; ++i)
            if (static_cast<double>(y(i)) > theta) {
                s2 += static_cast<double>(y(i));
                ++n2;
            }
        if (n2 == 0) break;             // degenerate: keep the current θ
        if (n2 == nact) break;          // consistent
        nact = n2;
        ssum = s2;
        theta = (ssum - P_total) / static_cast<double>(nact);
    }
    const Real th = static_cast<Real>(theta);
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < N; ++i)
        y(i) = (y(i) > th) ? (y(i) - th) : Real(0);
    return theta;
}

template double project_load_simplex<double>(VecT<double>&, double);
template double project_load_simplex<float>(VecT<float>&, double);

namespace {

// Independently recomputed state of one iterate (spec §4). Everything is
// accumulated in double regardless of the working precision.
struct Diag {
    double alpha = 0.0;   // reported approach = mean v over positive pressure
    double gmin = 0.0;    // min over the evaluated set of the centred gap
    double pk = 0.0;      // legacy  sum p|g| / (P g_ref)
    double G = 0.0;       // sum_i p_i (v_i - min v)   >= 0 by construction
    double fw = 0.0;      // G / (P g_ref)
    double pen = 0.0;     // max(0, -min gap) / g_ref
    double load = 0.0;    // |sum p - P| / P
    double psum = 0.0;
    double pv = 0.0;      // sum_i p_i v_i
    int nc = 0;           // number of positive-pressure points
    bool finite = true;
};

inline double default_load_tol(bool is_double) {
    return is_double ? 1e-12 : 5e-7;
}

} // namespace

// ── Full-grid Polonsky-Keer with certified termination ──────────────────────
template <class Real>
ContactResult solve_contact_impl(const MatVecIntoT<Real>& S,
                                 Eigen::Ref<const VecT<Real>> g0,
                                 Real p_bar, const SolveOptions& opt,
                                 const PrecondIntoT<Real>& precond,
                                 VecT<Real>* p_init) {
    using Vec = VecT<Real>;
    constexpr bool is_double = std::is_same_v<Real, double>;
    const int N = static_cast<int>(g0.size());
    if (N == 0 || !(p_bar > Real(0)))
        throw std::invalid_argument("solve_contact: empty gap or p_bar <= 0");
    if (!(opt.tol > 0.0))
        throw std::invalid_argument("solve_contact: tol must be positive");
    if (opt.max_iter < 0)
        throw std::invalid_argument("solve_contact: max_iter must be >= 0");
    // Reject a non-finite input before any operator call (spec A19): a
    // nonfinite *input* is invalid data, not a numerical termination state.
    if (!g0.allFinite())
        throw std::invalid_argument("solve_contact: g0 contains non-finite values");

    const double P_total = static_cast<double>(p_bar) * N;
    double g_ref = opt.scales.g_ref;
    if (!(g_ref > 0.0))
        g_ref = static_cast<double>(g0.maxCoeff()) -
                static_cast<double>(g0.minCoeff());
    if (!(g_ref > 0.0)) g_ref = 1.0;
    const double p_ref =
        (opt.scales.p_ref > 0.0) ? opt.scales.p_ref : static_cast<double>(p_bar);
    const double load_tol =
        (opt.load_tol > 0.0) ? opt.load_tol : default_load_tol(is_double);
    const double tol = opt.tol;
    const double req_tol = (opt.requested_tol > 0.0) ? opt.requested_tol : tol;

    Vec p;
    if (p_init) {
        if (static_cast<int>(p_init->size()) != N)
            throw std::invalid_argument("solve_contact: p_init size mismatch");
        // consume the warm start: its storage becomes the pressure iterate.
        p = std::move(*p_init);
        p = p.cwiseMax(Real(0));
        double s = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : s)
        for (int i = 0; i < N; ++i) s += static_cast<double>(p(i));
        if (s > 0.0) {
            const Real f = static_cast<Real>(P_total / s);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) p(i) *= f;
        } else {
            p.setConstant(p_bar);
        }
    } else {
        p = Vec::Constant(N, p_bar);
    }
    Vec t = Vec::Zero(N);
    Vec u(N), g(N), g_prev(N), z(N), r(N);
    std::vector<std::uint8_t> contact(N, 0);
    g_prev.setZero();
    Vec best_p;

    ContactResult res;
    long long mv = 0, pcount = 0;
    double G_old = 1.0;
    Real delta = 0.0; // conjugation switch: 0 restarts the direction

    // Evaluate v = Sp + g0 from a *fresh* u, centre it, and recompute every
    // acceptance diagnostic. g holds the centred gap on return.
    auto evaluate = [&](void) -> Diag {
        Diag d;
        double gsum = 0.0;
        int nc = 0;
        double vmin = std::numeric_limits<double>::infinity();
#pragma omp parallel for schedule(static) reduction(+ : gsum, nc) \
    reduction(min : vmin)
        for (int i = 0; i < N; ++i) {
            const Real v = u(i) + g0(i);
            g(i) = v;
            const double vd = static_cast<double>(v);
            if (vd < vmin) vmin = vd;
            if (p(i) > Real(0)) { gsum += vd; ++nc; }
        }
        d.nc = nc;
        d.alpha = nc ? gsum / nc : vmin;
        d.gmin = vmin - d.alpha;
        const Real alpha = static_cast<Real>(d.alpha);
        const Real gmin = static_cast<Real>(d.gmin);
        double e = 0.0, Gc = 0.0, psum = 0.0, pv = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : e, Gc, psum, pv)
        for (int i = 0; i < N; ++i) {
            const double pi = static_cast<double>(p(i));
            pv += pi * static_cast<double>(g(i));
            g(i) -= alpha;
            const double gi = static_cast<double>(g(i));
            e += pi * std::abs(gi);
            // non-negative summands: g(i) - gmin >= 0 by construction
            Gc += pi * static_cast<double>(g(i) - gmin);
            psum += pi;
        }
        d.pk = e / (P_total * g_ref);
        d.G = Gc;
        d.fw = Gc / (P_total * g_ref);
        d.pen = std::max(0.0, -d.gmin) / g_ref;
        d.psum = psum;
        d.pv = pv;
        d.load = std::abs(psum - P_total) / P_total;
        d.finite = std::isfinite(gsum) && std::isfinite(vmin) &&
                   std::isfinite(psum) && std::isfinite(e) && std::isfinite(Gc);
        return d;
    };

    // ── Feasible identification step (spec A02) ─────────────────────────────
    // p+ = Π_{p>=0, sum p = P}(p - s g), backtracking on the actual objective.
    // Π is invariant to an additive datum on the gradient, so the centred g
    // may be used. The step is preconditioner-independent by construction,
    // which is what makes activation scale invariant when the CG direction is
    // unusable. Scratch: z (trial point), r (its displacement) — the CG state
    // is invalid after this step anyway, so no extra memory is needed.
    double s_id = 0.0;
    bool u_fresh = false;
    auto identification_step = [&](const Diag& d) -> bool {
        double f_cur = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : f_cur)
        for (int i = 0; i < N; ++i)
            f_cur += static_cast<double>(p(i)) *
                     (0.5 * static_cast<double>(u(i)) +
                      static_cast<double>(g0(i)));
        if (!(s_id > 0.0)) {
            // initial inverse-stiffness estimate from the current iterate:
            // s = (p.p)/(p.Sp) has units pressure/displacement and is a lower
            // bound on 1/λ_min-style steps, so backtracking starts safe.
            double pp = 0.0, pu = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : pp, pu)
            for (int i = 0; i < N; ++i) {
                pp += static_cast<double>(p(i)) * static_cast<double>(p(i));
                pu += static_cast<double>(p(i)) * static_cast<double>(u(i));
            }
            s_id = (pu > 0.0 && pp > 0.0) ? pp / pu : 1.0;
        }
        for (int trial = 0; trial < 12; ++trial) {
            const Real s = static_cast<Real>(s_id);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) z(i) = p(i) - s * g(i);
            project_load_simplex<Real>(z, P_total);
            S(z, r);
            ++mv;
            double f_try = 0.0, gd = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : f_try, gd)
            for (int i = 0; i < N; ++i) {
                f_try += static_cast<double>(z(i)) *
                         (0.5 * static_cast<double>(r(i)) +
                          static_cast<double>(g0(i)));
                gd += static_cast<double>(g(i)) *
                      static_cast<double>(z(i) - p(i));
            }
            if (std::isfinite(f_try) && gd < 0.0 && f_try <= f_cur + 1e-4 * gd) {
                p.swap(z);
                u.swap(r);
                u_fresh = true;
                delta = Real(0);
                t.setZero();
                g_prev.setZero();
                G_old = 1.0;
                ++res.identification_steps;
                if (trial == 0) s_id *= 2.0; // the step was not limiting
                return true;
            }
            s_id *= 0.25;
        }
        (void)d;
        return false;
    };

    SolveStatus status = SolveStatus::max_iterations;
    std::string reason;
    double best_merit = std::numeric_limits<double>::infinity();
    double seen_merit = std::numeric_limits<double>::infinity();
    int stall = 0, forced_id = 0;
    const int stall_limit = opt.stall_limit > 0 ? opt.stall_limit : 200;
    Diag d;

    int it = 0;
    for (it = 0; it < opt.max_iter; ++it) {
        if (!u_fresh) { S(p, u); ++mv; }
        u_fresh = false;
        d = evaluate();
        if (!d.finite) { status = SolveStatus::nonfinite; break; }
        if (opt.record_history) res.error_history.push_back(d.pk);

        if (d.load <= load_tol && d.fw <= tol && d.pen <= tol) {
            status = SolveStatus::converged;
            u_fresh = true; // u corresponds to the returned p
            break;
        }

        const double merit = std::max(d.fw, d.pen);
        if (opt.keep_best && merit < best_merit) {
            best_merit = merit;
            best_p = p;
        }
        if (merit < seen_merit * (1.0 - 1e-4)) {
            seen_merit = merit;
            stall = 0;
        } else if (++stall >= stall_limit) {
            // before calling it stagnation, spend a guaranteed-descent
            // feasible step: it either makes progress or proves there is none
            if (forced_id < 3 && identification_step(d)) {
                ++forced_id;
                stall = 0;
                continue;
            }
            status = SolveStatus::stagnated;
            reason = "stall";
            break;
        }

        bool did_cg = false;
        if (d.nc >= 2) {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) contact[i] = (p(i) > Real(0)) ? 1 : 0;
            if (precond) {
                precond(g, contact, z);
                ++pcount;
            } else {
#pragma omp parallel for schedule(static)
                for (int i = 0; i < N; ++i) z(i) = contact[i] ? g(i) : Real(0);
            }
            double G = 0.0, G_pr = 0.0, gg = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : G, G_pr, gg)
            for (int i = 0; i < N; ++i)
                if (contact[i]) {
                    const double gi = static_cast<double>(g(i));
                    G += static_cast<double>(z(i)) * gi;
                    G_pr += static_cast<double>(z(i)) *
                            static_cast<double>(g(i) - g_prev(i));
                    gg += gi * gi;
                }
            const double beta_val =
                (G_old != 0.0)
                    ? (opt.use_pr ? std::max(0.0, G_pr / G_old) : G / G_old)
                    : 0.0;
            const Real beta = delta * static_cast<Real>(beta_val);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) {
                t(i) = contact[i] ? z(i) + beta * t(i) : Real(0);
                g_prev(i) = g(i);
            }
            G_old = G;

            // line search tau = <g,t>/<St,t> on the contact set; den keeps the
            // centred (r - rmean)*t form (the expanded variant cancels
            // catastrophically in float).
            S(t, r);
            ++mv;
            double rsum = 0.0, num = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : rsum, num)
            for (int i = 0; i < N; ++i)
                if (contact[i]) {
                    rsum += static_cast<double>(r(i));
                    num += static_cast<double>(g(i)) * static_cast<double>(t(i));
                }
            const Real rmean = static_cast<Real>(d.nc ? rsum / d.nc : 0.0);
            double den = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : den)
            for (int i = 0; i < N; ++i)
                if (contact[i])
                    den += static_cast<double>(r(i) - rmean) *
                           static_cast<double>(t(i));

            if (den > 0.0 && std::isfinite(den) && std::isfinite(num)) {
                const Real tau = static_cast<Real>(num / den);
                // Activation (overlap) step length.
                //   Under M^-1 -> c M^-1 (c>0) on an unchanged support the
                //   direction scales as t -> c t, so num -> c num and
                //   den -> c^2 den, hence tau -> tau/c: the CG update tau*t is
                //   invariant but the raw-gap update tau*g is NOT. Using
                //     tau_ov = num^2 / (den * |g|^2_A)
                //   restores invariance (num^2/den is invariant) and keeps the
                //   correct units (pressure per displacement). With no
                //   preconditioner and a restarted direction (t = g) it equals
                //   the Polonsky-Keer tau exactly; with beta != 0 and an exact
                //   line search g.t = |g|^2 + beta (g.t_prev) ~ |g|^2, so it
                //   stays within roundoff of the original algorithm.
                const Real tau_ov =
                    (gg > 0.0) ? static_cast<Real>(num * num / (den * gg)) : tau;
                int changed = 0;
#pragma omp parallel for schedule(static) reduction(| : changed)
                for (int i = 0; i < N; ++i) {
                    Real pi = p(i);
                    const bool was = (pi > Real(0));
                    if (was) pi = std::max(pi - tau * t(i), Real(0));
                    if (pi == Real(0) && g(i) < Real(0)) pi -= tau_ov * g(i);
                    if ((pi > Real(0)) != was) changed = 1;
                    p(i) = pi;
                }
                // restart conjugacy on ANY change of the support: a zero-sum
                // direction restricted to a smaller support is not zero-sum,
                // and a newly activated point carries no conjugacy history.
                delta = changed ? Real(0) : Real(1);

                double total = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : total)
                for (int i = 0; i < N; ++i) total += static_cast<double>(p(i));
                if (total > 0.0) {
                    const Real f = static_cast<Real>(P_total / total);
#pragma omp parallel for schedule(static)
                    for (int i = 0; i < N; ++i) p(i) *= f;
                } else {
                    p.setConstant(p_bar);
                }
                did_cg = true;
            }
        }

        if (!did_cg) {
            // Either the support cannot carry a zero-sum direction (|A| < 2)
            // or the direction lost positive curvature: hand control to the
            // feasible identification step instead of spinning.
            if (!identification_step(d)) {
                status = SolveStatus::nonpositive_curvature;
                break;
            }
        }
    }

    // ── final, freshly evaluated diagnostics on the RETURNED pressure ───────
    if (status != SolveStatus::converged && opt.keep_best && best_p.size() == N) {
        const double merit = std::max(d.fw, d.pen);
        if (!(merit <= best_merit)) {
            p = std::move(best_p);
            u_fresh = false;
            res.returned_best = true;
        }
    }
    best_p.resize(0);
    if (!u_fresh) { S(p, u); ++mv; }
    d = evaluate();
    if (!d.finite && status != SolveStatus::nonfinite)
        status = SolveStatus::nonfinite;
    if (status == SolveStatus::converged &&
        !(d.load <= load_tol && d.fw <= tol && d.pen <= tol)) {
        // the fresh check disagrees with the in-loop one (should not happen
        // with a deterministic operator, but never claim success on trust)
        status = SolveStatus::verification_failed;
        reason = "final_check";
    }
    if (status == SolveStatus::converged && req_tol < tol &&
        !opt.allow_tolerance_relaxation) {
        if (!(d.fw <= req_tol && d.pen <= req_tol)) {
            status = SolveStatus::stagnated;
            reason = "precision_limit";
        }
    }

    const double datum = opt.scales.datum;
    if (!opt.light) {
        res.displacement = u.template cast<double>();
        res.gap = g.template cast<double>();
    }
    res.approach = d.alpha + datum;
    double obj = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : obj)
    for (int i = 0; i < N; ++i)
        obj += static_cast<double>(p(i)) *
               (0.5 * static_cast<double>(u(i)) + static_cast<double>(g0(i)));
    res.objective = obj + datum * P_total;
    res.iterations = it;
    res.contact_fraction = double(d.nc) / N;
    res.mean_pressure = d.psum / N;
    res.status = status;
    res.status_reason = std::move(reason);
    res.converged = (status == SolveStatus::converged);
    res.error = d.pk;
    res.pk_error = d.pk;
    res.fw_gap = d.G;
    res.fw_error = d.fw;
    res.load_error = d.load;
    res.pressure_violation = 0.0; // the iterate is projected onto p >= 0
    res.penetration_error = d.pen;
    res.requested_tol = req_tol;
    res.effective_tol = tol;
    res.p_ref = p_ref;
    res.g_ref = g_ref;
    res.matvec_count = mv;
    res.precond_count = pcount;
    if constexpr (is_double)
        res.pressure = std::move(p);
    else
        res.pressure = p.template cast<double>();
    return res;
}

template ContactResult solve_contact_impl<double>(
    const MatVecIntoT<double>&, Eigen::Ref<const VecT<double>>, double,
    const SolveOptions&, const PrecondIntoT<double>&, VecT<double>*);
template ContactResult solve_contact_impl<float>(
    const MatVecIntoT<float>&, Eigen::Ref<const VecT<float>>, float,
    const SolveOptions&, const PrecondIntoT<float>&, VecT<float>*);

// ── Active-set (restricted) Polonsky-Keer ────────────────────────────────────
// Mirror of solve_contact_impl with every per-iteration O(N) loop driven by
// the candidate index list (for j: i = idx[j]). The certificate it reports is
// restricted to the candidate set; the caller combines it with a global
// streamed minimum gradient (spec A03).
template <class Real>
ContactResult solve_contact_active_impl(const MatVecIntoT<Real>& S,
                                        Eigen::Ref<const VecT<Real>> g0,
                                        Real p_bar, const SolveOptions& opt,
                                        const PrecondIntoT<Real>& precond,
                                        const std::vector<int>& idx,
                                        VecT<Real>* p_init,
                                        RestrictedCertificate* cert) {
    using Vec = VecT<Real>;
    constexpr bool is_double = std::is_same_v<Real, double>;
    const int N = static_cast<int>(g0.size());
    const int Nc = static_cast<int>(idx.size());
    if (N == 0 || Nc == 0 || !(p_bar > Real(0)))
        throw std::invalid_argument(
            "solve_contact_active: empty gap/candidate set or p_bar <= 0");
    if (!(opt.tol > 0.0))
        throw std::invalid_argument("solve_contact_active: tol must be positive");
    if (opt.max_iter < 0)
        throw std::invalid_argument("solve_contact_active: max_iter must be >= 0");
    if (!g0.allFinite())
        throw std::invalid_argument(
            "solve_contact_active: g0 contains non-finite values");
    int N_grid = opt.n_grid > 0 ? opt.n_grid : N;

    const double P_total = static_cast<double>(p_bar) * N_grid;
    double g_ref = opt.scales.g_ref;
    if (!(g_ref > 0.0))
        g_ref = static_cast<double>(g0.maxCoeff()) -
                static_cast<double>(g0.minCoeff());
    if (!(g_ref > 0.0)) g_ref = 1.0;
    const double p_ref =
        (opt.scales.p_ref > 0.0) ? opt.scales.p_ref : static_cast<double>(p_bar);
    const double load_tol =
        (opt.load_tol > 0.0) ? opt.load_tol : default_load_tol(is_double);
    const double tol = opt.tol;
    const double req_tol = (opt.requested_tol > 0.0) ? opt.requested_tol : tol;
    const Real p_unif = static_cast<Real>(P_total / Nc);

    Vec p = Vec::Zero(N);
    if (p_init) {
        if (static_cast<int>(p_init->size()) != N)
            throw std::invalid_argument("solve_contact_active: p_init size mismatch");
        double s = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : s)
        for (int j = 0; j < Nc; ++j) {
            const int i = idx[j];
            p(i) = std::max((*p_init)(i), Real(0));
            s += static_cast<double>(p(i));
        }
        p_init->resize(0); // consumed
        if (s > 0.0) {
            const Real f = static_cast<Real>(P_total / s);
#pragma omp parallel for schedule(static)
            for (int j = 0; j < Nc; ++j) p(idx[j]) *= f;
        } else {
#pragma omp parallel for schedule(static)
            for (int j = 0; j < Nc; ++j) p(idx[j]) = p_unif;
        }
    } else {
#pragma omp parallel for schedule(static)
        for (int j = 0; j < Nc; ++j) p(idx[j]) = p_unif;
    }

    Vec t = Vec::Zero(N);
    Vec u(N), g(N), z(N), r(N);
    Vec g_prev = Vec::Zero(N);
    std::vector<std::uint8_t> contact(N, 0);
    Vec best_p;

    ContactResult res;
    long long mv = 0, pcount = 0;
    double G_old = 1.0;
    Real delta = 0.0;

    auto evaluate = [&](void) -> Diag {
        Diag d;
        double gsum = 0.0;
        int nc = 0;
        double vmin = std::numeric_limits<double>::infinity();
#pragma omp parallel for schedule(static) reduction(+ : gsum, nc) \
    reduction(min : vmin)
        for (int j = 0; j < Nc; ++j) {
            const int i = idx[j];
            const Real v = u(i) + g0(i);
            g(i) = v;
            const double vd = static_cast<double>(v);
            if (vd < vmin) vmin = vd;
            if (p(i) > Real(0)) { gsum += vd; ++nc; }
        }
        d.nc = nc;
        d.alpha = nc ? gsum / nc : vmin;
        d.gmin = vmin - d.alpha;
        const Real alpha = static_cast<Real>(d.alpha);
        const Real gmin = static_cast<Real>(d.gmin);
        double e = 0.0, Gc = 0.0, psum = 0.0, pv = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : e, Gc, psum, pv)
        for (int j = 0; j < Nc; ++j) {
            const int i = idx[j];
            const double pi = static_cast<double>(p(i));
            pv += pi * static_cast<double>(g(i));
            g(i) -= alpha;
            const double gi = static_cast<double>(g(i));
            e += pi * std::abs(gi);
            Gc += pi * static_cast<double>(g(i) - gmin);
            psum += pi;
        }
        d.pk = e / (P_total * g_ref);
        d.G = Gc;
        d.fw = Gc / (P_total * g_ref);
        d.pen = std::max(0.0, -d.gmin) / g_ref;
        d.psum = psum;
        d.pv = pv;
        d.load = std::abs(psum - P_total) / P_total;
        d.finite = std::isfinite(gsum) && std::isfinite(vmin) &&
                   std::isfinite(psum) && std::isfinite(e) && std::isfinite(Gc);
        return d;
    };

    double s_id = 0.0;
    bool u_fresh = false;
    auto identification_step = [&]() -> bool {
        double f_cur = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : f_cur)
        for (int j = 0; j < Nc; ++j) {
            const int i = idx[j];
            f_cur += static_cast<double>(p(i)) *
                     (0.5 * static_cast<double>(u(i)) +
                      static_cast<double>(g0(i)));
        }
        if (!(s_id > 0.0)) {
            double pp = 0.0, pu = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : pp, pu)
            for (int j = 0; j < Nc; ++j) {
                const int i = idx[j];
                pp += static_cast<double>(p(i)) * static_cast<double>(p(i));
                pu += static_cast<double>(p(i)) * static_cast<double>(u(i));
            }
            s_id = (pu > 0.0 && pp > 0.0) ? pp / pu : 1.0;
        }
        // the trial point lives on the candidate set only: gather, project,
        // scatter (the projection must see exactly the candidate entries)
        Vec y(Nc);
        for (int trial = 0; trial < 12; ++trial) {
            const Real s = static_cast<Real>(s_id);
#pragma omp parallel for schedule(static)
            for (int j = 0; j < Nc; ++j) {
                const int i = idx[j];
                y(j) = p(i) - s * g(i);
            }
            project_load_simplex<Real>(y, P_total);
            z.setZero();
#pragma omp parallel for schedule(static)
            for (int j = 0; j < Nc; ++j) z(idx[j]) = y(j);
            S(z, r);
            ++mv;
            double f_try = 0.0, gd = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : f_try, gd)
            for (int j = 0; j < Nc; ++j) {
                const int i = idx[j];
                f_try += static_cast<double>(z(i)) *
                         (0.5 * static_cast<double>(r(i)) +
                          static_cast<double>(g0(i)));
                gd += static_cast<double>(g(i)) *
                      static_cast<double>(z(i) - p(i));
            }
            if (std::isfinite(f_try) && gd < 0.0 && f_try <= f_cur + 1e-4 * gd) {
                p.swap(z);
                u.swap(r);
                u_fresh = true;
                delta = Real(0);
                t.setZero();
                g_prev.setZero();
                G_old = 1.0;
                ++res.identification_steps;
                if (trial == 0) s_id *= 2.0;
                return true;
            }
            s_id *= 0.25;
        }
        return false;
    };

    SolveStatus status = SolveStatus::max_iterations;
    std::string reason;
    double best_merit = std::numeric_limits<double>::infinity();
    double seen_merit = std::numeric_limits<double>::infinity();
    int stall = 0, forced_id = 0;
    const int stall_limit = opt.stall_limit > 0 ? opt.stall_limit : 200;
    Diag d;

    int it = 0;
    for (it = 0; it < opt.max_iter; ++it) {
        if (!u_fresh) { S(p, u); ++mv; }
        u_fresh = false;
        d = evaluate();
        if (!d.finite) { status = SolveStatus::nonfinite; break; }
        if (opt.record_history) res.error_history.push_back(d.pk);

        if (d.load <= load_tol && d.fw <= tol && d.pen <= tol) {
            status = SolveStatus::converged;
            u_fresh = true;
            break;
        }

        const double merit = std::max(d.fw, d.pen);
        if (opt.keep_best && merit < best_merit) {
            best_merit = merit;
            best_p = p;
        }
        if (merit < seen_merit * (1.0 - 1e-4)) {
            seen_merit = merit;
            stall = 0;
        } else if (++stall >= stall_limit) {
            if (forced_id < 3 && identification_step()) {
                ++forced_id;
                stall = 0;
                continue;
            }
            status = SolveStatus::stagnated;
            reason = "stall";
            break;
        }

        bool did_cg = false;
        if (d.nc >= 2) {
#pragma omp parallel for schedule(static)
            for (int j = 0; j < Nc; ++j) {
                const int i = idx[j];
                contact[i] = (p(i) > Real(0)) ? 1 : 0;
            }
            if (precond) {
                precond(g, contact, z);
                ++pcount;
            } else {
#pragma omp parallel for schedule(static)
                for (int j = 0; j < Nc; ++j) {
                    const int i = idx[j];
                    z(i) = contact[i] ? g(i) : Real(0);
                }
            }
            double G = 0.0, G_pr = 0.0, gg = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : G, G_pr, gg)
            for (int j = 0; j < Nc; ++j) {
                const int i = idx[j];
                if (contact[i]) {
                    const double gi = static_cast<double>(g(i));
                    G += static_cast<double>(z(i)) * gi;
                    G_pr += static_cast<double>(z(i)) *
                            static_cast<double>(g(i) - g_prev(i));
                    gg += gi * gi;
                }
            }
            const double beta_val =
                (G_old != 0.0)
                    ? (opt.use_pr ? std::max(0.0, G_pr / G_old) : G / G_old)
                    : 0.0;
            const Real beta = delta * static_cast<Real>(beta_val);
#pragma omp parallel for schedule(static)
            for (int j = 0; j < Nc; ++j) {
                const int i = idx[j];
                t(i) = contact[i] ? z(i) + beta * t(i) : Real(0);
                g_prev(i) = g(i);
            }
            G_old = G;

            S(t, r);
            ++mv;
            double rsum = 0.0, num = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : rsum, num)
            for (int j = 0; j < Nc; ++j) {
                const int i = idx[j];
                if (contact[i]) {
                    rsum += static_cast<double>(r(i));
                    num += static_cast<double>(g(i)) * static_cast<double>(t(i));
                }
            }
            const Real rmean = static_cast<Real>(d.nc ? rsum / d.nc : 0.0);
            double den = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : den)
            for (int j = 0; j < Nc; ++j) {
                const int i = idx[j];
                if (contact[i])
                    den += static_cast<double>(r(i) - rmean) *
                           static_cast<double>(t(i));
            }

            if (den > 0.0 && std::isfinite(den) && std::isfinite(num)) {
                const Real tau = static_cast<Real>(num / den);
                const Real tau_ov =
                    (gg > 0.0) ? static_cast<Real>(num * num / (den * gg)) : tau;
                int changed = 0;
#pragma omp parallel for schedule(static) reduction(| : changed)
                for (int j = 0; j < Nc; ++j) {
                    const int i = idx[j];
                    Real pi = p(i);
                    const bool was = (pi > Real(0));
                    if (was) pi = std::max(pi - tau * t(i), Real(0));
                    if (pi == Real(0) && g(i) < Real(0)) pi -= tau_ov * g(i);
                    if ((pi > Real(0)) != was) changed = 1;
                    p(i) = pi;
                }
                delta = changed ? Real(0) : Real(1);

                double total = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : total)
                for (int j = 0; j < Nc; ++j)
                    total += static_cast<double>(p(idx[j]));
                if (total > 0.0) {
                    const Real f = static_cast<Real>(P_total / total);
#pragma omp parallel for schedule(static)
                    for (int j = 0; j < Nc; ++j) p(idx[j]) *= f;
                } else {
#pragma omp parallel for schedule(static)
                    for (int j = 0; j < Nc; ++j) p(idx[j]) = p_unif;
                }
                did_cg = true;
            }
        }

        if (!did_cg && !identification_step()) {
            status = SolveStatus::nonpositive_curvature;
            break;
        }
    }

    if (status != SolveStatus::converged && opt.keep_best && best_p.size() == N) {
        const double merit = std::max(d.fw, d.pen);
        if (!(merit <= best_merit)) {
            p = std::move(best_p);
            u_fresh = false;
            res.returned_best = true;
        }
    }
    best_p.resize(0);
    if (!u_fresh) { S(p, u); ++mv; }
    d = evaluate();
    if (!d.finite && status != SolveStatus::nonfinite)
        status = SolveStatus::nonfinite;
    if (status == SolveStatus::converged &&
        !(d.load <= load_tol && d.fw <= tol && d.pen <= tol)) {
        status = SolveStatus::verification_failed;
        reason = "final_check";
    }
    if (status == SolveStatus::converged && req_tol < tol &&
        !opt.allow_tolerance_relaxation) {
        if (!(d.fw <= req_tol && d.pen <= req_tol)) {
            status = SolveStatus::stagnated;
            reason = "precision_limit";
        }
    }

    double obj = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : obj)
    for (int j = 0; j < Nc; ++j) {
        const int i = idx[j];
        obj += static_cast<double>(p(i)) *
               (0.5 * static_cast<double>(u(i)) + static_cast<double>(g0(i)));
    }

    const double datum = opt.scales.datum;
    res.approach = d.alpha + datum;
    res.objective = obj + datum * P_total;
    res.iterations = it;
    res.contact_fraction = double(d.nc) / N_grid;
    res.mean_pressure = d.psum / N_grid;
    res.status = status;
    res.status_reason = std::move(reason);
    res.converged = (status == SolveStatus::converged);
    res.error = d.pk;
    res.pk_error = d.pk;
    res.fw_gap = d.G;
    res.fw_error = d.fw;
    res.load_error = d.load;
    res.penetration_error = d.pen;
    res.requested_tol = req_tol;
    res.effective_tol = tol;
    res.p_ref = p_ref;
    res.g_ref = g_ref;
    res.matvec_count = mv;
    res.precond_count = pcount;
    if (cert) {
        cert->pv_sum = d.pv;
        cert->vmin_local = d.gmin + d.alpha;
        cert->P_total = P_total;
        cert->g_ref = g_ref;
    }
    if constexpr (is_double)
        res.pressure = std::move(p);
    else
        res.pressure = p.template cast<double>();
    return res;
}

template ContactResult solve_contact_active_impl<double>(
    const MatVecIntoT<double>&, Eigen::Ref<const VecT<double>>, double,
    const SolveOptions&, const PrecondIntoT<double>&, const std::vector<int>&,
    VecT<double>*, RestrictedCertificate*);
template ContactResult solve_contact_active_impl<float>(
    const MatVecIntoT<float>&, Eigen::Ref<const VecT<float>>, float,
    const SolveOptions&, const PrecondIntoT<float>&, const std::vector<int>&,
    VecT<float>*, RestrictedCertificate*);

ContactResult solve_contact_opt(const MatVec& S, const Eigen::VectorXd& g0,
                                double p_bar, const SolveOptions& opt,
                                const Precond& precond,
                                const Eigen::VectorXd* p_init) {
    MatVecIntoT<double> Si = [&S](const Eigen::VectorXd& x, Eigen::VectorXd& y) {
        y = S(x);
    };
    PrecondIntoT<double> pi;
    if (precond)
        pi = [&precond](const Eigen::VectorXd& g,
                        const std::vector<std::uint8_t>& contact,
                        Eigen::VectorXd& z) { z = precond(g, contact); };
    // the impl consumes its p_init; copy to preserve the non-consuming
    // const-pointer contract (the copy simply becomes the pressure iterate).
    Eigen::VectorXd p0;
    if (p_init) p0 = *p_init;
    return solve_contact_impl<double>(Si, g0, p_bar, opt, pi,
                                      p_init ? &p0 : nullptr);
}

ContactResult solve_contact(const MatVec& S, const Eigen::VectorXd& g0,
                            double p_bar, double tol, int max_iter, bool use_pr,
                            const Precond& precond, const Eigen::VectorXd* p_init,
                            bool light, bool record_history) {
    SolveOptions opt;
    opt.tol = tol;
    opt.max_iter = max_iter;
    opt.use_pr = use_pr;
    opt.light = light;
    opt.record_history = record_history;
    return solve_contact_opt(S, g0, p_bar, opt, precond, p_init);
}

} // namespace hmc
