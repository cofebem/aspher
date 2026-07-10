#include "contact_solver.hpp"

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace hmc {

template <class Real>
ContactResult solve_contact_impl(const MatVecIntoT<Real>& S,
                                 Eigen::Ref<const VecT<Real>> g0,
                                 Real p_bar, Real tol, int max_iter, bool use_pr,
                                 const PrecondIntoT<Real>& precond,
                                 VecT<Real>* p_init, bool light,
                                 bool record_history) {
    using Vec = VecT<Real>;
    const int N = static_cast<int>(g0.size());
    if (N == 0 || p_bar <= Real(0))
        throw std::invalid_argument("solve_contact: empty gap or p_bar <= 0");

    const Real P_total = p_bar * N;
    Real g_scale = g0.maxCoeff() - g0.minCoeff();
    if (g_scale <= Real(0)) g_scale = Real(1);

    Vec p;
    if (p_init) {
        if (static_cast<int>(p_init->size()) != N)
            throw std::invalid_argument("solve_contact: p_init size mismatch");
        // consume the warm start: its storage becomes the pressure iterate,
        // so the caller's N-sized array doesn't sit idle beside its own copy
        // for the whole solve (see the header contract).
        p = std::move(*p_init);
        p = p.cwiseMax(Real(0)); // in-place, coefficient-wise
        const Real s = p.sum();
        if (s > Real(0))
            p *= P_total / s;
        else
            p.setConstant(p_bar);
    } else {
        p = Vec::Constant(N, p_bar);
    }
    Vec t = Vec::Zero(N);
    Vec u(N), g(N), g_prev(N), z(N), r(N);
    std::vector<std::uint8_t> contact(N, 0);
    g_prev.setZero();

    ContactResult res;
    double G_old = 1.0;
    Real delta = 0.0; // conjugation switch: 0 after the active set changed
    Real alpha = 0.0;

    // Stagnation guard: in low precision the complementarity error can plateau
    // above `tol` (float noise floor). If it stops improving for `stall_limit`
    // iterations, the achievable precision is reached — stop and report it as
    // converged. Double reaches `tol` monotonically first, so this never fires
    // there and leaves the validated double iteration counts unchanged.
    double best_err = 1e300;
    int stall = 0;
    const int stall_limit = 200;

    int it = 0;
    for (it = 0; it < max_iter; ++it) {
        S(p, u);

        // rigid-body shift: zero mean gap over the current contact set.
        // Grid-sized reductions accumulate in double regardless of Real: float
        // partial sums over ~1e8 terms lose the digits the CG update needs.
        double gsum = 0.0;
        int nc = 0;
#pragma omp parallel for schedule(static) reduction(+ : gsum, nc)
        for (int i = 0; i < N; ++i) {
            g(i) = u(i) + g0(i);
            if (p(i) > Real(0)) { gsum += static_cast<double>(g(i)); ++nc; }
        }
        alpha = static_cast<Real>(nc ? gsum / nc : 0.0);

        // complementarity error: sum p |g| (g = 0 where p > 0 at the solution)
        double e = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : e)
        for (int i = 0; i < N; ++i) {
            g(i) -= alpha;
            e += static_cast<double>(p(i)) * std::abs(static_cast<double>(g(i)));
        }
        res.error = e / (static_cast<double>(P_total) * static_cast<double>(g_scale));
        if (record_history) res.error_history.push_back(res.error);
        if (res.error < static_cast<double>(tol)) {
            res.converged = true;
            break;
        }
        if (res.error < best_err * (1.0 - 1e-4)) { best_err = res.error; stall = 0; }
        else if (++stall >= stall_limit) { res.converged = true; break; }

        // preconditioned residual z (z = g on contact when no preconditioner)
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) contact[i] = (p(i) > Real(0)) ? 1 : 0;
        if (precond) {
            precond(g, contact, z);
        } else {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) z(i) = contact[i] ? g(i) : Real(0);
        }

        // conjugate direction restricted to the contact set (M-inner product)
        double G = 0.0, G_pr = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : G, G_pr)
        for (int i = 0; i < N; ++i)
            if (contact[i]) {
                G    += static_cast<double>(z(i)) * static_cast<double>(g(i));
                G_pr += static_cast<double>(z(i)) *
                        static_cast<double>(g(i) - g_prev(i));
            }
        const double beta_val =
            use_pr ? std::max(0.0, G_pr / G_old) : G / G_old;
        const Real beta = delta * static_cast<Real>(beta_val);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            t(i) = contact[i] ? z(i) + beta * t(i) : Real(0);
            g_prev(i) = g(i);
        }
        G_old = G;

        // line search tau = <g, t> / <S t, t> on the contact set. den keeps
        // the centred (r - rmean)*t form: the expanded rt - rmean*ts variant
        // cancels catastrophically in float and stalls the float solve.
        S(t, r);
        double rsum = 0.0, num = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : rsum, num)
        for (int i = 0; i < N; ++i)
            if (p(i) > Real(0)) {
                rsum += static_cast<double>(r(i));
                num += static_cast<double>(g(i)) * static_cast<double>(t(i));
            }
        const Real rmean = static_cast<Real>(nc ? rsum / nc : 0.0);
        double den = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : den)
        for (int i = 0; i < N; ++i)
            if (p(i) > Real(0))
                den += static_cast<double>(r(i) - rmean) * static_cast<double>(t(i));
        if (den <= 0.0) { // direction lost positive curvature: steepest descent
            delta = 0.0;
            continue;
        }
        const Real tau = static_cast<Real>(num / den);

        // update, project onto p >= 0, and apply the overlap correction
        // (points at p = 0 but penetrating) in the same pass
        int overlap = 0;
#pragma omp parallel for schedule(static) reduction(| : overlap)
        for (int i = 0; i < N; ++i) {
            Real pi = p(i);
            if (pi > Real(0)) pi = std::max(pi - tau * t(i), Real(0));
            if (pi == Real(0) && g(i) < Real(0)) {
                pi -= tau * g(i);
                overlap = 1;
            }
            p(i) = pi;
        }
        delta = overlap ? Real(0) : Real(1);

        // enforce the load balance
        double total = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : total)
        for (int i = 0; i < N; ++i) total += static_cast<double>(p(i));
        if (total > 0.0) {
            const Real f = static_cast<Real>(static_cast<double>(P_total) / total);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) p(i) *= f;
        } else {
            p.setConstant(p_bar);
        }
    }

    S(p, u);
    double gsum = 0.0;
    int nc = 0;
#pragma omp parallel for schedule(static) reduction(+ : gsum, nc)
    for (int i = 0; i < N; ++i) {
        g(i) = u(i) + g0(i);
        if (p(i) > Real(0)) { gsum += static_cast<double>(g(i)); ++nc; }
    }
    alpha = static_cast<Real>(nc ? gsum / nc : 0.0);

    if (!light) {
        res.displacement = u.template cast<double>();
        res.gap = (g.array() - alpha).template cast<double>();
    }
    res.approach = static_cast<double>(alpha);
    res.objective = static_cast<double>(Real(0.5) * p.dot(u) + p.dot(g0));
    res.iterations = it;
    res.contact_fraction = double(nc) / N;
    res.mean_pressure = static_cast<double>(p.mean());
    // p is dead past this point: move it into the result when Real=double
    // instead of copying — the end-of-solve moment (all CG vectors + the
    // preconditioner scratch still allocated) is the peak RSS of the whole
    // solve, and the copy added an extra N-sized array (2.1 GiB at Ns=16384)
    // right there.
    if constexpr (std::is_same_v<Real, double>)
        res.pressure = std::move(p);
    else
        res.pressure = p.template cast<double>();
    return res;
}

// explicit instantiations
template ContactResult solve_contact_impl<double>(
    const MatVecIntoT<double>&, Eigen::Ref<const VecT<double>>, double, double,
    int, bool, const PrecondIntoT<double>&, VecT<double>*, bool, bool);
template ContactResult solve_contact_impl<float>(
    const MatVecIntoT<float>&, Eigen::Ref<const VecT<float>>, float, float,
    int, bool, const PrecondIntoT<float>&, VecT<float>*, bool, bool);

// ── Active-set (restricted) Polonsky–Keer ────────────────────────────────────
// Mirror of solve_contact_impl with every per-iteration O(N) loop driven by
// the candidate index list (for j: i = idx[j]). Kept as a separate function
// so the validated full-grid path stays byte-identical. All algorithmic
// choices (PR+, centred line-search denominator, overlap correction, load
// renormalisation, double accumulation of grid-length reductions, stagnation
// guard) are carried over unchanged.
template <class Real>
ContactResult solve_contact_active_impl(const MatVecIntoT<Real>& S,
                                        Eigen::Ref<const VecT<Real>> g0,
                                        Real p_bar, Real tol, int max_iter,
                                        bool use_pr,
                                        const PrecondIntoT<Real>& precond,
                                        const std::vector<int>& idx,
                                        VecT<Real>* p_init,
                                        bool record_history,
                                        int N_grid, Real g_scale) {
    using Vec = VecT<Real>;
    const int N = static_cast<int>(g0.size());
    const int Nc = static_cast<int>(idx.size());
    if (N == 0 || Nc == 0 || p_bar <= Real(0))
        throw std::invalid_argument(
            "solve_contact_active: empty gap/candidate set or p_bar <= 0");
    if (N_grid <= 0) N_grid = N;

    const Real P_total = static_cast<Real>(static_cast<double>(p_bar) * N_grid);
    if (g_scale <= Real(0)) g_scale = g0.maxCoeff() - g0.minCoeff();
    if (g_scale <= Real(0)) g_scale = Real(1);
    const Real p_unif = static_cast<Real>(static_cast<double>(P_total) / Nc);

    // pressure iterate: exactly zero outside the candidate set for the whole
    // solve (buffers are full-grid in this milestone; state compression to
    // O(N_c) is M3)
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
        p_init->resize(0); // consumed: off-candidate values are discarded
        if (s > 0.0) {
            const Real f = static_cast<Real>(static_cast<double>(P_total) / s);
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

    ContactResult res;
    double G_old = 1.0;
    Real delta = 0.0;
    Real alpha = 0.0;

    double best_err = 1e300;
    int stall = 0;
    const int stall_limit = 200;

    int it = 0;
    for (it = 0; it < max_iter; ++it) {
        S(p, u); // masked matvec: u is valid on candidate leaves only

        double gsum = 0.0;
        int nc = 0;
#pragma omp parallel for schedule(static) reduction(+ : gsum, nc)
        for (int j = 0; j < Nc; ++j) {
            const int i = idx[j];
            g(i) = u(i) + g0(i);
            if (p(i) > Real(0)) { gsum += static_cast<double>(g(i)); ++nc; }
        }
        alpha = static_cast<Real>(nc ? gsum / nc : 0.0);

        double e = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : e)
        for (int j = 0; j < Nc; ++j) {
            const int i = idx[j];
            g(i) -= alpha;
            e += static_cast<double>(p(i)) * std::abs(static_cast<double>(g(i)));
        }
        res.error = e / (static_cast<double>(P_total) * static_cast<double>(g_scale));
        if (record_history) res.error_history.push_back(res.error);
        if (res.error < static_cast<double>(tol)) {
            res.converged = true;
            break;
        }
        if (res.error < best_err * (1.0 - 1e-4)) { best_err = res.error; stall = 0; }
        else if (++stall >= stall_limit) { res.converged = true; break; }

#pragma omp parallel for schedule(static)
        for (int j = 0; j < Nc; ++j) {
            const int i = idx[j];
            contact[i] = (p(i) > Real(0)) ? 1 : 0;
        }
        if (precond) {
            precond(g, contact, z); // reads g on contact only; zeroes z elsewhere
        } else {
#pragma omp parallel for schedule(static)
            for (int j = 0; j < Nc; ++j) {
                const int i = idx[j];
                z(i) = contact[i] ? g(i) : Real(0);
            }
        }

        double G = 0.0, G_pr = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : G, G_pr)
        for (int j = 0; j < Nc; ++j) {
            const int i = idx[j];
            if (contact[i]) {
                G    += static_cast<double>(z(i)) * static_cast<double>(g(i));
                G_pr += static_cast<double>(z(i)) *
                        static_cast<double>(g(i) - g_prev(i));
            }
        }
        const double beta_val =
            use_pr ? std::max(0.0, G_pr / G_old) : G / G_old;
        const Real beta = delta * static_cast<Real>(beta_val);
#pragma omp parallel for schedule(static)
        for (int j = 0; j < Nc; ++j) {
            const int i = idx[j];
            t(i) = contact[i] ? z(i) + beta * t(i) : Real(0);
            g_prev(i) = g(i);
        }
        G_old = G;

        S(t, r); // t is supported on contact ⊂ candidate set
        double rsum = 0.0, num = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : rsum, num)
        for (int j = 0; j < Nc; ++j) {
            const int i = idx[j];
            if (p(i) > Real(0)) {
                rsum += static_cast<double>(r(i));
                num += static_cast<double>(g(i)) * static_cast<double>(t(i));
            }
        }
        const Real rmean = static_cast<Real>(nc ? rsum / nc : 0.0);
        double den = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : den)
        for (int j = 0; j < Nc; ++j) {
            const int i = idx[j];
            if (p(i) > Real(0))
                den += static_cast<double>(r(i) - rmean) * static_cast<double>(t(i));
        }
        if (den <= 0.0) {
            delta = 0.0;
            continue;
        }
        const Real tau = static_cast<Real>(num / den);

        int overlap = 0;
#pragma omp parallel for schedule(static) reduction(| : overlap)
        for (int j = 0; j < Nc; ++j) {
            const int i = idx[j];
            Real pi = p(i);
            if (pi > Real(0)) pi = std::max(pi - tau * t(i), Real(0));
            if (pi == Real(0) && g(i) < Real(0)) {
                pi -= tau * g(i);
                overlap = 1;
            }
            p(i) = pi;
        }
        delta = overlap ? Real(0) : Real(1);

        double total = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : total)
        for (int j = 0; j < Nc; ++j) total += static_cast<double>(p(idx[j]));
        if (total > 0.0) {
            const Real f = static_cast<Real>(static_cast<double>(P_total) / total);
#pragma omp parallel for schedule(static)
            for (int j = 0; j < Nc; ++j) p(idx[j]) *= f;
        } else {
#pragma omp parallel for schedule(static)
            for (int j = 0; j < Nc; ++j) p(idx[j]) = p_unif;
        }
    }

    S(p, u);
    double gsum = 0.0, obj = 0.0, total = 0.0;
    int nc = 0;
#pragma omp parallel for schedule(static) reduction(+ : gsum, nc, total)
    for (int j = 0; j < Nc; ++j) {
        const int i = idx[j];
        g(i) = u(i) + g0(i);
        total += static_cast<double>(p(i));
        if (p(i) > Real(0)) { gsum += static_cast<double>(g(i)); ++nc; }
    }
    alpha = static_cast<Real>(nc ? gsum / nc : 0.0);
    // objective over the candidate set only: p vanishes elsewhere, and u/g
    // hold stale values off the candidate leaves that must not be touched
#pragma omp parallel for schedule(static) reduction(+ : obj)
    for (int j = 0; j < Nc; ++j) {
        const int i = idx[j];
        obj += static_cast<double>(p(i)) *
               (0.5 * static_cast<double>(u(i)) + static_cast<double>(g0(i)));
    }

    res.approach = static_cast<double>(alpha);
    res.objective = obj;
    res.iterations = it;
    res.contact_fraction = double(nc) / N_grid;
    res.mean_pressure = total / N_grid;
    if constexpr (std::is_same_v<Real, double>)
        res.pressure = std::move(p);
    else
        res.pressure = p.template cast<double>();
    return res;
}

template ContactResult solve_contact_active_impl<double>(
    const MatVecIntoT<double>&, Eigen::Ref<const VecT<double>>, double, double,
    int, bool, const PrecondIntoT<double>&, const std::vector<int>&,
    VecT<double>*, bool, int, double);
template ContactResult solve_contact_active_impl<float>(
    const MatVecIntoT<float>&, Eigen::Ref<const VecT<float>>, float, float,
    int, bool, const PrecondIntoT<float>&, const std::vector<int>&,
    VecT<float>*, bool, int, float);

ContactResult solve_contact(const MatVec& S, const Eigen::VectorXd& g0,
                            double p_bar, double tol, int max_iter, bool use_pr,
                            const Precond& precond, const Eigen::VectorXd* p_init,
                            bool light, bool record_history) {
    MatVecIntoT<double> Si = [&S](const Eigen::VectorXd& x, Eigen::VectorXd& y) {
        y = S(x);
    };
    PrecondIntoT<double> pi;
    if (precond)
        pi = [&precond](const Eigen::VectorXd& g,
                        const std::vector<std::uint8_t>& contact,
                        Eigen::VectorXd& z) { z = precond(g, contact); };
    // the impl consumes its p_init; copy to preserve this function's
    // non-consuming const-pointer contract (the copy simply becomes the
    // solver's pressure iterate — no extra array at peak).
    Eigen::VectorXd p0;
    if (p_init) p0 = *p_init;
    return solve_contact_impl<double>(Si, g0, p_bar, tol, max_iter, use_pr,
                                      pi, p_init ? &p0 : nullptr, light,
                                      record_history);
}

} // namespace hmc
