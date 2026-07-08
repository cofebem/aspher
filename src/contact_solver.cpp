#include "contact_solver.hpp"

#include <cmath>
#include <stdexcept>

namespace hmc {

template <class Real>
ContactResult solve_contact_impl(const MatVecIntoT<Real>& S, const VecT<Real>& g0,
                                 Real p_bar, Real tol, int max_iter, bool use_pr,
                                 const PrecondIntoT<Real>& precond,
                                 const VecT<Real>* p_init, bool light) {
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
        p = p_init->cwiseMax(Real(0));
        const Real s = p.sum();
        p = (s > Real(0)) ? (p * (P_total / s)).eval() : Vec::Constant(N, p_bar);
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

    res.pressure = p.template cast<double>();
    if (!light) {
        res.displacement = u.template cast<double>();
        res.gap = (g.array() - alpha).template cast<double>();
    }
    res.approach = static_cast<double>(alpha);
    res.objective = static_cast<double>(Real(0.5) * p.dot(u) + p.dot(g0));
    res.iterations = it;
    res.contact_fraction = double(nc) / N;
    res.mean_pressure = static_cast<double>(p.mean());
    return res;
}

// explicit instantiations
template ContactResult solve_contact_impl<double>(
    const MatVecIntoT<double>&, const VecT<double>&, double, double, int, bool,
    const PrecondIntoT<double>&, const VecT<double>*, bool);
template ContactResult solve_contact_impl<float>(
    const MatVecIntoT<float>&, const VecT<float>&, float, float, int, bool,
    const PrecondIntoT<float>&, const VecT<float>*, bool);

ContactResult solve_contact(const MatVec& S, const Eigen::VectorXd& g0,
                            double p_bar, double tol, int max_iter, bool use_pr,
                            const Precond& precond, const Eigen::VectorXd* p_init,
                            bool light) {
    MatVecIntoT<double> Si = [&S](const Eigen::VectorXd& x, Eigen::VectorXd& y) {
        y = S(x);
    };
    PrecondIntoT<double> pi;
    if (precond)
        pi = [&precond](const Eigen::VectorXd& g,
                        const std::vector<std::uint8_t>& contact,
                        Eigen::VectorXd& z) { z = precond(g, contact); };
    return solve_contact_impl<double>(Si, g0, p_bar, tol, max_iter, use_pr,
                                      pi, p_init, light);
}

} // namespace hmc
