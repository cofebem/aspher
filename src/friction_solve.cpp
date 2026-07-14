#include "friction_solve.hpp"

#include <cmath>
#include <stdexcept>

namespace hmc {

// Interior test used for classification and the force-control mean pass.
// The clamp writes |q| = s up to one rounding, so "interior" needs a small
// relative guard band; 1e-9 is far above roundoff and far below any
// physically meaningful traction margin.
static inline bool interior_point(double qx, double qy, double si) {
    return std::hypot(qx, qy) < si * (1.0 - 1e-9);
}

TangentialResult solve_tangential(const TanMatVecInto& C,
                                  const Eigen::VectorXd& s,
                                  bool force_control,
                                  const Eigen::Vector2d& target, double tol,
                                  int max_iter, bool use_pr,
                                  const TanPrecondInto& precond,
                                  const Eigen::VectorXd* q_init) {
    const int N = static_cast<int>(s.size());
    if (N == 0) throw std::invalid_argument("solve_tangential: empty s");
    if (s.minCoeff() < 0.0)
        throw std::invalid_argument("solve_tangential: negative threshold");

    // candidate set A = { s > 0 }
    std::vector<std::uint8_t> active(N);
    int nA = 0;
    double Stotal = 0.0;
    for (int i = 0; i < N; ++i) {
        active[i] = (s(i) > 0.0) ? 1 : 0;
        if (active[i]) { ++nA; Stotal += s(i); }
    }
    if (nA == 0)
        throw std::invalid_argument("solve_tangential: s == 0 everywhere");

    Eigen::VectorXd q(2 * N);
    if (q_init) {
        if (static_cast<int>(q_init->size()) != 2 * N)
            throw std::invalid_argument("solve_tangential: q_init size");
        q = *q_init;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            if (!active[i]) { q(i) = 0.0; q(N + i) = 0.0; continue; }
            const double qn = std::hypot(q(i), q(N + i));
            if (qn > s(i)) {
                const double f = s(i) / qn;
                q(i) *= f;
                q(N + i) *= f;
            }
        }
    } else {
        q.setZero();
        if (force_control) {
            // uniform start carrying the load over A, clamped per disk
            const double f = static_cast<double>(N) / nA;
            const double q0x = target(0) * f, q0y = target(1) * f;
#pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) {
                if (!active[i]) continue;
                double qx = q0x, qy = q0y;
                const double qn = std::hypot(qx, qy);
                if (qn > s(i)) {
                    const double fr = s(i) / qn;
                    qx *= fr;
                    qy *= fr;
                }
                q(i) = qx;
                q(N + i) = qy;
            }
        }
    }

    Eigen::VectorXd u(2 * N), g(2 * N), g_prev(2 * N), z(2 * N), t(2 * N),
        r(2 * N);
    t.setZero();
    g_prev.setZero();
    Eigen::VectorXd q_best = q;
    double err_best = 1e300;

    TangentialResult res;
    double G_old = 1.0, delta_conj = 0.0, g_scale = 0.0;
    double best_err = 1e300;
    int stall = 0;
    const int stall_limit = 200;
    Eigen::Vector2d dt = target; // displacement control: fixed

    int it = 0;
    for (it = 0; it < max_iter; ++it) {
        C(q, u);

        if (force_control) {
            // rigid shift = interior (stick) mean of u; fall back to the
            // whole candidate set when nothing is interior
            double sx = 0.0, sy = 0.0, sax = 0.0, say = 0.0;
            int ni = 0;
#pragma omp parallel for schedule(static) reduction(+ : sx, sy, sax, say, ni)
            for (int i = 0; i < N; ++i) {
                if (!active[i]) continue;
                sax += u(i);
                say += u(N + i);
                if (interior_point(q(i), q(N + i), s(i))) {
                    sx += u(i);
                    sy += u(N + i);
                    ++ni;
                }
            }
            dt(0) = ni ? sx / ni : sax / nA;
            dt(1) = ni ? sy / ni : say / nA;
        }

        // residual g = u − δ_t on A; KKT stationarity error:
        //   interior: (s−|q|)·|g|      (PK's p·|g| analog)
        //   bound:    s·(|g_perp| + max(0, g·q̂))   (misalignment + wrong-sign
        //             multiplier — g = −w, so slip requires g·q̂ = −λ ≤ 0)
        double e = 0.0, gmax = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : e) reduction(max : gmax)
        for (int i = 0; i < N; ++i) {
            if (!active[i]) { g(i) = 0.0; g(N + i) = 0.0; continue; }
            const double gx = u(i) - dt(0), gy = u(N + i) - dt(1);
            g(i) = gx;
            g(N + i) = gy;
            const double gn = std::hypot(gx, gy);
            gmax = std::max(gmax, gn);
            const double qn = std::hypot(q(i), q(N + i));
            if (interior_point(q(i), q(N + i), s(i))) {
                e += (s(i) - qn) * gn;
            } else {
                const double qhx = q(i) / qn, qhy = q(N + i) / qn;
                const double gpar = gx * qhx + gy * qhy;
                const double gperp =
                    std::hypot(gx - gpar * qhx, gy - gpar * qhy);
                e += s(i) * (gperp + std::max(0.0, gpar));
            }
        }
        if (it == 0) g_scale = (gmax > 0.0) ? gmax : 1.0;
        res.error = e / (Stotal * g_scale);
        if (res.error < err_best) { err_best = res.error; q_best = q; }
        if (res.error < tol) {
            res.converged = true;
            break;
        }
        if (res.error < best_err * (1.0 - 1e-4)) {
            best_err = res.error;
            stall = 0;
        } else if (++stall >= stall_limit) {
            res.converged = true;
            break;
        }

        // preconditioned residual on the candidate mask
        if (precond) {
            precond(g, active, force_control, z);
        } else {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) {
                z(i) = active[i] ? g(i) : 0.0;
                z(N + i) = active[i] ? g(N + i) : 0.0;
            }
        }

        // Two-metric projection: at bound points the radial residual is the
        // Lagrange multiplier (g = −λ q̂, λ ≥ 0) — unreducible, and letting
        // it drive the search direction corrupts the line search (observed:
        // objective oscillation, stall at ~1e-2 error). Strip the outward
        // component (zpar < 0 means the −z update pushes outward); keep the
        // tangential part (rotates slip direction) and inward part (lets
        // wrong-sign points re-enter the interior).
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            if (!active[i]) continue;
            if (interior_point(q(i), q(N + i), s(i))) continue;
            const double qn = std::hypot(q(i), q(N + i));
            const double qhx = q(i) / qn, qhy = q(N + i) / qn;
            const double zpar = z(i) * qhx + z(N + i) * qhy;
            if (zpar < 0.0) {
                z(i) -= zpar * qhx;
                z(N + i) -= zpar * qhy;
            }
        }

        // conjugate direction over A (M-inner products in double)
        double G = 0.0, G_pr = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : G, G_pr)
        for (int i = 0; i < N; ++i)
            if (active[i]) {
                G += z(i) * g(i) + z(N + i) * g(N + i);
                G_pr += z(i) * (g(i) - g_prev(i)) +
                        z(N + i) * (g(N + i) - g_prev(N + i));
            }
        const double beta =
            delta_conj * (use_pr ? std::max(0.0, G_pr / G_old) : G / G_old);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            t(i) = active[i] ? z(i) + beta * t(i) : 0.0;
            t(N + i) = active[i] ? z(N + i) + beta * t(N + i) : 0.0;
            g_prev(i) = g(i);
            g_prev(N + i) = g(N + i);
        }
        G_old = G;

        // exact line search on A; centred denominator under force control
        // (the additive mean correction projects out uniform shifts, so the
        // effective operator acts on mean-zero directions)
        C(t, r);
        double num = 0.0, rsx = 0.0, rsy = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : num, rsx, rsy)
        for (int i = 0; i < N; ++i)
            if (active[i]) {
                num += g(i) * t(i) + g(N + i) * t(N + i);
                rsx += r(i);
                rsy += r(N + i);
            }
        const double rmx = force_control ? rsx / nA : 0.0;
        const double rmy = force_control ? rsy / nA : 0.0;
        double den = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : den)
        for (int i = 0; i < N; ++i)
            if (active[i])
                den += (r(i) - rmx) * t(i) + (r(N + i) - rmy) * t(N + i);
        if (den <= 0.0 || num <= 0.0) {
            // non-descent direction (or indefinite curvature on A):
            // restart from steepest descent next iteration
            delta_conj = 0.0;
            continue;
        }
        const double tau = num / den;

        // update + radial clamp; β-restart only when the stick/slip
        // PARTITION changes (an interior point hitting its bound, or a bound
        // point moving strictly inside). Re-clamping an already-bound point
        // is NOT a set change — it happens every iteration while slip
        // directions rotate, and resetting β there degrades the CG to
        // steepest descent (observed: stall at ~1e-2 error).
        int set_changed = 0;
#pragma omp parallel for schedule(static) reduction(| : set_changed)
        for (int i = 0; i < N; ++i) {
            if (!active[i]) continue;
            const bool was_int = interior_point(q(i), q(N + i), s(i));
            double qx = q(i) - tau * t(i);
            double qy = q(N + i) - tau * t(N + i);
            const double qn = std::hypot(qx, qy);
            const bool clamp = (qn > s(i));
            if (clamp) {
                const double f = s(i) / qn;
                qx *= f;
                qy *= f;
            }
            q(i) = qx;
            q(N + i) = qy;
            const bool is_int = !clamp && interior_point(qx, qy, s(i));
            if (was_int != is_int) set_changed = 1;
        }
        delta_conj = set_changed ? 0.0 : 1.0;

        if (force_control) {
            throw std::logic_error("force control lands in M4c");
        }
    }

    // Return the BEST iterate seen, not the last: near the metric floor
    // τ = num/den is noise-dominated and the iterate wanders off the
    // solution (measured: err reaches ~7e-6, then oscillates 1e-3–1e-2).
    // Everything downstream (final δ_t, classification, counts, q_mean,
    // res.q) works off the best iterate.
    q = q_best;
    res.error = err_best;

    // finalize: recompute u, δ_t (force control), classification, counts
    C(q, u);
    if (force_control) {
        double sx = 0.0, sy = 0.0, sax = 0.0, say = 0.0;
        int ni = 0;
#pragma omp parallel for schedule(static) reduction(+ : sx, sy, sax, say, ni)
        for (int i = 0; i < N; ++i) {
            if (!active[i]) continue;
            sax += u(i);
            say += u(N + i);
            if (interior_point(q(i), q(N + i), s(i))) {
                sx += u(i);
                sy += u(N + i);
                ++ni;
            }
        }
        dt(0) = ni ? sx / ni : sax / nA;
        dt(1) = ni ? sy / ni : say / nA;
    }
    res.delta_t = dt;
    res.q_mean =
        Eigen::Vector2d(q.head(N).mean(), q.tail(N).mean());
    res.state.assign(N, 0);
    int nst = 0, nsl = 0, nop = 0;
    for (int i = 0; i < N; ++i) {
        if (!active[i]) {
            ++nop;
            continue;
        }
        if (interior_point(q(i), q(N + i), s(i))) {
            res.state[i] = 1;
            ++nst;
        } else {
            res.state[i] = 2;
            ++nsl;
        }
    }
    res.n_stick = nst;
    res.n_slip = nsl;
    res.n_open = nop;
    res.iterations = it;
    res.q = std::move(q);
    return res;
}

} // namespace hmc
