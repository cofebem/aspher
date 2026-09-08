#include "friction_solve.hpp"

#include <cmath>
#include <stdexcept>

namespace hmc {

// Interior test used for classification and the direction projection.
// The clamp writes |q| = s up to one rounding, so "interior" needs a small
// relative guard band; 1e-9 is far above roundoff and far below any
// physically meaningful traction margin.
static inline bool interior_point(double qx, double qy, double si) {
    return std::hypot(qx, qy) < si * (1.0 - 1e-9);
}

// ── displacement-control core ────────────────────────────────────────────────
// Projected CG (vector Polonsky–Keer) for
//   min ½ qᵀC q − Σ_i q_i·δ_t   s.t. |q_i| ≤ s_i
// over the candidate set A = {s > 0}, with the two-metric projection (the
// outward-radial residual at bound points is the Lagrange multiplier and
// must not drive the search), β-restart on stick/slip partition changes,
// best-iterate tracking, and a steepest-descent fallback when the
// (preconditioned) direction loses descent. Force control is handled by an
// outer multiplier iteration in solve_tangential — this core never sees it,
// so the preconditioner is always applied without mean removal.
static TangentialResult
solve_disp(const TanMatVecInto& C, const Eigen::VectorXd& s,
           const std::vector<std::uint8_t>& active, double Stotal,
           const Eigen::Vector2d& dt, double tol, int max_iter, bool use_pr,
           const TanPrecondInto& precond, const Eigen::VectorXd* q_init,
           const Eigen::VectorXd* u_hist, double g_floor) {
    const int N = static_cast<int>(s.size());

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
    }

    Eigen::VectorXd u(2 * N), g(2 * N), g_prev(2 * N), z(2 * N), t(2 * N),
        r(2 * N);
    t.setZero();
    g_prev.setZero();
    Eigen::VectorXd q_best = q;
    const Eigen::VectorXd q_start = q; // post-clamp initial iterate
    double err_best = 1e300;

    TangentialResult res;
    double G_old = 1.0, delta_conj = 0.0, g_scale = 0.0;
    double best_err = 1e300;
    int stall = 0;
    const int stall_limit = 200;
    // The |k| tangential preconditioner is a large win in partial slip
    // (271 -> 126 iterations at Ns=64) but its metric mismatches the
    // strongly clamped near-gross-slip regime, where the iteration stalls
    // one to three orders of magnitude above the reachable residual
    // (measured on the A15 held-displacement relaxation: proj residual
    // 1.2e-2 preconditioned vs 2.5e-5 unpreconditioned on the same
    // problem). On the first stall, drop the preconditioner and continue
    // from the best iterate instead of giving up.
    bool use_precond = static_cast<bool>(precond);
    bool dropped_precond = false;

    // two-metric projection of the direction source z (see header comment)
    auto strip_bound_radial = [&]() {
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
    };

    int it = 0;
    for (it = 0; it < max_iter; ++it) {
        C(q, u);

        // residual g = u − δ_t on A; KKT stationarity error:
        //   interior: (s−|q|)·|g|      (PK's p·|g| analog)
        //   bound:    s·(|g_perp| + max(0, g·q̂))   (misalignment + wrong-sign
        //             multiplier — g = −w, so slip requires g·q̂ = −λ ≤ 0)
        double e = 0.0, gmax = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : e) reduction(max : gmax)
        for (int i = 0; i < N; ++i) {
            if (!active[i]) { g(i) = 0.0; g(N + i) = 0.0; continue; }
            const double hx = u_hist ? (*u_hist)(i) : 0.0;
            const double hy = u_hist ? (*u_hist)(N + i) : 0.0;
            const double gx = u(i) + hx - dt(0), gy = u(N + i) + hy - dt(1);
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
        if (it == 0) g_scale = std::max((gmax > 0.0) ? gmax : 1.0, g_floor);
        res.error = e / (Stotal * g_scale);
        if (res.error < err_best) {
            err_best = res.error;
            q_best = q;
        }
        if (res.error < tol) {
            res.converged = true;
            res.status = SolveStatus::converged;
            break;
        }
        if (res.error < best_err * (1.0 - 1e-4)) {
            best_err = res.error;
            stall = 0;
        } else if (++stall >= (dropped_precond ? 5 * stall_limit : stall_limit)) {
            // the unpreconditioned restart gets a generous window: it resumes
            // from a drifted iterate and the near-gross-slip angular problem
            // converges slowly but steadily (measured 1.8e-2 -> 3.4e-4)
            if (use_precond && !dropped_precond) {
                use_precond = false;
                dropped_precond = true;
                // restart from the ORIGINAL iterate, not from the drifted
                // preconditioned best: resuming from the drift converged an
                // order of magnitude worse on the measured case (1.3e-3 vs
                // 3.4e-4). q_best/err_best keep tracking across both phases,
                // so the returned state is the better of the two.
                q = q_start;
                stall = 0;
                best_err = 1e300;
                delta_conj = 0.0;     // restart conjugacy in the new metric
                G_old = 1.0;
                t.setZero();
                g_prev.setZero();
                continue;
            }
            // A15 §10.2: no stall-success exception. The best iterate is
            // still returned, but the caller learns the metric floor was hit
            // rather than the tolerance met; the final KKT check decides.
            res.status = SolveStatus::stagnated;
            res.status_reason = dropped_precond ? "stall (unpreconditioned)"
                                                : "stall";
            break;
        }

        // direction + line-search quantities; one fallback attempt with the
        // raw projected residual if the (preconditioned + conjugated)
        // direction is not a descent direction. Without the fallback the
        // num<=0 `continue` recomputes the identical direction forever
        // (observed deadlock: frozen iterate, endless restarts).
        double num = 0.0, den = 0.0;
        for (int attempt = 0;; ++attempt) {
            if (attempt == 0) {
                if (use_precond) {
                    precond(g, active, /*remove_mean=*/false, z);
                } else {
#pragma omp parallel for schedule(static)
                    for (int i = 0; i < N; ++i) {
                        z(i) = active[i] ? g(i) : 0.0;
                        z(N + i) = active[i] ? g(N + i) : 0.0;
                    }
                }
            } else {
                // steepest descent on the projected residual: num becomes
                // ||g_proj||² > 0 unless the projected residual vanishes
#pragma omp parallel for schedule(static)
                for (int i = 0; i < N; ++i) {
                    z(i) = active[i] ? g(i) : 0.0;
                    z(N + i) = active[i] ? g(N + i) : 0.0;
                }
                delta_conj = 0.0;
            }
            strip_bound_radial();

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
                delta_conj *
                (use_pr ? std::max(0.0, G_pr / G_old) : G / G_old);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) {
                t(i) = active[i] ? z(i) + beta * t(i) : 0.0;
                t(N + i) = active[i] ? z(N + i) + beta * t(N + i) : 0.0;
            }
            G_old = (G > 0.0) ? G : 1.0;

            // exact line search on A
            C(t, r);
            num = 0.0;
            den = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : num, den)
            for (int i = 0; i < N; ++i)
                if (active[i]) {
                    num += g(i) * t(i) + g(N + i) * t(N + i);
                    den += r(i) * t(i) + r(N + i) * t(N + i);
                }
            if (num > 0.0 || attempt == 1) break;
        }
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            g_prev(i) = g(i);
            g_prev(N + i) = g(N + i);
        }
        if (den <= 0.0 || num <= 0.0) {
            // projected residual vanished or curvature lost on A: nothing
            // productive this iteration; the stall guard bounds the exit
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
    }

    // Return the BEST iterate seen, not the last: near the metric floor
    // τ = num/den is noise-dominated and the iterate wanders off the
    // solution (measured: err reaches ~7e-6, then oscillates 1e-3–1e-2).
    // Everything downstream (classification, counts, q_mean, res.q) works
    // off the best iterate.
    q = q_best;
    res.error = err_best;

    res.delta_t = dt;
    res.q_mean = Eigen::Vector2d(q.head(N).mean(), q.tail(N).mean());
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

// ── final local KKT check on a candidate tangential state (A15 §10.2) ──────
// Everything is recomputed from scratch on res.q: cone feasibility, stick and
// slip residuals, the fixed-threshold projection residual
//   R_rho = q - Pi_{|q|<=s}[q - rho (C q + u_hist - E delta)]
// and the force balance. rho is dimensionally a traction per displacement and
// is fixed by the operator's own compliance scale (rho = max(s) / max|C(s,0)|),
// NOT by any preconditioner scaling and never shrunk to make a residual look
// small.
void check_tangential_kkt(const TanMatVecInto& C, const Eigen::VectorXd& s,
                          const Eigen::Vector2d& delta_t, bool force_control,
                          const Eigen::Vector2d& target,
                          const Eigen::VectorXd* u_hist,
                          TangentialResult& res) {
    const int N = static_cast<int>(s.size());
    if (static_cast<int>(res.q.size()) != 2 * N)
        throw std::invalid_argument("check_tangential_kkt: q size");
    Eigen::VectorXd qs = Eigen::VectorXd::Zero(2 * N), us(2 * N);
    qs.head(N) = s;
    C(qs, us);
    const double w_ref = std::max(us.head(N).cwiseAbs().maxCoeff(), 1e-300);
    const double s_ref = std::max(s.maxCoeff(), 1e-300);
    const double rho = s_ref / w_ref;
    Eigen::VectorXd u(2 * N);
    C(res.q, u);

    double cone = 0.0, stick = 0.0, slip = 0.0, proj = 0.0;
    double qsx = 0.0, qsy = 0.0;
#pragma omp parallel for schedule(static) reduction(max : cone, stick, slip, proj)     reduction(+ : qsx, qsy)
    for (int i = 0; i < N; ++i) {
        const double qx = res.q(i), qy = res.q(N + i);
        qsx += qx;
        qsy += qy;
        const double si = s(i);
        const double qn = std::hypot(qx, qy);
        if (si <= 0.0) { // open: q must be exactly zero
            cone = std::max(cone, qn / s_ref);
            continue;
        }
        cone = std::max(cone, (qn - si) / s_ref);
        const double hx = u_hist ? (*u_hist)(i) : 0.0;
        const double hy = u_hist ? (*u_hist)(N + i) : 0.0;
        const double gx = u(i) + hx - delta_t(0);
        const double gy = u(N + i) + hy - delta_t(1);
        // projection residual (dimensionally a traction)
        const double zx = qx - rho * gx, zy = qy - rho * gy;
        const double zn = std::hypot(zx, zy);
        const double f = (zn > si && zn > 0.0) ? si / zn : 1.0;
        proj = std::max(proj, std::hypot(qx - f * zx, qy - f * zy) / s_ref);
        // slip velocity w = -g
        const double wx = -gx, wy = -gy;
        if (interior_point(qx, qy, si)) {
            stick = std::max(stick, std::hypot(wx, wy) / w_ref);
        } else {
            const double qhx = qx / qn, qhy = qy / qn;
            const double wpar = wx * qhx + wy * qhy;
            const double wperp = std::hypot(wx - wpar * qhx, wy - wpar * qhy);
            slip = std::max(slip, (wperp + std::max(0.0, -wpar)) / w_ref);
        }
    }
    res.cone_violation = std::max(0.0, cone);
    res.stick_residual = stick;
    res.slip_residual = slip;
    res.proj_residual = proj;
    res.rho = rho;
    res.s_ref = s_ref;
    res.w_ref = w_ref;
    const Eigen::Vector2d qm(qsx / N, qsy / N);
    res.force_error =
        force_control ? (qm - target).norm() / s_ref : 0.0;
}

// ── public entry: displacement control directly; force control by an outer
// multiplier (Newton/Broyden) iteration on the rigid shift ─────────────────
TangentialResult solve_tangential(const TanMatVecInto& C,
                                  const Eigen::VectorXd& s,
                                  bool force_control,
                                  const Eigen::Vector2d& target, double tol,
                                  int max_iter, bool use_pr,
                                  const TanPrecondInto& precond,
                                  const Eigen::VectorXd* q_init,
                                  const Eigen::VectorXd* u_hist,
                                  double g_floor,
                                  Eigen::Matrix2d* K_io,
                                  const Eigen::Vector2d* delta_init,
                                  double kkt_tol) {
    const int N = static_cast<int>(s.size());
    if (N == 0) throw std::invalid_argument("solve_tangential: empty s");
    if (s.minCoeff() < 0.0)
        throw std::invalid_argument("solve_tangential: negative threshold");

    std::vector<std::uint8_t> active(N);
    int nA = 0;
    double Stotal = 0.0;
    for (int i = 0; i < N; ++i) {
        active[i] = (s(i) > 0.0) ? 1 : 0;
        if (active[i]) { ++nA; Stotal += s(i); }
    }
    if (nA == 0)
        throw std::invalid_argument("solve_tangential: s == 0 everywhere");

    // Final acceptance (A15 §10.2): local KKT recomputed on the RETURNED
    // state after every correction. The iteration metric alone — and, under
    // force control, an exact total force — cannot establish local
    // equilibrium, and the stall exit is not success.
    // Default local-KKT acceptance. This is NOT derived from `tol` (the
    // iteration metric and the projection residual have different
    // normalisations); it is the tangential solver's measured floor on the
    // hardest supported regime — near-gross-slip with rotating directions,
    // where the uniform-threshold Tresca reference sits at 7.3e-3 (test_uzawa)
    // and force control at ~1e-3..1e-4. Partial-slip problems land three to
    // five orders below it. Callers who need a stricter contract pass
    // kkt_tol explicitly and get an honest failure when it is not met;
    // lowering the DEFAULT needs the A16 semismooth-Newton branch, not a
    // looser residual.
    const double kt = (kkt_tol > 0.0) ? kkt_tol : 1e-2;
    auto finish = [&](TangentialResult& r, const Eigen::Vector2d& delta_used) {
        check_tangential_kkt(C, s, delta_used, force_control, target, u_hist, r);
        const bool kkt_ok = r.cone_violation <= 1e-9 &&
                            r.proj_residual <= kt &&
                            r.force_error <= kt && std::isfinite(r.proj_residual);
        if (!kkt_ok) {
            if (r.status == SolveStatus::converged)
                r.status = SolveStatus::verification_failed;
            r.status_reason = r.status_reason.empty() ? "local_kkt" : r.status_reason;
        }
        r.converged = kkt_ok && (r.status == SolveStatus::converged ||
                                 r.status == SolveStatus::stagnated);
        r.kkt_tol = kt;
        if (r.converged) {
            r.status = SolveStatus::converged;
            r.status_reason.clear();
            // Visibility, not a failure: a state that passes but sits within a
            // decade of its own acceptance threshold is riding the solver's
            // floor, and the caller should read proj_residual rather than
            // trust `converged` alone. Scaling the notice to kkt_tol keeps it
            // meaningful when the caller tightens the contract.
            if (r.proj_residual > 0.1 * kt)
                r.status_reason = "local_kkt_near_tolerance";
        }
    };

    if (!force_control) {
        TangentialResult r = solve_disp(C, s, active, Stotal, target, tol,
                                        max_iter, use_pr, precond, q_init,
                                        u_hist, g_floor);
        finish(r, target);
        return r;
    }

    // ── force control ──
    // δ_t is the Lagrange multiplier of mean(q) = q̄: solve F(δ) = 0 with
    // F(δ) = q_mean(δ) − q̄ by Newton/Broyden on the 2×2 stiffness. The map
    // δ → q_mean is monotone (SPD full-stick stiffness, softening with
    // slip), so with K initialized to the FULL-STICK stiffness (from two
    // ε-probe displacement solves, exact in that linear regime) the Newton
    // steps under-shoot and approach the root stably; Broyden updates
    // recover the softened slope as slip develops. Each inner solve is
    // warm-started from the previous iterate. This replaces the abandoned
    // per-iteration additive load correction, whose coupling with the
    // interior-mean δ_t estimate and the exact line search was an unstable
    // period-2 feedback at high loads (gain ≈ 1.5/iteration at 40% slip).
    const double q_limit = Stotal / N; // gross-slip bound: |mean q| < mean s
    if (target.norm() >= q_limit)
        throw std::invalid_argument(
            "solve_tangential: |q_bar| >= mean(s) — at or beyond the "
            "gross-slip limit, no equilibrium exists");
    TangentialResult res;
    // Trivial shortcut only holds cold and without history: with u_hist or a
    // warm start, a zero-target force solve (e.g. full unload to zero mean
    // traction after loading) is a REAL solve — the answer has locked-in
    // nonzero tractions with zero mean, not q = 0. Never fabricate K_io here
    // either way; leave it untouched (the caller's carried stiffness, if
    // any, stays valid for the next step).
    if (target.norm() == 0.0 && !u_hist && !q_init) { // trivial: q = 0, δ = 0
        res = solve_disp(C, s, active, Stotal, Eigen::Vector2d::Zero(),
                         tol, 1, use_pr, precond, nullptr, u_hist, g_floor);
        res.status = SolveStatus::converged;
        finish(res, Eigen::Vector2d::Zero());
        return res;
    }

    // full-stick 2×2 stiffness from two ε-probes (F(0) = −q̄ needs no solve:
    // δ = 0 ⇒ q = 0 exactly). Edge points slip lightly, so K is slightly soft
    // and the first Newton step overshoots; absorbed by Broyden updates and
    // floor detection. Robustness guard: if probes slip heavily (>20%), halve
    // eps and redo. If K_io is provided with positive determinant, skip probes
    // and use the provided K.
    Eigen::Matrix2d K;
    int total_probe_it = 0;
    const bool run_probes = !(K_io && K_io->determinant() > 0.0);
    if (run_probes) {
        // displacement scale for the ε-probes: one matvec on the gross-slip
        // x-directed field q = (s, 0) gives the compliance scale of the problem
        Eigen::VectorXd qs = Eigen::VectorXd::Zero(2 * N), us(2 * N);
        qs.head(N) = s;
        C(qs, us);
        const double u_scale = us.head(N).cwiseAbs().maxCoeff();
        if (u_scale <= 0.0) throw std::runtime_error("solve_tangential: degenerate tangential operator (C(s,0) == 0)");
        const double eps = 1e-4 * u_scale; // edge points slip lightly, K slightly soft, first Newton step overshoots

        if (!u_hist) {
            double eps_probe = eps;
            for (int tries = 0; tries < 2; ++tries) {
                TangentialResult px =
                    solve_disp(C, s, active, Stotal, Eigen::Vector2d(eps_probe, 0.0),
                               tol, max_iter, use_pr, precond, nullptr, u_hist, g_floor);
                TangentialResult py =
                    solve_disp(C, s, active, Stotal, Eigen::Vector2d(0.0, eps_probe),
                               tol, max_iter, use_pr, precond, nullptr, u_hist, g_floor);
                const int total_slip = px.n_slip + py.n_slip;
                const int total_cand = px.n_slip + px.n_stick + py.n_slip + py.n_stick;
                if (tries == 0 && total_slip > total_cand / 5) {
                    // probes slipping >20%, halve eps and retry (once only)
                    eps_probe *= 0.5;
                    continue;
                }
                // K always assigned: either guard passes on first try, or second try
                // always computes it. Slightly soft K on edge slip is fine—Broyden
                // updates absorb it.
                K.col(0) = px.q_mean / eps_probe;
                K.col(1) = py.q_mean / eps_probe;
                total_probe_it = px.iterations + py.iterations;
                break;
            }
        } else {
            // With history present, the residual at q=0 is dominated by the
            // history term, so a cold two-probe scheme (base F(0)=0 assumed
            // analytically, probes solved from q=0) measures K far from the
            // actual operating point. Probe DIFFERENTIALLY instead: solve a
            // base point at the intended starting shift, then two
            // warm-chained perturbed solves around it, and take the finite
            // difference of q_mean. No 20%-slip eps-halving retry here —
            // that guard exists to keep the cold probes in the linear regime
            // near q=0; differential probes at the operating point don't
            // need it.
            const Eigen::Vector2d delta0 =
                delta_init ? *delta_init : Eigen::Vector2d::Zero();
            TangentialResult base = solve_disp(
                C, s, active, Stotal, delta0, tol, max_iter, use_pr, precond,
                q_init, u_hist, g_floor);
            TangentialResult px = solve_disp(
                C, s, active, Stotal, delta0 + Eigen::Vector2d(eps, 0.0), tol,
                max_iter, use_pr, precond, &base.q, u_hist, g_floor);
            TangentialResult py = solve_disp(
                C, s, active, Stotal, delta0 + Eigen::Vector2d(0.0, eps), tol,
                max_iter, use_pr, precond, &px.q, u_hist, g_floor);
            K.col(0) = (px.q_mean - base.q_mean) / eps;
            K.col(1) = (py.q_mean - base.q_mean) / eps;
            total_probe_it = base.iterations + px.iterations + py.iterations;
            if (K.determinant() <= 0.0)
                throw std::runtime_error(
                    "solve_tangential: probe stiffness degenerate");
        }
    } else {
        K = *K_io;
    }

    // The inner solver resolves q_mean only to its metric floor (~1e-4
    // relative on hard partial-slip problems), so: stop the outer iteration
    // when F stops responding to delta (floor reached), never Broyden-update
    // on a noise secant (that collapses K: observed det 1.7e-1 -> 1e-17 ->
    // Newton blow-up), keep the best-|F| iterate, and meet the load EXACTLY
    // afterwards with a terminal additive correction on interior points
    // (single safe passes — iterating this correction inside the CG loop was
    // the abandoned unstable scheme).
    const double ftol = 1e-8;
    const int outer_max = 40;
    // Relative outer tolerances are normalized by target.norm(); that
    // degenerates to an unreachable exact-zero requirement for a target of
    // EXACTLY zero (e.g. full unload with history — Finding 1). Floor the
    // normalization at q_limit (the natural force scale, |mean q| < mean s)
    // for that edge case only; for any target.norm() > 0 this is identical
    // to the previous behavior.
    const double target_scale = (target.norm() > 0.0) ? target.norm() : q_limit;
    Eigen::Vector2d delta = delta_init ? *delta_init : Eigen::Vector2d(K.inverse() * target); // full-stick Newton start
    Eigen::Vector2d delta_prev = Eigen::Vector2d::Zero();
    Eigen::Vector2d F_prev = Eigen::Vector2d::Zero();
    bool have_prev = false;
    Eigen::VectorXd q_warm;
    if (q_init) q_warm = *q_init; // per-step warm start (M5 driver); solve_disp re-clamps
    int total_it = total_probe_it;
    bool best_converged = false;
    double F_best = 1e300;
    Eigen::Vector2d delta_best = delta;

    for (int outer = 0; outer < outer_max; ++outer) {
        TangentialResult inner = solve_disp(
            C, s, active, Stotal, delta, tol, max_iter, use_pr, precond,
            q_warm.size() ? &q_warm : nullptr, u_hist, g_floor);
        total_it += inner.iterations;
        const Eigen::Vector2d F = inner.q_mean - target;
        q_warm = inner.q; // warm-start the next inner solve (copy; res moves)
        if (F.norm() < F_best) {
            F_best = F.norm();
            delta_best = delta;
            best_converged = inner.converged;
            res = std::move(inner);
            res.delta_t = delta;
        }
        if (F.norm() <= ftol * target_scale) break;
        // floor detection: F no longer responds to delta at the inner
        // solver's resolution — a secant from here is pure noise
        if (have_prev && (F - F_prev).norm() <= 1e-3 * std::max(F.norm(), 1e-300)) break;

        // Broyden update of K from the observed secant, then Newton step
        //   K <- K + ((dF - K dd) dd^T) / |dd|^2,  delta <- delta - K^-1 F
        if (have_prev) {
            const Eigen::Vector2d dd = delta - delta_prev;
            const double dd2 = dd.squaredNorm();
            if (dd2 > 0.0) {
                const Eigen::Matrix2d K_upd =
                    K + ((F - F_prev) - K * dd) * dd.transpose() / dd2;
                // accept the update only while it stays safely non-degenerate
                if (K_upd.determinant() > 1e-6 * std::abs(K.determinant()))
                    K = K_upd;
            }
        }
        delta_prev = delta;
        F_prev = F;
        have_prev = true;
        const Eigen::Vector2d step = -K.inverse() * F;
        if (!step.allFinite())
            throw std::runtime_error(
                "solve_tangential: force-control step not finite");
        delta += step;
    }

    // terminal exact-load enforcement: distribute the residual load over the
    // interior (stick) points, re-clamping the few that spill past their
    // bound; two or three passes reach roundoff. The perturbation is
    // O(load residual) ~ 1e-4 |q_bar| and shows up in the KKT residuals at
    // the same harmless order.
    {
        Eigen::VectorXd& q = res.q;
        for (int pass = 0; pass < 3; ++pass) {
            double qsx = 0.0, qsy = 0.0;
            int ni = 0;
#pragma omp parallel for schedule(static) reduction(+ : qsx, qsy, ni)
            for (int i = 0; i < N; ++i) {
                qsx += q(i);
                qsy += q(N + i);
                if (active[i] && interior_point(q(i), q(N + i), s(i))) ++ni;
            }
            const double cx = target(0) * N - qsx, cy = target(1) * N - qsy;
            if (!ni || (std::abs(cx) + std::abs(cy)) <= 1e-14 * N * target_scale)
                break;
#pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) {
                if (!active[i] || !interior_point(q(i), q(N + i), s(i)))
                    continue;
                double qx = q(i) + cx / ni, qy = q(N + i) + cy / ni;
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
        res.q_mean = Eigen::Vector2d(q.head(N).mean(), q.tail(N).mean());
        // reclassify after the correction
        res.state.assign(N, 0);
        int nst = 0, nsl = 0, nop = 0;
        for (int i = 0; i < N; ++i) {
            if (!active[i]) { ++nop; continue; }
            if (interior_point(q(i), q(N + i), s(i))) { res.state[i] = 1; ++nst; }
            else { res.state[i] = 2; ++nsl; }
        }
        res.n_stick = nst;
        res.n_slip = nsl;
        res.n_open = nop;
    }
    // res.error still describes the pre-correction iterate; the local KKT
    // fields below are recomputed on the corrected q, which is what the
    // acceptance decision uses.
    res.delta_t = delta_best;
    res.iterations = total_it;
    res.status = (best_converged && F_best <= 1e-3 * target_scale)
                     ? SolveStatus::converged
                     : SolveStatus::stagnated;
    if (res.status != SolveStatus::converged)
        res.status_reason = "outer force iteration";
    finish(res, delta_best);
    if (K_io) *K_io = K;
    return res;
}

} // namespace hmc
