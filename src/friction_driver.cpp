#include "friction_driver.hpp"

#include <cmath>
#include <stdexcept>

namespace hmc {

FrictionDriver::FrictionDriver(int Ns, double L, double E_star, double nu,
                               const FrictionModel& model, bool precond)
    : Ns_(Ns), N_(Ns * Ns), L_(L), h_(L / Ns), model_(&model),
      precond_(precond), BK_(Ns, L, E_star), S_(BK_), CK_(Ns, L, E_star, nu),
      C_(CK_), Mn_(Ns), Mt_(Ns, nu) {
    S_.build();
    C_.build();
    reset();
}

void FrictionDriver::set_gap(const Eigen::VectorXd& g0) {
    if (static_cast<int>(g0.size()) != N_)
        throw std::invalid_argument("FrictionDriver::set_gap: size != Ns*Ns");
    g0_ = g0;
    reset();
}

void FrictionDriver::reset() {
    p_ = Eigen::VectorXd::Zero(N_);
    q_ = Eigen::VectorXd::Zero(2 * N_);
    u_t_ = Eigen::VectorXd::Zero(2 * N_);
    w_acc_ = Eigen::VectorXd::Zero(2 * N_);
    slip_prev_ = Eigen::VectorXd::Zero(2 * N_);
    delta_t_.setZero();
    K_.setZero();
    q_bar_last_.setZero();
    has_q_bar_last_ = false;
}

void FrictionDriver::fill_separated_normal(ContactResult& n) const {
    n.pressure = Eigen::VectorXd::Zero(N_);
    n.displacement = Eigen::VectorXd::Zero(N_);
    n.approach = g0_.minCoeff(); // just-touching convention (documented)
    n.gap = g0_.array() - n.approach;
    n.objective = 0.0;
    n.error = 0.0;
    n.pk_error = 0.0;
    n.fw_gap = 0.0;
    n.fw_error = 0.0;
    n.load_error = 0.0;
    n.pressure_violation = 0.0;
    n.penetration_error = 0.0;
    n.contact_fraction = 0.0;
    n.mean_pressure = 0.0;
    n.iterations = 0;
    n.status = SolveStatus::converged;
    n.converged = true;
    n.g_ref = std::max(g0_.maxCoeff() - g0_.minCoeff(), 1e-300);
    n.p_ref = 0.0;
}

FrictionStepResult FrictionDriver::step(const FrictionStepSpec& spec) {
    if (g0_.size() == 0)
        throw std::logic_error("FrictionDriver::step: set_gap first");
    if (spec.has_q_bar && spec.has_delta_t)
        throw std::invalid_argument(
            "FrictionDriver::step: q_bar and delta_t are exclusive");
    if (spec.dt <= 0.0)
        throw std::invalid_argument("FrictionDriver::step: dt <= 0");
    if (spec.max_threshold_iter <= 0)
        throw std::invalid_argument(
            "FrictionDriver::step: max_threshold_iter <= 0");
    if (spec.T && static_cast<int>(spec.T->size()) != N_)
        throw std::invalid_argument("FrictionDriver::step: T size");
    if (model_->velocity_dependent() && spec.max_threshold_iter < 2)
        throw std::invalid_argument(
            "FrictionDriver::step: with velocity-dependent model, "
            "max_threshold_iter must be >= 2");
    // ── normal load request (spec A15 §10.1) ──
    // "no update", "zero load" and "load X" are three different requests;
    // the old code mapped p_bar <= 0 to "skip", so asking for zero normal
    // load silently kept the previous pressure (review §13.2).
    bool want_normal;
    const double pb = spec.p_bar;
    if (spec.has_p_bar) {
        if (!(pb >= 0.0))
            throw std::invalid_argument(
                "FrictionDriver::step: p_bar must be >= 0 (0 = complete "
                "normal unloading); leave has_p_bar unset to hold the load");
        want_normal = true;
    } else if (pb > 0.0) {
        want_normal = true; // legacy positional use
    } else if (pb < 0.0) {
        want_normal = false; // legacy -1 sentinel
    } else {
        throw std::invalid_argument(
            "FrictionDriver::step: p_bar = 0 needs has_p_bar = true "
            "(complete normal unloading); leave p_bar unset (or -1) to hold "
            "the current normal load");
    }

    FrictionStepResult res;
    bool ok = true;

    // Everything below works on LOCAL candidates; persistent state (p_, q_,
    // u_t_, delta_t_, w_acc_, slip_prev_, K_) is committed only at the end,
    // and only if ok — see the transactional contract in the header.
    Eigen::VectorXd p_new = p_;

    // ── (0) complete normal unloading: p = q = u = 0 ──
    if (want_normal && pb == 0.0) {
        if (spec.has_q_bar && spec.q_bar.norm() > 0.0) {
            // a nonzero prescribed tangential force cannot be carried with
            // zero normal load in a nonadhesive model
            res.converged = false;
            res.status_reason = "nonzero q_bar at zero normal load";
            return res;
        }
        fill_separated_normal(res.normal);
        Eigen::Vector2d delta_new = delta_t_;
        if (spec.has_delta_t) delta_new = spec.delta_t; // rigid shift is free
        res.slip_inc = Eigen::VectorXd::Zero(2 * N_);
        res.dissipation = 0.0;
        res.converged = true;
        // Commit: zero fields, cleared traction-dependent warm state (so a
        // later recontact cannot resurrect stale tractions or a stale probe
        // stiffness), but the controlled rigid shift and the accumulated
        // slip history are preserved — separation does not undo what already
        // slipped (spec A15 §10.1).
        p_.setZero();
        q_.setZero();
        u_t_.setZero();
        slip_prev_.setZero();
        delta_t_ = delta_new;
        K_.setZero();
        has_q_bar_last_ = false;
        q_bar_last_.setZero();
        return res;
    }

    // ── (1) normal solve (uncoupled: p never feels q) ──
    if (want_normal) {
        MatVec Sop = [this](const Eigen::VectorXd& x) { return S_.matvec(x); };
        Precond Pn;
        if (precond_)
            Pn = [this](const Eigen::VectorXd& g,
                        const std::vector<std::uint8_t>& contact) {
                return Mn_.apply(g, contact);
            };
        const Eigen::VectorXd* warm = (p_.maxCoeff() > 0.0) ? &p_ : nullptr;
        res.normal = solve_contact(Sop, g0_, pb, spec.tol_normal,
                                   spec.max_iter, true, Pn, warm);
        ok = ok && res.normal.converged;
        if (!ok) {
            res.converged = false;
            res.status_reason = "normal solve: " + std::string(to_string(res.normal.status));
            return res;
        }
        p_new = res.normal.pressure;
    }

    Eigen::VectorXd q_new = q_, u_t_new = u_t_, w_acc_new = w_acc_,
                    slip_prev_new = slip_prev_;
    Eigen::Vector2d delta_t_new = delta_t_;
    Eigen::Matrix2d K_new = K_;

    // ── held tangential boundary condition on a normal-only step ──
    // Changing p changes the thresholds s = model(p,...), so stored shear
    // that now exceeds mu*p is out of equilibrium. The old code left it
    // untouched: after unloading p_bar 0.01 -> 0.0001 at Ns=32, 76 OPEN
    // points still carried shear with max excess |q| - mu p = 0.02362
    // (review §13.1). Clipping alone would enforce the cone but not
    // equilibrium, so the incremental problem is re-solved under the held
    // boundary condition.
    bool do_tan = spec.has_q_bar || spec.has_delta_t;
    bool tan_force = spec.has_q_bar;
    Eigen::Vector2d tan_target =
        spec.has_q_bar ? spec.q_bar : (spec.delta_t - delta_t_);
    bool is_hold = false;
    if (!do_tan && want_normal && q_.cwiseAbs().maxCoeff() > 0.0) {
        is_hold = true;
        do_tan = true;
        res.tangential_hold_applied = true;
        res.hold_applied = spec.tangential_hold;
        if (spec.tangential_hold == TangentialHold::force) {
            if (!has_q_bar_last_) {
                res.converged = false;
                res.status_reason =
                    "tangential_hold=force but no controlled force was ever "
                    "prescribed";
                return res;
            }
            tan_force = true;
            tan_target = q_bar_last_;
        } else {
            tan_force = false;
            tan_target.setZero(); // hold the TOTAL shift: increment is zero
        }
    }

    // ── (2)+(3) threshold loop + incremental tangential solve ──
    if (do_tan) {
        if (p_new.maxCoeff() <= 0.0)
            throw std::logic_error(
                "FrictionDriver::step: tangential step before any contact");
        TanMatVecInto Cop = [this](const Eigen::VectorXd& x,
                                   Eigen::VectorXd& y) { C_.matvec_into(x, y); };
        TanPrecondInto Mop;
        if (precond_)
            Mop = [this](const Eigen::VectorXd& g,
                         const std::vector<std::uint8_t>& mask, bool rm,
                         Eigen::VectorXd& z) { Mt_.apply_into(g, mask, rm, z); };

        const Eigen::VectorXd T =
            spec.T ? *spec.T : Eigen::VectorXd::Zero(N_);

        // increment targets: solver works in Δδ with u_hist = −uⁿ_t
        const Eigen::Vector2d target = tan_target;
        Eigen::VectorXd u_hist = -u_t_;
        // DISPLACEMENT units only. The old expression added |q_bar| — a
        // traction — to |u_t|, contradicting its own adjacent comment and
        // mixing units at non-unit E* (review §13.3). A cold first force
        // step gets floor 0; its it-0 gmax is an honest scale. If a
        // force-controlled floor is ever wanted it must come through a
        // documented compliance scale, not by adding a pressure to a
        // displacement.
        const double g_floor = 1e-6 * u_t_.cwiseAbs().maxCoeff();

        // velocity for the threshold: previous pass's slip increment
        // (first pass: previous STEP's — quasi-static continuation)
        Eigen::VectorXd v(N_), s;
        Eigen::VectorXd slip = slip_prev_;
        Eigen::VectorXd u_new(2 * N_);
        TangentialResult tan;
        double s_change = 1e300;
        int pass = 0;
        const int max_pass =
            model_->velocity_dependent() ? spec.max_threshold_iter : 1;
        Eigen::VectorXd s_old;
        Eigen::Vector2d dinit0;
        const Eigen::Vector2d* dinit0_p = nullptr;
        if (tan_force && K_new.determinant() > 0.0) {
            const Eigen::Vector2d q_mean_now(q_.head(N_).mean(),
                                             q_.tail(N_).mean());
            dinit0 = K_new.inverse() * (target - q_mean_now);
            dinit0_p = &dinit0;
        }
        Eigen::Vector2d dinit;
        const Eigen::Vector2d* dinit_p = nullptr;
        for (pass = 0; pass < max_pass; ++pass) {
            for (int i = 0; i < N_; ++i)
                v(i) = std::hypot(slip(i), slip(N_ + i)) / spec.dt;
            Eigen::VectorXd s_new;
            model_->threshold(p_new, v, T, s_new);
            if (pass == 0) {
                s = s_new;
            } else {
                s = 0.5 * (s + s_new); // damped fixed point
                s_change = (s - s_old).norm() / std::max(s.norm(), 1e-300);
            }
            s_old = s;
            // warm-start from the PREVIOUS PASS's solution (pass 0: the
            // previous step's committed q_ — quasi-static continuation)
            Eigen::VectorXd q_warm = (pass == 0) ? q_ : tan.q;
            try {
                tan = solve_tangential(Cop, s, tan_force, target,
                                       spec.tol_tangential, spec.max_iter, true,
                                       Mop, &q_warm, &u_hist, g_floor,
                                       tan_force ? &K_new : nullptr,
                                       pass == 0 ? dinit0_p : dinit_p,
                                       spec.tol_kkt);
            } catch (const std::invalid_argument& e) {
                // e.g. an infeasible held force (|q_bar| >= mean s after the
                // normal change). A request the caller did not make directly
                // fails transactionally instead of throwing.
                if (!is_hold) throw;
                res.converged = false;
                res.status_reason =
                    std::string("held tangential force infeasible: ") + e.what();
                return res;
            }
            dinit = tan.delta_t;
            dinit_p = &dinit;
            // slip increment of this candidate solution: Δw = Δδ − (u − uⁿ)
            C_.matvec_into(tan.q, u_new);
            for (int i = 0; i < N_; ++i) {
                slip(i) = tan.delta_t(0) - (u_new(i) - u_t_(i));
                slip(N_ + i) = tan.delta_t(1) - (u_new(N_ + i) - u_t_(N_ + i));
            }
            if (pass > 0 && s_change < spec.threshold_rtol) {
                ++pass;
                break;
            }
            if (!model_->velocity_dependent()) {
                ++pass;
                break;
            }
        }
        res.threshold_iters = pass;
        ok = ok && tan.converged &&
             (!model_->velocity_dependent() ||
              s_change < spec.threshold_rtol);

        // ── (4) candidate state update (u_new is the loop's final pass —
        // no need to recompute C q) ──
        res.slip_inc.resize(2 * N_);
        double D = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : D)
        for (int i = 0; i < N_; ++i) {
            const double dwx = tan.delta_t(0) - (u_new(i) - u_t_(i));
            const double dwy =
                tan.delta_t(1) - (u_new(N_ + i) - u_t_(N_ + i));
            res.slip_inc(i) = dwx;
            res.slip_inc(N_ + i) = dwy;
            D += tan.q(i) * dwx + tan.q(N_ + i) * dwy;
        }
        res.dissipation = D * h_ * h_;
        q_new = tan.q;
        u_t_new = u_new;
        delta_t_new += tan.delta_t; // solver's delta is the increment
        w_acc_new += res.slip_inc;
        slip_prev_new = res.slip_inc;
        res.tangential = std::move(tan);
    }

    res.converged = ok;
    if (ok) {
        p_ = std::move(p_new);
        q_ = std::move(q_new);
        u_t_ = std::move(u_t_new);
        delta_t_ = delta_t_new;
        w_acc_ = std::move(w_acc_new);
        slip_prev_ = std::move(slip_prev_new);
        K_ = K_new;
        if (tan_force && do_tan) {
            q_bar_last_ = tan_target;
            has_q_bar_last_ = true;
        }
        // carry a passing-but-near-the-floor notice up to the caller
        if (do_tan && res.status_reason.empty())
            res.status_reason = res.tangential.status_reason;
    } else if (res.status_reason.empty()) {
        res.status_reason = "tangential solve did not converge";
    }
    return res;
}

} // namespace hmc
