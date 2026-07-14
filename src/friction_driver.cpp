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
}

FrictionStepResult FrictionDriver::step(const FrictionStepSpec& spec) {
    if (g0_.size() == 0)
        throw std::logic_error("FrictionDriver::step: set_gap first");
    if (spec.has_q_bar && spec.has_delta_t)
        throw std::invalid_argument(
            "FrictionDriver::step: q_bar and delta_t are exclusive");
    if (spec.dt <= 0.0)
        throw std::invalid_argument("FrictionDriver::step: dt <= 0");
    FrictionStepResult res;
    bool ok = true;

    // ── (1) normal solve (uncoupled: p never feels q) ──
    if (spec.p_bar > 0.0) {
        MatVec Sop = [this](const Eigen::VectorXd& x) { return S_.matvec(x); };
        Precond Pn;
        if (precond_)
            Pn = [this](const Eigen::VectorXd& g,
                        const std::vector<std::uint8_t>& contact) {
                return Mn_.apply(g, contact);
            };
        const Eigen::VectorXd* warm = (p_.maxCoeff() > 0.0) ? &p_ : nullptr;
        res.normal = solve_contact(Sop, g0_, spec.p_bar, spec.tol_normal,
                                   spec.max_iter, true, Pn, warm);
        ok = ok && res.normal.converged;
        p_ = res.normal.pressure;
    }

    // ── (2)+(3) threshold loop + incremental tangential solve ──
    if (spec.has_q_bar || spec.has_delta_t) {
        if (p_.maxCoeff() <= 0.0)
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
        if (spec.T && static_cast<int>(spec.T->size()) != N_)
            throw std::invalid_argument("FrictionDriver::step: T size");

        // increment targets: solver works in Δδ with u_hist = −uⁿ_t
        const Eigen::Vector2d target = spec.has_q_bar
                                           ? spec.q_bar
                                           : (spec.delta_t - delta_t_);
        Eigen::VectorXd u_hist = -u_t_;
        const double g_floor =
            1e-6 * (target.norm() + u_t_.cwiseAbs().maxCoeff());

        // velocity for the threshold: previous pass's slip increment
        // (first pass: previous STEP's — quasi-static continuation)
        Eigen::VectorXd v(N_), s;
        Eigen::VectorXd slip = slip_prev_;
        TangentialResult tan;
        double s_change = 1e300;
        int pass = 0;
        const int max_pass =
            model_->velocity_dependent() ? spec.max_threshold_iter : 1;
        Eigen::VectorXd s_old;
        for (pass = 0; pass < max_pass; ++pass) {
            for (int i = 0; i < N_; ++i)
                v(i) = std::hypot(slip(i), slip(N_ + i)) / spec.dt;
            Eigen::VectorXd s_new;
            model_->threshold(p_, v, T, s_new);
            if (pass == 0) {
                s = s_new;
            } else {
                s = 0.5 * (s + s_new); // damped fixed point
                s_change = (s - s_old).norm() / std::max(s.norm(), 1e-300);
            }
            s_old = s;
            Eigen::VectorXd q_warm = q_;
            tan = solve_tangential(Cop, s, spec.has_q_bar, target,
                                   spec.tol_tangential, spec.max_iter, true,
                                   Mop, &q_warm, &u_hist, g_floor,
                                   spec.has_q_bar ? &K_ : nullptr, nullptr);
            // slip increment of this candidate solution: Δw = Δδ − (u − uⁿ)
            Eigen::VectorXd u_new(2 * N_);
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

        // ── (4) state update ──
        Eigen::VectorXd u_new(2 * N_);
        C_.matvec_into(tan.q, u_new);
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
        q_ = tan.q;
        u_t_ = u_new;
        delta_t_ += tan.delta_t; // solver's delta is the increment
        w_acc_ += res.slip_inc;
        slip_prev_ = res.slip_inc;
        res.tangential = std::move(tan);
    }

    res.converged = ok;
    return res;
}

} // namespace hmc
