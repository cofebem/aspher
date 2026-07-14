#pragma once

#include "boussinesq_kernel.hpp"
#include "cerruti_kernel.hpp"
#include "contact_solver.hpp"
#include "fft_operator.hpp"
#include "fourier_precond.hpp"
#include "friction_model.hpp"
#include "friction_solve.hpp"
#include "tangential_operator.hpp"

#include <Eigen/Dense>

namespace hmc {

struct FrictionStepSpec {
    double p_bar = -1.0;
    bool has_q_bar = false;
    Eigen::Vector2d q_bar = Eigen::Vector2d::Zero();
    bool has_delta_t = false;
    Eigen::Vector2d delta_t = Eigen::Vector2d::Zero();
    double dt = 1.0;
    const Eigen::VectorXd* T = nullptr;
    double tol_normal = 1e-8, tol_tangential = 1e-5;
    int max_iter = 20000, max_threshold_iter = 20;
    double threshold_rtol = 1e-3;
};

struct FrictionStepResult {
    ContactResult normal;
    TangentialResult tangential;
    Eigen::VectorXd slip_inc;
    double dissipation = 0.0;
    int threshold_iters = 0;
    bool converged = false;
};

// Incremental quasi-static frictional-contact driver (spec §6): per step,
// (1) normal Polonsky–Keer solve at spec.p_bar (warm-started; skipped when
// p_bar <= 0), (2) threshold field s = model(p, |Δw|/dt, T) with a damped
// fixed-point loop for velocity-dependent models, (3) incremental
// tangential solve (u_hist = −C qⁿ, targets are TOTAL loads/shifts,
// converted to increments internally; force-control stiffness K carried
// across steps), (4) state update: q, u_t, δ_t, accumulated slip, and the
// step dissipation D = h² Σ q·Δw (≥ 0 up to solver floor). FFT backend
// (exact operators); both spectral preconditioners on by default.
// History lives here; callers drive the load program step by step and may
// update the temperature field between steps. Not thread-safe.
//
// Transactional contract: step() works on local candidates and commits the
// persistent members (p, q, u_t, δ_t, accumulated slip, carried K) only
// when the step converges. A non-converged step (result.converged == false,
// including one where solve_tangential throws and step() rethrows) leaves
// the driver state exactly as it was before the call — the caller may
// inspect the failure and retry with a modified spec, or call reset().
//
// The carried force-control stiffness K is not invalidated when p (hence
// the threshold field s) changes between steps; the solver's Broyden
// updates and floor detection self-correct, at worst reporting
// converged=false rather than silently seeding a bad outer iterate.
class FrictionDriver {
public:
    FrictionDriver(int Ns, double L, double E_star, double nu,
                   const FrictionModel& model, bool precond = true);

    void set_gap(const Eigen::VectorXd& g0);
    FrictionStepResult step(const FrictionStepSpec& spec);
    void reset();

    const Eigen::VectorXd& pressure() const { return p_; }
    const Eigen::VectorXd& q() const { return q_; }
    const Eigen::VectorXd& u_t() const { return u_t_; }
    const Eigen::Vector2d& delta_t() const { return delta_t_; }
    const Eigen::VectorXd& w_acc() const { return w_acc_; }

private:
    int Ns_, N_;
    double L_, h_;
    const FrictionModel* model_;
    bool precond_;
    BoussinesqKernel BK_;
    FFTOperator S_;
    CerrutiKernel CK_;
    TangentialFFTOperator C_;
    FourierPreconditioner Mn_;
    TangentialFourierPreconditioner Mt_;
    Eigen::VectorXd g0_;
    // state
    Eigen::VectorXd p_, q_, u_t_, w_acc_, slip_prev_;
    Eigen::Vector2d delta_t_ = Eigen::Vector2d::Zero();
    Eigen::Matrix2d K_ = Eigen::Matrix2d::Zero(); // det 0 = no carry-over yet
};

} // namespace hmc
