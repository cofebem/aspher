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

#include <string>

namespace hmc {

// Tangential boundary condition held while the NORMAL load changes and no
// tangential target is given (spec A15 §10.1). These are physical boundary
// conditions, not a choice of numerical clipping: changing p changes the
// friction thresholds, so the stored shear must be re-solved, not clipped.
enum class TangentialHold {
    displacement, // hold the total tangential rigid shift delta_t (default)
    force,        // hold the previously controlled mean traction q_bar
};

struct FrictionStepSpec {
    // Normal load control (spec A15 §10.1):
    //   has_p_bar = false           -> no normal update; hold the current p
    //   has_p_bar = true, p_bar > 0 -> solve for that mean pressure
    //   has_p_bar = true, p_bar = 0 -> COMPLETE normal unloading (separation)
    //   p_bar < 0                   -> invalid
    // Compatibility: the historical p_bar = -1 sentinel (has_p_bar unset)
    // still means "no normal update", and a positive p_bar with has_p_bar
    // unset is still honoured as a load request. p_bar = 0 with has_p_bar
    // unset is rejected: it used to be silently skipped, which is exactly the
    // conflation of "zero load" with "no update" this API removes.
    bool has_p_bar = false;
    double p_bar = -1.0;
    bool has_q_bar = false;
    Eigen::Vector2d q_bar = Eigen::Vector2d::Zero();
    bool has_delta_t = false;
    Eigen::Vector2d delta_t = Eigen::Vector2d::Zero();
    TangentialHold tangential_hold = TangentialHold::displacement;
    double dt = 1.0;
    const Eigen::VectorXd* T = nullptr;
    double tol_normal = 1e-8, tol_tangential = 1e-5;
    // Local-KKT acceptance for the tangential state (0 -> the solver's
    // documented default). A step commits only when this is met on the
    // RETURNED tractions, after the terminal force correction.
    double tol_kkt = 0.0;
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
    // Set when a normal-only step re-solved the tangential problem under the
    // new thresholds (the held boundary condition), rather than the caller
    // driving it. `hold_relaxation` names which quantity was held.
    bool tangential_hold_applied = false;
    TangentialHold hold_applied = TangentialHold::displacement;
    // Empty on a clean success. On failure, why the step was refused (the
    // driver state is then unchanged). On success it may still carry
    // "local_kkt_near_tolerance", meaning the step committed but its local
    // equilibrium is only as good as tangential.proj_residual says.
    std::string status_reason;
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
    // Fill `res.normal` for the fully separated (p_bar = 0) state: zero
    // pressure and displacement, and the just-touching approach min(g0) —
    // the unconstrained separated approach is not unique, so this is the
    // documented convention and it keeps the reported gap >= 0.
    void fill_separated_normal(ContactResult& n) const;

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
    // last CONTROLLED mean traction, for tangential_hold = force
    Eigen::Vector2d q_bar_last_ = Eigen::Vector2d::Zero();
    bool has_q_bar_last_ = false;
};

} // namespace hmc
