#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <functional>
#include <vector>

namespace hmc {

// Result of one tangential (frictional) solve on the uncoupled tangential
// block u = C q (spec 2026-07-13-frictional-contact-design.md §5.1).
struct TangentialResult {
    Eigen::VectorXd q;                        // stacked [q_x; q_y] (2N)
    Eigen::Vector2d delta_t = Eigen::Vector2d::Zero(); // rigid shift
    Eigen::Vector2d q_mean = Eigen::Vector2d::Zero();  // grid-mean traction
    double error = 0.0;
    int iterations = 0;
    bool converged = false;
    int n_stick = 0, n_slip = 0, n_open = 0;
    std::vector<std::uint8_t> state; // 0 open (s == 0), 1 stick, 2 slip
};

using TanMatVecInto =
    std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>;
using TanPrecondInto = std::function<void(
    const Eigen::VectorXd& g, const std::vector<std::uint8_t>& mask,
    bool remove_mean, Eigen::VectorXd& z)>;

// Projected CG (vector Polonsky–Keer) for the per-step tangential QP
//   min ½ qᵀC q − Σ_i q_i·δ_t   s.t. |q_i| ≤ s_i            [displacement]
//   min ½ qᵀC q                 s.t. |q_i| ≤ s_i, mean(q)=q̄  [force]
// KKT ⇔ incremental stick/slip friction (stick: w_i = δ_t − u_i = 0;
// slip: |q_i| = s_i, w_i ∥ +q̂_i; open s_i = 0: q_i = 0).
//
// Algorithm notes (deviation from the spec §5.1 sketch, deliberate): the CG
// runs over the FULL candidate set A = {s_i > 0} with per-point radial disk
// clamping as the projection and a β-restart whenever a clamp fires
// (Tamaas PolonskyKeerTan-style). A strict stick-set CG would freeze the
// remaining direction DOF of bound (slipping) points; keeping them in the
// CG lets their direction converge. Force control enforces mean(q) = q̄ by
// an additive correction over interior points each iteration (a vector
// load cannot be rescaled like the scalar p̄); the constraint is met to
// solver tolerance at convergence, and δ_t is recovered as the interior
// mean of u. All grid-length reductions accumulate in double. target is
// δ_t (displacement control) or q̄ (force control). Double-only in M4.
TangentialResult solve_tangential(const TanMatVecInto& C,
                                  const Eigen::VectorXd& s,
                                  bool force_control,
                                  const Eigen::Vector2d& target,
                                  double tol = 1e-8, int max_iter = 5000,
                                  bool use_pr = true,
                                  const TanPrecondInto& precond = {},
                                  const Eigen::VectorXd* q_init = nullptr);

} // namespace hmc
