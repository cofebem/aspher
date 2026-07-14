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
// clamping as the projection and a β-restart on stick/slip partition changes
// (Tamaas PolonskyKeerTan-style). A strict stick-set CG would freeze the
// remaining direction DOF of bound (slipping) points; keeping them in the
// CG lets their direction converge. Force control solves F(δ_t) = mean(q(δ_t)) − q̄ = 0
// by an outer Newton/Broyden iteration on the rigid shift (δ_t is the Lagrange
// multiplier of the load constraint): the 2×2 stiffness is initialized
// from two small-shift probe solves, updated by noise-guarded Broyden
// secants, and the outer stops when F stops responding at the inner
// solver's resolution; the load is then met exactly by a terminal
// additive correction over interior points (one-shot, with re-clamping).
// δ_t reported is the best-|F| outer iterate's shift.
// All grid-length reductions accumulate in double. target is
// δ_t (displacement control) or q̄ (force control). Double-only in M4.
// converged=true is also set on the stagnation exit (200 non-improving iterations):
// the achievable metric floor was reached and the best iterate is returned —
// same semantics as solve_contact's stagnation guard.
//
// Incremental use (M5): u_hist (size 2N) is added to u in the residual,
//   g_i = (u_i + u_hist_i) − δ_t;
// for the step QP min ½(q−qⁿ)ᵀC(q−qⁿ) − Δδ·Σq s.t. |q|≤s pass
// u_hist = −C qⁿ and read target / delta_t as the INCREMENT Δδ_t.
// g_floor > 0 floors the it-0 error normalization (warm-started
// near-converged solves exit immediately instead of paying the stall
// window). K_io: force control only — if non-null with positive
// determinant it seeds the outer stiffness (ε-probes skipped); the final
// stiffness is written back. delta_init overrides the initial shift.
// With u_hist/q_init present F(0) = −q̄ no longer holds, so the Broyden
// loop simply skips its update on the first outer round.
TangentialResult solve_tangential(const TanMatVecInto& C,
                                  const Eigen::VectorXd& s,
                                  bool force_control,
                                  const Eigen::Vector2d& target,
                                  double tol = 1e-8, int max_iter = 5000,
                                  bool use_pr = true,
                                  const TanPrecondInto& precond = {},
                                  const Eigen::VectorXd* q_init = nullptr,
                                  const Eigen::VectorXd* u_hist = nullptr,
                                  double g_floor = 0.0,
                                  Eigen::Matrix2d* K_io = nullptr,
                                  const Eigen::Vector2d* delta_init = nullptr);

} // namespace hmc
