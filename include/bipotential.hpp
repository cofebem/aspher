// de Saxcé–Feng bipotential Uzawa reference solver (Mathl. Comput.
// Modelling 28 (1998) 225-245, eqs. (102)-(105); spec §5.2). Derivative-
// free predictor-corrector on the full (q, p) field, treating unilateral
// contact and friction SIMULTANEOUSLY through one implicit subnormality
// rule — the production staggered path's independent cross-check, and the
// designated strategy for the future coupled (dissimilar-material) case.
//
// Sign map from the paper to this codebase: our q is the traction ON the
// half-space and slip w = delta_t - C q satisfies w || +q̂; the paper's
// relative velocity is u̇_t = -w and its separation velocity u̇_n = g =
// u_z + g0 - alpha >= 0. Substituting into eq. (102),
//   tau = r - rho (u̇_t + (u̇_n + mu |u̇_t|) n),
// gives the predictor used here:
//   tau_t = q + rho w,     tau_n = p - rho (g + mu |w|).

#ifndef HMC_BIPOTENTIAL_HPP
#define HMC_BIPOTENTIAL_HPP

#include <Eigen/Dense>
#include <functional>

namespace hmc {

// Analytic projection onto the Coulomb cone K_mu (de Saxcé–Feng eq. 105).
// In/out: (tx, ty, tn) -> the projected (qx, qy, p). Exposed for tests.
void project_coulomb_cone(double mu, double& tx, double& ty, double& tn);

struct BipotentialResult {
    Eigen::VectorXd p;   // N
    Eigen::VectorXd q;   // 2N stacked
    double error = 0.0;  // last relative update norm
    int iterations = 0;
    bool converged = false;
    int n_open = 0, n_stick = 0, n_slip = 0;
};

// Displacement-controlled bipotential Uzawa reference (spec §5.2): imposed
// approach alpha and rigid shift delta_t; friction = Coulomb(mu) when
// s_frozen is null, otherwise the frozen-threshold cylinder (Tresca /
// generic-with-frozen-s): p >= 0 and |q_i| <= s_frozen_i independently.
// rho = rho_scale / lambda_max(stacked operator), lambda_max from
// npower power iterations. First-order reference: expect O(1e3-1e4)
// iterations; tol is on the relative iterate update per sweep.
BipotentialResult solve_bipotential(
    const std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>& S,
    const std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>& C,
    const Eigen::VectorXd& g0, double mu, double alpha,
    const Eigen::Vector2d& delta_t, double tol = 1e-8, int max_iter = 200000,
    double rho_scale = 1.0, const Eigen::VectorXd* s_frozen = nullptr,
    int npower = 20);

} // namespace hmc

#endif // HMC_BIPOTENTIAL_HPP
