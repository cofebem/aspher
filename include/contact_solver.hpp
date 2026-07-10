#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <functional>
#include <vector>

namespace hmc {

struct ContactResult {
    Eigen::VectorXd pressure;
    Eigen::VectorXd displacement; // u = S p
    Eigen::VectorXd gap;          // u + g0 - approach (>= 0, = 0 in contact)
    double approach = 0.0;        // rigid-body shift (mean gap over contact)
    double objective = 0.0;       // W = 1/2 p.u + p.g0
    double error = 0.0;
    int iterations = 0;
    bool converged = false;
    double contact_fraction = 0.0;
    double mean_pressure = 0.0;
    std::vector<double> error_history; // per-iteration complementarity error;
                                        // empty unless record_history requested
};

template <class Real> using VecT = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
template <class Real> using MatVecT = std::function<VecT<Real>(const VecT<Real>&)>;
template <class Real>
using PrecondT = std::function<VecT<Real>(const VecT<Real>& g,
                                          const std::vector<std::uint8_t>& contact)>;

// Allocation-free variants used by the CG loop itself: the operator and the
// preconditioner write into caller-owned buffers, so no N-sized temporary is
// allocated per iteration (2 operator applies + 1 preconditioner apply).
template <class Real>
using MatVecIntoT = std::function<void(const VecT<Real>&, VecT<Real>&)>;
template <class Real>
using PrecondIntoT = std::function<void(const VecT<Real>& g,
                                        const std::vector<std::uint8_t>& contact,
                                        VecT<Real>& z)>;

using MatVec = MatVecT<double>;

// Optional preconditioner: z = M^-1 g, given the gradient g and the contact
// mask (contact[i] != 0). Returns z restricted to the contact set. An empty
// Precond means unpreconditioned (z = g on the contact set).
using Precond = PrecondT<double>;

// Polonsky & Keer (Wear 231, 1999) projected CG for the constrained problem
//   min 1/2 p^T S p + p^T g0   s.t.  p >= 0,  mean(p) = p_bar.
// Two operator applications per iteration (gradient + line search).
// use_pr=true (default): Polak-Ribière+ β; use_pr=false: Fletcher-Reeves.
// precond (optional): spectral preconditioner applied to the gradient.
// p_init (optional): warm-start pressure (renormalised to the load); nullptr
//   starts from the uniform field p_bar.
// Scalar-templated implementation (Real = double or float). The result is
// always returned in double regardless of the working precision.
// light=true skips storing the displacement and gap fields (saves two N-sized
// double arrays in the result); pressure and all scalars are still filled.
// Takes the allocation-free (into-style) functors; solve_contact adapts the
// by-value ones.
// g0 is a read-only view (Eigen::Ref): the caller keeps ownership and no
// N-sized copy is made — at Ns=16384 double the gap is a 2.1 GiB array, and
// the nested solve passes the Python-owned numpy buffer straight through.
template <class Real>
ContactResult solve_contact_impl(const MatVecIntoT<Real>& S,
                                 Eigen::Ref<const VecT<Real>> g0,
                                 Real p_bar, Real tol, int max_iter, bool use_pr,
                                 const PrecondIntoT<Real>& precond,
                                 const VecT<Real>* p_init, bool light = false,
                                 bool record_history = false);

ContactResult solve_contact(const MatVec& S, const Eigen::VectorXd& g0,
                            double p_bar, double tol = 1e-8,
                            int max_iter = 5000, bool use_pr = true,
                            const Precond& precond = {},
                            const Eigen::VectorXd* p_init = nullptr,
                            bool light = false, bool record_history = false);

} // namespace hmc
