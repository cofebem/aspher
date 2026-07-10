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
    int active_rounds = 0;      // active-set driver: verification rounds used
                                // (0 = standard full-grid solve)
    bool active_fallback = false; // active-set driver gave up after
                                  // active_max_rounds and ran the full solve
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
// p_init, when non-null, is CONSUMED: its storage is moved into the solver's
// pressure iterate at initialization and *p_init is left moved-from (only
// reassign or destroy it afterwards). This keeps the warm-start vector (a
// full N-sized array, 2.1 GiB at Ns=16384 double) from sitting idle beside
// its own copy for the whole solve. solve_contact preserves the
// non-consuming const-pointer contract by copying.
template <class Real>
ContactResult solve_contact_impl(const MatVecIntoT<Real>& S,
                                 Eigen::Ref<const VecT<Real>> g0,
                                 Real p_bar, Real tol, int max_iter, bool use_pr,
                                 const PrecondIntoT<Real>& precond,
                                 VecT<Real>* p_init, bool light = false,
                                 bool record_history = false);

ContactResult solve_contact(const MatVec& S, const Eigen::VectorXd& g0,
                            double p_bar, double tol = 1e-8,
                            int max_iter = 5000, bool use_pr = true,
                            const Precond& precond = {},
                            const Eigen::VectorXd* p_init = nullptr,
                            bool light = false, bool record_history = false);

// Active-set (restricted) Polonsky-Keer: the same algorithm as
// solve_contact_impl, but every per-iteration O(N) loop runs over the
// candidate index list idx (flat grid indices, each in [0, N)). The pressure
// iterate is kept exactly zero outside idx for the whole solve, so this
// solves the QP restricted to the candidate set; the load constraint stays
// global (mean over the FULL grid = p_bar). Contracts:
//  - S is expected to be the masked matvec with src = tgt = candidate mask:
//    its output is only ever read at idx entries (u/r are stale elsewhere).
//  - precond (optional) receives the full-grid gradient and the contact mask
//    (a subset of idx); FourierPreconditioner::apply_into reads g only at
//    masked entries and zeroes z elsewhere, so the stale entries never leak.
//  - p_init, when non-null, is CONSUMED (storage freed right after its idx
//    entries are gathered); values outside idx are discarded.
// The result is always light (no displacement/gap fields: the operator
// output is not valid off the candidate leaves) — the caller computes full
// fields from its verification matvec. iterations/error/approach/objective/
// contact_fraction/mean_pressure are filled as usual; pressure has g0's
// length (zero off idx).
//
// O(N_c) compressed mode (M3): g0 and all state may be COMPRESSED vectors
// (slot-blocked, see H2Mask) rather than full-grid ones — the algorithm only
// ever addresses them through idx, so it cannot tell the difference. Two
// physical quantities must then be supplied because they are properties of
// the grid, not of the compressed vector: N_grid (the physical cell count,
// for the load constraint P = p_bar·N_grid and the reported fractions;
// 0 → g0.size()) and g_scale (the full-grid gap scale max−min used to
// normalise the complementarity error; <=0 → computed from g0).
template <class Real>
ContactResult solve_contact_active_impl(const MatVecIntoT<Real>& S,
                                        Eigen::Ref<const VecT<Real>> g0,
                                        Real p_bar, Real tol, int max_iter,
                                        bool use_pr,
                                        const PrecondIntoT<Real>& precond,
                                        const std::vector<int>& idx,
                                        VecT<Real>* p_init,
                                        bool record_history = false,
                                        int N_grid = 0,
                                        Real g_scale = Real(0));

} // namespace hmc
