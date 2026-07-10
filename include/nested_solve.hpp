#pragma once

#include "contact_solver.hpp"

#include <Eigen/Dense>

#include <string>

namespace hmc {

struct NestedParams {
    int coarsest = 64;        // coarsest grid side (power of two, divides Ns)
    int q = 6;                // H2 Chebyshev order on every level
    int leaf_side = 8;        // H2 leaf side on every level
    bool precond = true;      // |q| spectral preconditioner per level
    double coarse_tol = 1e-4; // cascadic: looser tolerance on coarse levels
    bool single_precision = false; // run each level's solve in float (~half RAM)
    bool light_result = false;     // skip displacement/gap in the result (~2 N arrays)
    std::string backend = "h2";    // per-level operator: "h2" or "fft"
    bool record_error_history = false; // finest-level per-iteration error trace

    // Active-set finest-level solve (requires backend "h2" and Ns > coarsest):
    // restricted Polonsky-Keer on a candidate set C = dilate(prolonged coarse
    // contact, active_halo) ∪ {coarse gap < active_delta·(gap scale)}, then a
    // full-grid verification matvec per round; violations (negative gap
    // outside C beyond -tol·scale) are dilated into C and the solve resumes
    // warm-started. After active_max_rounds uncertified rounds the driver
    // falls back to the standard full solve (result flags active_fallback).
    // active_delta is deliberately generous by default: a tight δ can pass
    // verification while boundary pressures are subtly wrong (prototype
    // study, experiments/active_set_results.md Q2).
    bool active_set = false;
    double active_delta = 0.05; // gap threshold, fraction of the level gap scale
    int active_halo = 2;        // dilation radius for candidate/violation sets
    int active_max_rounds = 5;  // verification rounds before full-solve fallback
};

// Single-entry nested-grid (cascadic / full-multigrid) contact solve. Builds
// the grid hierarchy coarsest..Ns by doubling, restricts the fine gap g0 to
// each level (2x2 block average), and solves coarse->fine: each level uses an
// H2 operator and the |q| preconditioner, warm-started by injecting the
// previous (coarser) pressure. Returns the finest-level ContactResult (its
// .iterations is the finest-level count). Ns must equal coarsest * 2^k.
// g0 is a read-only view (Eigen::Ref): the caller keeps ownership and the
// finest level solves directly on it — no N-sized copy is made anywhere in
// the chain. At Ns=16384 double the gap is a 2.1 GiB array, and the Python
// binding passes the numpy buffer straight through; only the coarse levels
// (~N/3 total) are materialised internally.
ContactResult solve_contact_nested(int Ns, double L, double E_star,
                                   Eigen::Ref<const Eigen::VectorXd> g0,
                                   double p_bar,
                                   double tol = 1e-8, int max_iter = 20000,
                                   bool use_pr = true,
                                   const NestedParams& np = {});

} // namespace hmc
