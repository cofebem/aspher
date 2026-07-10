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
};

// Single-entry nested-grid (cascadic / full-multigrid) contact solve. Builds
// the grid hierarchy coarsest..Ns by doubling, restricts the fine gap g0 to
// each level (2x2 block average), and solves coarse->fine: each level uses an
// H2 operator and the |q| preconditioner, warm-started by injecting the
// previous (coarser) pressure. Returns the finest-level ContactResult (its
// .iterations is the finest-level count). Ns must equal coarsest * 2^k.
// g0 is taken by value and moved into the finest-level slot internally (not
// copied): at Ns=16384 double precision a redundant extra N-sized copy here
// was one of the things standing between fitting and OOMing on a 32 GiB
// node. Callers that can give up their own copy should std::move(g0) in.
ContactResult solve_contact_nested(int Ns, double L, double E_star,
                                   Eigen::VectorXd g0, double p_bar,
                                   double tol = 1e-8, int max_iter = 20000,
                                   bool use_pr = true,
                                   const NestedParams& np = {});

} // namespace hmc
