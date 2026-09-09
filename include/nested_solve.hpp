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
    // ── A09: precision policy ─────────────────────────────────────────────
    // `double_only`       every stage in double (default).
    // `float_only`        every stage in float — the historical
    //                     single_precision=true. Float cannot drive the
    //                     certificate below ~2e-7, so a request tighter than
    //                     `float_floor` is NOT met; such a solve now reports
    //                     `stagnated` with reason "precision_limit" unless
    //                     allow_tolerance_relaxation is set.
    // `float_then_double` identify the contact in float at `float_floor`,
    //                     then POLISH in double, warm-started, to the
    //                     requested tolerance. This buys accuracy, not
    //                     memory: the polish carries the double path's
    //                     working set, so the large-grid recipe that uses
    //                     float for RAM should stay on float_only.
    enum class Precision { double_only, float_only, float_then_double };
    Precision precision = Precision::double_only;
    // Legacy spelling of Precision::float_only; when true and `precision` is
    // still the default, it selects float_only.
    bool single_precision = false;
    // Certificate level float can actually reach. Measured on a Ns=256 rough
    // surface: 2e-7 converges in 102 iterations, anything tighter stalls at
    // 1.88e-7, and the PRESSURE error against double saturates at ~5e-6
    // regardless — so tightening this buys iterations, not accuracy.
    double float_floor = 2e-6;
    // Permit success at an effective tolerance looser than the requested one
    // (recorded in the result either way). Off by default: an unmet request
    // is a failure, not a silently relaxed success.
    bool allow_tolerance_relaxation = false;
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
    // Restrict EVERY level that has a coarser level beneath it, not only the
    // finest. Measured at Ns=16384 (doc/bench/2026-09-09-ns16384-rebaseline.md):
    // with the finest-only default the 8192 level alone is 42% of the whole
    // solve, because it runs a standard full-grid solve and pays the matvec
    // cost the finest level no longer pays. Opt-in until the paired A/B
    // clears the promotion gate.
    bool active_all_levels = false;
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
