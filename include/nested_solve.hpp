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
    // Which implementation applies the |q| preconditioner.
    //   stencil — a 13-tap real-space disc evaluated only on the contact set,
    //             O(taps·N_c) with no grid allocation (the default).
    //   fft     — the historical full-grid transform, O(N log N) per apply.
    //             Kept as the equivalence reference and an escape hatch; it
    //             was 77–81% of an Ns=16384 active-set run.
    // Both apply the same operator; see
    // doc/specs/2026-09-09-stencil-preconditioner-design.md.
    enum class PrecondEngine { stencil, fft };
    PrecondEngine precond_engine = PrecondEngine::stencil;
    // l2 disc radius of the stencil, in cells. 2 (13 taps) matched or beat
    // the full transform in every regime measured; 1 (5 taps) costs up to 13%
    // more iterations. Must not exceed leaf_side.
    int precond_radius = 2;
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
    // with the finest-only behaviour the 8192 level alone was 42% of the whole
    // solve, because it ran a standard full-grid solve and paid the matvec
    // cost the finest level no longer pays. Removing that is worth -43…-48%
    // of total wall time. Default on, gated by the occupancy rule below; set
    // false to restrict only the finest level (the pre-2026-09-09 behaviour).
    bool active_all_levels = true;
    // Occupancy gate (B04, doc/bench/2026-09-09-b04-candidate-density.md).
    // Restriction wins below ~40% contact and loses above it — at 99% contact
    // the masked matvec costs 6.4x the unmasked one, because once the mask
    // skips nothing its per-box guards and compressed indirection are pure
    // overhead. Occupancy is not known before solving, but the cascade
    // measures it: each level's contact fraction predicts the next one's. A
    // level is therefore restricted only when the level BELOW it reported a
    // fraction under this threshold; the coarsest level always solves in full.
    // 0.4 is deliberately conservative — 48% still wins at Ns=2048 — because
    // the cost of being wrong is asymmetric: forgoing a 15% gain is cheaper
    // than taking a 6x loss. Set >= 1.0 to disable the gate.
    double active_occupancy_max = 0.4;
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
