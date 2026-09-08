#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace hmc {

// Termination state of a normal contact solve (spec 2026-09-08 §3.2/§4).
// `converged` is exactly `status == converged`; nothing else may set it.
enum class SolveStatus : int {
    converged = 0,          // every acceptance criterion met on a fresh apply
    stagnated,              // progress stopped above the target (see reason)
    max_iterations,         // iteration budget exhausted
    nonpositive_curvature,  // no usable descent direction remained
    nonfinite,              // non-finite operator output or iterate
    verification_failed,    // an outer/global check rejected the iterate
    resource_limit,         // a required fallback did not fit its budget
};

const char* to_string(SolveStatus s);

struct ContactResult {
    Eigen::VectorXd pressure;
    Eigen::VectorXd displacement; // u = S p
    Eigen::VectorXd gap;          // u + g0 - approach (>= 0, = 0 in contact)
    double approach = 0.0;        // rigid-body shift (mean gap over contact)
    double objective = 0.0;       // W = 1/2 p.u + p.g0
    double error = 0.0;           // legacy PK complementarity diagnostic
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

    // ── A01: independently recomputed termination diagnostics ───────────────
    // All of these are evaluated on the RETURNED pressure with a fresh
    // operator application (never on a recursively maintained residual).
    SolveStatus status = SolveStatus::max_iterations;
    std::string status_reason;      // e.g. "precision_limit", "stall"

    double pk_error = 0.0;          // == error; sum p|g| / (P g_ref)
    double fw_gap = 0.0;            // G = sum_i p_i (v_i - min_j v_j), v = Sp+g0
    double fw_error = 0.0;          // G / (P g_ref)   (Frank-Wolfe certificate)
    double load_error = 0.0;        // |sum p - P| / P
    double pressure_violation = 0.0;  // max(0, -min p) / p_ref
    double penetration_error = 0.0;   // max(0, -min gap) / g_ref

    double requested_tol = 0.0;     // what the caller asked for
    double effective_tol = 0.0;     // what this stage actually targeted
    double p_ref = 0.0;             // pressure scale used for normalisation
    double g_ref = 0.0;             // displacement scale used for normalisation

    // Scope of the certificate: which operator produced the diagnostics.
    // "solve_operator" (default) or "reference_operator".
    std::string validation_scope = "solve_operator";
    // "unavailable" | "estimate" | "bound"; never report 0 for "unavailable".
    std::string operator_error_kind = "unavailable";
    double operator_error = 0.0;
    bool operator_error_valid = false;
    double objective_error_bound = 0.0;
    bool objective_error_bound_valid = false;

    // ── A07: library-side allocation accounting and phase timing ───────────
    // These are the SOLVER's own buffers, computed from the sizes it actually
    // allocated. They are not process RSS, which also carries the caller's
    // gap and surface arrays, the operator, and the Python heap.
    struct Memory {
        long long cg_state = 0;      // iterate, direction, residual, gradient
        long long best_iterate = 0;  // 0 when keep_best is off
        long long contact_mask = 0;
        long long output_arrays = 0; // pressure (+ displacement and gap)
        long long peak = 0;          // largest simultaneous total
    } memory;
    double time_total = 0.0;        // seconds inside the solve
    double time_matvec = 0.0;       // in operator applications
    double time_precond = 0.0;      // in preconditioner applications
    double time_build = 0.0;        // operator construction (nested driver)
    double time_coarse = 0.0;       // coarse-level solves (nested driver)
    double time_verification = 0.0; // streamed global checks (active set)
    double time_output = 0.0;       // materialising the full-grid fields

    // One entry per solve stage (coarse level, finest, polish). Spec §3.2
    // lists this as opt-in; it is O(number of stages) — well under a
    // kilobyte — so it is always populated, which keeps benchmark provenance
    // complete by construction rather than by remembering a flag.
    struct Stage {
        std::string name;      // "coarse:128", "finest:float", "polish:double"
        std::string precision; // "float" | "double"
        int q = 0;
        double requested_tol = 0.0, effective_tol = 0.0;
        int iterations = 0;
        long long matvec_count = 0;
        double seconds = 0.0;
        SolveStatus status = SolveStatus::max_iterations;
        double fw_error = 0.0, penetration_error = 0.0;
    };
    std::vector<Stage> stage_stats;

    long long matvec_count = 0;              // operator applies inside the solve
    long long verification_matvec_count = 0; // applies spent on verification
    long long precond_count = 0;             // preconditioner applications
    int identification_steps = 0;            // feasible projected-gradient steps
    bool returned_best = false;  // returned the best checked iterate (not last)
};

// Physical scales used for normalising diagnostics and tolerances (spec §3.1).
// Both are computed in double and are invariant to an additive gap datum.
//
// A04: adding a constant c to g0 cannot change p* (it only adds c to the
// approach and cP to the objective), but it destroys the small variations
// that carry the physics once the field is rounded — a 1e6 offset changed the
// float nested solution by relative error 1.08 (review §4). The solver
// therefore always works on a centred gap and restores the datum into the
// reported approach and objective.
struct SolveScales {
    double p_ref = 0.0; // pressure scale; <= 0 -> p_bar
    double g_ref = 0.0; // displacement/gap scale; <= 0 -> range(g0)

    enum class DatumMode {
        automatic, // solver picks c = midpoint(min g0, max g0) and subtracts it
                   // on the fly (no extra N-sized buffer, zero-copy preserved)
        solver,    // subtract the supplied `datum` on the fly
        caller,    // `datum` was already removed from g0 by the caller (e.g.
                   // fused into a float cast); restore it only
    };
    DatumMode datum_mode = DatumMode::automatic;
    double datum = 0.0;
};

// Solver policy. `tol` is the target this stage works to (effective);
// `requested_tol` is what the caller originally asked for (0 -> same as tol).
struct SolveOptions {
    double tol = 1e-8;
    double requested_tol = 0.0;
    int max_iter = 5000;
    bool use_pr = true;         // Polak-Ribiere+ beta (false: Fletcher-Reeves)
    bool light = false;         // skip the displacement/gap result arrays
    bool record_history = false;
    bool keep_best = true;      // retain the best checked iterate (1 N-vector)
    double load_tol = 0.0;      // 0 -> 1e-12 (double) / 5e-7 (float)
    int stall_limit = 200;      // insufficient-improvement window
    // Success at an effective tolerance looser than the requested one is only
    // permitted when this is set; otherwise such a solve reports `stagnated`
    // with reason "precision_limit".
    bool allow_tolerance_relaxation = false;
    SolveScales scales;
    // Restricted (candidate-set) solves only: physical grid cell count used
    // for the load constraint P = p_bar * n_grid and the reported fractions.
    // 0 -> g0.size().
    int n_grid = 0;
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

// Projection onto the load simplex {p >= 0, sum p = P}: p_i = max(y_i-θ, 0).
// In-place, O(N) expected (Michelot/Condat threshold iteration). Exposed for
// the identification step and for tests (an independent sorting projector is
// the oracle). Returns the threshold θ.
template <class Real>
double project_load_simplex(VecT<Real>& y, double P_total);

// Polonsky & Keer (Wear 231, 1999) projected CG for the constrained problem
//   min 1/2 p^T S p + p^T g0   s.t.  p >= 0,  mean(p) = p_bar,
// with the corrections of spec 2026-09-08 A01/A02:
//  - success requires the Frank-Wolfe certificate AND the penetration and
//    load-feasibility checks, recomputed on the returned pressure from a
//    fresh operator application; stagnation is a failure status;
//  - the overlap (activation) step uses a preconditioner-scale-invariant
//    step length (see the derivation in the .cpp);
//  - a degenerate restricted direction (no positive curvature, or a support
//    too small to carry a zero-sum direction) hands control to a feasible
//    simplex-projected gradient step with objective backtracking, instead of
//    spinning until the iteration limit.
// Two operator applications per CG iteration (gradient + line search).
// Scalar-templated implementation (Real = double or float). The result is
// always returned in double regardless of the working precision.
// g0 is a read-only view (Eigen::Ref): the caller keeps ownership and no
// N-sized copy is made — at Ns=16384 double the gap is a 2.1 GiB array, and
// the nested solve passes the Python-owned numpy buffer straight through.
// p_init, when non-null, is CONSUMED: its storage is moved into the solver's
// pressure iterate at initialization and *p_init is left moved-from (only
// reassign or destroy it afterwards).
template <class Real>
ContactResult solve_contact_impl(const MatVecIntoT<Real>& S,
                                 Eigen::Ref<const VecT<Real>> g0,
                                 Real p_bar, const SolveOptions& opt,
                                 const PrecondIntoT<Real>& precond,
                                 VecT<Real>* p_init);

ContactResult solve_contact(const MatVec& S, const Eigen::VectorXd& g0,
                            double p_bar, double tol = 1e-8,
                            int max_iter = 5000, bool use_pr = true,
                            const Precond& precond = {},
                            const Eigen::VectorXd* p_init = nullptr,
                            bool light = false, bool record_history = false);

// Same, with full control over the solver policy (scales, tolerances,
// best-iterate retention). p_init is NOT consumed here.
ContactResult solve_contact_opt(const MatVec& S, const Eigen::VectorXd& g0,
                                double p_bar, const SolveOptions& opt,
                                const Precond& precond = {},
                                const Eigen::VectorXd* p_init = nullptr);

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
// The certificate reported here is RESTRICTED to the candidate set: fw_gap
// uses min v over the candidates only, so `status == converged` means the
// candidate subproblem is solved. The caller MUST combine it with a global
// streamed minimum gradient before accepting the result (spec A03); the
// quantities needed for that are returned in `fw_pv_sum` / `fw_vmin_local`
// through the extra out-parameter below.
//
// O(N_c) compressed mode: g0 and all state may be COMPRESSED vectors
// (slot-blocked, see H2Mask) rather than full-grid ones — the algorithm only
// ever addresses them through idx, so it cannot tell the difference. Supply
// opt.n_grid (physical cell count) and opt.scales.g_ref (full-grid gap scale)
// because they are properties of the grid, not of the compressed vector.
struct RestrictedCertificate {
    double pv_sum = 0.0;    // sum_i p_i v_i over the candidate set (p=0 outside)
    double vmin_local = 0.0; // min v over the candidate set
    double P_total = 0.0;    // p_bar * n_grid
    double g_ref = 1.0;
};

template <class Real>
ContactResult solve_contact_active_impl(const MatVecIntoT<Real>& S,
                                        Eigen::Ref<const VecT<Real>> g0,
                                        Real p_bar, const SolveOptions& opt,
                                        const PrecondIntoT<Real>& precond,
                                        const std::vector<int>& idx,
                                        VecT<Real>* p_init,
                                        RestrictedCertificate* cert = nullptr);

} // namespace hmc
