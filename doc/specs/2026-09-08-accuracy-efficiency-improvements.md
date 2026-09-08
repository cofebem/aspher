# ASPHER accuracy and efficiency improvements — implementation specification

Status: **D1 (correctness release) implemented 2026-09-08** on branch
`feat/accuracy-efficiency` — A01, A02, A03, A04, A15 and the protective parts
of A19, each with its gates (see the companion plan's test IDs and the
"Accuracy/efficiency roadmap" bullet in `CLAUDE.md`). D2–D5 not started.
Date: 2026-09-08. Baseline source: `002b20d`.
Basis: [review and proofs](../review_20260908.md), its linked experiments and
recorded measurements, and the existing operator/solver specifications.
Companion: [test and benchmark plan](../plans/2026-09-08-accuracy-efficiency-validation.md).

This specification covers every improvement in the review. Requirements
labelled MUST are acceptance requirements; SHOULD allows a documented,
measured alternative. Research items have experiment gates before promotion
to production. Numerical thresholds in the test plan are proposed acceptance
targets, not claims about what the current implementation already satisfies.

## 1. Scope and deliverables

Preserve the free-space, uniform-grid, constant-source-cell normal BEM as
the default physical model. Improve correctness, accuracy control, memory,
and time to an independently checked solution. Preserve dense and FFT
references, and maintain normal/tangential kernel parity conventions.

Deliver:

1. Honest termination states and independently recomputed diagnostics.
2. Robust activation, load feasibility, precision scaling and verification.
3. Stable kernel evaluation, compact H² construction and complete memory
   accounting.
4. Measured operator/solver optimizations and optional accuracy schedules.
5. Correct friction state transitions and final local equilibrium checks.
6. Regression/oracle tests, reproducible benchmarks and updated user/theory
   documentation.

New Galerkin, periodic, multigrid, sparse-preconditioner and semismooth
Newton formulations are explicitly separate research deliverables. Their
experiments MUST NOT silently alter the default model or accuracy contract.

## 2. Work packages and dependencies

| ID | Package | Class | Depends on | Principal files |
|---|---|---|---|---|
| A01 | Certificates, statuses, final-result consistency | Correctness | — | contact_solver, result bindings |
| A02 | Feasible activation and scale-safe contact steps | Correctness | A01 | contact_solver, fourier_precond |
| A03 | Global candidate verification and expansion | Correctness | A01–A02 | nested_solve, test_active |
| A04 | Gap centering, scales, float diagnostics | Correctness | A01 | nested_solve, contact_solver, bindings |
| A05 | Stable Love/Cerruti evaluation | Accuracy | — | boussinesq_kernel, cerruti_kernel |
| A06 | Compact near-offset construction | Memory | —; integrate A05 later | kernel ownership, H² constructors, drivers |
| A07 | Accurate memory/cost instrumentation | Measurement | — | H²/FFT info, benchmarks |
| A08 | Displacement recurrence | Optimization | A01–A03, A07 | contact_solver |
| A09 | Staged precision/q and operator-error control | Accuracy/optimization | A01–A05, A07 | nested_solve, operator factory |
| A10 | M2L compression/batching/symmetry; transfer/leaf tuning | Optimization experiments | A05, A07, A09 | h2_operator, cheb_basis |
| A11 | Occupied traversal, compact M/L and screening | Optimization | A03, A07; A09 for certified screening | h2_operator, H2Mask |
| A12 | Two-phase simplex/reduced-CG solver | Solver experiment | A01–A04, A07 | new solver implementation |
| A13 | Sparse preconditioner with coarse correction | Solver experiment | A11–A12 | new preconditioner implementation |
| A14 | Bounded ACA storage and H-matrix symmetry/matvec | Secondary backend | A01, A07 | hmatrix, cluster_tree |
| A15 | Friction diagnostics, loading semantics and transactions | Correctness | A01–A04 | friction_solve, friction_driver, bindings |
| A16 | Projection-residual tangential Newton solver | Solver experiment | A05, A15 | friction_solve, bipotential references |
| A17 | Shared vector H² and friction backend selection | Optimization | A05–A07, A15 | tangential_operator, friction_driver |
| A18 | Physical observables and alternative discretizations | API/research | A01, A05, A09 | area utilities, theory, experimental operators |
| A19 | Validation, lifecycle, ownership and concurrency | Correctness/API | —; integrate with other packages | public C++/Python entry points, fft_engine |

A06 MUST be a separate refactor from A05: first establish equivalence with
the existing numerical coefficients, then change their evaluation with an
independent oracle. Similarly, A10/A11 traversal changes MUST be isolated
from changes to the solver algorithm.

## 3. Common numerical and result contracts

### 3.1 Scales and feasibility

For positive normal load, P=N p_bar. Geometry-aware drivers choose

\[
 p_{ref}=p_{bar},\qquad
 g_{ref}=\max\{\operatorname{range}(g_0),\ p_{bar}L/E^*\}.
\]

Both scales are computed in double before casting. They are invariant to
an additive gap datum and transform consistently under a change of units.
The low-level operator-functor API accepts an explicit `SolveScales` value;
its compatibility adapter may derive a positive displacement scale from
gap variation and an initial displacement, documenting that fallback.
Nested levels MUST use a documented consistent physical scale when their
diagnostics are compared. Candidate vector size MUST NOT replace N in load
or area normalization.

All accepted pressures MUST be finite and nonnegative. Load error MUST
satisfy an explicit feasibility tolerance. Projection/load normalization
MUST precede final operator evaluation and diagnostic computation; no
pressure edits are allowed after the final check without invalidating it.

Proposed numerical feasibility defaults: relative load error ≤10⁻¹² for
double and ≤5×10⁻⁷ for float working pressure, accumulated in double.
Explicitly supplied stricter tolerances MUST either be met or return a
non-converged result. They MUST NOT be silently enlarged. Use compensated
or pairwise double reductions where required to meet the large-grid gate.

The mathematical objective bound assumes exact load feasibility. Numerical
results MUST state that finite-precision feasibility and roundoff qualify
that bound; an interval-certified upper bound requires including those
errors explicitly. Do not label a tolerance-based floating result a formal
interval certificate.

### 3.2 Additive result API

Add equivalent C++ and Python fields; retain existing array properties,
`error`, `error_history`, `iterations`, `active_rounds`, and
`active_fallback` for compatibility:

| Field | Required meaning |
|---|---|
| `status` | `converged`, `stagnated`, `max_iterations`, `nonpositive_curvature`, `nonfinite`, `verification_failed`, or `resource_limit` |
| `converged` | Exactly `status == converged`; never a separate mutable success condition |
| `error`, `pk_error` | Existing normalized pressure-weighted complementarity diagnostic, recomputed on returned fields |
| `fw_gap`, `fw_error` | Unnormalized certificate and certificate/(P g_ref), positive-load case |
| `load_error`, `pressure_violation`, `penetration_error` | Independently defined feasibility/penetration diagnostics |
| `requested_tol`, `effective_tol` | User target and explicit working-stage target; neither is inferred from `error` |
| `validation_scope` | `solve_operator` or `reference_operator`; identify backend, precision, q and compression settings |
| `operator_error_kind` | `unavailable`, `estimate`, or `bound` |
| `operator_error`, `objective_error_bound` | Optional values with units, scope and validity flags; unavailable is not zero |
| `matvec_count`, `verification_matvec_count`, `precond_count` | Actual calls; reference and recursive corrections are accounted for |
| `level_stats`, `stage_stats` | Opt-in per-level/stage counts, tolerances, statuses and time |

Invalid input raises an exception before computation; it is not a numerical
termination status. `light_result=True` omits field storage, not final
diagnostics or required verification. Existing `iterations` retains its
documented finest-level meaning; aggregate work is available separately.
New convergence-history data SHOULD be a separate opt-in trace, preserving
the existing PK history's interpretation.

Default validation remains against the chosen solve operator to avoid
unexpected FFT memory allocation. Requesting reference-operator accuracy
MUST perform that validation or return failure; it MUST NOT downgrade the
requested scope. `converged` with H² scope never implies exact-Love or
continuum accuracy.

### 3.3 Compatibility policy

The meaning of success becomes stricter. Old scripts remain callable, but
cases formerly accepted at a noise floor can now return `stagnated`.
`tol` controls the new stopping contract; `error` remains the legacy
diagnostic, so users must inspect `fw_error` and feasibility fields rather
than assuming `error < tol` is sufficient. Document this behavioral change
in release notes and examples.

Preserve the existing float effective-stage floor initially, expose it,
and add an explicit final-double-polish option in A09. Proposed policies
are `float_only` and `float_then_double`; the existing single_precision
flag maps to `float_only`. The final requested tolerance is strict by
default. A float-only stage reaching its floor above that target returns
`stagnated` with reason `precision_limit`, unless a final fresh check meets
the requested target. Only explicit `allow_tolerance_relaxation=True`
permits success at a relaxed effective tolerance, which must be recorded.
Thus previously clamped calls remain callable but may now correctly fail;
tests/examples needing relaxed float accuracy must request it explicitly.

## 4. A01 — trustworthy convergence

For v=Sp+g₀ and feasible positive-load p, evaluate

\[
 \mathcal G=\sum_i p_i(v_i-v_{min}),\quad v_{min}=\min_i v_i.
\]

Use the nonnegative-summand form, not subtraction of large totals. Compute
v in centered coordinates (A04). Retain the approach convention of the
mean v over strictly positive pressure for compatibility; compute the gap
and penetration from that reported approach. `penetration_error` is
max(0,−min gap)/g_ref. Avoid silently thresholding positive pressures to
define contact during certification.

A normal positive-load success MUST meet all of:

1. Finite fields and scalars; nonnegative pressure; load feasibility.
2. `fw_error <= effective_tol` and
   `penetration_error <= effective_tol` for the selected scope.
3. Final diagnostics from a fresh operator application or a streamed
   application of the returned pressure; no recursively maintained residual
   alone may authorize success.
4. If reference validation is required, that validation also passes A09.

Stagnation returns a failure status and the best checked feasible iterate
when available. Define merit by the maximum normalized failing criterion,
not PK error alone. Keeping a best iterate costs a pressure buffer: memory
accounting MUST include it. If a memory-limited mode cannot retain one,
return the last feasible iterate and explicitly mark `returned_best=false`;
do not promise a best iterate without storing or reconstructing it.

At iteration limit, reevaluate diagnostics on the final returned pressure.
A zero direction with violated KKT conditions transfers control to
activation; it is not convergence. Nonpositive curvature permits one
documented descent/restart recovery, followed by an honest failure if the
problem persists. No repeated identical-iterate loop to the stall counter.

The proof obligations are those in the review:
\(0\le f(p)-f(p_*)\le\mathcal G\) and
\(\|p-p_*\|_S^2\le2\mathcal G\), conditional on convexity/feasibility.
For approximate nonsymmetric operators, the implementation MUST NOT claim
an objective-gap proof without an appropriate symmetric model/error bound.

Tests: T01–T05, T31. Files: `include/contact_solver.hpp`,
`src/contact_solver.cpp`, `python/bindings.cpp`, normal/friction callers.

## 5. A02–A03 — feasible steps and global activation

### 5.1 Repair the existing normal path

Maintain a clear distinction between the candidate set C and positive
pressure support A. A negative-gap zero-pressure point MUST be eligible
for activation even when the old PK error is zero.

For restricted PCG, directions MUST lie in
\(\{d:\operatorname{supp}d\subset A,\ \mathbf1^Td=0\}\).
Restart conjugacy after additions or removals from A and after changes to
operator/preconditioner/scaling. Reproject a retained direction if needed
to remove summation drift. The denominator for a feasible direction is
dᵀSd; any alternative centered evaluation MUST have an explicit equality
argument and preserve the stable finite-precision reduction.

The raw-gap overlap update MUST NOT reuse a preconditioned line-search
step with an arbitrary preconditioner scale. Initial implementation SHOULD
use a separate feasible simplex-projection identification step with
backtracking on the actual objective; a reduced-CG step is truncated at its
first blocking bound and restarted. A retained PK-style alternative must
give a scale-invariance derivation and pass T04; normalizing the Fourier
symbol alone does not establish robustness for arbitrary preconditioners.

Scalar multiplication M⁻¹→cM⁻¹, c>0, MUST leave the mathematical feasible
step invariant within an unchanged CG phase. Complete solves must satisfy
the same accuracy contract across the scaling sweep, although roundoff may
change active-set history/iteration count. No denominator/activation fix
may substitute contact-area agreement for pressure/KKT validation.

### 5.2 Candidate rounds

The finest driver MUST follow this logical sequence:

1. Construct C from coarse pressure/gap and gather compressed state.
2. Solve the candidate problem with its complete internal KKT checks.
3. Stream all physical targets, including candidates, and recompute the
   global minimum v, global penetration, and operator-error diagnostics.
4. If violations require new points, expand C, map old pressures, explicitly
   identify a new feasible support, clear conjugacy/recurrence validity,
   and solve again.
5. Accept only when the restricted solve and global check both pass.
6. If the round budget is exhausted, use the corrected full solver only
   when its memory budget permits; otherwise return `resource_limit` with
   the last feasible result. Never turn an unavailable fallback into success.

All newly added points may start at zero, but the identification phase MUST
actually inspect and activate their violations before a convergence exit.
Positive seeding is permitted as initialization, not as the correctness
argument. Verify non-converged restricted results cannot be accepted merely
because the outside-C list is empty.

Combine the candidate contribution to G with the global streamed v_min;
because p=0 outside C, no full pressure/displacement workspace is required
for this scalar computation. Streaming may use two passes or cache the
small candidate gradient as needed to avoid unstable subtraction.

Tests: T02–T06, T14, T15. Files: `contact_solver`, `nested_solve`.

## 6. A04 — centering and precision policy

Choose a finite scalar datum c in double, e.g. the midpoint of min/max g₀
computed without overflow. Restrict centered geometry and convert
(g₀−c)/g_ref to float; never cast the original large-offset field first.
Full-grid double input ownership/zero-copy behavior MUST be retained where
possible through an offset-aware view or fused loops. Centering MUST NOT
require an extra persistent N-double buffer on the finest double path.

Restore approach by adding c, leave physical displacement unchanged, and
restore the legacy objective with the cP contribution, evaluated in double.
Optimization/merit checks use the centered objective to avoid cancellation.
For optional nondimensional variables use p_scale=E* g_ref/L and
S_scaled=(E*/L)S. Keep p_ref for reporting distinct from p_scale.

Warm-load sums, means, objective, extrema and all convergence reductions
MUST use double accumulation for both working precisions. Float output
converted to double remains labelled as a float solution.

Tests: T07–T09, including offset restoration, flat gaps and returned fields.

## 7. A05–A07 — kernels, compact construction and measurement

### 7.1 Stable kernels

Preserve exact source-cell definitions, prefactors and parity. Implement
analytic self/near terms with removable edge/corner limits; evaluate far
terms using a stable analytic form, positive quadrature, or controlled
moment expansion. The switch criterion MUST be dimensionless in separation
and element dimensions, with a stated accuracy target. No blanket point
1/r substitution and no presumed exact cross-level homogeneity at fixed h.

The kernel contract MUST cover signed continuous offsets, integer offsets,
self/edge/corner cases and near-axis limits. Normal and Cerruti diagonals
are even; the Cerruti xy component is odd in each coordinate and even under
joint reversal. Preserve component swap/elastic-parameter conventions.

Proposed double target for supported aspect ratios: ≤10⁻¹² relative error
for positive normal/diagonal values on the validated domain. Near zeros of
xy use absolute error ≤10⁻¹² times the local normal-kernel scale (including
consistent prefactors); never divide by a symmetry-enforced zero. Validate
the transition from both sides. High-precision analytic evaluation alone
does not independently validate an incorrect analytic formula: include
adaptive/singularity-aware quadrature as a second oracle.

### 7.2 Compact near cache

Separate geometry/material metadata from full coefficient-table ownership.
For H² use a near-offset extent
b=min(Ns,(near_radius+1)leaf_side), requiring b² normal coefficients.
Preserve zero-outside-domain lookup behavior. Far evaluation depends only
on immutable geometry/material/kernel data.

Apply the refactor to explicit `ContactSolver(backend="h2")`, nested solves
and tangential H². Dense/H-matrix/FFT paths retain the coefficient access
they require; no hidden full-table construction is allowed before backend
dispatch. Backing kernel/callback ownership MUST remain valid across build
and subsequent rebuilds, or callbacks must be released after a one-shot
build with a documented lifecycle.

Before A05 integration, compare compact/full-table variants with identical
arithmetic and require bitwise equality on the same build for near entries
and matvec outputs. After A05, compare both using the same improved kernel;
do not demand bitwise identity to unstable historical coefficients.

### 7.3 Accounting and timing

Report bytes for immutable kernel data, tree nodes, near/far interactions,
CSR offsets, masks/maps, interpolation/transfers, couplings, near stencils,
double/float copies, and scratch actually allocated. Count each shared
allocation once. Distinguish logical elements, allocated capacity and peak
transient estimates. Do not include unallocated lazy scratch as resident
memory. A separate `estimated_next_apply_bytes` may predict first use.

Solver statistics include CG state, best-iterate storage, coarse fields,
candidate state, preconditioner scratch, verification workspaces and output
arrays. Distinguish library allocation estimates from process RSS and
Python surface-generation peaks. Instrument build, passes, preconditioning,
verification, coarse solves, conversion and output materialization.

Tests: T10–T13, T18, T23. Benchmarks B01–B03.

## 8. A08–A11 — optimized solves and operators

### 8.1 Displacement recurrence

Reuse u_new=f(u−τSd) only when the actual pressure update is exactly the
corresponding linear update apart from floating arithmetic. Invalidate
after clipping, activation, remapping, operator/precision/q changes or an
unaccounted normalization. Initially refresh every ≤20 reuse steps and on
any convergence candidate; expose refresh/call counts for benchmarking.
Store validity with the iterate/operator generation, not just an iteration
counter. Always verify the final returned pressure freshly.

Sparse nonlinear corrections using SΔp are a subsequent experiment. Their
cost, including new mask construction, must beat a full apply before use.

### 8.2 Staged accuracy and operator validation

Introduce an optional structured accuracy policy, leaving fixed q/precision
available. A stage specifies precision, q, compression tolerance, stopping
target and validation scope. Start coarse/cheap and polish with a fixed
accurate operator; restart CG and invalidate recurrence at every switch.
The candidate set may expand during polishing.

For a justified bound ||Sp−S_tilde p||∞≤ε_u, propagate
G_S≤G_tilde+2Pε_u. Allocate, by default, at most half the final objective-gap
budget to operator error and half to iteration error; these fractions are
policy values that must sum to at most one. Penetration must also be checked
with an appropriate reference/error allowance; a mean-contact approach
estimated using the approximate operator can shift by up to ε_u, so a
2ε_u gap allowance is conservative for the same pressure support.

An FFT reference apply validates the discrete coefficients it uses. A
q-versus-q+2 discrepancy is an estimate, not a rigorous interpolation bound.
When rigorous operator-error bounds are unavailable, label the result
accordingly. Small reference problems establish operator symmetry/coercivity;
local SVD tolerances alone MUST NOT be represented as global pressure-error
or SPD guarantees.

Memory policy must prevent retaining unnecessary full precision/q cache
generations. If a requested FFT verification exceeds its budget, fail
explicitly or use a caller-selected alternative scope; do not allocate it
unexpectedly at the end of a successful large H² solve.

### 8.3 FMM tuning experiments

Implement individually selectable variants for measurement:

- Per-coupling SVD only when its measured cost/storage beats dense.
- Reverse-offset transpose sharing and correct D4 permutations/signs.
- Shared compressed bases and batching by level/offset.
- Tensor M2M/L2L versus dense transfers; retain dense q=4/6 until measured
  otherwise. P2M/L2P already exploit tensor products.
- Joint leaf/q/near-radius/precision/occupancy tuning; near-field batching
  and exact block-padded local convolution as optional candidates.

Every compressed reverse interaction MUST use consistent transpose factors.
Changing summation order changes the equivalence test from bitwise equality
to an error budget. Keep a deterministic reference variant. Autotuning must
return/log its choice and operate inside the selected accuracy and memory
budget; no network or persistent personal configuration is required.

### 8.4 Sparse traversal and screening

Cache occupied box indices by level and candidate generation; reuse
`slot_leaf` for occupied-leaf traversals. Build reduced interaction walks
when their amortized cost is favorable. Preserve parent closure separately
for source and target masks. Solve mode may compact M/L columns; full-target
verification uses a distinct workspace/traversal with explicit costs.

Safe screening is optional and disabled without a valid objective-gap bound.
For the exact SPD operator, let η_i=√(2S_ii G); certify p*_i=0 only when
v_i−min v>η_i+max_jη_j. With an approximate v and a uniform bound ε_u, use
the bound on true G plus at least 2ε_u of additional gradient margin.
Use strict inequalities and numerical error allowances. Never treat an
empirical q discrepancy as authorization for mathematically safe screening.

Tests: T14–T20, T31. Benchmarks B02–B05. Default changes require the
benchmark promotion gate in the companion plan.

## 9. A12–A14 — solver and secondary-backend research

### 9.1 Two-phase normal solver

Add an explicit experimental `algorithm="simplex_cg"` option. Identification
uses projection onto p≥0, sum p=P with an objective-decreasing line search;
reduced minimization uses the support's zero-sum tangent space and an SPD
projected preconditioner. A sorting projection is the oracle; production
may use selection or safeguarded threshold solving. Treat degenerate
zero-pressure/zero-gap points without requiring strict complementarity.

Specify the switch between identification and reduced minimization using
both free and binding residuals, informed by the single-equality QP method
referenced in the review. Prove feasibility, descent/globalization and the
conditions needed for reduced PCG. Retain the corrected normal reference
path until benchmarks justify replacing its default.

### 9.2 Sparse preconditioning

Prototype symmetric additive overlapping block solves plus a coarse space:
Q(ΣR_jᵀB_j⁻¹R_j+ZB_0⁻¹Zᵀ)Q. Show positive definiteness on the covered
zero-sum tangent space and document how coarse modes span disconnected
patches/long-range load redistribution. Do not identify (S_CC)⁻¹ with
(S⁻¹)_CC. No mesh-independent-convergence claim without analysis or a
resolution/occupancy sweep. Keep full Fourier preconditioning available.

### 9.3 H-matrix fallback

Grow ACA factor capacity incrementally with checked allocation/rank budgets.
At a rank budget, subdivide or report a controlled fallback/failure; never
silently violate requested tolerance. Add sampled residual checks for ACA
stopping on adversarial blocks. Share symmetric transpose blocks where
possible and schedule race-free target-owned matvec work to avoid one
N-vector per worker and a global critical reduction. Account for costs of
both halves even if factors share storage.

Tests: T19, T21–T23, T31. Benchmarks B04–B06.

## 10. A15–A17 — friction correctness and scalability

### 10.1 Explicit load and hold semantics

Proposed new API: `p_bar=None` means no normal update; p_bar=0 means actual
complete normal unloading; p_bar>0 requests that load; negative values are
invalid except a documented compatibility adapter for the old −1 sentinel.
Use equivalent explicit presence in C++. Update examples/tests together.

For normal-only changes after shear loading, introduce
`tangential_hold="displacement"|"force"`. Proposed default is held total
tangential displacement; held force must be selected explicitly and stores
the previous controlled force. Solve tangential relaxation under the new
normal thresholds before committing. These are physical boundary conditions,
not a choice of numerical clipping. Infeasible held-force requests fail
transactionally; never silently change the held quantity.

At zero normal load, require p=q=u_normal=u_t=0 for the nonadhesive model.
Choose the normal approach min(g₀) as the documented just-touching convention
(the unconstrained separated approach is not unique), returning gap≥0.
A nonzero prescribed tangential force is infeasible. Preserve the controlled
rigid shift and cumulative history outputs, clear traction-dependent warm
state/probe stiffness, and ensure recontact cannot resurrect stale traction.
Document the open-point convention for accumulated slip; do not overwrite
previously accumulated contact-slip history merely to enforce separation.

### 10.2 Final friction checks

After every final force correction, threshold update or state change,
recompute displacement, cone feasibility, force balance, stick/slip
residuals and dissipation from the returned candidate state. The increment
in history is computed from the old committed state, not a previous trial.
Commit only if normal, tangential, threshold and finite-state checks pass.
Failed calls MUST preserve pressures, tractions, history, rigid shifts and
carried stiffness. Stage-local trial caches must not contaminate retries.

For fixed thresholds, use the projection residual
Rρ=q−Π_disks[q−ρ(Cq+u_hist−EΔδ)]. Choose and record a dimensionally correct
ρ (pressure/displacement), independent of arbitrary preconditioner scaling;
divide by a documented pressure scale for reporting. Supplement the global
residual with maximum local cone/stick/slip violations and force balance.
Reevaluate at the final q; a tiny force residual cannot replace local KKT.

The velocity-dependent loop MUST reevaluate thresholds from the final slip
and temperature, checking the undamped model fixed-point residual. A small
damped update alone is insufficient. Include a mixed absolute/relative
threshold criterion when thresholds vanish. Any negative dissipation beyond
a scale-aware numerical tolerance fails validation and prevents commit.

Remove pressure-valued terms from displacement floors. If force must set a
displacement scale, pass it through a documented compliance scale/probe.
No stall-success exception is permitted for rotating slip directions.

### 10.3 Newton and vector H² experiments

The optional semismooth solver uses the projection residual and its local
2×2 generalized derivatives. Add globalization and a merit criterion.
The full Jacobian is not assumed SPD; use GMRES or a derived symmetric
reduced formulation. Validate directional derivatives away from kinks and
appropriate one-sided/generalized derivatives at kinks. Keep bipotential
Uzawa as an independent cross-check for its supported problems.

Vector H² shares immutable tree/transfers and propagates two components
together with correctly signed tensor couplings. Preserve q/u alias safety.
Add explicit FFT/H² selection to the driver so this optimization is actually
reachable. Keep the uncoupled similar-material model explicit; adding
dissimilar-material cross kernels is a separate physical-model project.

Tests: T24–T29, T31. Benchmark B07.

## 11. A18 — observables and physical-model experiments

Expose raw area, contact/noncontact interface length and optional corrected
area using the existing estimator. Define edge policy explicitly:
nonperiodic internal transitions by default; periodic wrap only for a
periodic model; external loaded-domain edges reported separately. Preserve
the raw pressure>0 definition and document sensitivity to boundary pressure
uncertainty. Corrected area is an estimator, not a changed contact set or a
pressure correction. Flag out-of-range estimates rather than silently
clamping and concealing estimator breakdown.

Research deliverables:

- Galerkin source/target cell-average kernel, with consistent cell-average
  geometry, energy positivity argument and isolated FFT/H² implementations.
- Periodic elastic operator with an explicit mean/DC and rigid-approach
  convention. Periodic surface generation does not select it implicitly.
- Constrained multigrid/coarse correction accounting for S_c versus R S_f P,
  beyond the current cascadic initialization.

Each starts with a small oracle/analytic test and a fixed-realization mesh
study. Hold physical spectral cutoffs fixed on all compared grids and align
sample positions. Separate geometry/discretization error from algebraic and
operator errors. Include anisotropy, edge contact, isolated asperities and
near-percolation behavior before expanding applicability claims.

Tests: T30–T32. Benchmark B08. These models remain opt-in research branches
until a separate model-specific acceptance report is written.

## 12. A19 — input, lifecycle, API ownership and concurrency

Validate dimensions, finite arrays, positive material/geometry parameters,
iteration/tolerance ranges, grid ratios, q, radii and leaf choices before
large allocation or an iterative loop. H² near_radius MUST be ≥1 unless
a separately validated alternative admissibility rule is implemented.
Validate Poisson-ratio/material ranges for the selected physical model.
Reject coarsest=0 immediately. Use checked multiplication and index types
for N, 2N, padded grids and byte counts; large-size arithmetic tests must
not require allocating those arrays.

Define build as idempotent for immutable parameters: repeat calls are
no-ops after success. Build failure leaves an unbuilt valid object; apply
before build throws. A changed kernel/parameter set creates a new immutable
operator generation. This avoids appended interactions and stale float
caches. Public apply/mask functions validate dimensions and index ranges.

Preserve copying result properties initially; add explicit read-only NumPy
view accessors with owner/capsule lifetimes. Test lifetime after deletion of
the Python result wrapper, GC and repeated access. Do not hand out mutable
views into buffers reused by later solves. Route explicit solves through
into-style operators while preserving the public nonconsuming warm start.

Separate immutable operator data from an exclusive workspace, or add a
per-instance guard that rejects simultaneous use with a clear exception.
Do not hold a lock while waiting for the GIL in a way that deadlocks callbacks.
Distinct instances with distinct workspaces may run concurrently. In FFTW
builds serialize planning/destruction and planner thread settings across
instances; execution of distinct workspaces remains parallel.

Remove automatic process-wide malloc policy changes from library calls.
Provide an explicit application-level helper/benchmark option if measured
necessary. Do not edit or reload user shell/configuration files as part of
this work. Document copying, thread safety, actual allocation and statuses.

Tests: T09, T13, T23, T29, T33–T36.

## 13. Delivery order and promotion gates

1. **D0: oracle/measurement harness.** Add independent diagnostics and
   manufactured cases, preserve failing examples and baseline artifacts.
   Tests may be staged in a failing-feature branch but not disabled in a
   final release to make it green.
2. **D1: correctness release.** A01–A04, A15 and the protective parts of A19.
   Existing physics gates plus T01–T09/T24–T29 pass with honest statuses.
   A result that now correctly fails is not fixed by weakening its test.
3. **D2: kernel/memory release.** A06 before A05, plus A07 and A14 storage
   protections. Pass independent kernel and memory gates.
4. **D3: measured optimization release.** A08–A11 and A17. Promote each
   variant independently only after equal-accuracy full-solve benchmarks.
5. **D4: solver research.** A12–A13/A16, benchmark against the corrected
   reference and publish go/no-go findings.
6. **D5: model research.** A18 alternative operators/multigrid. The area API
   may ship earlier once its boundary conventions pass tests.

For each package provide code, its tests, result/API docs, a benchmark or
accuracy report when applicable, and explicit remaining limitations. Update
`CLAUDE.md`, README and theory descriptions after behavior is validated;
preserve historical benchmark context instead of overwriting it with
unqualified new numbers. Follow repository backup rules before modifying
any pre-existing configuration file.

Completion of this roadmap means all correctness contracts are implemented
and all optimization/research proposals have either passed their promotion
gates or have a documented measured no-go outcome. It does not require
shipping an optimization that the experiments show is slower or less accurate.
