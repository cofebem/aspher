# Accuracy and efficiency improvements — test and benchmark plan

Status: **partially implemented (2026-09-08)**. Landed: T01–T09 (in
`tests/test_certification.cpp` and `tests/test_precision.cpp`, with the
independent oracle in `tests/contact_oracle.hpp`), T21's projector oracle,
T31, T06 (in `tests/test_active.cpp`), T24–T26 (in `tests/test_friction.cpp`
and `tests/test_driver.cpp`), T33–T34 (in `tests/test_contracts.cpp`), and
Python coverage in `tests/test_certification_py.py` / `tests/test_friction_py.py`.
Not yet implemented: T10–T20, T22–T23, T27–T30, T32, T35–T36 and every
benchmark B01–B08.
Date: 2026-09-08.
Normative behavior: [implementation specification](../specs/2026-09-08-accuracy-efficiency-improvements.md).
Evidence: [review](../review_20260908.md),
[Python probes](../../experiments/review_20260908.py),
[C++ probes](../../experiments/review_20260908.cpp).

The baseline passed 12 CTest groups and 9 selected Python integration tests.
The probes demonstrate missing coverage. Promote their mathematical cases
into independent assertions, rather than asserting their historical failure
values or exact iteration counts. No current probe is a substitute for a
regression test merely because it prints a diagnostic.

## 1. Test architecture and independent oracles

### 1.1 Proposed files and registration

| File/group | Purpose | Test IDs |
|---|---|---|
| `tests/contact_oracle.hpp` | Small dense oracle, manufactured solutions and independent diagnostics | helper |
| `tests/test_certification.cpp` | KKT, statuses, active-support and scale regressions | T01–T05, T31 |
| `tests/test_active.cpp` | Expansion, fallback, masks and streaming extensions | T06, T14–T15 |
| `tests/test_precision.cpp` | Centering, units, stages and reductions | T07–T09, T17 |
| `tests/test_kernel_stability.cpp` | Stable source-cell integration and parity | T10–T12 |
| `tests/test_operator_contracts.cpp` | Compact caches, symmetry, lifecycle, sizes | T13, T18–T19, T33–T34 |
| `tests/test_recurrence.cpp` | Linear update reuse and invalidation | T16 |
| `tests/test_solver_variants.cpp` | Sparse/alternative solver and H-matrix contracts | T20–T23 |
| existing friction/driver test groups | Local KKT and state transitions | T24–T29 |
| `tests/test_observables.py` | Area/perimeter/model conventions | T30 |
| `tests/test_api_contracts.py` | Views, copies, validation and concurrency | T33–T36 |
| `tests/generate_kernel_reference.py` | Offline high-precision fixture generation | oracle |
| `experiments/accuracy_efficiency_bench.py` | Slow accuracy/performance/model studies | B01–B08, T32 |

These paths are proposed; small cases may join existing files when that
keeps dependencies simpler. Register C++ groups with CTest and Python tests
with the repository's test workflow. Use separate labels for fast, numerical,
threading, sanitizer, performance and large-memory tests. Do not place wall
time pass/fail assertions in ordinary CI tests.

### 1.2 Dense QP oracle

For tiny SPD matrices with n≤12, independently enumerate nonempty supports
A and solve

\[
 \begin{bmatrix}S_{AA}&-\mathbf1\\\mathbf1^T&0\end{bmatrix}
 \begin{bmatrix}p_A\\\alpha\end{bmatrix}
 =\begin{bmatrix}-g_{0,A}\\P\end{bmatrix}.
\]

Retain candidates with nonnegative pressure, nonnegative outside gaps and
the prescribed load within tight oracle tolerances. Resolve degeneracy by
objective/pressure equivalence, not by demanding one particular support
representation. The enumeration and diagnostic helper MUST NOT call the
production solver, production certificate helper or production projector.
Cross-check oracle linear residuals and positive eigenvalues.

For larger discrete fixtures, manufacture p*≥0 and g*≥0 with p*ᵢg*ᵢ=0,
then choose
g₀=α*1−S p*+g* and p_bar=mean(p*).
The KKT conditions prove p* is optimal for SPD S. Include separated patches,
singleton support, full contact, disconnected islands, and degenerate
p*=g*=0 points. Construct g₀ using dense S at small sizes or exact FFT at
moderate sizes; tests against H² explicitly include operator approximation
error. Never manufacture with H² and then call agreement with the same H²
an independent exact-Love check.

The oracle evaluates nonnegativity, load, v, reported approach/gap,
complementarity, global minimum, G and objective from the *returned pressure*.
Use the same declared physical normalization but separate numerical code.

### 1.3 Kernel oracle and fixtures

Generate immutable high-precision values with mpmath at ≥80 digits, and
validate selected values with independent adaptive quadrature. Self/edge/
corner quadrature must split or transform the integrable singularity;
ordinary tensor Gauss rules across it are not an adequate oracle.
For far points use quadrature with a convergence check between orders.
Store input decimals, output decimals, units/scaling, generator revision,
precision and cross-check error in a small fixture file.

Require normal test runs only to read fixtures; mpmath need not become a
runtime dependency of ASPHER or every C++ CI job. Fixture regeneration is
an explicit developer command, never an automatic overwrite when a test
fails. The review's NumPy mirror is useful diagnosis, but the new kernel
tests MUST exercise the actual compiled C++ functions.

## 2. Normal solver and certification regressions

### T01 — certificate and known solution

Package A01. Run tiny SPD and manufactured Love problems, with cold and
valid/invalid warm starts. Compare pressure and objective to the independent
oracle; directly recompute all result diagnostics. Test uniform, singleton,
partial and full support, including a known solution supplied as warm start.
An already valid warm start may finish at iteration zero; an invalid one
must not. Prove the test's reference by its KKT construction.

For well-conditioned normalized tiny fixtures, target pressure relative
L2≤10⁻⁸ with requested tol=10⁻¹². If a fixture is ill-conditioned, compare
against the explicit energy/λ_min certificate bound instead of silently
using that universal pressure threshold. Diagnostic fields must match the
independent recomputation within a documented double roundoff allowance.

### T02 — singleton warm-start counterexample

Packages A01–A02. Preserve the Ns=8, L=E*=1 parabolic gap,
p_bar=0.01, all-load-at-corner fixture from the review. Dense and FFT solves
must activate the missing contact and meet the final oracle checks. Run
with no preconditioner and Fourier preconditioning. Repeat with positive
load on a central point and on a wrong disconnected asperity.

Do not assert a particular repaired iteration count. Assert that the
historical pressure is rejected by the independent checker, and that any
successful returned solution satisfies the requested conditions.

### T03 — termination/result consistency

Package A01. Use max_iter=0 and 1, an unattainable precision target, injected
NaN/Inf operator output, and a direction with zero/negative curvature.
Trigger deterministic stagnation through a test seam/policy with a bounded
stall window, not a flaky expectation about OpenMP noise at tol=10⁻²⁵.

Check status/converged consistency, fresh diagnostics of the actual
returned iterate, best-versus-last metadata, light-result parity, and that
a curvature restart either makes progress or exits honestly. An injected
nonfinite callback should return the documented numerical failure; a
nonfinite input should be rejected before the callback runs. Reject a
candidate where load is exact but penetration or G exceeds tolerance.

### T04 — preconditioner scale and feasible directions

Package A02. Use the Ns=64 wavy fixture from the C++ probe. Sweep positive
preconditioner multipliers 10⁻⁶, 10⁻³, 1, 10³, 10⁶, plus identity and
Fourier baselines. All successful solves must meet identical oracle
tolerances; the c=10⁻³ historical stalled answer must no longer pass.

On a manufactured fixed-support problem, instrument d, dᵀSd and step
length. Check sum(d)=0 to a scale-aware roundoff bound and invariance of the
actual step under positive scalar preconditioning. On changing supports,
check feasibility and final solution, not bitwise iteration identity.

### T05 — removal/addition and degenerate activation

Packages A02/A12. Construct steps that remove a pressure point without
adding one, add one without removing one, and swap multiple points.
Verify conjugacy restarts and zero-sum directions after each transition.
Include a support of size one with violated outside gaps, p*=g*=0 degeneracy,
and nearly equal asperity heights. A zero restricted direction must invoke
identification when global KKT fails; it must not spin to the stall limit.

### T06 — candidate expansion cannot hide a violation

Package A03. Test the round controller using an injected initial candidate
set or a small independently testable driver seam. Start with a converged
restricted solution whose true solution has contact outside C. Force at
least one expansion; new pressures start at zero. Check the newly included
violating points are reconsidered before termination. Include a second
remote violation requiring another round.

Also cover: empty outside violation list with an unconverged inner solve;
violations inside C; all-grid C; exhausted rounds; permitted full fallback;
fallback over a deliberately small memory budget; empty candidate recovery;
and `light_result` streaming. Only accept after both restricted and global
checks pass. Compare to a dense manufactured oracle, then repeat a moderate
rough surface against FFT. The full fallback test must not depend on the
historical seeding workaround for correctness.

## 3. Precision and kernel tests

### T07 — gap datum invariance

Package A04. On the fixed Ns=128 review surface, use g₀+c with c=0, ±10³,
±10⁶, both float-only and double. Run explicit and nested APIs, warm and
cold. For float explicitly select a relaxed tolerance when appropriate;
separately test strict requested tolerance and double polishing.

For the review fixture, target pressure relative error≤10⁻⁴ against the
double reference in the relaxed float mode. Compare centered and shifted
solutions using the known float/geometry uncertainty; do not require exact
equality when adding c already rounded the input double field.
Check approach shifts by c, physical displacement is unchanged, gap is
unchanged, and the objective shifts by cP with an appropriate absolute
roundoff allowance. Test min/max near large finite values to catch overflow
in datum selection. A caller-supplied already-rounded float array cannot
be required to recover lost information.

### T08 — units and geometry/material scaling

Packages A04/A15. Change L→aL, g₀→ag₀ and E*→bE*, p_bar→bp_bar:
pressure must scale by b, displacement/approach by a, and contact topology
must agree away from explicitly degenerate boundaries. Use a,b spanning
10⁻³…10³, flat and rough gaps, and both working precisions. For friction,
scale traction thresholds/force by b and imposed shifts/history by a;
adjust rate-law dimensional parameters consistently. Detect any mixture of
traction and displacement in residual floors.

### T09 — reductions, zero load and declared precision

Packages A01/A04/A19. Exercise warm-load normalization and final mean/
objective with vectors containing large dynamic ranges and millions of
entries without building a huge dense matrix. Compare reductions to a
compensated/long-double reference. Check strict versus explicitly relaxed
float tolerance semantics and float-to-double result metadata.

For zero-load API support, require p=u=0, gap≥0, the specified just-touching
approach, and finite zero-load diagnostics without division by P. A cold
flat positive-load gap must also have a positive meaningful g_ref. These
tests are independent of the full-separation friction transition tests.

### T10 — compiled kernel high-precision sweep

Package A05. Test normalized offsets r/h from near-cell distances through
1, 8, 32, 10³, 16384 and 10⁶, including axis, near-axis, diagonal and
generic angles in every quadrant. Include actual Chebyshev-node
differences on several levels. Sweep supported rectangular aspect ratios,
initially 1/16…16 if the public kernel retains general a,b support.

Meet the spec's 10⁻¹² relative normal/diagonal target; use the stated
normal-kernel-scaled absolute target for xy near zero. The large-separation
stress points are kernel tests, not requests for grids of that size.

### T11 — self, edge, corner and branch continuity

Package A05. Self term must match 4h log(1+√2)/(πE*) on square cells.
Evaluate exactly at every edge/corner and at signed perturbations on both
sides, including nextafter neighbors. Require finite values and agreement
with singularity-aware references. Evaluate both sides of any far-method
switch and verify accuracy/continuity within the oracle tolerance.

### T12 — parity, component swaps and homogeneity

Package A05. Check normal and diagonal evenness, xy oddness per axis,
joint reversal, xx/yy coordinate swaps, nu=0 cross-term vanishing, and
simultaneous scaling of x,y,a,b. Scaling all lengths by a must multiply
the integrated geometric kernel by a. Separately test that scaling only
box separation at fixed cell size is NOT assumed to give exact 1/r scaling.
Use force/displacement prefactors independently from geometric formulas.

### T13 — compact-cache equality and backend construction

Packages A06/A19. For Ns=8,16,32,64; leaf sides 4,8,16 when valid; radii
1,2,3; and q=4,6, compare every required compact entry to the full table.
Check maximum requested offset, outside-domain zeros and the Ns=leaf case.
Compare full/masked/streamed matvecs in both precisions using identical
coefficients. Before kernel changes, require bitwise equality on the same
compiler/build. Instrument table allocation to prove H² construction does
not first build and then discard an Ns² coefficient table.

## 4. Operators, variants and memory

### T14 — occupancy and traversal equivalence

Packages A03/A11. Test full, empty-source, singleton, clustered, fragmented
and edge masks. Source and target masks may differ. Verify parent closure,
mapped indices, sources with zero pressure, compressed slot sizes, and
stale-buffer poisoning. Change masks repeatedly and compare with unmasked
reference output on all requested targets. Same-order traversal changes
retain bitwise checks; reordered/batched paths use a declared roundoff
budget. Instrument visited boxes to verify sparse lists actually avoid
full-tree scans in solve mode, without using timing assertions.

### T15 — streamed final fields and global reductions

Packages A03/A11. Compare streamed displacement, gap, global v_min,
certificate and reported approach against materialized reference results.
Include the minimum inside C, outside C, and on a domain edge. Verify
thread-safe reductions and no reads of invalid off-target scratch. Light
and full results must have matching scalar diagnostics, and light mode
must not allocate full displacement/gap outputs for verification.

### T16 — displacement recurrence and refresh

Package A08. Compare reuse on/off on fixed support and changing-support
fixtures. Count actual operator calls; require a reduction on the fixed
support case and identical final accuracy, not a fixed speedup. Inject
clipping, load scaling, support remapping, precision/q changes, a forced
restart and a controlled accumulated residual perturbation. Each must
invalidate or correctly update recurrence state. Verify periodic refresh
and a fresh final check with a counting operator wrapper. Float tests
include drift that the final check must detect and repair or reject.

### T17 — accuracy stages and validation scope

Package A09. Exercise fixed q, staged q=4→6→8, float→double, active-set
polishing and an exact FFT final check. Check stage metadata, restarted
directions, released obsolete caches, expanded support, requested/effective
tolerances and all call counters. A deliberately inadequate q=4 solve at
strict reference tolerance must refine or fail verification even if its
own residual is tiny. Reject a reference validation over budget without
silently substituting solve-operator scope. An empirical q discrepancy
must retain `operator_error_kind=estimate`.

### T18 — M2L/transfer identities and compression budgets

Package A10. For every child orientation verify
(B⊗A)vec(X)=vec(AXBᵀ) and the corresponding adjoints using random and basis
vectors at q=4,6,8,10. Verify all reverse-offset factors/permutations,
including odd tensor component signs. Compare full and compressed M2L
blocks against dense SVD residuals; skip compression when not beneficial.

Test complete matvecs with impulses, signed zero-mean vectors, near-Nyquist
modes, contact-like pressure and random inputs. Propagate/check a complete
operator error budget; a passed local singular-value cutoff is not enough.
No new variant is accepted solely because the contact area matches.

### T19 — reciprocity, coercivity and energy

Packages A09/A10/A13/A14. Explicitly assemble small operator matrices by
applying them to basis vectors. Measure ||A−Aᵀ||/||A|| and the minimum
eigenvalue of the symmetric part. For exact/reference operators require
symmetry at roundoff and positive eigenvalues above numerical uncertainty.
For approximations, require their documented symmetry/error budget and
positive tangent-space curvature on tested supported settings; a failed
case must trigger refinement or controlled failure, not pass as CG success.

Apply random and adversarial zero-sum directions and test positive energy.
Repeat for masked restrictions, coarse-space preconditioners, and vector
kernels. Finite small-grid eigenvalue tests are a regression check, not a
proof of SPD for every grid or interpolation order.

### T20 — certified screening

Package A11. On tiny manufactured SPD problems, compute G and the screening
bound independently; no screened point may have p*>0. Include nearly tight
inequalities and nonuniform diagonals. Perturb operator application by a
known bounded error and verify the enlarged margin remains safe. Reject
screening from an empirical-only error estimate. Check screening disabled
when feasibility, convexity or certificate prerequisites are unavailable.

### T21 — simplex solver and projector oracle

Package A12. Compare production projection to an independent sorting
projector for repeated values, singleton output, full output, widely scaled
inputs and degenerate thresholds. Check exact nonnegativity, sum tolerance,
idempotence and the projection variational inequality. Solve the tiny QPs
against support enumeration. Instrument feasibility/descent and phase
switches; include strict and non-strict complementarity. Compare corrected
reference and experimental solvers at equal final certificates.

### T22 — sparse preconditioner and coarse modes

Package A13. Check linearity, symmetry and positive tangent-space energy.
Use two far-separated contact patches with a load-transfer mode and many
fragmented patches. Verify coarse correction affects the intended global
modes and handles rank-deficient coarse basis proposals safely. Compare
iteration/cost scaling to Fourier in benchmarks; do not assert an arbitrary
iteration ceiling in small CI tests.

### T23 — allocations, ACA budgets and memory accounting

Packages A06/A07/A14/A19. Instrument logical/capacity bytes before build,
after build, after double/float apply, after release and after destruction.
Check shared data counted once and lazy data not counted as resident.
Account for best-iterate, stages and verification scratch.

Force ACA beyond its initial capacity and to a rank/allocation limit using
a small difficult block. Verify safe growth/subdivision/failure and an
independent residual. Enforce an allocation budget to catch dense-sized ACA
temporary factors without trying to cause a real OOM. For the H-matrix
matvec, show scratch no longer scales as N×workers in the proposed target-
owned implementation. Test allocator accounting separately from noisy RSS.

## 5. Friction regressions and alternative solver tests

### T24 — final traction KKT and force correction

Package A15. Independently recompute Cq, cone violation, stick/slip
alignment, wrong-sign slip and load after the final force correction. Use
full stick, partial slip, all slip under displacement control, and a
force-controlled case near the friction limit. Force balance alone cannot
pass. Deliberately perturb a returned q with an exact-load correction that
worsens local equilibrium and verify the checker rejects it. Check the
projection residual with a fixed dimensionally specified ρ, rather than
shrinking ρ to make a residual look small.

### T25 — normal unloading after shear, held displacement/force

Package A15. Preserve the Ns=32 review sequence p_bar=0.01,
q_bar=(0.001,0), then p_bar=0.0001. Under held displacement, the driver must
relax shear: |q|≤μp and q=0 on open points, with local KKT/history checks.
Under held force, the same large unload is infeasible and must fail without
state mutation. Include a smaller feasible unload for that mode, subsequent
reload, and contact patch splitting/merging. Check the documented default
hold policy and an explicit override in both C++ and Python.

### T26 — zero load, separation and recontact

Package A15. Distinguish p_bar omitted, zero and invalid negative input.
Omission retains normal load; zero unloads. Require the specified zero
tractions/displacements, gap/approach convention and finite diagnostics.
Nonzero held tangential force at complete separation must fail. Recontact
must not restore stale q, Cq or carried stiffness; cumulative history
outputs must follow the documented separation convention.

### T27 — dimensional floors and load-path references

Packages A04/A15. Extend T08 to friction at non-unit E*, L and nonzero
history. Check equivalent dimensionless errors and histories. Retain
full-stick Mindlin stiffness, nu=0 Cattaneo–Mindlin/Ciavarella–Jäger and
Mindlin unloading checks, but compute final KKT independently. For vector
coupling, add oblique loading, rotating tangential direction and load
reversal at representative nu≠0. Do not claim the nu=0 superposition
reference proves all vector cases.

### T28 — final nonlinear threshold and dissipation

Package A15. Use constant Coulomb/Tresca and documented rate-dependent laws.
Create a law/iteration where damping makes the update small while the
undamped model residual is not small. Require final s=model(p,v_final,T)
within declared mixed tolerance, and nonnegative dissipation up to a
dimensional numerical allowance. Include vanishing thresholds, temperature
changes, callback exceptions/nonfinite outputs under the documented model
policy, and a deliberately nonconvergent law. Failures must not commit.

### T29 — transactions, Newton derivatives and vector H²

Packages A15–A17. Snapshot every committed member, including stiffness and
history, then force normal, tangential, threshold, callback and verification
failures; compare the full snapshot after failure. Test a successful retry.

For the Newton branch compare Jacobian-vector products to finite differences
away from projection kinks and validate one-sided/generalized behavior at
them. Compare tiny fixed-threshold problems to an independent convex solve
and supported displacement cases to bipotential Uzawa. Require merit/global
convergence safeguards and honest failure at iteration limits.

For vector H² compare dense/FFT/scalar-H² references, both q/u alias and
nonalias calls, component swaps/parities, and backend selection through the
actual driver API. Verify the test really exercises H² rather than a hidden
FFT fallback.

## 6. Physical observables, mathematical bounds and API safeguards

### T30 — area/perimeter and boundary policy

Package A18. Use empty/full masks, one pixel, rectangles, a checkerboard,
edge-touching patches and periodic-wrap patches with hand-counted internal
transitions. Check h and L normalization, external-edge reporting and raw
area consistency. Reproduce the existing estimator exactly under matching
conventions; flag impossible corrected estimates rather than conceal them.
Do not require correction to improve every tiny or anisotropic shape.

### T31 — objective/error-bound mathematics

Packages A01/A09/A11. On tiny exact SPD QPs verify
0≤f(p)−f(p*)≤G and ||p−p*||²_S≤2G for many feasible nonoptimal p. Include
the one-point counterexample and check G=0 on the manufactured optimum.
Evaluate a known symmetric perturbation E and verify
G_S≤G_(S+E)+2P||Ep||∞. Demonstrate that a nonfeasible pressure cannot be
given an unqualified feasible-QP bound. Cross-check screening with T20.
These tests validate the numerical implementation of the derivations;
the derivations and their assumptions remain part of the specification.

### T32 — controlled model/refinement studies

Package A18. Slow studies hold realization, bandwidth and physical load
fixed while refining the same surface; do not merely reuse a random seed
on different independently generated grids. Track pressure, displacement,
compliance, energy, raw/corrected area and perimeter. Ensure algebraic and
operator errors lie below the discretization changes under study.

Compare collocation and Galerkin only with matching observable definitions
and consistent averaged geometry. For periodic operators, test one Fourier
mode, the explicit DC convention and translation invariance under cyclic
shift. Separately test free-space padding without wrap contamination. For
multigrid research, check constraint-preserving transfer and the declared
Galerkin/rediscretization relation before testing convergence rates.

### T33 — bad inputs and checked sizes

Package A19. Parameterize zero/negative/nonfinite Ns, coarsest, q, radii,
leaf sizes, L, E*, tolerances, loads and invalid material ranges. Reject
coarsest=0 before entering level generation. Check non-power-of-two H²
grids and invalid nesting ratios; retain supported non-power-of-two FFT
grids. Validate wrong array sizes, invalid masks/indices and noncontiguous
Python inputs according to the documented copy/conversion rules.

Exercise checked N, 2N, padded-grid and byte-count calculations at int32 and
size_t boundaries, including Ns=32768, without allocating the arrays.
Use a subprocess timeout as a harness safety limit, but assert the intended
exception rather than counting timeout/OOM as a successful rejection.

### T34 — build/apply lifecycle

Package A19. Apply before build, build twice, float before double and vice
versa, cache release/reuse and a simulated build failure. Repeated build
must follow the idempotent contract, with no duplicate interactions or
stale precision cache. Wrong-sized apply must throw before touching output.
Destroy backing wrappers and verify retained immutable data follows the
documented ownership contract; no dangling kernel callbacks.

### T35 — Python ownership and explicit solve buffers

Package A19. Existing copy properties stay independent; new views are
read-only and retain their owner after wrapper deletion and GC. Repeated
view access should share data without N-sized copies. A subsequent solve
must not mutate an earlier result view. Explicit p_init remains unchanged.
Check contiguous float64, strided input and float32 conversion behavior and
instrument the expected allocation differences. Verify explicit and nested
into-style paths produce equivalent fields/diagnostics.

### T36 — concurrency and allocator policy

Package A19. In a subprocess, start simultaneous calls on one instance and
require documented rejection or safe distinct workspaces. Distinct instances
must agree with serial references. Repeat with callbacks that reacquire the
GIL. Stress FFTW planner creation/destruction/thread settings separately
from plan execution under an FFTW-enabled test job. Use bounded timeouts
and sanitizer tools to detect races/deadlocks; account for known OpenMP
sanitizer limitations explicitly rather than disabling all concurrency
coverage. Verify ordinary solves do not call the process-wide malloc
configuration helper; invoke that helper only in an explicit opt-in test.

## 7. Numerical acceptance tiers

| Tier | Scope | Required acceptance |
|---|---|---|
| Fast CI | Tiny oracles, manufactured cases, API/status/lifecycle, core parity | Deterministic success/failure assertions; ≤10⁻¹² double load error; selected tol=10⁻¹⁰…10⁻¹² |
| Numerical CI | Ns≤256, precision/active/friction matrices | Independent final criteria; double/float targets declared per fixture; no hidden tolerance relaxation |
| Extended | Multiple seeds, high-precision kernels, eigenvalue sweeps, threaded/FFTW, sanitizers | All supported-setting assertions and failure paths; serialized large jobs |
| Performance | Ns=256…4096, optional isolated larger grids | Equal final validation scope/accuracy; full cost/memory statistics and promotion decision |
| Model research | Fixed-surface refinement, Galerkin/periodic/multigrid | Model-specific definitions and error separation; published limitations |

For deterministic normal reference problems use requested normalized
certificate/penetration tolerances 10⁻¹⁰ or 10⁻¹². Float-only demonstration
fixtures may explicitly request effective 2×10⁻⁶, but MUST also satisfy
their load/pressure/penetration checks; the numerical floor is not an
automatic exemption. Pressure-error comparisons use independent oracle
bounds or fixture-specific targets. Contact-area equality and exact
iteration counts are never the sole acceptance criteria.

Existing physics tolerances can remain as discretization gates when still
justified, but formerly weak friction/stall gates must be supplemented by
final local KKT and status checks. Updating an expectation to `stagnated`
is valid only when the test is testing honest failure; it does not fulfill
a test whose purpose is to solve a requested physical load path.

## 8. Benchmarks and promotion rules

### B01 — compact cache and memory

Compare full-table and compact H² build at Ns=256,1024,2048,4096 with ℓ=8,16,
q=4,6; separate coefficient generation, coupling build and first apply.
Measure actual allocations and process peak. Confirm the exact table-byte
removal formula. Include small Ns where b is clipped, and both explicit
and nested construction paths.

### B02 — recurrence at equal final accuracy

Run corrected solver reuse on/off, full and compressed contact, double and
float. Report fresh/recursive/verification calls, corrections, refreshes,
total time and peak memory. Include fixed-support late iterations and
rapidly changing contact. The review's 45→37 count is a mechanism reference,
not a promised universal target.

### B03 — FMM pass tuning

Sweep q=4,6,8 (10 optionally), ℓ=4,8,16, valid near radii 1,2, precision,
thread count and occupied-leaf fraction. Compare dense/tensor transfers,
M2L dense/SVD/shared-basis/batched variants and near-field variants.
Report time per pass and full solve; include build/rank-storage overhead
and signed-direction accuracy. Do not promote a variant based only on
FLOP count or full-grid throughput when its intended use is sparse contact.

### B04 — candidate density and preconditioner cost

Vary candidate/occupied-leaf fractions through approximately 0.1%, 1%, 10%,
50% and 100%, using clustered and fragmented geometry. Measure traversal
construction/amortization, M/L resident storage, FFT preconditioning,
global verification and fallback. Compare Fourier and optional local/coarse
preconditioners on identical candidates. Include a case with evolving C
where preprocessing cannot be amortized over many iterations.

### B05 — algorithm and accuracy schedule

Compare corrected normal solver and simplex-CG; fixed q/precision and
staged policies; active and full paths; FFT and H² where affordable. Use
fixed physical surfaces with H=0.3,0.8,1.0 and at least three fixed seeds.
Select loads spanning dilute contact, intermediate contact and near-full
contact; report achieved area rather than assuming loads produce fixed
fractions across surfaces. Include a Hertz fixture and a flat loaded patch.

### B06 — H-matrix fallback

Measure ACA peak temporary/resident storage, rank, approximation error,
build time and threaded matvec scratch under the same requested tolerance.
Include blocks that exceed the initial rank capacity and cases using
subdivision/fallback. Compare symmetric storage with independently assembled
reference blocks and validate the effect on contact certificates.

### B07 — friction

Compare existing corrected tangential solver and Newton experiment at
matched final force/local-KKT tolerances, including rotating directions,
history/reversal, normal unloading, varying nu and rate laws. Compare FFT,
scalar H² and shared vector H² through the driver. Count probes, inner
solves, Broyden/Newton steps, threshold passes, verification and all operator
applications; do not report only the last inner iteration count.

### B08 — discretization/observable studies

Execute T32 with stored surface provenance and matched sampling/averaging.
Report algebraic/operator errors alongside mesh changes. Separate periodic
and free-space physics. The outcome may be a no-go for an estimator/model
in a particular regime; that is a valid documented research result.

### Measurement protocol

Record revision, compiler/flags, Eigen/FFT engine, CPU, threads/affinity,
working precision, surface/hash/bandwidth, all solver parameters, validation
scope, requested/effective targets, status and independent final errors.
Use fresh-process measurements for peak memory and separate warm/cold cache
timings. Run variants in alternating or randomized paired order, avoiding
concurrent performance jobs. Include all construction/coarse/verification/
output costs in full-solve totals.

For short benchmarks use ≥5 paired samples and report medians plus spread
or a bootstrap confidence interval. Expensive jobs may use fewer samples
only with an explicit uncertainty limitation. Large grids beyond 4096 are
manual, serialized jobs with a preflight memory budget; no routine test
should attempt to OOM a workstation. A new larger-grid measurement is not
required to verify the exact compact-table storage formula.

### Promotion decision

Mandatory correctness fixes ship when correctness gates pass; an increased
iteration count needed to reject false convergence is not a regression in
time to a correct solution. Rebaseline timings against the corrected solver.

For optional speed defaults, require a reproducible benefit on the declared
target workload, with the confidence interval favoring the change and no
unexplained >5% median regression on the representative supported suite.
A 10% median full-solve improvement is the initial practical target for a
new default; a smaller consistent improvement may be accepted with a
written complexity/maintenance justification. Memory-oriented variants may
instead meet an explicit storage target with their measured time tradeoff.
These numbers are promotion policy, not numerical correctness tolerances.

Every promotion requires equal independent final accuracy and scope. If a
variant is useful only at high q, dense occupancy or a particular precision,
select it conditionally and publish that regime. Keep it experimental or
record a no-go when it fails; do not silently lower accuracy to create a
speedup. Preserve reports and immutable numerical fixtures, not machine-
specific timing expectations, in regression coverage.

## 9. Completion checklist

- Each A01–A19 package links its implemented tests and measured decision.
- All reproduced false-success cases become passing corrective regressions.
- Public status/tolerance/zero-load/hold/view/threading behavior is documented
  and tested through Python as well as C++.
- Every advertised certificate states its operator and precision scope.
- Kernel fixtures and contact oracles are independent of the code under test.
- Reported memory includes global FFT/tree and verification costs.
- Experimental methods have a numerical/benchmark go or no-go report.
- Existing tests run alongside the new suite; updated expectations explain
  intentional corrected semantics rather than hiding failures.
