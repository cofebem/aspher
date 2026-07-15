# Frictional Contact Extension — Design

**Date**: 2026-07-13
**Status**: approved design, pending user review of this document
**Branch (planned)**: `feat/friction`

## 1. Goal

Extend ASPHER from frictionless normal contact to quasi-static frictional
contact on the half-space, with three friction models:

1. **Tresca**: stick until `|q| = τ_c` (pressure-independent threshold);
2. **Coulomb**: stick until `|q| = μ p`;
3. **Generic user model**: threshold `s = τ_c(p, |v|, T)` supplied as a
   vectorized Python callable (pressure–slip-velocity–temperature dependent).

New unknowns: tangential tractions `(q_x, q_y)` and the induced tangential
surface displacements `(u_x, u_y)`, alongside the existing `p`, `u_z`.

### Decisions fixed during brainstorming

| Decision | Choice |
|---|---|
| Elastic normal–tangential coupling | **Uncoupled v1** (similar materials, Dundurs β=0); operator layer structured as 3×3 blocks so Cerruti off-diagonals can be added later. |
| Loading | **Incremental quasi-static driver** from the start (sequence of load steps, state carried). |
| Control | **Both**: force-controlled (impose `Q`, rigid shift `δ_t` is the multiplier) and displacement-controlled (impose `δ_t` per step). |
| Temperature | **Input field** per step (user may update it in Python between steps); frictional-heating solver out of scope. |
| User friction model | **Vectorized Python callable**; optional derivative callables; no AD framework (see §7). |
| Backends | **FFT and H2 both in v1** (dense for tests). |
| Solver strategy | **A + B**: staggered projected-CG (production) + de Saxcé–Feng bipotential Uzawa (reference/cross-check, future coupled-case candidate). |

### Non-goals of v1 (documented follow-ups)

- Elastic normal↔tangential coupling (dissimilar materials). Reference for
  later: Wang et al. 2022 (`REF/Tangential/Wang_2022.pdf`), three-level
  coupled iteration; the bipotential Uzawa path is coupling-proof by design.
- Nested (cascadic) continuation and active-set driver for the *tangential*
  solve (warm start across load steps covers most practical gain in v1).
- Single-precision tangential path (interfaces must not block it).
- Frictional heating / thermal solver; adhesion; finite sliding of the
  geometry (gap field is not advected by `δ_t` in v1).

## 2. Continuum and incremental formulation

### 2.1 Setting

Elastic half-space (`E*`, and reduced shear modulus `G*`; for identical
materials `1/G* = (2−ν)/(4G)·2`, Johnson's convention — constants pinned in
§3), rigid rough counterpart. Surface tractions `t = (q_x, q_y, p)` on the
contact plane; surface displacements `u = (u_x, u_y, u_z)`.

Uncoupled (β=0) influence operator, block-diagonal:

```
u_z = S p                    (existing Love/Boussinesq operator, untouched)
(u_x, u_y) = C (q_x, q_y)    (new symmetric 2×2-block Cerruti operator)
```

Normal problem: exactly today's QP (Polonsky–Keer PCG), solved first each
step; `p` never feels `q` in v1.

### 2.2 Slip kinematics and friction law

Counterpart rigid tangential shift `δ_t` (2-vector). Local relative slip of
the counterpart w.r.t. the surface point:

```
w_i = δ_t − (C q)_i        (sign: q is the traction ON the half-space —
                            friction drags the surface along the slip)
```

Friction law (per point, on the *total* traction):

- stick: `|q_i| < s_i` and `ẇ_i = 0`;
- slip:  `|q_i| = s_i` and `q_i` parallel to `ẇ_i` (traction drags the
  surface along the relative motion; reaction on the counterpart opposes it);
- open (`p_i = 0`): `s_i = 0` hence `q_i = 0`.

Threshold: `s_i = τ_c(p_i, |ẇ_i|, T_i)` — Tresca `τ_c`, Coulomb `μ p_i`,
generic callable.

### 2.3 Incremental problem (one load step n → n+1)

Backward-Euler slip increment `Δw_i = Δδ_t − (C (q − qⁿ))_i`, slip velocity
`|Δw_i|/Δt`. For a **frozen threshold field** `s`, the step's tangential
problem is the convex QP

```
min over q :  J(q) = ½ (q−qⁿ)ᵀ C (q−qⁿ) − Δδ_t · Σᵢ qᵢ
subject to    |qᵢ| ≤ sᵢ  ∀i
[force control: Δδ_t unknown, constraint h² Σᵢ qᵢ = Qⁿ⁺¹, Δδ_t = multiplier]
```

KKT ⇔ incremental friction law: interior points stick (`Δw_i = 0`), bound
points slip with `Δw_i = λ_i q_i/|q_i|`, `λ_i ≥ 0` (correct direction).
Per-step dissipation `D = Σ q_i·Δw_i = Σ λ_i s_i ≥ 0` (test assertion).
This is the standard incremental variational statement (Tresca-type problem
per step); Coulomb with known `p` is Tresca with spatially varying `s`.

History enters only through `u_tⁿ = C qⁿ` (stored state; one matvec per step
or maintained incrementally).

### 2.4 Threshold outer loop (generic model only)

`s` depends on `|Δw|/Δt` ⇒ fixed point: `s⁽ᵏ⁺¹⁾ = τ_c(p, |Δw⁽ᵏ⁾|/Δt, T)`
with Aitken relaxation, inner QP warm-started from the previous pass.
Expected ≤ 5–10 passes; max-outer cap + stagnation guard with honest
`converged` reporting (existing conventions). Tresca/Coulomb: exactly one
pass (threshold does not depend on the tangential solution).

> **Implementation note (M5)**: plain damped fixed point (s ← ½(s + s_new),
> stop on rel change < threshold_rtol, default 1e-3) instead of Aitken —
> measured 7 passes on the rate-weakening smoke test; Aitken remains a
> follow-up if a production law needs it.

## 3. Kernels (input: literature closed forms; user cross-check)

### 3.1 Element-integrated stencils

Uniform rectangular elements, translation invariance ⇒ per-component
Ns×Ns lookup tables exactly like the Love table today. Closed forms are
**taken from the literature, not re-derived**:

- **Pohrt & Li 2014** (`REF/Tangential/Complete-boundary-element-…pdf`),
  eqs. (12)–(21): all nine discrete influence coefficients on a rectangular
  grid — `K_xx` (17), `K_yy` (20), `K_xy = K_yx` (18)/(21) for v1;
  `K_xz`, `K_yz`, `K_zx`, `K_zy` (14)–(16)/(19) enter only with coupling
  (∝ 1−2ν; recorded now, used later).
- **Dydo & Busby 1995** (`REF/Tangential/Dydo_et_al_1995.pdf`), eqs.
  (7)–(9), (16)–(18) with harmonic functions (25)–(27): independent
  closed-form source (also subsurface/linear loading, useful beyond v1).

Verification: each stencil entry vs adaptive numerical quadrature of the
point-force Cerruti kernel (`C_xx ∝ (1−ν)/ρ + ν x²/ρ³`, `C_xy ∝ ν xy/ρ³`)
over the source rectangle, including self-terms; plus parity/symmetry checks
(`K_xx` even in x,y; `K_xy` odd in x and y; x↔y maps `K_xx`→`K_yy`).
**The user will supply/cross-check the analytical formulas** (offered during
brainstorming) — any discrepancy with the papers resolved before M2.

### 3.2 Fourier symbols

Tangential block has the 2×2 matrix symbol (anisotropic, still ∝ 1/|k|):

```
Ĉ(k) = (2/(E*|k|)) [ α I − β k⊗k/|k|² ],   α, β set by ν (pinned with §3.1)
```

Used for: (i) the spectral preconditioner (per-mode 2×2 analytic inverse of
the symbol, applied to the stick-masked residual — direct generalization of
`fourier_precond`); (ii) spectral validation of the stencil tables; (iii) ρ
estimate for the Uzawa reference (operator norm from the symbol).

### 3.3 Equation cross-check — RESOLVED 2026-07-13 (spec §12 gate)

Verified against two independent sources supplied by the user:

- **Duquesne thesis 2025, Annexe G**
  (`REF/Tangential/ThesePBRemyDUQUESNE.pdf`, eqs. G.19–G.22): corner-form
  primitives `U^qx_x = −(1−ν²)/(πE)·x ln(ρ̄−y) − (1+ν)/(πE)·y ln(ρ̄−x)` and
  `U^qx_y = −ν(1+ν)/(πE)·ρ̄` reduce, under the second mixed difference (the
  y-independent `2x ln|x|` term cancels), EXACTLY to our
  `(1/(2πG))[(1−ν)X_log + Y_log]` and `(ν/(2πG))[R(k,n)−R(k,m)+R(l,m)−R(l,n)]`
  with `1/(2πG) = (1+ν)/(πE)`. This independently confirms the Pohrt & Li
  eq. (18) h²-typo resolution. G.22 gives the (1−2ν) normal↔tangential
  coupling primitive for the post-v1 coupled case.
- **Tamaas `src/model/influence.hh`** (`Boussinesq<3,0>::applyU0`, Fourier
  space): with d± = (∓i q̂x, ∓i q̂y, 1), d₂ = (i q̂x, i q̂y, 0), the operator
  `B = [2I + (1−2ν) d₊d₋ᵀ + d₂d₂ᵀ − e₃e₃ᵀ]/(2μ|q|)` has tangential block
  `[I − ν q̂⊗q̂]/(μ|q|)` ≡ our symbol (since `1/μ = 2/(E*(1−ν))`), zz block
  `2/(E*|q|)` (Love), and imaginary antisymmetric xz coupling ∝(1−2ν).
- **Tamaas cone projection** (`Kato::enforcePressureCoulomb`,
  `src/solvers/kato.hh`): identical three-case orthogonal projection onto
  K_μ as de Saxcé–Feng eq. (105) — our planned shared projection module.
  Note: Tamaas's general coupled solver (`Condat`) solves the *associated*
  Coulomb law (docstring caveat: normal/tangential slip coupled); our
  bipotential path targets the true non-associated law — a differentiator
  to keep.
- **Validation option for M4/M5**: cross-validate the tangential solve
  against Tamaas `PolonskyKeerTan` (Coulomb `solve` / `solveTresca`) in the
  `fluidpaper` env, following the `tamaas-comparison` skill pattern (mind
  the dcfft quirks of CLAUDE.md for the non-periodic case).

## 4. Operators

- **FFT backend** (`fft_operator`): three new kernel tables (`C_xx`, `C_yy`,
  `C_xy`), spectra precomputed once; blocked tangential matvec
  `(q_x,q_y) → (u_x,u_y)` = 2 forward + 2 inverse padded transforms with the
  2×2 spectral multiply in between. Exact to roundoff (the validation anchor
  for H2), same zero-padded Hockney scheme, structurally-zero line skipping,
  object-owned scratch, `matvec_into` contract.
- **H2 backend** (`h2_operator`): **kernel-functor generalization** — the
  operator takes a far-field kernel `g(dx,dy)` + near-field element stencil
  per component instead of hardcoding Love. Then three tangential instances
  (or one vector-valued instance) sharing one `UniformQuadTree`. This
  refactor lands FIRST, gated by the existing normal-path test suite
  (bit-for-bit or stated tolerance). Chebyshev machinery unchanged (Cerruti
  kernels are as smooth as Love off the origin); accuracy gates mirror the
  normal path (q=4 ≈ 1e-4, q=6 ≈ 3e-6 vs dense).
- **Dense reference** for small-Ns tests, as today.

## 5. Solvers

### 5.1 Production: staggered projected-CG (per load step)

1. **Normal solve**: existing PK PCG (warm-started from previous step).
2. **Threshold**: `s = model.threshold(p, |Δw|/Δt, T)` (outer loop §2.4 if
   velocity-dependent).
3. **Tangential QP**: vector Polonsky–Keer projected CG —
   - *free set* = stick candidates (`|q_i| < s_i`), where CG drives the
     residual (`Δw_i`) to zero;
   - *projection* = radial clamp of each 2-vector `q_i` onto its disk of
     radius `s_i` (replaces the `p ≥ 0` clamp);
   - *release check* (analog of PK's overlap correction — omitting it breaks
     convergence, as in the normal problem): bound points whose multiplier
     sign is wrong (`Δw_i · q_i < 0`) return to the free set with a gradient
     step;
   - *β*: PR+ (FR option), exact line search on the free set, all scalar
     reductions accumulated in double (existing float-safety conventions);
   - *preconditioner*: 2×2 spectral symbol inverse on the stick-masked
     residual, mean-zeroed per component under force control;
   - *force control*: `mean(q) = q̄` enforced exactly like `mean(p) = p̄`
     today (rigid shift `δ_t` = multiplier); *displacement control*: no mean
     constraint, `Δδ_t` sits in the RHS (strictly simpler).
4. **State update** (§6) + per-step result.

> **Implementation note (M4, 2026-07-14)**: the shipped algorithm differs
> from the sketch above in three load-bearing ways, all forced by measured
> failures. (1) The projected CG runs over the FULL candidate set
> A = {s_i > 0} with radial disk clamping (Tamaas PolonskyKeerTan-style),
> not over the strict stick set: bound points retain a direction DOF that a
> stick-set CG never updates. (2) A **two-metric projection** strips the
> outward-radial residual (the Lagrange multiplier −λq̂) from the CG
> direction at bound points — letting it drive the search corrupts the
> exact line search (objective oscillation, stall at ~1e-2 error); β
> restarts only on stick/slip partition changes, the best-error iterate is
> returned, and a steepest-descent fallback handles non-descent
> preconditioned directions. (3) **Force control is an outer
> Newton/Broyden iteration on the rigid shift δ_t** (the Lagrange
> multiplier of mean(q) = q̄): F(δ) = q_mean(δ) − q̄, stiffness initialized
> from two small-shift probe solves, noise-guarded Broyden updates, outer
> stop when F stops responding at the inner resolution, then a terminal
> additive interior correction meets the load exactly (~1e-16). The
> per-iteration additive load correction sketched above is UNSTABLE
> (period-2 feedback with the interior-mean δ_t estimate and the exact
> line search, gain ≈1.5/it at 40% slip) and was abandoned.

### 5.2 Reference: de Saxcé–Feng bipotential Uzawa

From `REF/BiPotential/Feng_Saxce_bipotential.pdf`, eqs. (102)–(105),
transplanted to the BEM operators; shares the cone-projection module and the
operators with §5.1. Per iteration, for every point:

```
predictor:  τ_i = r_i − ρ ( Δw_i + (Δu_{n,i} + μ|Δw_i|) n )
corrector:  r_i = proj(τ_i, K_μ)      (analytic, 3 statuses:
            separation r=0 / stick r=τ / slip: radial-cone return, eq. 105)
```

Solves normal+tangential **simultaneously** (no staggering), derivative-free,
indifferent to operator symmetry ⇒ the designated strategy once coupling
arrives. `ρ` from the operator-norm estimate via the symbols; optional
preconditioned metric. Load constraints via multiplier update on `δ`
(normal) and `δ_t` (tangential) per sweep.
Cone generalizations: Tresca ⇒ cylinder of radius `τ_c`; generic model ⇒
cylinder with `s` frozen from the predictor state per iteration.

> **Implementation note (M6, 2026-07-15)**: shipped as `solve_bipotential`
> (`bipotential.{hpp,cpp}`), displacement-controlled (imposed α, δ_t — the
> cross-check obtains both from a production run); ρ = rho_scale/λ_max with
> λ_max from 20 power iterations on the stacked operators; Coulomb cone +
> frozen-threshold cylinder (Tresca / generic-with-frozen-s). Measured at
> Ns=32 (Hertz, partial slip): **790 sweeps** to rel-update 1e-9, agreeing
> with the production staggered path to rel-L2 5.8e-8 (p) / 4.3e-6 (q), and
> passing a direct KKT self-check at the 1e-10 level — the two independent
> solver families confirm each other. Per sweep = 1 normal + 1 tangential
> matvec (vs the production path's ~2 per CG iteration but far fewer
> iterations); at Ns=32 the reference costs ~10× the production wall time.
> Force-controlled and coupled (dissimilar-material) Uzawa variants remain
> the documented future path.

**Contract in v1**: reference implementation — must reproduce §5.1 solutions
(rel-L2 gate) on small/medium grids for all three models; performance at
scale is *benchmarked and reported, not promised* (first-order method; the
paper demonstrates it at FEM scales of 10²–10³ nodes, not 10⁶–10⁸).

## 6. Incremental driver and state

C++ `FrictionDriver` owning the operators, the friction model, and state:

| State | Meaning |
|---|---|
| `p`, `q` | total tractions at last accepted step |
| `u_t` | tangential elastic displacement `C q` (avoids a rebuild matvec) |
| `w_acc` | accumulated slip (output/diagnostics) |
| `δ_t`, `δ_z` | rigid shifts |

`step(Δload, Δt, T)` executes §5.1 (or §5.2 on request) and returns a
per-step result; `reset()` clears history. Python drives the step sequence
and may update `T` (user-built thermal loops) or the gap between steps.
History lives in C++; nothing is recomputed from Python between steps.

> **Implementation note (M5)**: implemented as `FrictionDriver`
> (`friction_driver.{hpp,cpp}`, FFT backend) with
> `FrictionStepSpec`/`FrictionStepResult`; `solve_tangential` gained
> `u_hist` (per-point history offset — the incremental QP residual is
> g = u − uⁿ_t − Δδ), `g_floor`, and `K_io`/`delta_init` carry-over
> (per-step force solves skip the ε-probes: measured 1475 → 14 inner
> iterations). Targets are TOTAL loads/shifts; the driver converts to
> increments internally. `step()` is TRANSACTIONAL: a non-converged step
> leaves the driver state unchanged. Dissipation D = h²Σ q·Δw reported per
> step, gated ≥ 0 in tests.

## 7. Friction-model interface (and the autodiff decision)

C++ `FrictionModel`: `threshold(p, v, T) -> s` vectorized over contact
points (called once per outer pass — negligible next to matvecs), plus flags
(`velocity_dependent`, `pressure_dependent`).

- Built-ins `Tresca(τ_c)`, `Coulomb(μ)` in C++ (templated scalar so
  dual-number exact derivatives are available later if a Newton solver is
  ever added).
- `UserFriction(fn, d_dp=None, d_dv=None)` wraps a vectorized Python
  callable (numpy arrays in/out; GIL taken once per outer pass). Users may
  pass JAX/sympy-generated functions.

**Autodiff decision**: no AD framework. The chosen solvers are
derivative-free (thresholds only). Derivatives appear solely in the local
scalar root-solve when the law is strongly velocity-dependent (implicit
`s(|Δw|)` during the point projection): use the user-supplied `d_dv` if
given, else a safeguarded secant. Rationale: an AD dependency (CppAD/Enzyme)
buys nothing for a scalar law of 3 arguments and complicates the build;
revisit only if a semi-smooth Newton solver is added.

## 8. Python API

```python
import aspher as hc

model = hc.CoulombFriction(mu=0.3)           # hc.TrescaFriction(tau_c=0.01)
# hc.UserFriction(lambda p, v, T: mu0*p/(1+v/v0), d_dp=None, d_dv=None)

fs = hc.FrictionSolver(grid_size=1024, backend="fft",   # or "h2" (+ q, leaf_side)
                       E_star=1.0, nu=0.3, model=model, precond=True)
fs.set_gap(g0)                                # gap = -surface, as today

r1 = fs.step(p_bar=0.05)                      # normal-only step
r2 = fs.step(q_bar=(0.01, 0.0), dt=1.0, T=T)  # force-controlled tangential
r3 = fs.step(delta_t=(1e-4, 0.0), dt=1.0)     # displacement-controlled
fs.reset()
```

`FrictionStepResult`: `pressure, qx, qy, ux, uy` (light-mode skippable),
`status` (0 open / 1 stick / 2 slip), `slip_x/slip_y` (increment),
`delta_t` **or** `Q` (whichever was not imposed), `dissipation`,
`iterations` (normal, tangential, outer), `converged`, normal-solve scalars
as today. `fs.step(..., solver="uzawa")` selects the reference path.
Existing `ContactSolver` / `solve_nested` are untouched.

## 9. Code layout

| File | Contents |
|---|---|
| `include/cerruti_kernel.hpp`, `src/cerruti_kernel.cpp` | point + rectangle-integrated tangential kernels, tables, symbols (constants from §3) |
| `include/friction_model.hpp` | Tresca/Coulomb, user-callback wrapper, cone projection (shared §5.1/§5.2) |
| `include/friction_solve.hpp`, `src/friction_solve.cpp` | tangential projected CG, threshold loop, `FrictionDriver`, Uzawa reference |
| touched | `fft_operator` (tangential tables + blocked matvec), `h2_operator` (kernel-functor refactor — first, regression-gated), `fourier_precond` (2×2 symbol), `bindings.cpp`, `CMakeLists.txt`, `CLAUDE.md` |
| tests | `test_cerruti` (stencils/symbols), `test_friction` (solvers/physics), extensions to `test_fft`/`test_h2` |

## 10. Validation gates

1. **Stencils**: closed forms vs adaptive quadrature of the point kernel
   (incl. self-terms); parity/symmetry; Fourier symbol vs table FFT.
2. **Operators**: FFT tangential matvec vs dense ≈ 1e-15 (double); H2
   tangential vs dense (q=4/q=6 gates); H2 refactor reproduces the entire
   existing normal test suite.
3. **Cattaneo–Mindlin** (flagship): force-controlled sphere, Coulomb —
   stick radius `c/a = (1 − Q/(μP))^{1/3}`, traction distribution vs
   analytic; Mindlin tangential compliance.
4. **Ciavarella–Jäger on rough surfaces** (Paggi et al. 2014, eqs. 15–19):
   monotonic tangential loading ⇒ `q = μ[p(F_z) − p*(Q*)]` with
   `Q* = F_z − F_x/μ`, and `A_stick/A_0 = 1 − F_x/(μF_z)` — validated
   against **two runs of the existing normal solver** (Coulomb only: the
   superposition needs the corrective term to be a scaled normal solution,
   which does not hold for Tresca).
5. **Limits**: `τ_c → ∞` ⇒ full stick (tangential flat-punch analog);
   `Q → μP` ⇒ gross sliding onset; `μ = 0` ⇒ zero shear + normal result
   identical to existing solver (regression).
6. **Cross-check**: Uzawa vs production path, rel-L2 gate, all three models.
7. **Physics sanity**: per-step dissipation ≥ 0 (assert); Mindlin cyclic
   load/unload hysteresis qualitatively correct; rough-surface smoke test at
   Ns=1024, iteration counts recorded in `CLAUDE.md` Validated Numbers.

## 11. Milestones (for the implementation plan)

- **M1** — H2 kernel-functor refactor, regression-gated (no new physics).
- **M2** — Cerruti stencils + symbols + `test_cerruti` (user equation
  cross-check closes here).
- **M3** — Tangential operators (FFT + H2 + dense) + operator tests.
- **M4** — Tangential projected CG + 2×2 preconditioner; displacement
  control first, then force control; Cattaneo–Mindlin + Ciavarella–Jäger
  gates.
- **M5** — Incremental `FrictionDriver` + friction-model interface
  (Tresca/Coulomb/user) + threshold outer loop; cyclic Mindlin test.
- **M6** — Bipotential Uzawa reference + cross-check gate + scale benchmark.
- **M7** — Bindings, examples (`example_friction.py`), docs (`CLAUDE.md`,
  theory notes in `doc/theory/`).

## 12. Open items

- ~~The exact constants (`G*`, symbol `α, β`) pinned at M2 against the papers
  and the user's analytical formulas~~ — RESOLVED, see §3.3 (Duquesne annex G
  exact match; Tamaas `influence.hh` symbol match; 2026-07-13).
- Uzawa `ρ` selection heuristic (symbol-based estimate + safety factor) —
  tuned at M6, reported honestly.
- `REF/BiPotential` was said to contain two references; only
  `Feng_Saxce_bipotential.pdf` is present. If the second (e.g., de Saxcé &
  Feng 1991, Mech. Struct. & Mach.) matters for the Uzawa variant, add it
  before M6.
