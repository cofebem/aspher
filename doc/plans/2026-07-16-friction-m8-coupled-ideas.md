# Friction M8 — Fully Coupled Dissimilar-Material Contact (first ideas)

**Status:** IDEAS / NOTES ONLY — not a finalized implementation plan. Captures
direction so M8 can be picked up later; deferred by the user 2026-07-16.
Turn into a proper spec+plan (brainstorming → writing-plans) when work starts.

**Date:** 2026-07-16
**Prereqs:** friction M1–M7 complete and merged (uncoupled v1). See
`doc/specs/2026-07-13-frictional-contact-design.md` and its §1 non-goals.

---

## 1. What M8 is, and when it matters

v1 (M1–M7) is deliberately **uncoupled** (Dundurs β = 0): the influence
operator is block-diagonal, `u_z = S p` and `(u_x,u_y) = C (q_x,q_y)`
independent, so pressure never feels shear and vice versa. For **identical /
elastically similar** materials (or ν = 1/2) that is *exact*.

Coupling matters only for **dissimilar materials** (β ≠ 0), where:
- normal traction `p` induces tangential surface displacement, and
- shear `(q_x,q_y)` induces normal surface displacement.

Physical consequences M8 would capture: the contact pressure itself changes
under tangential loading (the normal and tangential problems must be solved
*together*, not staggered); asymmetric stick/slip; the classical
Cattaneo–Mindlin / Ciavarella–Jäger superposition **no longer holds** (it
relied on the corrective shear field being a scaled normal solution — true
only at β = 0). Reference physics: Johnson §7; Wang et al. 2022
(`REF/Tangential/Wang_2022.pdf`) — coupled partial slip of dissimilar rough
surfaces.

## 2. What already exists (foundation laid in v1 — verified in code 2026-07-16)

- **Closed-form coupling kernels are derived and cross-checked on paper**,
  just not coded:
  - Pohrt & Li 2014 eqs. (14)–(16)/(19): discrete `K_xz, K_yz, K_zx, K_zy`
    coefficients (all ∝ 1−2ν).
  - Duquesne thesis 2025 Annexe G, eq. G.22: the (1−2ν) normal↔tangential
    coupling primitive in corner form (this is the one cross-checked in
    spec §3.3).
  - Tamaas `src/model/influence.hh` `Boussinesq<3,0>`: the coupled Fourier
    operator — the xz/zx blocks are **imaginary, antisymmetric, ∝(1−2ν)**
    (recorded during the M6 equation cross-check).
- **Operator layer is coupling-ready**: the H2 kernel-functor constructor
  (M1) is kernel-agnostic; offsets are documented target−source *specifically
  because* the coupling kernels are odd in one variable (unlike the even
  Love/xx/yy and jointly-even xy). The design was structured as 3×3 blocks.
- **A coupling-proof solver already exists**: the bipotential Uzawa (M6,
  `bipotential.{hpp,cpp}`) treats normal + tangential simultaneously through
  one cone projection and is **indifferent to operator symmetry** — it was
  explicitly built as the designated strategy for the coupled case. It
  currently only wires the block-diagonal `S` and `C`.

## 3. What M8 needs to build

1. **Coupling kernel tables** — code + verify `K_xz, K_yz` (normal→tangential)
   and `K_zx, K_zy` (shear→normal), ∝ 1−2ν. Verification mirrors M2:
   closed form vs adaptive quadrature of the point kernel, parity/symmetry
   (these are ODD in one variable — the xy-guard machinery and target−source
   convention from M1/M2 already anticipate this), and a Fourier-symbol
   cross-check against the imaginary ∝(1−2ν) off-diagonal from Tamaas
   `influence.hh`. Reciprocity check: `K_xz` and `K_zx` are related by the
   Betti symmetry of the stacked operator (sign per the paper's convention —
   pin it against Duquesne G.22).
2. **Full 3×3-block operator** `[q_x;q_y;p] → [u_x;u_y;u_z]` with the four
   off-diagonal blocks. FFT first (exact, mirrors M3's `TangentialFFTOperator`
   — but the xz/zx spectra are IMAGINARY, so the per-mode 3×3 mix is complex,
   not the all-real 2×2 of the tangential block); H2 as the O(N) follow-up
   (three→nine kernel-functor instances, or a vector-valued instance).
   The full operator is **symmetric** (Betti) but **indefinite in the naive
   stacking** unless signs are set so it's the SPD compliance — decide the
   sign/stacking convention up front.
3. **Solver strategy — the crux.** The production staggered projected-CG
   CANNOT handle p↔q coupling (it solves the normal QP first, assuming p ⊥ q).
   Two candidate paths:
   - **(preferred) Promote the bipotential Uzawa to production**: wire the
     coupled 3×3 operator into `solve_bipotential`'s predictor
     (`τ_n = p − ρ(g + μ|w|)` already couples through the *cone*; the
     *elastic* coupling enters through the operator applies `u_z = f(p, q)`,
     `u_t = f(q, p)`). It's coupling-proof by design and already validated
     against the uncoupled production path — the smallest conceptual jump.
     Cost: first-order, slow — M8 would need to make it fast enough
     (better ρ, a coupled preconditioner, possibly a 3×3 spectral
     accelerator) or accept it as the reference-grade coupled solver.
   - **(alternative) Wang et al. 2022 three-level coupled iteration**: keeps
     the fast CG structure but adds an outer coupling loop (global shear-
     traction adjustment + convergence on lateral load balance). More
     machinery, closer to the production path's speed.
4. **3×3 coupled preconditioner** — extend `TangentialFourierPreconditioner`
   to the full symbol inverse (now with the imaginary xz/zx blocks). Only
   needed if the CG-style path (3b) is taken; the Uzawa uses ρ, not this.
5. **Validation** — dissimilar-material partial slip. Primary reference:
   Wang et al. 2022 (coupled rough dissimilar surfaces). Also: a coupled
   Hertzian sphere (Spence / Hills dissimilar-material cylinder solutions),
   and the *breakdown* of Ciavarella–Jäger at β ≠ 0 as a sanity signature
   (the uncoupled gate should degrade smoothly as β → 0⁺).

## 4. Key open questions / decisions to settle at brainstorming time

- **Solver choice (3a vs 3b)** — the single biggest decision. Uzawa is the
  low-risk, coupling-proof, already-validated path but slow; Wang three-level
  is fast but more machinery and a new convergence structure. Likely: land
  the Uzawa coupled path first (correctness + reference), then decide if speed
  warrants the three-level.
- **Parameterization** — expose `(E1,ν1,E2,ν2)` and compute β internally, or
  take `(E*, ν, β)` directly? Dundurs β is the single parameter the coupling
  scales with; but users think in per-body constants. Probably per-body with
  β derived, defaulting to the identical-material (β=0) case = exactly v1.
- **Backward compatibility** — β=0 must reduce to the shipped uncoupled path
  bit-for-bit (a hard regression gate; the coupled operator's off-diagonal
  blocks vanish at ν→½ / identical materials, and the driver should route to
  the fast uncoupled solver when β=0 to avoid paying the Uzawa cost
  needlessly).
- **API** — a `coupled=True` flag / `beta` argument on `FrictionSolver`, or a
  separate `CoupledFrictionSolver`? The incremental driver, models, and
  Python surface (M5/M7) should carry over unchanged; only the operator +
  inner solver swap.
- **Incremental formulation** — the step QP (spec §2.3) assumed the normal
  problem is history-free within a step; with coupling, `p` and `q` co-evolve
  each step, so the `u_hist = −C qⁿ` bookkeeping generalizes to a stacked
  3-field history. Check the FrictionDriver's transactional/threshold-loop
  structure still holds (it should — it's operator-agnostic above the solve).

## 5. Rough milestone shape (when it becomes a real plan)

- M8a: coupling kernel tables + verification (mirror M2).
- M8b: full 3×3-block FFT operator + dense-reference gate (mirror M3a).
- M8c: coupled bipotential Uzawa (wire the 3×3 operator into `solve_bipotential`)
  + β=0 ≡ uncoupled regression + a dissimilar-material validation gate.
- M8d (optional): coupled preconditioner / speed, or the Wang three-level path.
- M8e: H2 coupled operator (O(N)); M8f: driver/Python `beta`/coupled option.

## 6. Non-goals to keep in mind

Finite sliding / geometry advection, frictional heating, adhesion, and the
single-precision tangential path remain out of scope (v1 non-goals carry
over). M8 is specifically the *elastic* normal–tangential coupling, nothing
more.
