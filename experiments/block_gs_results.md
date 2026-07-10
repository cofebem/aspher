# FETI-style block Gauss-Seidel/Jacobi for Boussinesq contact — experiment results

**Script**: `block_gs_feti.py`. Ns=1024, self-affine surface (seed 42, rms 0.02),
prescribed-approach formulation (α = α* from a converged load-controlled
reference, `hc.solve_nested`, tol 1e-10), tiles solved with an exact local Love
operator (one small `ContactSolver(backend="fft")` reused for every tile by
translation invariance), local LCPs by projected CG (PK minus load constraint),
inter-tile coupling through one global FFT matvec per sweep (frozen far field).

**Validation**: 1×1 "tiling" reproduces the reference to rel-L2 1.3e-5 with the
load recovered to 1.000 without any load constraint — the fixed-α formulation
and the Python LCP solver are correct.

## Gluing error of fully independent tiles (interaction OFF)

| load | area | 2×2 tiles | 4×4 | 8×8 |
|---|---|---|---|---|
| p̄=0.002 (dilute) | 0.76% | rel-L2 0.21, load +13% | 0.62, +52% | 0.81, +84% |
| p̄=0.02 (moderate) | 7.1% | rel-L2 0.59, load +49% | 1.33, +153% | 2.28, +347% |

Ignoring inter-tile elasticity is never acceptable: even the friendliest case
(4 large tiles, dilute contact) mis-predicts pressure by 21% and load by 13%.

## Frozen-far-field iteration (Jacobi, ω=1; damped ω=0.5 where noted)

Asymptotic contraction factor ρ per sweep (rel-L2 pressure error vs p*):

| load | 2×2 | 4×4 | 8×8 |
|---|---|---|---|
| p̄=0.002 | 0.62 | 0.69 (oscillatory) | ~0.87 (near-stall, oscillatory) |
| p̄=0.02 | 0.75 (oscillatory) | **limit cycle** (load 2.4×↔0.09×); ω=0.5 → 0.87 | **period-2 divergence** (area 22%↔0%); ω=0.5 → 0.90 |

- Undamped Jacobi diverges into contact-set limit cycles as soon as coupling
  dominates (more/smaller tiles, higher load). Damping (ω=0.5) always
  stabilized it but with ρ ≈ 0.87–0.90.
- ρ degrades with tile count and with contact density — the smooth (small-q)
  modes carry the operator's largest eigenvalues (Ŝ ∝ 1/|q|), so long-range
  coupling is the *strongest* part of the problem, and block-local solves
  leave exactly that part untreated. This is the L/H coarse-space gap of
  domain decomposition, at its worst because the kernel is nonlocal.
- Metric caveat: under damping, `area(p>0)` is contaminated by exponentially
  decaying stale entries — only the rel-L2 column is meaningful there.

## Cost accounting (vs the existing solver)

Reference: nested+|q|-preconditioned PCG = 58 finest-level iterations
(116 global matvecs) to err 1e-10 at this surface/load.

- Sweeps needed for rel-L2 1e-6: ~29 at ρ=0.62 (best case) to ~100+ at
  ρ=0.87–0.90. Each sweep costs ≥1 global matvec **even if local solves were
  free** — so the scheme is at best break-even with PCG on matvec count, and
  in the full-tile implementation the local solves dominate wall time
  (measured 2–12 s/sweep vs 0.02 s/global matvec).
- The one potentially interesting corner: dilute contact + patch-restricted
  local solves (active area 0.76% → local work ~free), giving ~29 global
  matvecs to 1e-6 vs PCG's 116 to 1e-10 — a possible small win, but it
  requires the 2×2-like coupling regime (few, well-separated patch clusters)
  and evaporates as contact densifies or fragments.

## Conclusion

FETI-style tearing does not pay for this operator. The coupling is not an
interface effect — it is the dominant part of the spectrum — so block-local
solves either diverge (undamped) or crawl (damped), and every sweep needs the
same global matvec the existing PCG spends its iterations on. The productive
descendant of this idea is **active-set localization** (solve only near the
contact patches, glue with the existing H2/FFT far field, keep the global
coarse/spectral treatment of smooth modes) — already on the What-Is-Left list —
rather than geometric domain decomposition.
