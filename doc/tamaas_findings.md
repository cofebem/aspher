# Tamaas 2.8.1 findings (affect any "non-periodic" Tamaas comparison)

Notes collected while validating ASPHER's benchmark against
[Tamaas](https://gitlab.com/tamaas/tamaas) 2.8.1; verified against Hertz
theory. They matter for anyone comparing a non-periodic (free-space) contact
solver with Tamaas.

1. `tm.PolonskyKeerRey` **ignores** `ModelFactory.registerNonPeriodic` unless
   you also call `solver.setIntegralOperator("dcfft")` — otherwise it
   silently solves the periodic problem (pressures bit-identical to a model
   without the registration).
2. The `dcfft` operator's effective modulus is **2 E²** instead of `E`:
   Hertz with `E = 1` gives `p_max/p0 = 1.583 ≈ 2^(2/3)`; with
   `E = 1/sqrt(2)` it matches theory to 0.1%. `tamaas_reference.py`
   compensates accordingly.
3. Its influence coefficients deviate from the exact Love values by
   oscillating ±2–8% at 1–3 cell separations (Gibbs-like), which bounds the
   achievable pressure-field agreement at roughly 3% L2 on a 64-grid rough
   surface. ASPHER's `compare_tamaas.py` therefore asserts L2 < 5%, and the
   observed ~3.3% difference is dominated by the Tamaas coefficients, not by
   ASPHER's hierarchical approximation.

Reproduction: `tamaas_reference.py` (runs in the `fluidpaper` conda env) and
`compare_tamaas.py` / `compare_tamaas_h2.py` (build env); see README.
