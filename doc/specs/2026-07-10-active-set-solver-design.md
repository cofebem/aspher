# Active-set contact solver (masked H2 + restricted PK) — design

Status: M1 implemented and gate-passed 2026-07-10; M2/M3 per
`doc/plans/2026-07-10-active-set-solver.md`.
Basis: prototype GO in `experiments/active_set_results.md` (restricted
Polonsky–Keer on gap-proximity candidate sets recovers the exact solution in
one verification round; the |q| preconditioner stays; candidate sets must
include gap-threshold points).

## 1. Masked H2 matvec (M1)

`H2Mask` (`include/h2_operator.hpp`) is a flat per-box occupancy bitmap over
the complete `UniformQuadTree` — no index lists. `H2Operator::build_mask`
(grid-mask and flat-index-list overloads) marks the leaf boxes intersecting
the marked set via `tree_.box_id(leaf_level, ix/ls, iy/ls)` and OR-propagates
to ancestors in one reverse sweep over the level-ordered box array
(O(N + nbox)); the occupied set is therefore parent-closed, which L2L relies
on.

`matvec_masked_into(x, y, src, tgt)` (`matvec_masked_impl<S>` in the header,
next to and sharing all caches with the untouched `matvec_impl`; float mirror
`matvec_masked_single_into`):

- **P2M** computes moments only for `src`-occupied leaves; skipped columns of
  the shared M scratch hold stale data that no later pass reads.
- **M2M** runs only on `src`-occupied parents, combining `src`-occupied
  children (occupancy is the OR of the children's, so an occupied parent has
  at least one).
- **M2L** computes locals only for `tgt`-occupied boxes (`tgt == nullptr` →
  all boxes: the verification mode) and skips non-`src` sources inside the
  CSR walk — one byte load per interaction (the branchy version; prebuilt
  masked lists were not needed, see the gate numbers).
- **L2L/L2P/near** run only on `tgt` boxes; near sums skip non-`src` source
  leaves. Loops use dynamic OpenMP schedules (occupied boxes are clustered;
  static chunks would idle most threads).

**Contract**: `x` must vanish outside `src`-occupied leaves; `y` is valid
only on `tgt`-occupied leaves (no O(N) zeroing per apply). For such `x` the
masked output is **bit-for-bit identical** to the unmasked matvec on the
target leaves: every skipped contribution is an exact zero (adding a zero
vector never changes IEEE bits) and kept contributions keep their summation
order. `tests/test_active.cpp` asserts exact equality at Ns=256 on a
clustered random mask (islands + isolated pixels), double and float, for
tgt=all / tgt=src / full-mask, each masked call on a fresh operator so the
shared scratch starts as garbage and a wrongly-read stale column fails
loudly.

## 2. M1 gate measurement — PASSED

`tests/bench_masked.cpp` driven by `bench_masked_gate.py`: candidate masks
from a real quick solve (seed-42 self-affine surface, H=0.8, rms=0.02,
p̄=0.002, contact set dilated by 2 — the M2 candidate-halo construction); the
Ns=16384 mask is the 4096 contact mask upsampled 4× (same island geometry,
no 24 GiB solve). Timings are best-of-reps on the 20-core workstation
(2026-07-10, load < 1; ratios are the quantity of interest).

| Ns | ls | q | cand/N | leaf occ | restricted f64 | verify f64 | restricted f32 | verify f32 |
|---|---|---|---|---|---|---|---|---|
| 2048 | 8 | 6 | 1.08e-2 | 2.1% | 1.24 ms (**1.8%**) | 4.8 ms (7.2%) | 1.18 ms (2.4%) | 4.6 ms (9.4%) |
| 2048 | 16 | 4 | 1.08e-2 | 2.9% | 3.69 ms (**2.1%**) | 7.7 ms (4.3%) | 1.59 ms (1.9%) | 3.1 ms (3.7%) |
| 4096 | 8 | 6 | 1.06e-2 | 1.8% | 4.41 ms (**1.8%**) | 23.1 ms (9.5%) | 3.72 ms (2.0%) | 19.1 ms (10.2%) |
| 4096 | 16 | 4 | 1.06e-2 | 2.3% | 12.9 ms (**1.8%**) | 22.9 ms (3.1%) | 6.10 ms (1.8%) | 11.0 ms (3.1%) |
| 16384 | 16 | 4 | 7.6e-3 | 1.3% | 112 ms (**0.93%**) | 248 ms (2.1%) | 48 ms (0.86%) | 119 ms (2.1%) |

Percentages are relative to the full unmasked matvec at the same (Ns, ls, q,
precision). Gate criterion was ≲5% at N_c/N ≈ 5e-3: measured **≤2.4% at
double that density** and ~0.9% at 16384 — the cost model's assumed 1–3%
constant is confirmed; the branchy CSR skip suffices, no prebuilt masked
lists needed. Verification-mode applies (full target grid) cost 2–10% of a
full matvec, consistent with their O(N) target-side floor; they happen once
or twice per solve, not per iteration.

At Ns=16384 f64 this projects per-iteration operator cost 12.0 s → 0.11 s
(restricted), matching the prototype cost model's 0.05–0.15 s estimate.

## 3. Active-set driver (M2) and O(N_c) memory (M3)

See the plan for the milestone specifications; this section will be updated
as they land.

Key M2 decisions carried from the prototype:
- candidate set C = dilate_halo(prolonged coarse contact) ∪ {prolonged
  coarse gap < δ}, δ = `active_delta` × level gap scale, default 0.05 and
  deliberately generous: δ too tight can pass full-grid verification while
  boundary pressures are subtly wrong (0.5–0.9% error) — the certificate can
  be clean while the active set is wrong;
- the |q| preconditioner stays, applied full-grid (restriction does not
  improve conditioning: contact islands span the domain);
- verification = one masked-source/full-target matvec + gap scan; extend C
  with dilated violations, re-solve warm-started; after `active_max_rounds`
  fall back to the standard full solve.
