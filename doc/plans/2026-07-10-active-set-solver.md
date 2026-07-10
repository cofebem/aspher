# Active-set contact solver — C++ implementation (masked H2 + restricted PK driver)

## Context

The prototype study (`experiments/active_set_results.md`, commit f5c782c) gave
a GO: restricted Polonsky–Keer on gap-proximity candidate sets recovers the
exact solution in one verification round, with projected 5–20× wall-time and
3–8× memory gains at Ns=16384 (measured baseline 1168 s / ~24 GiB double).
Established by the prototype and binding on this design: the |q| preconditioner
**stays** (restriction does not improve conditioning); candidate sets **must**
include gap-threshold points (~30% of fine islands have no coarse precursor);
tight δ can pass verification while boundary pressures are subtly wrong, so δ
defaults generous (0.05–0.1 of the level's gap scale) and is exposed as a knob.
The model's one unmeasured constant is the masked-matvec cost — hence M1 gates
everything.

Branch: `feat/active-set` (repo convention). Spec doc
`doc/specs/2026-07-10-active-set-solver-design.md` written alongside M1.

## M1 — Masked H2 matvec + the gate measurement

**`include/h2_operator.hpp` / `src/h2_operator.cpp`:**
- `struct H2Mask { std::vector<std::uint8_t> box_occ; /* size nbox */ }` built
  by `H2Operator::build_mask(const std::vector<std::uint8_t>& grid_mask)` (or
  an index-list overload): mark leaf boxes via
  `tree_.box_id(leaf_level, ix/ls, iy/ls)`, OR-propagate to ancestors via
  `boxes()[b].parent`. O(N_c + nbox). The complete-tree layout
  (`uniform_quadtree.hpp`: `box_id`, `level_begin`) makes this a flat bitmap —
  no index lists.
- `matvec_masked_into(x, y, const H2Mask& src, const H2Mask* tgt)` (`tgt ==
  nullptr` → all targets: the verification mode). New
  `matvec_masked_impl<S>` in the header next to `matvec_impl`
  (`h2_operator.hpp:120`), sharing all cached matrices; the validated unmasked
  path is not touched. Masking per pass: P2M zeroes/skips non-`src` leaves;
  M2M combines occupied children into occupied parents only; M2L computes L
  only for `tgt`-occupied boxes and skips non-`src` sources inside the CSR
  loop (one bool load per interaction — the branchy version first; prebuilt
  masked lists only if profiling demands); L2L/L2P/near only on `tgt` boxes,
  near skipping non-`src` source leaves. Contract: **y is valid only on tgt
  leaves** (no O(N) zeroing per apply). Float variant mirrors it
  (`matvec_masked_single_into`).
- **Exactness test** (`tests/test_active.cpp`, new ctest target): for x
  supported on the mask, skipped terms are exact zeros and kept terms keep
  their summation order → masked output must equal unmasked output
  **bit-for-bit** on target leaves. Test at Ns=256 with a clustered random
  mask, double and float.
- **Gate benchmark**: real contact masks (quick solve, threshold + dilate) at
  Ns=2048/4096 (+16384, operator + 2 N-arrays ≈ 5 GiB, co-tenancy OK):
  masked/unmasked time ratio vs N_c/N. **Proceed to M2 if masked ≲ 5% of full
  at N_c/N ≈ 5e-3** (the cost model assumed 1–3%); else stop and re-model.

## M2 — Active-set PK driver (full-grid buffers; the speed win)

**`src/contact_solver.cpp` + `include/contact_solver.hpp`:**
- New `solve_contact_active_impl<Real>` — the PK algorithm (PR+, centred
  line-search denominator, overlap correction, load renormalisation, double
  accumulation of reductions, stagnation guard — mirror `solve_contact_impl`)
  with every O(N) loop driven by a candidate index list (`for j: i = idx[j]`),
  matvec through the masked functor, preconditioner through the existing
  `FourierPreconditioner::apply_into` unchanged (full-grid buffers in M2; the
  contact mask it already takes is exactly the restricted contact set). The
  validated `solve_contact_impl` is not modified (bit-for-bit default path).
- **Driver in `solve_contact_nested`** (`src/nested_solve.cpp`), behind
  `NestedParams{bool active_set=false; double active_delta=0.05; int
  active_halo=2; int active_max_rounds=5;}`:
  1. Coarse levels: unchanged full solves, but keep the next-to-finest level's
     gap field (drop its `light=true`; N/4-sized, negligible).
  2. Candidate set C = dilate_halo(prolonged coarse contact) ∪ {prolonged
     coarse gap < δ}, δ = active_delta × (level gap scale, `max−min` as in
     `solve_contact_impl`'s `g_scale`).
  3. Restricted solve on C (warm start = prolonged pressure, consumed as in
     the existing path).
  4. Verification: one full-target masked-source matvec + gap scan; extend C
     with dilated violations, resolve warm-started; after `active_max_rounds`
     fall back to the standard full solve (safety, logged in the result).
- Python: `hc.solve_nested(..., active_set=True, active_delta=...)`
  (`python/bindings.cpp`, plumb NestedParams).
- **Equivalence tests** (extend `tests/test_active.cpp`): rough seed-42
  surface (has orphan islands by construction), Ns=256: active-set nested vs
  standard nested — equal contact area, pressure rel-L2 ≤ 1e-6, clean
  certificate; both precisions; plus a δ-too-tight regression case asserting
  the fallback triggers rather than returning silently-wrong pressures.

## M3 — O(N_c) memory (the Ns=32768 enabler)

- Compressed CG state: vectors of size |C| + one index array in
  `solve_contact_active_impl`; extend `H2Mask` with an occupied-leaf → slot
  map and add gather/scatter P2M/L2P/near addressing so the restricted applies
  need **no N-sized x/y** (leaf-blocked dense slots, ~2–4·N_c).
- Verification without a full y: stream it — per target leaf compute the
  L2P+near output into an ls² buffer, scan for violations, discard (O(N)
  compute, O(ls²) memory).
- Preconditioner: stays full-grid (own scratch ≈ 2N reals) through 16384;
  at 32768 double that's ~17 GiB → 32768 runs float, or `precond=false`
  (documented trade-off; the reduced-frequency idea from arXiv:2103.10203
  remains future work if this hurts).
- Measured targets: Ns=16384 double active-set ≤ ~8 GiB solve RSS and ≥5×
  faster than the 1168 s baseline (same surface/load/seed, fresh-reboot
  protocol); stretch M4: one Ns=32768 float attempt.

## Verification (each milestone)

- `ctest` stays 6/6 green + new `test_active` target; the default
  (non-active-set) path must remain bit-for-bit: rerun the ref_solve.py
  identity check (scratchpad protocol from the memory-pass work) after every
  milestone that touches shared files.
- M1 gate: masked-matvec ratio table (Ns, N_c/N, ratio) recorded in the spec.
- M2: equivalence tests above + a Ns=1024 Python cross-check against
  `experiments/active_set_proto.py`'s reference numbers (same surface: 132/64
  global iters, area 0.00759) — the C++ driver should land within a few
  iterations of the prototype's restricted-fourier arm (56–57).
- M3: RSS A/B via `/usr/bin/time -v` at 4096 (fast) and the 16384 fresh-reboot
  case (user protocol); results folded into `data/backend_precision_study.jsonl`
  conventions if we add a benchmark case.

## Files touched (summary)

`include/h2_operator.hpp`, `src/h2_operator.cpp` (M1); `tests/test_active.cpp`
+ `CMakeLists.txt` test registration (M1/M2); `include/contact_solver.hpp`,
`src/contact_solver.cpp`, `include/nested_solve.hpp`, `src/nested_solve.cpp`,
`python/bindings.cpp` (M2); same files again for M3 compression;
`doc/specs/2026-07-10-active-set-solver-design.md`, CLAUDE.md updates at the
end. Existing pieces reused: `UniformQuadTree` box addressing, cached
couplings/stencils/R matrices, `FourierPreconditioner::apply_into`, the nested
cascade's restriction/prolongation and warm-start plumbing.
