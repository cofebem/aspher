# Active-set contact solver (masked H2 + restricted PK) — design

Status: M1 implemented and gate-passed 2026-07-10; M2 implemented and
cross-checked 2026-07-10; M3 per `doc/plans/2026-07-10-active-set-solver.md`.
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

## 3. Active-set driver (M2) — implemented

`solve_contact_active_impl<Real>` (`src/contact_solver.cpp`): the identical
Polonsky–Keer algorithm (PR+, centred line-search denominator, overlap
correction, load renormalisation, double-accumulated reductions, stagnation
guard) with every per-iteration O(N) loop driven by the candidate index list
(`for j: i = idx[j]`). The pressure iterate is exactly zero off the candidate
set; the load constraint stays global (mean over the full grid = p̄). The
operator is the masked matvec (src = tgt = C); its output is stale off the
candidate leaves and is never read there. `FourierPreconditioner::apply_into`
is used unchanged: it reads the gradient only on the contact mask (⊂ C) and
zeroes its output elsewhere, so stale entries never leak. The result is
always light; the driver fills displacement/gap from its verification
matvec. The validated `solve_contact_impl` is untouched (default path stays
bit-for-bit, checked by the ref_solve identity protocol).

Driver (`active_finest<Real>` in `src/nested_solve.cpp`, behind
`NestedParams{active_set, active_delta=0.05, active_halo=2,
active_max_rounds=5}`, Python-plumbed through `hc.solve_nested`):
1. coarse levels unchanged; the next-to-finest level keeps its gap field
   ((Ns/2)², sampled by injection — the prolonged fine copy is never
   materialised);
2. C = dilate_halo(prolonged coarse contact) ∪ {coarse gap < δ},
   δ = `active_delta` × fine gap scale (max−min), deliberately generous: a
   tight δ can pass verification while boundary pressures are subtly wrong
   (prototype Q2);
3. restricted solve on C, warm start = prolonged coarse pressure (consumed);
4. verification: one masked-source/full-target matvec + gap scan (violation:
   gap < −tol·scale outside C, the prototype criterion); violations are
   dilated into C and the solve resumes warm-started; after
   `active_max_rounds` uncertified rounds → full-solve fallback
   (`ContactResult.active_fallback`; `active_rounds` reports the count).

**Fallback subtlety (found by the δ-too-tight regression test)**: the PK
complementarity error Σ p|g| is blind to p=0 ∧ g<0 points, so a fallback
warm start that is converged on the old C but penetrating outside it makes
the full solve "converge" at iteration 0 with the penetration unfixed
(1.5e-2 pressure error observed). The driver therefore seeds the dilated
violating points with p̄ in the warm start — they join the active set from
iteration 0, and the full solve (exact for any p ≥ 0 start) proceeds
normally (measured: rel-L2 2.8e-6 vs reference, ΔArea 0 — normal
different-path scatter at tol 1e-8).

Verification (2026-07-10, `tests/test_active.cpp` M2 section + Python):
- Ns=256 rough multi-scale surface: active vs standard nested — ΔArea 0,
  rel-L2 1.2e-14 (f64, 1 round) / 1.7e-4 vs f64 (f32, 1 round), clean
  certificates; fallback regression (δ=0, halo=0, 1 round) triggers and
  still returns correct pressures.
- Ns=1024 cross-check against `experiments/active_set_proto.py` (seed-42
  surface, p̄=0.002, tol 1e-8): area 0.00759 == prototype; finest 51
  iterations vs the prototype's restricted-fourier 56–57 (ours warm-started,
  ~15% cut expected); 1 round, rel-L2 3e-14 vs standard nested; wall
  2.54 → 1.05 s (2.4×) already at Ns=1024.

## 4. O(N_c) memory (M3) — implemented

The active-set driver now works entirely on **compressed slot-blocked
vectors**: `H2Mask` carries an occupied-leaf → slot map, and a compressed
vector stores each occupied leaf's ls² elements contiguously
(`slot·ls² + ly·ls + lx`; ~2–4 N_c entries for contact-like masks).

- `H2Operator::matvec_masked_compressed_into` (+float): P2M and the near
  field read the slot blocks directly (an `Eigen::Map` of the block — the
  compressed layout removes the strided grid gathers), sharing the
  M2M/M2L/L2L passes with the grid-masked matvec (extracted verbatim into
  `masked_upward/coupling/downward`). Bit-for-bit identical to the grid
  masked matvec (asserted in test_active, both precisions). The M/L scratch
  stays full-size — it is O(N/ls²·q²) (~0.36 GiB at Ns=16384, q=4, ls=16)
  and the verification pass needs all target locals anyway.
- `H2Operator::matvec_masked_stream(_single)`: the verification pass —
  full-target matvec whose L2P+near output is computed per target leaf into
  an ls² tile and handed to a thread-safe sink, never materialising an
  N-sized u. Bit-for-bit identical to the grid masked tgt=all matvec.
- `FourierPreconditioner::apply_into_indexed(_single)`: scatters the
  compressed contact-masked residual into the owned full FFT grid (zeroed
  per apply; the full-grid FFT is the documented M3 trade-off), transforms,
  gathers back compressed. Matches the full-grid apply to roundoff (the
  contact-mean reduction order differs; measured 3.5e-18). Plus
  `release_scratch()`: the driver frees the ~4N-real FFT scratch after the
  last CG iteration, before the final full-grid pressure scatter.
- `solve_contact_active_impl` is unchanged algorithmically — it is
  index-driven and size-agnostic, so it runs on compressed vectors as-is;
  it only gained `N_grid` and `g_scale` parameters because the load
  constraint (P = p̄·N_grid) and the error normalisation are physical-grid
  quantities that can no longer be inferred from the (compressed) g0.
- Driver: candidate mask and violation mask stay full-grid uint8 (N bytes
  each); the warm start is remapped slot-block-wise when the candidate set
  is extended (old slots are a subset of the new ones); the fallback path
  scatters to the full grid and runs the standard solve (it is the
  memory-disaster escape hatch, not the normal route); non-light results
  re-stream the tiles into the displacement/gap arrays.

Measured (co-tenant workstation, seed-42 rough surface, p̄=0.002, tol 1e-8,
q=4, ls=16, light result, RSS sampled during the solve only):

| case | standard nested | active-set nested |
|---|---|---|
| Ns=4096 f64 wall / iters / area | 172.9 s / 91 / 0.005403 | **46.5 s / 91 / 0.005403** (3.7×, 1 round) |
| Ns=4096 f64 solve-peak RSS | 1593 MB | **841 MB** (1.9×) |

(The Ns=4096 standard numbers match the recorded bench_fft.py reference
177 s / 91 it / 0.005403.) Ns=16384 measurements below.

**Ns=16384 co-tenant A/B (2026-07-10, full-band k^-1.8 float32-generated
surface, p̄=0.002, area ≈ 0.0047 — 10× the rfgen study surface's contact, so
a HARDER case than the official baseline; solve-phase RSS, load ~15):**

| case | wall | iters | area | solve-peak RSS |
|---|---|---|---|---|
| standard f32 | 1444.9 s | 86 | 0.004704 | 18.28 GiB |
| **active f32** | **308.0 s (4.7×)** | 89 | 0.004705 | **10.94 GiB (1.67×)** |
| active f64 | 1458.2 s | 210 (tol 1e-8) | 0.004683 | 12.50 GiB |
| standard f64 | — cannot run (>24 GiB, OOM on this 31 GiB machine co-tenant) | | | |

All active runs: 1 verification round, no fallback. The f64 row is the
enabler result: double precision to tol 1e-8 at Ns=16384 in 12.5 GiB on a
surface with 10× the study contact. The plan's formal M3 targets (≤ ~8 GiB
and ≥5× vs the 1168 s baseline, rfgen surface, fresh-reboot protocol) still
need the official user-protocol run —
`bench_backend_precision_study.py --backend h2 --precision double
--ns 16384 --active-set`; on that 10×-sparser contact the candidate set and
compressed state shrink accordingly, so the projection is favourable.

**δ-scale sensitivity (measured 2026-07-10, full-band k^-1.8 surface,
p̄=0.002)**: the default `active_delta=0.05` is defined against the level gap
scale (max−min), which is ~5·rms on this surface → δ ≈ 0.25·rms. The
coarse-gap-below-δ fraction is then 6.2% of the grid (|C| ≈ 9·N_c at contact
0.67%), vs 3.4% at δ=0.10·rms and 1.8% at δ=0.03·rms — the prototype's
validated range gives |C| ≈ 3–5·N_c. The generous default is intentional
(prototype Q2: a tight δ can produce a CLEAN certificate with 0.5–0.9%
pressure error — that failure is invisible, whereas a fat candidate set only
costs memory/time and is tunable via `active_delta`); on memory-critical
runs with rich spectra, `active_delta≈0.02` (≈0.1·rms here) roughly halves
the compressed state at the prototype-validated safety margin.
