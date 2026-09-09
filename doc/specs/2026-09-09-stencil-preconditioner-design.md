# Sparse stencil preconditioner — design

Status: agreed in discussion 2026-09-09, not yet implemented.
Supersedes the A13 "sparse preconditioner + coarse space" sketch in
`doc/specs/2026-09-08-accuracy-efficiency-improvements.md`: the coarse-space
half of that idea is **measured dead** (§3.2) and the sparse half turns out to
need no coarse space at all.

Goal: replace the full-grid FFT application of the `|k|` preconditioner with a
13-tap real-space stencil evaluated only on the contact set, turning the last
O(N log N) component of an otherwise O(N_c) solve into O(N_c) — measured
4–5× total speed-up at Ns=16384, and −8 GiB of resident scratch at Ns=32768.

---

## 1. Why

After `active_all_levels` (A11) removed the coarse cascade, the preconditioner
is what the large-grid solve spends its time on. At Ns=16384, `active_all`,
rough-H0.8 at p̄=0.002 (ledger `data/bench_ledger.jsonl`,
`data/bench_ledger_memfix.jsonl`):

| variant | wall | matvec | **precond** | coarse | per apply |
|---|---|---|---|---|---|
| `h2-f32-active-all` | 65.1 s | 4.7 s (7.3%) | **50.2 s (77.5%)** | 5.2 s (8.0%) | 1.07 s |
| `h2-f64-active-all` | 175.4 s | 18.2 s (10.4%) | **141.1 s (80.6%)** | 10.3 s (5.9%) | 1.96 s |

The contact fraction on that run is 4.4×10⁻⁴, so **N_c ≈ 1.2×10⁵ against
N = 2.7×10⁸**. Every apply zeroes an Ns² grid, runs a forward r2c transform,
multiplies a full half-spectrum, runs an inverse c2r transform and gathers —
four full-grid passes and two FFTs — to precondition a residual with 10⁵
nonzero entries. It does roughly 2000× more work than the vector it acts on.

The active-set work made the matvec, the CG state and the verification all
O(N_c). The preconditioner is the one piece that was left behind.

---

## 2. Mathematical core

The Boussinesq operator has symbol `Ŝ(k) ∝ 1/|k|`, so the preconditioner uses
the inverse symbol `m(k) = |k|` (overall scale irrelevant — CG is invariant
under `M⁻¹ → cM⁻¹`, which A02 made true in implementation as well as in
theory). Written out, the current apply is

```
z = P_C · F⁻¹ diag(|k|) F · P_C g ,      P_C = restriction to the contact set
```

i.e. a circular convolution of the masked residual with the real-space kernel
`w = F⁻¹|k|`, sampled on `C`.

**`|k|` is a positive-order symbol, so `w` is short-ranged.** This is the whole
design. A symbol that *grows* with `|k|` has a kernel concentrated at the
origin; 74.6% of `Σ|w|` lies within radius 1 and 83.9% within radius 2. Truncate
`w` to a disc of radius `R`:

```
w_R(d) = w(d) if |d| ≤ R else 0 ,        z_i = Σ_{|d| ≤ R} w_R(d) · g_{i+d}
```

Two properties make this more than an approximation of convenience:

* **On `C` the stencil form is *exact*, not approximate.** `g` is zero off the
  contact set, so the full-grid convolution restricted to `C` and the local
  sum over neighbours in `C` are the same sum. The only approximation
  introduced anywhere is truncating `w` itself.
* **`ŵ_R` stays strictly positive**, so `M⁻¹` remains SPD and the CG theory is
  untouched. Measured min/max over non-DC modes: 0.160 (R=1), 0.137 (R=2),
  0.065 (R=4), 0.031 (R=8) — against 6.9×10⁻⁴ for the untruncated `|k|` at
  Ns=2048, i.e. the truncated symbol is *flatter*, not more singular.

### 2.1 The weights are universal

`w(d)/w(0,0)` converges to grid-independent constants, identical to six
significant figures from Ns=512 upward (computed at Ns = 128…4096):

| offset | (1,0) | (1,1) | (2,0) | (2,1) | (2,2) | (3,0) | (4,0) |
|---|---|---|---|---|---|---|---|
| `w/w(0,0)` | −0.180875 | −0.032860 | +0.021189 | −0.003912 | −0.003198 | −0.015473 | +0.006277 |

with `w(0,0)/Ns → 0.382598`. The `Ns` factor in the overall scale is
irrelevant by the scale-invariance above. The **R=2 disc is 13 points**
(`dx²+dy² ≤ 4`): the centre, the four edge neighbours, the four diagonals and
the four two-cell axial neighbours — weights `1`, `−0.180875`, `−0.032860`,
`+0.021189`. (R=1 is the 5-point cross, not a 3×3 block.)

---

## 3. Measured evidence

All measurements 2026-09-09, revision `77c4296`, pocketfft, 20 threads,
rough-H0.8, `coarsest=64`, `tol=1e-8`. Raw records:
`experiments/precond_probe_20260909.md` (all four probes) and
`data/precond_probe.jsonl` (per-run JSON for §3.1). Single reps on a shared
desktop, so ratios are more reliable than absolutes.

### 3.1 The current preconditioner is a net loss at large Ns and low occupancy

Nested active-set path (`active_set=True`, `q=4`, `leaf_side=16`), p̄ = 0.002,
wall seconds (iterations):

| Ns | f32 on | f32 off | f64 on | f64 off |
|---|---|---|---|---|
| 1024 | **0.33** (11) | 0.35 (23) | **0.51** (15) | 0.67 (31) |
| 2048 | 0.68 (14) | **0.58** (34) | 1.13 (19) | **0.95** (43) |
| 4096 | 2.19 (23) | **1.76** (80) | 3.87 (27) | **3.17** (92) |
| 8192 | 11.26 (39) | **7.83** (152) | 22.96 (43) | **21.21** (166) |
| 16384 | 68.26 (47) | **35.03** (213) | 183.99 (72) | **78.80** (233) |

Contact area identical and `status == converged` in every row. The governing
quantity is `T_precond_per_apply / T_matvec_per_iter`; turning the
preconditioner off wins exactly when that ratio exceeds `m − 1`, where `m` is
the iteration multiplier (measured 1.8–4.6, growing with Ns). Occupancy sweep
at Ns=2048, float, same path, which locates the crossover at ≈0.4% contact:

| p̄ | contact | on | off | ratio |
|---|---|---|---|---|
| 0.0005 | 0.025% | 0.45 (12) | **0.24** (22) | 0.53 |
| 0.002 | 0.092% | 0.68 (14) | **0.58** (34) | 0.85 |
| 0.01 | 0.41% | **1.20** (27) | 1.26 (48) | 1.05 |
| 0.05 | 1.97% | **5.07** (80) | 7.08 (148) | 1.40 |
| 0.15 | 5.63% | **30.06** (241) | 35.63 (324) | 1.19 |

### 3.2 A coarse-grid preconditioner is dead — recorded no-go

Band-limiting the symbol to `|k| ≤ k_max/m` emulates the spectral content a
coarse-grid correction could carry. Ns=1024, single level, p̄=0.002:

| band limit | full | ≤k_max/2 | ≤k_max/4 | ≤k_max/8 | ≤k_max/16 |
|---|---|---|---|---|---|
| iterations | 27 | **3985** | 2261 | 2222 | 3437 |
| status | converged | stagnated | stagnated | stagnated | stagnated |

(`precond="none"` is 34 iterations throughout — the band-limited preconditioner
is *far worse than no preconditioner at all*.) Coarsening removes precisely the
part of the operator that does the work: `M⁻¹` acts most strongly at high `k`,
so truncating the top of the spectrum annihilates the subspace it exists to
correct. **A13's coarse space should be struck from the roadmap.**

### 3.3 Real-space truncation costs nothing

Ns=1024, **single level** (`ContactSolver(backend="h2", q=6)`, not the
cascade — so the counts are not comparable with §3.1), p̄=0.002; mass is the
fraction of `Σ|w|` retained:

| | full | R=1 | R=2 | R=4 | R=8 | R=16 | R=64 |
|---|---|---|---|---|---|---|---|
| mass | 100% | 74.6% | 83.9% | 91.1% | 95.2% | 97.5% | 99.4% |
| iterations | 27 | 27 | **25** | 26 | 26 | 27 | 27 |

Across grid size and occupancy, all single level:

| case | contact | full | R=1 | R=2 | R=4 |
|---|---|---|---|---|---|
| Ns=1024, p̄=0.002 | 0.11% | 27 | 27 | **25** | 26 |
| Ns=2048, p̄=0.002 | 0.09% | 33 | 36 | **33** | 34 |
| Ns=1024, p̄=0.05 | 2.45% | 53 | 60 | 60 | 56 |
| Ns=512, p̄=0.15 | 8.70% | 78 | 83 | 83 | **77** |
| Ns=512, p̄=0.5 | 24.4% | 221 | **196** | 198 | 206 |

R=2 matches or beats the full FFT in every regime tested; R=1 costs +9% and
+13% iterations in two of five. Since 13 taps on 10⁵ points is sub-millisecond
against a 105 ms matvec, there is no trade to make: **R=2 is the default.**

---

## 4. What is built

### 4.1 Structure

A new `StencilPreconditioner` (`include/stencil_precond.hpp`,
`src/stencil_precond.cpp`) mirroring `FourierPreconditioner`'s interface
exactly:

```cpp
class StencilPreconditioner {
public:
    explicit StencilPreconditioner(int Ns, int radius = 2);
    void apply_into(const VectorXd&, const std::vector<uint8_t>&, VectorXd&) const;
    void apply_single_into(const VectorXf&, const std::vector<uint8_t>&, VectorXf&) const;
    void apply_into_indexed(const VectorXd& gc, const std::vector<uint8_t>& contact_c,
                            const std::vector<int>& grid_index, VectorXd& zc) const;
    void apply_single_into_indexed(...) const;
    void release_scratch() const;          // no-op: there is no grid scratch
};
```

`FourierPreconditioner` is **kept untouched** as the equivalence reference and
as an escape hatch. Dispatch happens at the call sites in `nested_solve.cpp`
and `python/bindings.cpp`, which already pass `PrecondIntoT<Real>` functors —
`solve_contact_impl` does not change at all.

Selection. The two entry points spell the option differently today and keep
doing so: `hc.solve_nested(precond=True)` takes a bool, and
`ContactSolver.solve(precond="fourier"|"none")` takes a string. The stencil
becomes what both existing spellings mean, and the string form gains
`"fourier-fft"` for the old implementation:

| call | before | after |
|---|---|---|
| `solve_nested(precond=True)` | FFT | **stencil** |
| `solve(precond="fourier")` | FFT | **stencil** |
| `solve(precond="fourier-fft")` | — | FFT (reference / escape hatch) |
| `solve(precond="none")` | none | unchanged |

`solve_nested` gains `precond_engine="stencil"|"fft"` for the same choice, and
both gain `precond_radius` (default 2), plumbed through `NestedParams`.

Naming note: the option keeps the name `"fourier"` because it is the same
operator — the spectral symbol is where the weights come from. The class name
does not, because the implementation is no longer a transform.

### 4.2 The weights

Computed once at construction by one FFT on a grid of side `min(Ns, 512)`
(a few ms, ~2 MB) and truncated to the disc: no transcribed magic numbers, any
`R` supported, and exact for the coarse levels where `Ns < 512`. The six
universal constants of §2.1 become the **test oracle**, not the source.

Storage: `(2R+1)²` doubles plus an offset list — 169 doubles at R=2.

### 4.3 The three apply paths

**Full grid, masked** (`apply_into`, standard solve). Loop the contact set;
for each contact point sum the taps over neighbours that are themselves in
contact, with periodic wrap; write zero elsewhere in `z`. Cost O(13·N_c) plus
the O(N) zero-fill of `z` that the current code already does.

**Compressed** (`apply_into_indexed`, active-set path) — §4.4.

**Float.** Same code templated on `Real`. The 13-tap per-point sum accumulates
in `double` even when `Real=float` (13 terms; free at this cost) and the
contact-mean reduction stays `double`, as now.

The contact-mean removal is unchanged: it is part of the operator's definition
(the load constraint fixes the mean), and it is already O(N_c).

### 4.4 The active-set path: leaf-tile halo gather

The compressed state is slot-blocked by leaf (`H2Mask`, side `ls` = 8 or 16),
and `R = 2 ≪ ls`, so a leaf's stencil reaches only its eight immediate
neighbour leaves — the same neighbourhood the H² near field already indexes.
Per occupied leaf:

1. gather an `(ls+2R)²` tile — 20×20 at `ls=16`, 400 doubles on the stack:
   interior from the leaf's own slots, halo from neighbour leaves' slots where
   occupied, **zero elsewhere**;
2. each gathered value is `g_c[slot]` if `contact_c[slot]`, else zero;
3. apply the 13 taps over the `ls²` interior, writing `zc` at the leaf's slots;
4. periodic wrap at the domain boundary.

Then the contact-mean reduction over the compressed set, in double, as now.

No hash map, no grid→slot table, O(1) extra memory beyond the mask, parallel
over leaves, cache-resident tiles. **Exactness**: on the restricted path
`contact ⊆ C` by construction, so a neighbour outside the candidate set is
provably a zero of the residual — a skipped term, not a dropped one. This is
the same argument that makes the masked matvec bit-for-bit exact.

Precondition to assert: `R ≤ ls`. Rejected alternative: a per-round neighbour
index table (`13·N_c` int32 ≈ 6 MB at Ns=16384) — simpler to write but needs a
grid→slot map to build and is less cache-friendly.

### 4.5 The safety valve

Retained from the "just switch it off" option, which §3.1 shows is worth 2×
against the *FFT* preconditioner. After the first iteration of a level,
compare the measured apply cost against the measured matvec cost; if
`t_precond > gate · t_matvec` (default `gate = 1.5`), drop the preconditioner
for that level and restart the CG direction — machinery the solver already has
for support changes. Reported in `stage_stats`.

Threshold from §3.1: off wins when the ratio exceeds `m − 1`, `m ∈ [1.8, 4.6]`;
1.5 catches the Ns=2048 case (ratio 1.81) without firing at parity (0.71).

With the stencil this **never fires** — the expected ratio is ~10⁻³. It exists
to protect the `"fourier-fft"` path and any future regime where a
preconditioner stops paying for itself. That it is inert on the default path is
the point, not a defect.

---

## 5. Cost and memory

Per apply at Ns=16384, N_c ≈ 1.2×10⁵:

| | FFT | stencil |
|---|---|---|
| full-grid passes | 4 (zero, symbol, 2 transform I/O) | 0 |
| FFTs | 2 (Ns²) | 0 |
| arithmetic | ~4×10¹⁰ bytes of traffic | 13·N_c ≈ 1.6×10⁶ FMA |
| measured / projected | 1.07 s (f32), 1.96 s (f64) | ~1 ms |

Projected totals at Ns=16384, from the measured per-phase breakdown of the
current runs (matvec and coarse unchanged, preconditioner → ~0):

| | today | precond off | **stencil** |
|---|---|---|---|
| f32 | 68.3 s | 35.0 s | **≈16 s** |
| f64 | 184.0 s | 78.8 s | **≈34 s** |

i.e. **4–5× against today** and ~2× against simply disabling it.

Memory: the stencil has no grid scratch, so `(N + 2(Ns/2+1)Ns)·sizeof(Real)`
disappears — **2.0 GiB at Ns=16384 f32, 4.0 GiB f64, 8.0 GiB at Ns=32768 f32**.
That is the largest single solver-side item in the Ns=32768 budget, which the
preflight puts at 41.2 GiB with a float32 surface; removing it leaves ≈33 GiB.
Honest framing: this is the largest single step toward Ns=32768 but not
sufficient on its own — the gap, the prolonged warm start and the returned
pressure are 8 GiB each and are a separate question.

---

## 6. Implementation plan

1. **`StencilPreconditioner`** with weight construction (§4.2) and the
   full-grid masked apply (§4.3), double and float. Gate: agreement with
   `FourierPreconditioner` fed the same truncated symbol.
2. **Compressed apply** via the leaf-tile halo gather (§4.4). Gate:
   bit-for-bit equality with the full-grid apply on the same data, both
   precisions — the `test_active` masked-matvec gate is the template.
3. **Plumbing**: `precond` selector and `precond_radius` through
   `NestedParams`, `solve_contact_nested`, `ContactSolver` and the bindings;
   default flipped to the stencil.
4. **Safety valve** (§4.5) with its `stage_stats` reporting.
5. **Benchmarks**: paired A/B stencil vs `fourier-fft` vs `none` at
   Ns ∈ {1024, 4096, 16384} × {f32, f64} × occupancy {0.1%, 1%, 10%}, wall and
   peak RSS, through `bench/harness.py` as a new variant set.
6. **Docs**: CLAUDE.md PCG section and validated-numbers table;
   `doc/theory/pcg.tex` §"Spectral Preconditioning" gains the truncation
   analysis; record §3.2 as a no-go against A13.

---

## 7. Validation gates

| id | check | criterion |
|---|---|---|
| S1 | weights vs the §2.1 oracle table | 6 significant figures, Ns ≥ 512 |
| S2 | `ŵ_R` positivity, R ∈ {1,2,4}, several Ns | min over non-DC modes > 0 |
| S3 | full-grid apply ≡ compressed apply | **bit-for-bit**, double and float |
| S4 | solve equivalence vs `fourier-fft` | contact area identical; pressure rel-L2 ≤ 1e-6 (f64) |
| S5 | iteration count vs `fourier-fft` | within +20% on the regression suite, documented per case (worst measured in §3.3 is +13%, Ns=1024 at 2.45% contact) |
| S6 | existing suite | `ctest` green, including `test_precond`, `test_active`, `test_certification` |

S5 is a band rather than an equality because the stencil is a *different*
(valid) preconditioner: §3.3 shows it sometimes converges faster.

---

## 8. Risks and open questions

- **Boundary wrap.** The preconditioner is periodic; the stencil must wrap, and
  a contact patch touching the domain edge is the natural bug. S3 covers it
  only if the test data includes edge-touching contact — make sure it does.
- **`R ≤ ls`.** Asserted, not assumed. `h2_leaf_side` is user-settable and can
  be 8; R=2 is safe, a future R=8 would not be.
- **High occupancy is unmeasured above 24%.** §3.3 stops at 24.4% contact.
  Near-full contact is where the low-`k` behaviour of `ŵ_R` differs most from
  `|k|`. The B04 sweep infrastructure covers this; add 50% and 99% points.
- **Coarse levels** use the same preconditioner and the same change; their
  `Ns < 512` weights are computed exactly rather than taken from the limit.
- **Out-of-range index, pre-existing.** The band-limit probe of §3.2 crashed
  the nested/active path with an Eigen index assertion on a
  `Matrix<float,-1,1>`, rather than merely converging badly. The trigger was a
  deliberately degenerate preconditioner and it may be unreachable in practice,
  but an out-of-range index is never benign. **Trace it during step 2**; if it
  is a real bounds bug it is a defect independent of this work.
- **Friction.** `TangentialFourierPreconditioner` has the same positive-order
  structure (`|k|(I + γ k̂k̂ᵀ)`) and should get the same treatment, with a 2×2
  matrix-valued stencil. Explicitly **out of scope** here; note as a follow-on
  once the scalar version is validated.
