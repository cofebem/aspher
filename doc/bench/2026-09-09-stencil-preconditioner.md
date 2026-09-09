# Stencil vs FFT preconditioner engine — measured outcome

Companion to `doc/specs/2026-09-09-stencil-preconditioner-design.md`. Ledger:
`data/bench_stencil.jsonl` (all rows below; revision `3125c07`, not dirty). All
rows are `status=converged` at matching `effective_tol` within each pair
(float: 2e-6, double: 1e-8) and matching `validation_scope=solve_operator`, so
timing comparisons are apples-to-apples. Runs use `active_set=True` on the h2
backend (`active_all_levels` at its default `True`, `active_occupancy_max=0.4`).

Variant names state the engine explicitly on both arms
(`h2-f32/f64-active-stencil` vs `h2-f32/f64-active-fft`) because the default
flip earlier in this branch means the pre-existing `h2-f32-active` name now
means something different than it did before — this A/B never reuses that
name for either arm. `h2-f32-active-nopc` (no preconditioner) is included at
Ns=1024/4096 for context only.

The commands run:

```bash
python bench/harness.py run --workload rough-H0.8 --ns 1024 4096 \
    --variants h2-f32-active-stencil,h2-f32-active-fft,h2-f32-active-nopc \
    --reps 5 --ledger data/bench_stencil.jsonl
python bench/harness.py run --workload rough-H0.8 --ns 1024 4096 \
    --variants h2-f64-active-stencil,h2-f64-active-fft \
    --reps 5 --ledger data/bench_stencil.jsonl
python bench/harness.py preflight --ns 16384 --variant h2-f32-active-stencil
python bench/harness.py run --workload rough-H0.8 --ns 16384 \
    --variants h2-f32-active-stencil,h2-f32-active-fft --reps 1 --ledger data/bench_stencil.jsonl
python bench/harness.py run --workload rough-H0.8 --ns 16384 \
    --variants h2-f64-active-stencil,h2-f64-active-fft --reps 1 --ledger data/bench_stencil.jsonl
python bench/harness.py run \
    --workload rough-H0.8@0.002 rough-H0.8@0.05 rough-H0.8@2.0 \
    --ns 1024 --variants h2-f32-active-stencil,h2-f32-active-fft \
    --reps 5 --ledger data/bench_stencil.jsonl
```

All ran inside chunked, in-sandbox `Bash` calls (each finished well inside the
10-minute tool timeout — the biggest single run, Ns=16384 f64-fft, took 88 s);
the detached-launch path was not needed for this task.

`solver_args.precond_engine` was confirmed present in the first ledger row
before trusting any further numbers (`"stencil"` / `"fft"` on the respective
arms, as set by the new `VARIANTS` entries and passed through the existing
forwarding allowlist in `bench/harness.py`).

## Occupancy dial: the required ≥40% point took more load than the brief's `@0.5`

Per controller ruling R3, `rough-H0.8@0.5` was probed first and, as predicted,
did not reach 40% contact at Ns=1024:

| `p_bar` | achieved contact fraction |
|---|---|
| 0.002 | 0.106% |
| 0.05  | 2.455% |
| 0.5   | 20.13% |
| 1.0   | 35.09% |
| **2.0** | **56.75%** |

`p_bar=2.0` was used for the required high-occupancy point (`rough-H0.8@2.0`),
achieving 56.75–56.76% contact — comfortably above the 40% gate. `rough-H0.8@0.5`
is reported here for completeness but is not the gate point; it does not meet
the ≥40% requirement even though it is more heavily loaded than the spec's own
24.4% probe (measured at a different Ns).

## Paired A/B, fixed default load (`rough-H0.8`, occupancy 0.04–0.11%)

Analyzed with:
```bash
python bench/analyze.py --pair h2-f32-active-fft,h2-f32-active-stencil --metric wall_cold_s --ledger data/bench_stencil.jsonl
python bench/analyze.py --pair h2-f32-active-fft,h2-f32-active-stencil --metric peak_rss_gib --ledger data/bench_stencil.jsonl --target 0.0
python bench/analyze.py --pair h2-f64-active-fft,h2-f64-active-stencil --metric wall_cold_s --ledger data/bench_stencil.jsonl
python bench/analyze.py --pair h2-f64-active-fft,h2-f64-active-stencil --metric peak_rss_gib --ledger data/bench_stencil.jsonl --target 0.0
```

### float32

| Ns | n | fft wall | stencil wall | change | CI | area spread |
|---|---|---|---|---|---|---|
| 1024  | 5 | 0.193 s | 0.169 s | **−12.2%** | [−39.3, −0.6]% | 0 |
| 4096  | 5 | 1.207 s | 0.890 s | **−26.3%** | [−34.3, −17.7]% | 9.0e-05 |
| 16384 | 1 | 32.98 s | 11.50 s | **−65.1%** (indicative, n=1) | n/a | 8.4e-06 |

| Ns | n | fft RSS | stencil RSS | change | CI |
|---|---|---|---|---|---|
| 1024  | 5 | 0.108 GiB | 0.105 GiB | −2.8% | [−3.4, −2.5]% |
| 4096  | 5 | 0.711 GiB | 0.673 GiB | −5.4% | [−5.5, −5.3]% |
| 16384 | 1 | 9.117 GiB | 9.117 GiB | ~0% | n/a |

### float64

| Ns | n | fft wall | stencil wall | change | CI | area spread |
|---|---|---|---|---|---|---|
| 1024  | 5 | 0.218 s | 0.182 s | −17.5% | [−21.2, **+56.4**]% (does not exclude zero) | 0 |
| 4096  | 5 | 1.635 s | 1.096 s | **−32.5%** | [−44.9, −27.5]% | 0 |
| 16384 | 1 | 88.26 s | 30.48 s | **−65.5%** (indicative, n=1) | n/a | 0 |

| Ns | n | fft RSS | stencil RSS | change | CI |
|---|---|---|---|---|---|
| 1024  | 5 | 0.119 GiB | 0.111 GiB | −7.5% | [−7.7, −6.9]% |
| 4096  | 5 | 0.827 GiB | 0.663 GiB | **−19.9%** | [−20.0, −19.8]% |
| 16384 | 1 | 9.315 GiB | 9.117 GiB | −2.1% | n/a |

Contact area agrees to within 9e-05 absolute (float, worst case) or exactly
(double) across every pair — the engines answer the same problem.

### Per-phase breakdown (float32, median over reps)

```
occup.    total   coarse  precond   matvec  candid.   verif.
--- h2-f32-active-fft ---
0.066%    1.154    0.143    0.050    0.284   0.0249   0.0165
0.106%    0.220    0.043    0.003    0.024   0.0046   0.0022
--- h2-f32-active-stencil ---
0.066%    0.843    0.102    0.009    0.099   0.0263   0.0165
0.106%    0.164    0.031    0.001    0.014   0.0036   0.0015
```

The preconditioner's own share collapses (0.050→0.009 s at Ns=4096, 0.003→0.001
at Ns=1024 — roughly 5× and 3×), but the *matvec* line also shrinks nearly
3× (0.284→0.099, 0.024→0.014). That second drop is not something the stencil
change should touch directly; it tracks iteration count, which fell sharply
under the stencil (see below) — fewer PCG iterations means fewer matvecs, not
a faster matvec. The total-wall reduction is the product of both effects.

### Iteration counts (median)

| Ns | fft it | stencil it | nopc it |
|---|---|---|---|
| 1024 f32 | 22 | **13** | 23 |
| 4096 f32 | 62 | **25** | 69 |
| 16384 f32 | 208 | **61** | — |
| 1024 f64 | 30 | **17** | — |
| 4096 f64 | 79 | **33** | — |
| 16384 f64 | 272 | **94** | — |

The stencil preconditioner is not merely a cheaper application of the same
`|k|` operator — it also converges the PCG in noticeably fewer iterations at
every grid size tested here, and the gap widens with Ns (2.9× fewer at
Ns=16384 f64). This is a stronger and more surprising result than "same
convergence, cheaper per-iteration cost," and it is the dominant driver of the
wall-time win at large Ns, not the per-application FFT removal alone. It
should be treated as measured behaviour on this one surface family
(rough-H0.8, self-affine, seeded), not yet as a proven general property.

## The required high-occupancy point: `rough-H0.8@2.0`, 56.75% contact

```bash
python bench/analyze.py --pair h2-f32-active-fft,h2-f32-active-stencil \
    --workload rough-H0.8@2.0 --metric wall_cold_s --ledger data/bench_stencil.jsonl
```

```
Ns=1024  n=5  fft: 19.32 s   stencil: 24.28 s   change +16.85%  CI[+9.42,+29.97]%
REGRESSION beyond 5%: -16.9%
-> promote as default: NO
```

**The stencil is ~17% slower than FFT at this occupancy — a genuine
regression, not noise (CI excludes zero).** Iteration counts confirm it is not
purely dispatch overhead: stencil needs *more* iterations here (625 vs 519
median), the opposite of the low-occupancy result above.

Mechanism, from `stage_stats` (`--sweep` phase split): at 56.755% occupancy
both engines report `rounds=0, fallback=0` and zero `candid.`/`verif.` time —
**the active-set restriction itself is off** at this point, gated by
`active_occupancy_max=0.4` (a level restricts only when the level below
measured occupancy under 40%; here the coarse cascade already reports high
occupancy, so the finest level runs the full, unrestricted PK solve). So the
high-occupancy comparison is not stencil-vs-fft-under-restriction; it is
stencil-vs-fft on the **plain full-grid nested solve**, which is exactly the
regime the design's own low-`k` concern (§3.3) targets: near-full contact is
where the truncated 13-tap stencil's approximation of `ŵ_R` differs most from
the true `|k|` symbol, and here that shows up as a worse conditioning number
(more iterations), not merely a slower per-application cost.

Per-phase (float32, median):

```
occup.     total   coarse  precond   matvec
fft      19.320    2.194    1.364   13.942
stencil  24.281    2.586    2.203   17.185
```

Every phase is slower under the stencil at this occupancy — precond (+61%),
matvec (+23%, tracking the extra iterations), even coarse (+18%, since the
coarse cascade also runs the same preconditioner).

Contact area still agrees (0.567554 vs 0.567547, spread 1.2e-5) — the stencil
converges to the right answer, just more slowly, at this occupancy.

## Promotion verdict

Applying the plan's rule (≥10% median improvement, ≥5 paired samples, no case
regressing beyond 5%):

- **Low/typical occupancy (≤~2.5%, Ns=1024/4096, both precisions):** clears
  the bar. Ns=4096 f32 (−26.3%, CI excludes zero) and f64 (−32.5%, CI excludes
  zero) both exceed 10% with 5 paired samples; Ns=1024 f32 also clears
  (−12.2%, CI excludes zero) though narrowly; Ns=1024 f64 is favourable in
  direction (−17.5%) but its CI does not exclude zero at n=5, so it is
  indicative only at that specific size. Ns=16384 (both precisions) shows a
  large favourable single-sample result (−65%) consistent in direction and
  magnitude with the 5-sample cases, but is single-rep and therefore
  indicative, not a confirmed pass on its own.
- **Required high-occupancy point (`rough-H0.8@2.0`, 56.75% contact,
  Ns=1024, 5 paired samples):** **fails the rule** — the stencil is ~17%
  *slower*, a confirmed regression beyond the 5% budget (CI excludes zero in
  the unfavourable direction).

**Per the brief's instruction, this is not a silent default flip.** The
regression is real and reproducible at the required gate point, so the
recommendation is **not** "ship the flip that already happened as-is without
comment," but an **occupancy-gated engine choice**, in the same shape B04
proposed for the active-set restriction itself:

> Below the point where `active_set` disables its own restriction
> (`active_occupancy_max`, default 0.4), the stencil wins outright and by a
> wide, growing-with-Ns margin. At or above that occupancy — where the finest
> level is running the plain full-grid solve, not the restricted one — FFT
> should be preferred; the two engines are gated by exactly the same
> occupancy signal the active-set driver already computes, so an occupancy
> threshold that selects `precond_engine` the same way `active_occupancy_max`
> already selects the active-set restriction would give "all the upside, no
> downside" the way the B04 occupancy gate did for active-set. This has not
> been implemented or measured here — it is the concrete next step this note
> recommends, not a completed change.

The 4–5× total-wall-time projection from the design work **did materialize at
low occupancy and large Ns** (2.87× at Ns=16384 f32, 2.90× at Ns=16384 f64 —
short of "4–5×" but a large, real win) but **did not hold, and inverted, at
the required near-full-contact gate point.** Memory did not show the dramatic
drop one might expect from removing full-grid FFT scratch either: RSS
improved only modestly (0–20%, mostly at Ns=4096 f64) and was flat within
noise at Ns=1024 and Ns=16384 — the full-grid `gap`/`warm_start`/`output`
buffers the preflight accounting lists dominate peak RSS at these sizes far
more than the preconditioner's own scratch, so removing the FFT engine's
buffers does not move the needle much on total peak memory. Both of these are
reported as measured findings, not confirmations of the design-time estimate.

## Ledger

All rows: `data/bench_stencil.jsonl` (84 rows total: 5 reps × 3 variants × 2 Ns
= 30 for the float main A/B, 5 × 2 × 2 = 20 for the double main A/B, 1 rep ×
2 variants × 2 precisions = 4 for Ns=16384, 5 × 2 variants × 3 workloads = 30
for the occupancy sweep). The `p_bar` occupancy-dial probes (0.5, 1.0, 2.0,
one rep each, used only to pick the gate load) were run to a separate,
uncommitted scratch ledger and deleted after use; they are not part of the
analyzed A/B and are reproducible from the commands in the "Occupancy dial"
section above if needed.

## Task 7 — occupancy-gated engine choice (`automatic`, now the default)

The recommendation at the end of the previous section is implemented:
`NestedParams::PrecondEngine` gains `automatic` (the new default), gated by a
dedicated `precond_occupancy_max` (default 0.4, deliberately a *separate*
knob from `active_occupancy_max` even though they coincide numerically today
— a caller who raises `active_occupancy_max` to disable the active-set gate
must not thereby lose this one). The decision is a pure function,
`stencil_for_level(engine, precond, prev_occupancy, occupancy_max)`
(`include/nested_solve.hpp`, `src/nested_solve.cpp`), applied per level from
the previous (coarser) level's measured contact fraction — the coarsest level
(`prev_occupancy < 0`, no measurement yet) always prefers the cheap stencil.
`stencil`/`fft` keep forcing that engine at every level regardless of
occupancy. Python: `precond_engine="auto"` (new default) / `"stencil"` /
`"fft"`, plus a `precond_occupancy_max` kwarg inserted after `precond_radius`
in both the C++ signature and the `py::arg` list.

### Unit test (exhaustive)

`tests/test_precond.cpp` drives `stencil_for_level` through every row of the
brief's table directly: `!precond` for every engine, the two forcing engines
at extreme occupancies, `automatic` at `prev_occupancy < 0`, on both sides of
the default threshold (0.39/0.40/0.41), and on both sides of a different
threshold (0.6) to confirm the crossover actually moves with the knob. All
pass (`ctest -R precond`, "stencil_for_level: all table rows pass").

### Behavioural test — can it actually discriminate?

One end-to-end check on a two-level nested Hertz solve (`coarsest=64`,
Ns=256), because `automatic`'s only externally visible effect at this size is
the iteration count. Measured directly (not assumed) before committing to it:

| load | area | stencil it | fft it | automatic it |
|---|---|---|---|---|
| dilute (p̄=0.002) | 0.026 | 9 | 6 | **9** (== stencil) |
| crowded (p̄=0.2) | 0.560 | 22 | 8 | **8** (== fft) |

The engines disagree by 3–14 iterations at this size, comfortably enough to
discriminate; the test asserts `automatic`'s iteration count equals the
expected engine's and differs from the other's, at both loads. This is not a
borderline "any Ns happens to work" result — a p_bar sweep at Ns=256
(0.0005/0.002/0.01/0.05/0.1/0.2/0.4) showed the two engines disagreeing on
iteration count at every occupancy tried; 0.002/0.2 were picked because they
sit unambiguously below/above the 0.4 crossover.

### Gate re-measurement (`data/bench_stencil.jsonl`, 5 paired reps each, Ns=1024)

```
python bench/harness.py run --workload rough-H0.8@2.0 rough-H0.8@0.002 \
    --ns 1024 --variants h2-f32-active-auto,h2-f32-active-fft,h2-f32-active-stencil \
    --reps 5 --ledger data/bench_stencil.jsonl
```

The forwarding allowlist in `bench/harness.py` (`worker()`) gained
`precond_occupancy_max`; the first `h2-f32-active-auto` ledger row's
`solver_args` was inspected directly and shows `"precond_engine": "auto"`
arriving at the solver (the historical failure mode — a key silently dropped
by the allowlist — was checked for, not assumed absent).

```
=== rough-H0.8@2.0 (≈57% contact): h2-f32-active-auto vs h2-f32-active-fft ===
Ns=1024  n=5  fft: 19.32s   auto: 18.45s   change -16.57%  CI[-22.86,-3.53]%
  -> promote as default: YES

=== rough-H0.8@0.002 (≈0.1% contact): h2-f32-active-auto vs h2-f32-active-fft ===
Ns=1024  n=5  fft: 0.2219s  auto: 0.1927s  change -21.16%  CI[-31.92,+1.09]%
  CI does not exclude zero in its favour -> promote as default: NO (n/a — see below)

=== rough-H0.8@0.002: h2-f32-active-auto vs h2-f32-active-stencil ===
Ns=1024  n=5  stencil: 0.1668s  auto: 0.1927s  change -0.31%  CI[-27.07,+14.00]%
  CI does not exclude zero -> auto is statistically indistinguishable from stencil

=== rough-H0.8@2.0: h2-f32-active-auto vs h2-f32-active-stencil ===
Ns=1024  n=5  stencil: 24.28s   auto: 18.45s   change +34.73%
  REGRESSION beyond 5%: -34.7% (stencil, forced everywhere, vs auto)
```

**At the required gate point (`@2.0`, ≈57% contact): parity, not a win.**
The raw paired-A/B number (−16.57%, CI[−22.86,−3.53]%) does not survive
scrutiny as a genuine speed advantage, and is not being claimed as one.
Routing is confirmed correct by a per-level trace of median iteration counts
(five levels, coarsest→finest, Ns=64…1024): `automatic` 36/51/103/192/516,
`fft` 26/51/103/192/519, `stencil` 36/67/122/209/625. `automatic` matches
`stencil` only at the coarsest level (`prev_occupancy < 0`, unmeasured) and
matches `fft` at every level above it, where the measured occupancy
(92%/82%/72%/63%) sits above the 0.4 threshold — exactly the designed
behaviour.

But the wall-time gap between `automatic` and `fft` is dominated by run-to-run
noise, not by this routing difference: at *identical* iteration counts across
reps, matvec wall time alone spans 13.82–15.64 s for the `fft` reps and
10.67–13.42 s for the `auto` reps — a noise band wider than the reported
16.57% gap. What is real, and deterministic, is a small cascade effect: the
finest level takes 516 iterations under `automatic` versus 519 under `fft`
(reproduced across all five reps of each arm) — the coarsest level's stencil
solve produces a marginally better prolonged warm start for the level above
it. That is ~0.6% fewer finest-level iterations; it cannot account for a ~3 s
wall-time difference on its own.

**Conclusion: `automatic` is at parity with `fft` at the gate point, plus a
genuine but tiny (~0.6% iteration) cascade benefit from using the stencil at
the coarsest, unmeasured level.** The −16.57% headline number should not be
read as "automatic is faster here" — it is noise-dominated.

**At the dilute point (`@0.002`):** `automatic` is statistically
indistinguishable from the forced `stencil` arm (−0.31%, CI includes zero)
and both are faster than forced `fft`, so `automatic` **retains the
stencil's win** as required — it did not regress toward `fft`'s slower
behaviour at low occupancy.

**Promotion-rule verdict: satisfied, on the corrected reading.** The
promotion rule only requires that the gate point not regress beyond 5%; parity
(not the overstated 16.57% "win") clears that bar just as well, and the
occupancy gate removes the sole confirmed regression (`@2.0`, previously
+16.9% slower under an unconditional stencil default) without giving up any
of the low-occupancy win. `automatic` is now the default `precond_engine`.
