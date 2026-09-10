# Stencil vs FFT preconditioner engine — measured outcome

> **CORRECTION (2026-09-10, final whole-branch review, F1-F3).** The
> low-occupancy A/B below (default `rough-H0.8` workload, the section
> "Paired A/B, fixed default load") was run under a runtime "cost gate"
> (`SolveOptions::precond_cost_gate`, default `1.5`) that took one wall-clock
> timing sample after the first preconditioner apply and, on this restricted
> active-set path (masked matvec = 1-2% of a full matvec), fired on **every**
> `fft`-arm run — silently disabling the FFT preconditioner after iteration 1
> and leaving the remaining iterations unpreconditioned. Every `*-active-fft`
> row at the dilute default workload has `precond_count == 1` while
> `*-active-stencil` rows show `precond_count == iterations`; nothing in the
> original analysis compared the two, so the corruption went unnoticed. The
> resulting headline claim — **"the stencil is a better preconditioner, not
> merely a cheaper one"** — is retracted: it described an artefact. The gate
> now defaults OFF, `precond_dropped`/`precond_count` are checked by
> `bench/analyze.py`'s contract guard, and the low-occupancy A/B has been
> re-run with the gate genuinely disabled; see "Paired A/B" and "Why the
> first measurement was wrong" below for the corrected numbers and the
> honest finding. The high-occupancy point (`rough-H0.8@2.0`, ≈57% contact)
> and the `automatic`-engine gate re-measurement further down were **not**
> affected — at that occupancy the active-set restriction is off, the matvec
> is the full O(N) H2 apply, and `precond_count == iterations` holds in the
> original rows too (verified directly); those sections are unchanged.

Companion to `doc/specs/2026-09-09-stencil-preconditioner-design.md`. Ledger:
`data/bench_stencil.jsonl` (original rows: revision `3125c07`; re-baseline
rows after the correction above: revisions `279a871`..`35ac7d6`, all not
dirty). All
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

The default-workload (`rough-H0.8`) rows were **re-run after the correction**
with `precond_cost_gate` at its new default (`0.0`, disabled) once F1 landed:

```bash
python bench/harness.py run --workload rough-H0.8 --ns 1024 \
    --variants h2-f32-active-fft,h2-f32-active-stencil --reps 5 --force \
    --ledger data/bench_stencil.jsonl
python bench/harness.py run --workload rough-H0.8 --ns 1024 \
    --variants h2-f64-active-fft,h2-f64-active-stencil --reps 5 --force \
    --ledger data/bench_stencil.jsonl
python bench/harness.py run --workload rough-H0.8 --ns 4096 \
    --variants h2-f32-active-fft,h2-f32-active-stencil --reps 5 --force \
    --ledger data/bench_stencil.jsonl
python bench/harness.py run --workload rough-H0.8 --ns 4096 \
    --variants h2-f64-active-fft,h2-f64-active-stencil --reps 5 --force \
    --ledger data/bench_stencil.jsonl
python bench/harness.py run --workload rough-H0.8 --ns 16384 \
    --variants h2-f32-active-fft,h2-f32-active-stencil --reps 1 --force \
    --ledger data/bench_stencil.jsonl
python bench/harness.py run --workload rough-H0.8 --ns 16384 \
    --variants h2-f64-active-fft,h2-f64-active-stencil --reps 1 --force \
    --ledger data/bench_stencil.jsonl
```

`--force` was required because the ledger's dedup key (workload, Ns, variant,
rep) matched the original, gate-corrupted rows; every new row was verified
(`precond_count == iterations`, `precond_dropped == False`) before use. The
22 gate-corrupted `*-active-fft` rows at this workload (`precond_count == 1`
with `iterations > 1`) were then removed from `data/bench_stencil.jsonl`
outright, and their now-superseded `*-active-stencil` and `*-active-fft`
duplicates from the same reruns were deduplicated to the freshest row per
key — an append-only ledger is the right default, but a row that is actively
wrong (and, per F1's `analyze.py` contract guard, would otherwise sit at a
different `precond_dropped` value than its sibling and silently disqualify
future comparisons) does not belong in the record it is meant to protect.

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

## Paired A/B, fixed default load (`rough-H0.8`, occupancy 0.04–0.11%) — CORRECTED

Re-measured with the cost gate at its new default (disabled). Analyzed with:
```bash
python bench/analyze.py --pair h2-f32-active-fft,h2-f32-active-stencil --metric wall_cold_s --ledger data/bench_stencil.jsonl
python bench/analyze.py --pair h2-f32-active-fft,h2-f32-active-stencil --metric peak_rss_gib --ledger data/bench_stencil.jsonl --target 0.0
python bench/analyze.py --pair h2-f64-active-fft,h2-f64-active-stencil --metric wall_cold_s --ledger data/bench_stencil.jsonl
python bench/analyze.py --pair h2-f64-active-fft,h2-f64-active-stencil --metric peak_rss_gib --ledger data/bench_stencil.jsonl --target 0.0
```

Every row below has `precond_count == iterations` and `precond_dropped ==
False` on both arms, verified before use.

### float32

| Ns | n | fft wall | stencil wall | change | CI | area spread |
|---|---|---|---|---|---|---|
| 1024  | 5 | 0.231 s | 0.180 s | −20.6% | [−49.3, +10.2]% (does not exclude zero) | 0 |
| 4096  | 5 | 2.037 s | 0.912 s | **−54.9%** | [−56.0, −52.1]% | 0 |
| 16384 | 1 | 64.88 s | 11.63 s | **−82.1%** (indicative, n=1) | n/a | 1.7e-05 |

| Ns | n | fft RSS | stencil RSS | change | CI |
|---|---|---|---|---|---|
| 1024  | 5 | 0.107 GiB | 0.105 GiB | −2.5% | [−3.2, −2.3]% |
| 4096  | 5 | 0.711 GiB | 0.672 GiB | −5.4% | [−5.5, −5.3]% |
| 16384 | 1 | 9.117 GiB | 9.118 GiB | ~0% | n/a |

### float64

| Ns | n | fft wall | stencil wall | change | CI | area spread |
|---|---|---|---|---|---|---|
| 1024  | 5 | 0.310 s | 0.222 s | **−23.7%** | [−29.6, −16.5]% | 0 |
| 4096  | 5 | 3.365 s | 1.045 s | **−69.3%** | [−71.2, −67.3]% | 0 |
| 16384 | 1 | 178.1 s | 28.16 s | **−84.2%** (indicative, n=1) | n/a | 0 |

| Ns | n | fft RSS | stencil RSS | change | CI |
|---|---|---|---|---|---|
| 1024  | 5 | 0.119 GiB | 0.110 GiB | −7.6% | [−8.0, −7.1]% |
| 4096  | 5 | 0.826 GiB | 0.663 GiB | **−19.8%** | [−20.0, −19.7]% |
| 16384 | 1 | 9.315 GiB | 9.117 GiB (previous rep) | −2.1% | n/a |

Contact area agrees to within 1.7e-05 absolute (float, worst case) or exactly
(double) across every pair — the engines answer the same problem. RSS numbers
are essentially unchanged from the retracted measurement (the cost gate
affects which code path a solve takes, not how much memory it allocates), so
the RSS story is unaffected by the correction: the stencil's memory edge is
real but modest (0–20%), smallest at Ns=1024/16384, largest at Ns=4096 f64.

### Iteration counts (median) — the corrected direction

| Ns | fft it | stencil it | nopc it | stencil/fft ratio |
|---|---|---|---|---|
| 1024 f32  | **11** | 13 | 23 | +18% |
| 4096 f32  | **19** | 25 | 69 | +32% |
| 16384 f32 | **47** | 61 | —  | +30% |
| 1024 f64  | **15** | 17 | —  | +13% |
| 4096 f64  | **25** | 33 | —  | +32% |
| 16384 f64 | **72** | 94 | —  | +31% |

**This is the opposite of what the retracted measurement reported.** The
exact full-grid FFT application of the `|k|` symbol is the *stronger*
preconditioner per iteration — it needs 13–32% *fewer* iterations than the
truncated 13-tap stencil, and the gap widens with `Ns`, exactly as the
truncated symbol's flatter, less-faithful approximation of `|k|` would
predict (see the "why the first measurement was wrong" section below, and
`doc/theory/pcg.tex`'s "positivity of the truncated symbol" paragraph for the
spectral argument: `min/max` of the truncated symbol is 0.137 vs 6.9e-4 for
the true `|k|` symbol — a genuinely worse-conditioned reweighting). The
stencil wins on WALL TIME anyway, and decisively (18–84%, growing with `Ns`),
because a full-grid FFT application costs one to two orders of magnitude more
than a truncated real-space sum restricted to the active-set candidate set,
where the masked matvec itself is only 1–2% of a full one. The correct
one-line summary: **the stencil is a somewhat weaker preconditioner per
iteration but one to two orders of magnitude cheaper per application, so it
wins decisively on wall time at low occupancy** — not "a better
preconditioner, not merely a cheaper one," which is retracted.

Against the `nopc` baseline (no preconditioner at all — unaffected by the
cost-gate bug, since there is nothing for the gate to drop), the stencil
still wins outright: **−17.8%** at Ns=1024 f32 (0.180 s vs 0.219 s no-precond,
from the existing, uncorrupted ledger rows) and **−21.4%** at Ns=4096 f32
(0.912 s vs 1.160 s; `bench/analyze.py`'s own guard disqualifies this specific
pair as "different accuracy contract" only because the legacy `nopc` rows
predate the `precond_dropped` field entirely — `None` vs `False` — not
because of any real difference in what was measured; medians computed
directly from the ledger). The full-grid FFT arm, by contrast, is *slower than no
preconditioner at all* here at Ns=4096 f32 (2.037 s vs 1.160 s, +76%) despite
needing far fewer iterations (19 vs 69): on this restricted path its
per-application cost is high enough to erase the entire benefit of
preconditioning. That is itself a useful, previously invisible finding: a
correct, effective preconditioner can still be a net loss if its application
cost is mismatched to the size of the matvec it is meant to speed up.

### Why the first measurement was wrong

The retracted numbers were not fabricated or mis-transcribed — they were
measured, honestly, from a solver whose numerical path silently depended on
wall-clock timing. `SolveOptions::precond_cost_gate` (default `1.5` before
this fix) samples one preconditioner-apply-to-matvec cost ratio after the
first iteration and, if it exceeds the threshold, drops the preconditioner
for the rest of the solve. On the active-set restricted path the masked
matvec is 1–2% of a full one, so a full-grid FFT apply — unrestricted, it
still touches every grid point — routinely costs many times a single masked
matvec, and the gate fired on **every** `*-active-fft` run at this workload.
The result: what the ledger labelled "the FFT-preconditioned solve" ran
correctly preconditioned for exactly one iteration, then unpreconditioned
for the rest — a fundamentally different (and much slower-converging) solve
than the one the variant name claimed. `precond_count` — 1, instead of the
iteration count — was in the ledger the whole time, but nothing compared it
against `iterations`, so a >1 order-of-magnitude corruption to the headline
comparison went unnoticed through a full write-up, a promotion verdict, and
three separate documents (this one, `CLAUDE.md`, `doc/theory/pcg.tex`).

Two structural fixes came out of this, both now in place: (1)
`precond_cost_gate` defaults to `0.0` (disabled) — a solver's numerical
trajectory should not depend on wall-clock timing, and the gate is now
opt-in for the regime it was actually designed to protect (a full-grid FFT
preconditioner with no active-set restriction); (2) `precond_dropped` is
bound end-to-end (Python, per-stage in `stage_stats`, the bench ledger) and
`bench/analyze.py`'s accuracy-contract guard now disqualifies a pair whose
arms disagree on it — so a repeat of exactly this failure mode would fail
loudly (a "DISQUALIFIED: different accuracy contract" line) rather than
silently producing a wrong headline number. The lesson for any future
benchmark note: **every field the ledger records for a "why did this run
converge the way it did" reason belongs in the accuracy-contract check, not
just the ones that were designed in from the start** — `precond_count` was
exactly such a field, present from the first commit of this benchmark,
unused until the incident that made its absence from the guard obvious.

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
  the bar, using the corrected (post-F1) numbers. Ns=4096 f32 (**−54.9%**, CI
  excludes zero) and f64 (**−69.3%**, CI excludes zero) both exceed 10% with 5
  paired samples; Ns=1024 f64 also clears (**−23.7%**, CI excludes zero);
  Ns=1024 f32 is favourable in direction (−20.6%) but its CI does not exclude
  zero at n=5 (wide interval from only 5 samples at a small, noisy absolute
  wall time), so it is indicative only at that specific size. Ns=16384 (both
  precisions) shows a large favourable single-sample result (−82%/−84%)
  consistent in direction and magnitude with the 5-sample cases, but is
  single-rep and therefore indicative, not a confirmed pass on its own. The
  win is now understood to come entirely from the stencil's far cheaper
  per-application cost, not from a lower iteration count — the corrected
  iteration counts run the *other* way (stencil needs 13–32% more iterations
  than fft at this occupancy; see "Paired A/B" above).
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
the required near-full-contact gate point.** These two ratios are the
*whole-nested-solve* comparison against the pre-stencil standard path
(`doc/bench/2026-09-09-ns16384-rebaseline.md`), not this note's
`*-active-fft`-vs-`*-active-stencil` isolate — they were not re-verified
against `precond_count`/`precond_dropped` as part of this correction, and
should be read with the same caution the rest of this note now applies to
any pre-F1 wall-time number until spot-checked. The finest-level, isolated
comparison directly above (18–84% depending on Ns and precision, all
confirmed `precond_count == iterations`) is the number to trust for the
finest-level engine choice; treat the whole-solve 2.87×/2.90× as plausible
but unaudited. Memory did not show the dramatic
drop one might expect from removing full-grid FFT scratch either: RSS
improved only modestly (0–20%, mostly at Ns=4096 f64) and was flat within
noise at Ns=1024 and Ns=16384 — the full-grid `gap`/`warm_start`/`output`
buffers the preflight accounting lists dominate peak RSS at these sizes far
more than the preconditioner's own scratch, so removing the FFT engine's
buffers does not move the needle much on total peak memory. Both of these are
reported as measured findings, not confirmations of the design-time estimate.

## Ledger

All rows: `data/bench_stencil.jsonl` (94 rows after the F3 correction: the
original 84 minus the 22 gate-corrupted `*-active-fft` rows at the default
`rough-H0.8` workload, plus 32 fresh rows from the re-run — 5 reps × 4
(fft/stencil × f32/f64) at Ns=1024, 5 reps × 4 at Ns=4096, 1 rep × 4 at
Ns=16384 — deduplicated against their own superseded `--force` reruns down
to one row per (workload, Ns, variant, rep)). The `p_bar` occupancy-dial
probes (0.5, 1.0, 2.0, one rep each, used only to pick the gate load) were
run to a separate, uncommitted scratch ledger and deleted after use; they
are not part of the analyzed A/B and are reproducible from the commands in
the "Occupancy dial" section above if needed.

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

**F3 note on this subsection's `@0.002` numbers.** The `@2.0` (≈57% contact)
rows below are confirmed unaffected by the cost-gate bug
(`precond_count == iterations` on every row, checked directly — that
occupancy runs the plain full-grid solve, where the gate's ratio rarely
crosses its threshold). The `@0.002` (≈0.1% contact) `h2-f32-active-fft` row
IS one of the gate-corrupted rows (`precond_count == 1` against 22
iterations) and was not re-run as part of this correction (out of the scope
directed for F3, which targeted the default `rough-H0.8` workload at
Ns=1024/4096). This does not change the conclusion drawn from it below,
which already treated the `@0.002` auto-vs-fft comparison as statistically
inconclusive (CI does not exclude zero) and promoted nothing on its basis —
but the exact `-21.16%` figure should not be read as a confirmed number, and
a future re-measurement of this specific row would be needed before citing
it standalone.

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
