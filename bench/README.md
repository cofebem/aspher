# ASPHER benchmark harness

Implements the measurement protocol and promotion rule of
[the validation plan](../doc/plans/2026-09-08-accuracy-efficiency-validation.md) §8.
It exists because the D3 optimisation packages (A08, A10, A11, A17) are
*experiments with gates*, not features: each one ships only if a paired
measurement says so, and a documented no-go is a valid result.

## Why the ceremony

Three things on this workstation make casual timing worthless, and the
harness answers each:

| Problem | Answer |
|---|---|
| Wall clock swings ~30% with desktop co-tenancy | paired ABBA ordering + medians + bootstrap CI on the **difference** |
| Two builds write the same `python/aspher*.so` (pocketfft vs FFTW), and a wrong-engine `.so` has contaminated a study before | the FFT engine is read from `ldd` and recorded on every row |
| A faster variant that quietly solved an easier problem is not faster | accuracy is a **precondition**: rows are disqualified before timings are compared |

Above Ns=4096 add a fourth: an OOM costs half an hour, so nothing launches
without a preflight budget.

## Use

```bash
python bench/harness.py list                 # workloads, variants, provenance
python bench/harness.py preflight --ns 16384 --variant h2-f32-active
python bench/harness.py run --workload rough-H0.8 --ns 512 1024 \
    --variants h2-f64,h2-f64-active --reps 5
python bench/analyze.py --pair h2-f64,h2-f64-active --metric wall_cold_s
python bench/analyze.py --pair h2-f64,h2-f64-active --metric peak_rss_gib --target 0.0
```

A long job must be launched **outside the agent sandbox**. Every sandboxed
shell invocation runs in its own PID namespace, so when the invocation ends
the kernel kills everything in it — `nohup`, `disown` and `setsid` all act on
sessions and job tables, not namespaces, and none of them help. Verified with
a bare `sleep 900`, which does not survive either. From a normal terminal
this is simply:

```bash
OMP_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 setsid nohup \
    python bench/harness.py run --workload rough-H0.8 --ns 16384 \
    --variants h2-f32-active,h2-f64-active,h2-f32,h2-f64 \
    > data/bench_16384.log 2>&1 < /dev/null & disown
```

Two consequences worth knowing:

* the ledger is the ground truth for whether a job ran. A killed job leaves
  **no record at all**, which is distinguishable from a failed one: the
  harness records `oom` / `timeout` / `error` rows for anything that fails
  *inside* it.
* `ps` from inside the sandbox cannot see a host job, so an empty process list
  there is not evidence of death. Use the log's mtime and the ledger, which
  are visible either way — the filesystem is shared, only the PID namespace
  is not.

The ledger (`data/bench_ledger.jsonl`) is append-only and resumable: a case
already recorded with `run_status == "ok"` is skipped, so an interrupted sweep
resumes without repeating work.

## What a row carries

Everything needed to defend or refute the number later: revision (and a
refusal to run against uncommitted *tracked* source), compiler and flags,
build type, FFT engine, CPU model, cores, CPU affinity, thread counts,
available memory, workload parameters **and a blake2b hash of the generated
surface**, every solver argument, and from the result — status and reason,
requested vs effective tolerance, validation scope, operator-error kind, the
independently recomputed `fw_error` / `penetration_error` / `load_error`,
iteration and matvec and preconditioner counts, per-stage statistics,
per-phase timings, the solver's own byte accounting, peak RSS from a fresh
process, and a hash of the returned pressure.

RSS and the library's accounting are kept apart on purpose: RSS includes the
Python heap and the surface, and conflating them is how a memory claim gets
overstated.

## Timing and memory conventions

* **Peak memory** comes from a fresh process (`RUSAGE_SELF` after the solve),
  never from a process that has already run other cases.
* **Cold** is the first in-process solve; **warm** is the median of the rest
  (`--inner-reps`). They are reported separately because they answer different
  questions.
* **Pairing** is ABBA across reps, so drift cancels between arms.
* Above `--manual-above` (default 4096) a case runs once, serialised, behind
  the preflight gate.

## Preflight model

H2 storage is **not** proportional to N — the near stencils alone are 4.5 MiB
of fixed cost at `leaf_side=16`, so a single bytes/DOF figure overestimates by
gigabytes. The estimate therefore *constructs* the operator at two small grids
and fits `a + b·N` to its A07 itemised accounting, then adds the exact
solver-side buffer set (CG state, best iterate, mask, outputs), the caller's
gap, the coarse restrictions, the prolonged warm start, and the Python surface
generation transient — which at large Ns is the real peak.

The surface term is **measured, not assumed**. It was written as `3 × 8N`
(24 B/DOF) on the reasoning that generation holds a few grid-sized temporaries;
the real figure for `rfgen` is **36.5 B/DOF**, because `numpy.fft.rfftn` and
`irfftn` transform one axis at a time and allocate a fresh complex array per
axis, so two full half-spectra are live at once on each of the two passes.
That 50% shortfall was the single largest error in the budget — at Ns ≥ 8192
the surface is the biggest item in it, and at Ns=16384 it *is* the peak: the
generator alone reaches 9.115 GiB, and a whole float active-set case was
recorded at 9.116 GiB. `SURFACE_BYTES_PER_DOF` now carries one measured
constant per workload kind.

Validated against the Ns=16384 measurements on record:

| variant | measured RSS | predicted (3×8N) | predicted (measured) |
|---|---|---|---|
| `h2-f32-active` | 10.9 GiB | 8.29 GiB | **11.12 GiB** |
| `h2-f32` | 18.3 GiB | 14.82 GiB | **16.82 GiB** |
| `h2-f64-active` | 12.5 GiB | 8.99 GiB | **12.99 GiB** |
| `h2-f64` | OOM on a 31 GiB machine | 21.82 GiB | **25.82 GiB** |

The old model was uniformly 20–25% *under* measured RSS, and the default
`1.25` headroom was that bias made explicit — which meant the gate had no
actual margin. The corrected model sits within ±8%, and over-predicts on the
two active-set variants, so `1.25` is now headroom for the Python heap and
allocator slack rather than a fudge factor for a known error. For the
active-set variants the estimate uses the certified O(N_c) state and
*separately* reports what the full-solve fallback would need, since that path
can still fire.

## Promotion rule

`analyze.py` applies the plan's rule rather than leaving it to judgement:

1. **Accuracy gate.** All rows must have converged, with identical
   `(status, validation_scope, requested_tol, effective_tol)`. Contact-area
   spread is printed alongside. A run at a different contract is disqualified,
   not discounted.
2. **Direction.** The bootstrap CI on the paired difference must exclude zero
   in the change's favour.
3. **Magnitude.** Median improvement ≥ the target — 10% for a new default,
   `--target 0.0` for a memory-oriented variant that only needs to not regress.
4. **No hidden cost.** Any case regressing beyond 5% is called out.

Fewer than 5 paired samples is flagged as indicative rather than silently
accepted.

## Adding a case

Workloads are physical problems (surface + load); variants are solver
arguments only. Keep them separate — that separation is what lets one variant
be compared across workloads and one workload across variants. Add to
`WORKLOADS` / `VARIANTS` in `harness.py`; both are recorded verbatim in every
row, so an old ledger stays interpretable after the tables change.
