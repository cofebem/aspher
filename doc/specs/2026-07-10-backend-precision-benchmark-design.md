# Backend × precision × Ns performance and convergence study — design

Status: agreed in discussion 2026-07-10, not yet implemented.
Goal: a complete, reproducible sweep of the rough-contact problem from
`example_rough_contact.py` across `backend ∈ {h2, fft}` × `precision ∈
{double, float}` × `Ns ∈ {256, 512, 1024, 2048, 4096, 8192, 16384}`,
recording wall time, peak memory, and PCG convergence behaviour for each
of the 28 cases, plus plots and a short written summary.

## 1. Problem setup (fixed across the sweep)

- Surface: `rfgen.selfaffine_field(dim=2, N=Ns, Hurst=0.8, k_low=12/Ns,
  k_high=0.33, plateau=False, noise=True, rng=np.random.default_rng(42))`,
  rescaled so `rms=0.02` (`roughness *= 0.02/np.std(roughness)`). Pure
  roughness, **no** added paraboloid — matches the already-validated
  `bench_fft.py` reference case (Ns=4096 h2 f64: 177s/91it/area 0.005403),
  not the Hertz+roughness combination in `example_rough_contact.py`.
  `k_low` scales with `1/Ns` so the low-frequency (large-wavelength)
  content is resolution-independent; `k_high=0.33` is a fixed fraction of
  each grid's own Nyquist, so finer grids resolve more physical high-frequency
  detail than coarser ones — the surfaces at different Ns are *related* but
  not identical resamplings of one master surface. This is a deliberate
  simplification (matches how `example_rough_contact.py` scales `k_high` by
  `Nmax` too) and should be called out as a caveat wherever iteration counts
  are compared across Ns.
- Single load step: `p_bar=0.002`, `E_star=1.0`, `L=1.0`, one-shot
  `solve_nested` call (no incremental loading).
- 7 Ns values × 2 backends × 2 precisions = 28 cases total. Some may fail
  (most likely double precision at Ns=16384, which is expected to approach
  or exceed this machine's 31 GiB RAM); failures are recorded, not fatal to
  the sweep (§5).

## 2. Solver settings

- `coarsest=64` (all Ns values are `64 * 2^k`).
- `precond=True`, `light_result=True`, `tol=1e-8`, `coarse_tol=1e-4`,
  `max_iter=20000` — `solve_nested` defaults; float precision internally
  clamps the finest-level tolerance to `2e-6` (existing behaviour).
- H2 backend: `q=6, leaf_side=8` for Ns ≤ 4096; `q=4, leaf_side=16` for
  Ns ∈ {8192, 16384} (CLAUDE.md's validated large-grid recipe — reduces
  OOM risk at the top end at the cost of lower far-field accuracy there;
  noted as a caveat, not treated as a confound to hide).
- FFT backend: no q/leaf_side (n/a).
- Repetitions: 3 reps (report min wall time + spread) for Ns ≤ 512; 1 rep
  for Ns ≥ 1024 (solves become too expensive to repeat).
- `record_error_history=True` always (§3) — captures the finest level's
  per-iteration complementarity error.

## 3. C++ / bindings addition: opt-in convergence history

Non-intrusive, off-by-default addition so the study can plot residual decay,
not just endpoint iteration counts:

- `include/contact_solver.hpp`: `ContactResult` gains
  `std::vector<double> error_history;` (empty unless requested).
- `solve_contact_impl<Real>` gains a trailing `bool record_history = false`
  parameter. When true, one `error_history.push_back(res.error)` per
  iteration, at the point `res.error` is already computed
  (`src/contact_solver.cpp`, current line 74). Zero allocation/branch cost
  in the hot loop when false (single `if` check).
- `solve_contact` (double convenience wrapper) forwards the new parameter
  with default `false`.
- `include/nested_solve.hpp`: `NestedParams` gains
  `bool record_error_history = false;`.
- `src/nested_solve.cpp`: `solve_contact_nested` passes
  `record_history = finest && np.record_error_history` into both the
  double and float `solve_contact_impl` calls — mirrors the existing
  `finest`/`light` pattern, so only the finest level's trace is captured
  (consistent with `res.iterations` already being finest-level-only).
- `python/bindings.cpp`: `ContactResult.error_history` exposed as a numpy
  array (empty when not requested); `solve_nested(..., record_error_history=
  False)` new kwarg threaded to `NestedParams`.
- Test coverage: extend `tests/test_contact.cpp` with one case asserting
  `!error_history.empty()` and `error_history.back() == result.error`
  when the flag is set, and that `error_history` stays empty when it
  isn't. (Not `size() == iterations`: the existing loop reports
  `iterations = it`, the loop-counter value at break time, which is one
  less than the number of error evaluations on the early-converged path
  but equal to it on the max-iter-exhausted path — a pre-existing
  off-by-one quirk between those two branches, not something this
  feature should encode into a fragile test.) No new test file needed.

## 4. Scripts

Two files, following `bench_h2_cputime.py`'s proven dual-mode pattern
(subprocess-per-case isolates crashes/OOM and gives clean peak RSS via
`resource.getrusage`).

### `bench_backend_precision_study.py`

- **Worker mode** (`--backend {h2,fft} --precision {double,float} --ns N`):
  runs one case in-process.
  - Builds the surface memory-leanly: float32 broadcasting (not
    `np.meshgrid`) for Ns ≥ 8192, `gc.collect()` + `ctypes` `malloc_trim(0)`
    before the solve — mirrors `example_rough_contact.py`'s Ns=16384 path.
  - Calls `hc.solve_nested(...)` with the settings from §2, timed with
    `time.perf_counter()`, repeated per the §2 repetition policy (min
    reported, all values kept).
  - Measures `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` once,
    after the (last) solve.
  - Prints one line `JSON {...}` (see §6 for the schema) to stdout.
- **Orchestrator mode** (no case-selecting args, or `--max-ns`): iterates
  all 28 cases in increasing Ns (cheapest first), spawning each worker as a
  subprocess via `subprocess.run(..., timeout=...)`.
  - Appends each result as one line to `data/backend_precision_study.jsonl`.
  - **Resumable**: before running a case, skips it if a matching
    `(backend, precision, Ns)` entry already exists in that file, unless
    `--force` is passed. Matters because a 28-case sweep with Ns up to
    16384 can run for a long time and may need to be restarted.
  - Per-case timeout, default 1800 s (`--timeout`): on timeout, kill the
    subprocess and record `status="timeout"` instead of hanging the sweep.
  - Non-zero exit (crash/OOM) → record `status="oom"` if the return code
    indicates a signal consistent with the OOM-killer (`returncode == -9`,
    i.e. `SIGKILL` — the common case; the kernel logs to `dmesg`, not the
    subprocess's stderr, so stderr is typically empty here) or if stderr
    contains `std::bad_alloc`/`MemoryError`; `status="error"` for any other
    non-zero exit, with a `stderr_tail` field regardless. The sweep
    continues to the next case either way.

### `analyze_backend_precision_study.py`

Reads `data/backend_precision_study.jsonl`, writes to
`doc/backend_precision_study/`:

- `fig_walltime_vs_ns.png` — log-log wall time vs Ns, one line per
  (backend, precision) (4 lines).
- `fig_memory_vs_ns.png` — log-log peak RSS (GiB) vs Ns, same 4 lines.
- `fig_iterations_vs_ns.png` — iteration count vs Ns, same 4 lines.
- `fig_convergence_curves.png` — `error_history` vs iteration (log-y),
  small multiples at 2–3 representative Ns (e.g. 1024 and 4096), one
  subplot per Ns, 4 lines per subplot.
- A short `summary.md` with a results table and a few sentences of
  findings (which backend/precision wins where, where things failed and
  why, any surprises vs the CLAUDE.md validated numbers).
- Failed/missing cases are shown as gaps in the lines (not silently
  dropped), with a printed list of what failed and why.

## 5. Failure handling

Any case that OOMs, times out, or crashes must not stop the sweep or
corrupt already-collected data. The orchestrator treats each subprocess
independently, appends incrementally (not a single end-of-run write), and
records failures as data (`status` field) rather than exceptions.

## 6. Output data schema (one JSON object per line)

```
backend, precision ("double"|"float"), Ns, N,
q, leaf_side,                      # H2 only; null for fft
wall_time_s,                       # min over reps
wall_time_all_s,                   # list, all reps
rss_gib,
iterations, converged, final_error, contact_area, mean_pressure,
error_history,                     # list[float], finest level, [] if not requested
status,                            # "ok" | "oom" | "timeout" | "error"
stderr_tail,                       # only when status != "ok"
seed, p_bar, rms, Hurst, k_low, k_high,   # problem parameters (redundant but self-describing)
timestamp
```

## 7. Out of scope

- Modifying `example_rough_contact.py` itself.
- Per-iteration history for coarse nested-solve levels (only the finest
  level is captured, matching existing `iterations` semantics).
- Automated re-tuning of q/leaf_side/tol per case — the §2 settings are
  fixed inputs to the study, not something the study searches over.
