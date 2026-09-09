# Review 2026-09-08 — recorded baseline artifacts

`review_20260908_results.jsonl` and `review_20260908_cpp_results.txt` are the
**pre-fix** measurements produced by `review_20260908.py` and
`review_20260908.cpp` against baseline `002b20d`, on this workstation with
`OMP_NUM_THREADS=4`. They are kept because the validation plan
(`doc/plans/2026-09-08-accuracy-efficiency-validation.md` §1) requires
preserving the failing examples and baseline artifacts rather than only the
prose that describes them: a regression test asserts the *repaired* behaviour,
so without these files the size of what was repaired is lost.

Do not regenerate them in place. They are a fixed reference point; a new
measurement belongs in a new file with its own provenance.

## What each probe recorded, and what superseded it

| Probe | Recorded baseline | Superseded by | Current |
|---|---|---|---|
| `warm_start` | `converged=True` at **iteration 0**, pressure rel error **2.236**, fw_gap 0.2946, penetration 0.4603, old PK error **0.0** | `d82ad9c` (A01+A02) | repaired in 3 iterations, rel 1.5e-16 |
| `simplex_repair` | independent projected-gradient prototype: 162 iterations, λ_min 0.0737, error vs cold 3.2e-11 | — (oracle, not production) | production needs 3 iterations for the same repair |
| `precond_scale` | c=1: 22 it. **c=1e-3: 208 it, `converged=True`, PK error 0.196, pressure rel 2.133.** c=1e3: 22 it | `d82ad9c` (A02) | 23 it at every c ∈ [1e-6, 1e6], pressure rel ≤ 1.5e-15 |
| `stagnation` | requested tol 1e-25 → `converged=True` after 542 iterations | `d82ad9c` (A01) | `stagnated`/`max_iterations`, or success only when the certificate is exactly zero |
| `float_gap_offset` | shift 1e6 uncentered: **1 iteration, reported error 0.0, pressure rel 1.0795**; shift 1e3: 211 it, rel 1.26e-2 | `42f346a` (A04) | 14 it and rel 3.0e-6 at every offset; approach shifts by exactly the offset |
| `float_scale` | uniform ×1e3/×1e6 rescaling already correct (rel 4.84e-5) | — | unchanged; gated by T08 |
| `kernel` | Love relative error **1.86e-7** at (16384,16384)h, **0.99996** at (−1e6,0)h, 1.11e-4 at (+1e6,0)h | `5bb3c23` (A05) | worst 3.5e-14 over the 472-point 80-digit fixture |
| `kernel_edge` | **NaN** at the cell corner | `5bb3c23` (A05) | finite; gated by T11 |
| `table_free` | full table 49.8 ms + build 26.2 ms vs compact build 25.1 ms; `equal=1` | `aab8cf2` (A06) | shipped; still `equal=1, rel=0` with the A05 kernel |
| `friction_normal_unload` | `converged=True` with **cone excess 0.02362** and shear on **76 open points** | `7c9f6c7` (A15 §10.1) | cone excess 1.1e-19, 0 shear-carrying open points |
| `friction_zero_normal_load` | `p_bar=0` silently skipped: `pressure_unchanged=True`, mean pressure 1e-4 | `7c9f6c7` (A15 §10.1) | p = q = u_t = 0, approach = min(g₀) |
| `fft_contact`, `h2_contact` | reference certificates; H² bilinear symmetry defect 1.44e-13 (q=4) … 5.77e-14 (q=8); identical contact area at all q while pressure error spans 5.1e-5 … 2.4e-8 | — | still the reference; feeds A09 and T19 |
| `h2_memory` | reported 15 365 952 B vs omitted metadata **≥ 7 072 776 B** | **open — A07** | unchanged: the accounting understates by ≥46% |
| `transfer` | tensor vs dense M2M/L2L: q=4 **0.654×**, q=6 **0.609×**, q=8 1.202× | **open — A10 evidence** | reconfirmed 0.70× / 0.69× / 1.16×: a measured no-go at production q |
| `m2l_rank` | factored/dense **flop** ratio: q=4 1.175 (1e-6) and 1.60 (1e-8); q=6 0.53; q=8 0.30 | **open — A10 evidence** | unchanged; flop ratios are not predictive of time here (see `transfer`) |
| `residual_recurrence` | 45 → 37 matvecs (18%) but 22.50 → 21.56 ms (**4.2%**), with only **8 recurrences in 22 iterations** | **open — A08 evidence** | unchanged; measure before implementing |

## Reproduction

```bash
conda run -n fenicsx-env cmake --build build -j4
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  conda run -n fenicsx-env python experiments/review_20260908.py
conda run -n fenicsx-env /usr/bin/g++ -std=c++17 -O3 -march=native -fopenmp \
  -Iinclude -I$CONDA_PREFIX/include/eigen3 \
  experiments/review_20260908.cpp build/libaspher_core.a -o /tmp/aspher-review
OMP_NUM_THREADS=4 /tmp/aspher-review
```

Running these today reproduces the *repaired* values in the last column, not
the recorded baseline — that is the point of keeping the files.
