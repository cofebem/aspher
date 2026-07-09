# ASPHER — Accelerated SPectral and HiERarchical contact solver

*(pronounced "asper", as in **asper**ity — or as the Latin* asper*, "rough")*

> ***ad astra per ASPHERa*** — yes, the misspelling of *aspera* is entirely
> deliberate: through roughness, to the stars.

+ **Author:** Claude Fable (foundation), Claude Opus 4.8
+ **Coordinator:** V.A. Yastrebov

Formerly **Hcontact**. C++17 boundary-integral solver for frictionless normal
contact of an elastic half-space: exact Love/Boussinesq influence coefficients
applied through hierarchical operators (H-matrix ACA and a matrix-free
H²/bbFMM), a |q| spectral preconditioner (FFTW), nested-grid continuation and
a mixed-precision Polonsky–Keer CG, with a Python interface.

- **Kernel**: Love (1929) closed-form integration of the Boussinesq solution
  over square elements — exact for every entry, served from an O(N) lookup
  table (translation invariance).
- **H-matrix**: quad-tree cluster tree, Chebyshev admissibility
  `min(diam) <= eta * dist`, partially pivoted ACA for admissible blocks,
  dense leaves otherwise; OpenMP-parallel assembly and matvec.
- **Contact solver**: Polonsky & Keer (Wear 231, 1999) projected CG with
  overlap correction and load normalisation.

## Install (pip)

```bash
pip install aspher            # once published on PyPI
# or from a checkout / the sdist:
pip install .
```

Building from source needs a C++17 compiler, CMake >= 3.18 and **FFTW3**
(double + float, e.g. `apt install libfftw3-dev` or `conda install fftw`);
Eigen is found on the system or fetched automatically. If the default
compiler misbehaves (see the conda note below), override it with
`CMAKE_ARGS="-DCMAKE_CXX_COMPILER=/usr/bin/g++" pip install .`.
Wheels are built without `-march=native` (portable); local dev builds keep it.

Licensed **GPL-3.0-or-later** (linking FFTW, which is GPL, requires a
GPL-compatible license for distributed binaries).

## Build (development)

Requires Eigen 3.4, OpenMP, FFTW3, pybind11, CMake >= 3.18. On this machine the
`fenicsx-env` conda env provides Eigen and Python, pybind11 comes from the
`dolfinx-010` env, and the system `/usr/bin/g++` must be used (the conda gcc
fails on `Python.h` + `<ctime>` interaction):

```bash
conda activate fenicsx-env
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
      -Dpybind11_DIR=$(conda run -n dolfinx-010 python -m pybind11 --cmakedir)
cmake --build build -j && ctest --test-dir build
```

The Python module `aspher*.so` is placed in `python/` (an
`hmatrix_contact` import alias is kept for existing scripts).

## Usage

```python
import numpy as np, sys; sys.path.insert(0, "python")
import aspher as hmc   # `import hmatrix_contact` still works (alias)

solver = hmc.ContactSolver(grid_size=64, domain_size=1.0, E_star=1.0,
                           eta=2.0, aca_tol=1e-6, leaf_size=32)

u = solver.matvec(np.ones(64 * 64))          # influence-matrix product

result = solver.solve(gap=gap_array,          # (64, 64) or flat, >= 0,
                      p_nominal=0.05,         # 0 at deepest indenter point
                      tol=1e-8, max_iter=5000)
result.pressure       # (64, 64), mean == p_nominal
result.displacement   # (64, 64)
result.gap            # (64, 64), >= 0, == 0 in contact
result.contact_area   # fraction of elements in contact
result.objective      # 1/2 p.u + p.g0

solver.hmatrix_info() # block counts, ranks, compression ratio
```

`ContactSolver(..., use_hmatrix=False)` assembles the dense matrix instead
(useful for verification on small grids).

## Tests and benchmark

- `ctest --test-dir build` — kernel values vs analytics, H-matvec vs dense
  (< 1e-5 at `aca_tol=1e-6`), H2-matvec vs dense (~3e-6 at `q=6`), Hertz
  contact vs theory (a and p_max within 5%, actual ~1.6% / 0.2% on a 64-grid).
- `python compare_tamaas.py` — rough-surface benchmark against Tamaas
  (n=64, Hurst 0.8, seed 12345, p=0.05). The Tamaas reference runs
  automatically in the `fluidpaper` env via `conda run`; cached in `data/`
  (`--regen` to refresh). Result: contact fractions agree to 0.0005,
  pressure fields to 3.3% L2.
- `python bench_h2.py` — matrix-free **H2/FMM** backend vs the classical
  H-matrix across grid sizes. At Ns=512 the H2 operator stores 5.3 MiB vs
  6194 MiB (1169× less) and builds in 0.03 s vs 24 s, at ~7e-6 rel error.
- `python bench_h2_memory.py [--plot]` — H2 memory scaling: O(N), ~13 B/DOF
  asymptotically (51 MiB at Ns=2048 vs 128 TiB dense).
- `python bench_h2_cputime.py` — CPU-time scaling of the rough-contact problem
  from Ns=128 to 16384. Build and matvec are O(N); the 2.7e8-DOF operator
  builds in 14 s, applies in 6 s, fits in 14 GiB (full solves listed to 1024).
- `python experiments/bench_cpp_precond.py` — convergence acceleration:
  `solve(precond="fourier")` and the single-entry nested-grid solve
  `hc.solve_nested(grid_size, gap, p_nominal, ...)` vs the unpreconditioned
  solver. At Ns=1024: 4× fewer iterations (180→45) and ~1.6× faster wall time,
  identical solution (ΔArea 0, rel-L2 ~5e-7).
- `hc.solve_nested(..., single_precision=True, light_result=True)` runs the
  solve in `float` and drops the displacement/gap outputs, cutting the solver's
  peak RAM by ~⅓ with the preconditioner on (Ns=4096: 2.31→1.57 GiB →
  Ns=16384 ≈ 25 GiB, fits 32 GiB); solution matches the double solve to rel-L2
  ~2e-5. Keep the preconditioner on (single precision needs it to converge; the
  PCG has a stagnation guard so it stops at the float floor rather than spinning
  to max_iter). At the largest grids the Python surface generation often
  dominates memory — build it in float32 via broadcasting. See
  `example_rough_contact.py` (Ns=16384). When `light_result=True`, the result's
  `displacement`/`gap` are `None`.
- `OMP_NUM_THREADS=1 python compare_tamaas_h2.py --max-n 512` — H2 vs Tamaas
  dcfft FFT, accuracy and timing. Single-thread per-matvec is 0.78→0.69× the
  FFT as Ns grows (O(N) vs O(N log N)); end-to-end solve ~1.5× faster;
  pressure L2 ~3.3% (Tamaas near-field error).
- `python example_rough_contact.py` — end-to-end rough-surface contact on the
  H2 backend (self-affine surface → applied mean pressure → contact-area map),
  writing `example_rough_contact.png`. Scales to Ns=1024 (N ≈ 10⁶, where a
  dense matrix would need ≈ 8.8 TB): solves in ~50 s within ~0.2 GB RAM,
  Ac/A ≈ 0.17.

## Tamaas 2.8.1 findings (affect any "non-periodic" Tamaas comparison)

Verified against Hertz theory while building the benchmark:

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
   surface.

## Layout

```
include/, src/      boussinesq_kernel, cluster_tree, hmatrix, contact_solver
python/bindings.cpp pybind11 module `aspher` (.so lands here too; `hmatrix_contact` alias)
tests/              C++ unit/integration tests (CTest) + tamaas_test.py
compare_tamaas.py   benchmark vs Tamaas; tamaas_reference.py runs in fluidpaper
doc/plans/          implementation plan
```
