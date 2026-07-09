<p align="center">
  <img src="https://raw.githubusercontent.com/cofebem/aspher/main/extras/logo.png"
       alt="ASPHER logo: through roughness, toward the star" width="180">
</p>

# ASPHER — Accelerated SPectral and HiERarchical contact solver

*(pronounced "asper", as in **asper**ity — or as the Latin *asper*, "rough")*

> ***ad astra per ASPHERa*** - through roughness, to the stars.

+ **Author:** Claude Fable (foundation), Claude Opus 4.8, chatGPT 5.5 (initial theory)
+ **Coordinator:** V.A. Yastrebov

C++17 boundary-integral solver for frictionless normal
contact of an elastic half-space: exact Love/Boussinesq influence coefficients
applied through hierarchical operators (H-matrix ACA and a matrix-free
H²/bbFMM), a |q| spectral preconditioner, nested-grid continuation and
a mixed-precision Polonsky–Keer CG, with a Python interface.

The design combines an *exact* kernel with *fast, matrix-free* operators:

- **Kernel**: Love (1929) closed-form integration of the Boussinesq point
  solution over square elements — exact for every influence coefficient,
  served from an O(N) lookup table (translation invariance).
- **Operator (default): matrix-free H²/FMM** — black-box fast multipole
  method with Chebyshev interpolation (Fong & Darve 2009), all transfer and
  coupling operators cached by translation invariance. **O(N) memory and
  matvec**: 5.3 MiB at 512², ~13 B/DOF asymptotically; grids up to 16384²
  (2.7×10⁸ DOFs) run on a 32 GiB workstation. A classical H-matrix (ACA)
  backend and a dense backend are kept for validation and small grids.
- **Solver**: Polonsky & Keer (1999) projected CG with overlap correction,
  accelerated by a |q| spectral preconditioner (half-spectrum real FFT),
  nested-grid (cascadic/FMG) continuation, warm starts, and an optional
  single-precision mode with double-accumulated reductions — up to ~15×
  faster than the cold double solve at equal contact-area accuracy.

## Install

```bash
pip install aspher            # once published on PyPI
# or from a checkout / the sdist:
pip install .
```

Building from source needs a C++17 compiler and CMake ≥ 3.18; Eigen is found
on the system or fetched automatically, and the FFT engine (pocketfft) is
bundled. If the default compiler misbehaves (see the conda note in
CLAUDE.md), override it with
`CMAKE_ARGS="-DCMAKE_CXX_COMPILER=/usr/bin/g++" pip install .`.
Wheels are built without `-march=native` (portable); local dev builds keep it.

Development build (see `CLAUDE.md` for the environment specifics):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j && ctest --test-dir build
```

The Python module `aspher*.so` is placed in `python/` (an `hmatrix_contact`
import alias is kept for existing scripts).

### Release to PyPI (maintainers)

```bash
python -m pip install -U build twine
python -m build --sdist          # -> dist/aspher-X.Y.Z.tar.gz
python -m twine check dist/*
python -m twine upload dist/aspher-*.tar.gz   # sdist only: PyPI rejects
                                              # non-manylinux binary wheels
```

Bump `version` in `pyproject.toml` and `CITATION.cff` before releasing. The
sdist is all-BSD (bundled pocketfft); binary wheels would need `cibuildwheel`
(manylinux + `auditwheel`) and are a later step.

## Quick start

```python
import numpy as np, sys; sys.path.insert(0, "python")
import aspher as hc   # `import hmatrix_contact` still works (alias)

# a rigid rough indenter pressed onto an elastic half-space: gap0 = -height
Ns = 1024                       # power of two for the H2 operator
gap = -surface.ravel()          # your (Ns, Ns) height field

# one-call nested-grid solve (coarse -> fine, preconditioned, warm-started)
res = hc.solve_nested(grid_size=Ns, gap=gap, p_nominal=0.005,
                      coarsest=64, q=6)
print(res.contact_area, res.iterations, res.converged)

# large grids: single precision + light result (~half the memory, ~3x faster)
res = hc.solve_nested(grid_size=4096, gap=gap4k, p_nominal=0.005,
                      single_precision=True, light_result=True)

# or the explicit operator interface (backend="h2" | "hmatrix" | "dense")
solver = hc.ContactSolver(grid_size=Ns, backend="h2", q=6)
res = solver.solve(gap, p_nominal=0.005, tol=1e-8, precond="fourier")
res.pressure        # (Ns, Ns), mean == p_nominal
res.contact_area    # Ac/A
```

`python example_rough_contact.py` runs an end-to-end rough-surface demo
(self-affine surface → applied pressure → contact map).

## Performance highlights

Measured on a 20-core workstation (fixed-band self-affine roughness):

| Quantity | Value |
|---|---|
| H² operator memory at Ns=2048 (4.2M DOFs) | 51 MiB (dense: 128 TiB) |
| H² build / matvec at Ns=512 | 0.03 s / 9 ms (H-matrix: 24 s / 144 ms) |
| Nested solve Ns=1024, double, tol 1e-8 | 1.2 s |
| Nested solve Ns=4096, double, tol 1e-8 | ~40 s (50 it) |
| Nested solve Ns=4096, float + light result | ~11 s (18 it, area matches double to 0.2%) |
| Ns=16384 (2.7×10⁸ DOFs), float, 32 GiB node | ~25 GiB peak |

## Tests and benchmarks

- `ctest --test-dir build` — kernel vs analytics, H/H² matvec vs dense,
  Hertz contact vs theory, preconditioner and mixed-precision consistency.
- `python compare_tamaas.py` / `compare_tamaas_h2.py` — benchmark against
  [Tamaas](https://gitlab.com/tamaas/tamaas): contact fractions agree to
  0.0005, pressure fields to ~3.3% L2 — a bound set by Tamaas's dcfft
  coefficients, not by ASPHER's operators; see
  [doc/tamaas_findings.md](doc/tamaas_findings.md) for the three Tamaas
  pitfalls anyone comparing against it should know.
- `python bench_h2.py`, `bench_h2_memory.py`, `bench_h2_cputime.py` — H² vs
  H-matrix and O(N) scaling sweeps up to Ns=16384.
- `python experiments/bench_cpp_precond.py` — preconditioner + nested-grid
  acceleration (e.g. 180→45 iterations at Ns=1024).

## License

The ASPHER source is **BSD-3-Clause** (see `LICENSE`).

The spectral preconditioner's FFT runs, by default, on the bundled
[pocketfft](https://github.com/mreineck/pocketfft) (BSD-3-Clause) — so the
default build, including binary wheels, is permissively licensed throughout.
Configuring with **`-DASPHER_USE_FFTW=ON`** switches to FFTW3 plans, which
are slightly faster end-to-end (measured at Ns=4096: ~16% on a double
solve, ~5% on a float solve, identical results) — but **FFTW is GPL**, so
binaries built with that option are governed by the GPL as a whole. Choose
per deployment:

```bash
CMAKE_ARGS="-DASPHER_USE_FFTW=ON" pip install .   # faster, GPL binaries
pip install .                                      # default, all-BSD
```

## References

The methods implemented here are described in:

- J. Boussinesq, *Application des potentiels à l'étude de l'équilibre et du
  mouvement des solides élastiques*, Gauthier-Villars, 1885 — half-space
  point-load solution.
- A.E.H. Love, "The stress produced in a semi-infinite solid by pressure on
  part of the boundary", *Phil. Trans. R. Soc. A* 228 (1929) — exact element
  integrals.
- I.A. Polonsky, L.M. Keer, "A numerical method for solving rough contact
  problems based on the multi-level multi-summation and conjugate gradient
  techniques", *Wear* 231 (1999) 206–219 — the projected CG with overlap
  correction.
- W. Fong, E. Darve, "The black-box fast multipole method", *J. Comput.
  Phys.* 228 (2009) 8712–8725 — the Chebyshev H²/FMM operator.
- M. Bebendorf, *Hierarchical Matrices*, Springer, 2008; W. Hackbusch,
  *Hierarchical Matrices: Algorithms and Analysis*, Springer, 2015 — the
  H-matrix/ACA backend.
- V.A. Yastrebov, G. Anciaux, J.-F. Molinari, "From infinitesimal to full
  contact between rough surfaces: evolution of the contact area", *Int. J.
  Solids Struct.* 52 (2015) 83–102 — rough-contact context and validation
  methodology.
- L. Frérot, G. Anciaux, V. Rey, S. Pham-Ba, J.-F. Molinari, "Tamaas: a
  library for elastic-plastic contact of periodic rough surfaces", *J. Open
  Source Softw.* 5(51) (2020) 2121 — the FFT-based solver used for
  cross-validation.

Solver theory (projected CG, spectral preconditioning, finite-precision
implementation) is documented in `doc/theory/pcg.tex`; the H²/FMM operator in
`doc/theory/h2_fmm_detailed.tex`.

## Citing

If ASPHER contributes to your research, please cite it (see also
`CITATION.cff`, which GitHub renders as a "Cite this repository" button):

```bibtex
@software{aspher,
  author  = {Yastrebov, Vladislav A. and {Claude (Anthropic)}},
  title   = {{ASPHER}: Accelerated {SP}ectral and {H}i{ER}archical contact solver},
  url     = {https://github.com/cofebem/aspher},
  version = {0.1.0},
  year    = {2026},
}
```

## Layout

```
include/, src/       kernel, cluster tree, H-matrix, H2/FMM operator,
                     spectral preconditioner, nested solve, contact solver
python/bindings.cpp  pybind11 module `aspher` (+ `hmatrix_contact` alias)
third_party/         bundled pocketfft (BSD-3-Clause)
tests/               C++ unit/integration tests (CTest) + tamaas_test.py
doc/theory/          PCG and H2/FMM theory references (LaTeX)
doc/tamaas_findings.md  Tamaas 2.8.1 pitfalls for non-periodic comparisons
```
