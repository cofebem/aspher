# CLAUDE.md — ASPHER Project

**ASPHER** — *Accelerated SPectral and HiERarchical contact solver*
(pronounced "asper", as in *asperity* / Latin *asper* = rough; formerly
**Hcontact**). Motto: *ad astra per ASPHERa* (misspelling intentional).

## What This Is

A C++17 hierarchical (H-matrix / H2-FMM) BEM contact solver with pybind11 Python bindings.
The Python module is `aspher` (built as `aspher.cpython-312-*.so` in `python/`);
`import hmatrix_contact` still works via the `python/hmatrix_contact.py` alias shim.
All paths below are relative to this repository's root.

---

## Conda Environments (critical — each serves one role)

| Env | Role |
|-----|------|
| `fenicsx-env` | **Build env**: Eigen3, numpy, Python 3.12. Use for cmake, ctest, Python scripts. **No pybind11 here.** |
| `dolfinx-010` | **pybind11 only**: used to get `pybind11_DIR` at cmake time, nothing else. |
| `fluidpaper` | **Tamaas 2.8.1**: only for `tamaas_reference.py`. Do not use for building. |

The Python `.so` module (`aspher.cpython-312-*.so`) lands in `python/` and must be imported from there (`hmatrix_contact` remains as an alias).

---

## Build Commands

```bash
conda activate fenicsx-env
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -Dpybind11_DIR=$(conda run -n dolfinx-010 python -m pybind11 --cmakedir)
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

**Why `/usr/bin/g++` and not conda gcc?**
Conda's gcc 12 sets `_POSIX_C_SOURCE` in a way that hides `timespec_get` from `<ctime>`,
breaking pybind11 header inclusion. System gcc 11.4 (`/usr/bin/g++`) works fine.

**Why the explicit `pybind11_DIR`?**
`fenicsx-env` has no pybind11. We borrow the cmake config from `dolfinx-010`.

**FFT engine: bundled pocketfft by default, FFTW3 opt-in (2026-07)**
The spectral preconditioner's transforms run on the vendored
`third_party/pocketfft/pocketfft_hdronly.h` (BSD-3-Clause; keeps binaries
all-BSD) — decomposed r2c/c2r + in-place c2c, parallelised over slabs inside
one OpenMP region (pocketfft's own std::thread pool contends with OpenMP's
spin-waiting workers, and its multi-axis c2r copies the whole spectrum —
both deliberately avoided; `POCKETFFT_CACHE_SIZE` enabled).
`-DASPHER_USE_FFTW=ON` switches to FFTW3 2-D plans (`HMC_USE_FFTW`): ~16%
faster double / ~5% float end-to-end at Ns=4096, identical results, but FFTW
is GPL so such binaries carry GPL terms. In FFTW mode CMake needs `fftw3.h`
+ `libfftw3`/`libfftw3f` (the `*_omp` variants enable `HMC_FFTW_THREADS`);
`FFTW_ESTIMATE` plans by default, `HMC_FFTW_MEASURE=1` opts into measured
plans (not worth it for single solves).

---

## Run Tests

```bash
ctest --test-dir build --output-on-failure
# Or individually:
build/test_kernel
build/test_hmatrix
build/test_contact
```

---

## Theory Summary

### Boussinesq BEM
`u_z(x) = ∫ G(x−x') p(x') dA'`,   `G(r) = 1/(π E* |r|)`

Discretise on Ns×Ns uniform grid, element side h = L/Ns.
Influence matrix `S_ij` = Love (1929) exact integral of G over a square element of half-size a = h/2:

```
L(x,y,a) = (x+a)·ln[(y+a+R++)/(y-a+R+-)] + (y+a)·ln[(x+a+R++)/(x-a+R-+)]
          + (x-a)·ln[(y-a+R--)/(y+a+R-+)] + (y-a)·ln[(x-a+R--)/(x+a+R+-)]
```
where `R±± = sqrt((x±a)²+(y±a)²)`.

Self-term: `S_ii = 4h·ln(1+√2) / (π E*)`.

Translation invariance: `S_ij` depends only on `|ix-jx|, |iy-jy|` → Ns×Ns lookup table, O(1) per entry.

### H-Matrix
- **Cluster tree**: recursive quad-tree, midpoint split, leaf ≤ leaf_size elements. Default `leaf_size=64` (optimal; 32 gives same rank but identical block structure at Ns=64/128).
- **Admissibility**: block (t,s) is low-rank when `min(diam(t),diam(s)) ≤ η·dist(t,s)`, η = 2.0.
- **ACA**: partially-pivoted, stopping criterion `‖u_k‖·‖v_k‖ ≤ ε_aca·‖A_k‖_F`.
- **ACA-GP** (`use_acagp=True`): geometric first pivot (center-facing) + central-subset pivot search for subsequent ranks (Yastrebov 2025). Gives ~5% lower rank at 2× assembly cost on this smooth translation-invariant kernel. Enable with `use_acagp=True, central_fraction=0.3`.
- **SVD recompression** (`solver.recompress(svd_tol)`): post-ACA truncated SVD via QR factorisation of U and V. Drops singular values below `svd_tol * σ_max` per block. Dramatically reduces rank: 12→2.3 (tol=0.01) or 12→1 (tol=0.5), with 47%–54% memory reduction and <0.4% change in contact area.
- **Matvec**: OpenMP over blocks; dense blocks use GEMV, low-rank blocks use `U(V'p)`.
- **Visualization**: `visualize_hmatrix.py` → `doc/slides/figures/fig_hmatrix_blocks.pdf` (blue=low-rank, red=dense).
- **Leaf-size sweep**: `leaf_size_bench.py` benchmarks leaf sizes 8–128 for Ns=64/128.

### H2/FMM operator (`backend="h2"`) — preferred for large Ns
Matrix-free black-box FMM (Chebyshev interpolation, Fong & Darve 2009). **No blocks stored**: shares bases per cluster and couplings per interaction, all cached by `(level, relative offset)` via translation invariance. O(N) memory, O(N) matvec.
- **Tree**: `UniformQuadTree` — balanced quad-tree to square leaves of side `h2_leaf_side` (default 8); stores index *ranges*, no index lists. `Ns`, `leaf_side` must be powers of two.
- **Far field**: tensor-product Chebyshev interpolation, order `q` (default 4; r=q² nodes). Passes `P2M → M2M → M2L → L2L → L2P`. Coupling `K[a,b]=g(ξ_a−ξ_b)` cached by `(level,dx,dy)`; M2M/L2L are 4 cached q²×q² matrices (scale-invariant).
- **Near field**: exact Love stencils for leaves within `near_radius` (default 1, the 3×3 neighborhood), cached by relative leaf offset. Uses the same `love_uz` kernel as the far field (consistent; far error is interpolation-only).
- **Kernel**: far kernel `g(dx,dy) = love_uz(dx,dy,h/2,h/2)/(πE*)` (continuous offsets); near via `BoussinesqKernel::entry_offset`.
- **Accuracy**: rel L2 vs dense ≈ 1.3e-4 (q=4), 3e-6 (q=6); converges with q. Plugs into the same PCG (`MatVec` functor) — reproduces Hertz area/pressure exactly.
- **Bench**: `bench_h2.py` (H2 vs H-matrix). At Ns=512: 5.3 MiB vs 6194 MiB (1169× less), build 0.03s vs 24s, matvec 8.9ms vs 144ms.
- Spec/plan: `doc/specs/2026-06-27-h2-fmm-operator-design.md`, `doc/plans/2026-06-27-h2-fmm-operator.md`.

### FFT-convolution operator (`backend="fft"`) — exact, O(N log N)
Exact zero-padded (Hockney) circular convolution of the pressure with the Love element table on a (2Ns)² grid (`fft_operator.hpp/.cpp`, shared square r2c/c2r engine in `src/fft_engine.hpp` — pocketfft default, FFTW3 under `-DASPHER_USE_FFTW=ON`). **Matches the dense matvec to roundoff** (rel L2 ~1e-15 double, ~1.4e-7 float; no interpolation, no Gibbs — that exactness is its main value), unlike H2's ~1e-4 (q=4) interpolation error. ~10 N reals double scratch (kernel half-spectrum 2N + padded grid 4N + complex half-spectrum work 4N), object-owned and reused; single-precision caches via `build_single_caches`/`matvec_single_into` (same contract as H2). The padded transforms skip structurally-zero forward lines and unread inverse lines (2026-07 perf commit). **Measured performance** (bench_fft.py, 20-core, desktop co-tenancy — ratios more reliable than absolutes): matvec modestly faster than H2 (q=6) at Ns ≤ 2048 (1.6×/1.5× at 1024/2048), ≈parity at Ns=4096 (337 vs 331 ms) — the padded transforms are bandwidth-bound, not flop-bound, so the spec's 2–3× estimate did not materialise. H2 remains preferred for very large Ns (O(N) vs O(N log N), ~5× smaller working set). Available as `ContactSolver(backend="fft")` and `hc.solve_nested(..., backend="fft")`. Spec: `doc/specs/2026-07-09-fft-convolution-backend-design.md`.

### Polonsky–Keer (1999) PCG
Projected CG for the QP `min ½p'Sp + p'g₀  s.t. p≥0, mean(p)=p_bar`.
Default β formula: **Polak-Ribière+** (`use_pr=true`); Fletcher-Reeves available via `use_pr=false`.
**Convergence acceleration** (2026-06): the iteration count grows ~√Ns from the operator's `1/|q|` spectral conditioning (κ(S)∼Ns), plus active-set cost.
- **Spectral preconditioner** (`precond="fourier"`, `fourier_precond.hpp`): `M⁻¹` with symbol `∝|q|` (inverse of `Ŝ∝1/|q|`) applied by FFT to the contact-masked residual, mean-zeroed, DC zeroed. Only the CG direction/β change (M-inner product); exact line search untouched; `precond="none"` follows the original algorithm exactly (identical solution; since the 2026-07 OpenMP reductions the floating-point summation order differs, so no longer bit-for-bit). ~1.7–2.9× fewer iterations (more at larger Ns).
- **Warm start** (`p_init=`): start PCG from a given pressure (renormalised to the load).
- **Nested-grid (cascadic/FMG) continuation** — single C++ entry point `hc.solve_nested(grid_size, gap, p_nominal, coarsest=64, q=6, ...)` (`nested_solve.hpp`): builds the coarse→fine hierarchy and per-level H2 operators internally, restricts the gap (2×2 average), and warm-starts each level by injecting the prolonged coarse pressure (sharp contact boundary; injection beats bilinear). `grid_size` must be `coarsest·2^k`. Combined with the preconditioner → up to 4× fewer iterations at Ns=1024 (180→45), full solve cheaper than one cold solve. Prototypes in `experiments/`; design in `doc/specs/2026-06-30-spectral-preconditioner-design.md`.
- **Single precision** (`hc.solve_nested(..., single_precision=True)`): runs each level's H2 matvec + PCG (and the |q| preconditioner's FFT) in `float`. `solve_contact` is templated (`solve_contact_impl<Real>`, float/double); `H2Operator::matvec_single`/`build_single_caches` hold float cache copies; `FourierPreconditioner::apply_single` uses a float FFT (symbol stored as float). Float's arithmetic floor is ~1e-6, so the finest tol is clamped to 2e-6 (solution matches double to rel-L2 ~2e-5, ΔArea ~4e-6). **Keep the preconditioner ON with single precision** — the float solve stalls (and returns a *wrong* answer) without it. Default `False`. Since the 2026-07 perf pass all CG scalar reductions (dots, sums, means, line-search num/den) accumulate in **double** even when Real=float, and the line-search denominator keeps the centred `(r−rmean)·t` form (the expanded `Σrt − rmean·Σt` form cancels catastrophically in float): the float solve now converges in ~the same iteration count as double (e.g. Ns=2048 fixed-band rough: 21 it, 3.4 s vs 214 stalled it, 52 s before) instead of grinding at the noise floor.
- **Light result** (`hc.solve_nested(..., light_result=True)`): skip the `displacement`/`gap` result arrays (2 of the 3 double N-sized outputs); `pressure` + all scalars still filled. Same flag on `solve_contact(..., light=)`. Through the Python bindings, `result.displacement`/`result.gap` are `None` when light.
- **Stagnation guard** (`solve_contact_impl`): if the complementarity error plateaus for 200 iterations (float noise floor), stop and report `converged` instead of spinning to `max_iter`. With the double accumulators (2026-07) the float path normally reaches its clamped tol directly, so this is now a safety net rather than the usual float exit path.
- **Preallocated solve buffers (2026-07)**: the hot loop is allocation-free in steady state. `H2Operator` owns its multipole/local scratch (`Mbuf_`/`Lbuf_`, q²×nbox, lazily sized per precision — the old per-matvec `vector<VectorXd> M,L` was ~2·nbox small mallocs per apply) and exposes `matvec_into(x, y)`; `FourierPreconditioner` owns its grid/spectrum scratch and exposes `apply_into(g, contact, z)`; `solve_contact_impl` takes into-style functors (`MatVecIntoT`/`PrecondIntoT`) and reuses `u`/`r`/`z` across iterations (public `solve_contact` adapts the old by-value functors). One `H2Operator`/`FourierPreconditioner` must not be applied from two threads concurrently.
- **mallopt in `solve_contact_nested`** (`M_MMAP_THRESHOLD`/`M_TRIM_THRESHOLD` = 128 KB): kept as a belt-and-braces guard, but with the preallocated buffers the per-iteration large-allocation traffic (and the mmap/munmap page-fault churn it caused) is gone from the steady-state loop.
- **Zero-copy gap path (2026-07)**: `solve_contact_nested` and `solve_contact_impl` take `g0` as `Eigen::Ref<const ...>`; `py_solve_nested` passes an `Eigen::Map` view of the numpy buffer straight through (the finest level solves on it directly, only coarse restrictions ~N/3 are materialised). Plus `res.pressure = std::move(p)` when `Real=double` (the end-of-solve copy was the peak-RSS moment). Together −2 N-sized double arrays at peak on the double path: measured 2272→2017 MB at Ns=4096, −4.3 GiB at Ns=16384 — h2-double at 16384 was ~2 GiB short post-9929c60 and is worth a fresh-reboot retest (`16k_test.py`). Beware: a caller passing a non-C-contiguous or non-float64 array still gets a forcecast temporary (correct, just not zero-copy).
- **Consuming warm start (2026-07)**: `solve_contact_impl`'s `p_init` is now a non-const pointer that is **CONSUMED** (moved into the pressure iterate at init; caller may only reassign/destroy it afterwards) — the prolonged warm start no longer sits idle beside its own copy for the whole finest nested solve. The nested float path also frees the double `p_init` right after casting to `p0f`. Public `solve_contact` keeps its non-consuming const-pointer contract by copying (no peak cost — the copy becomes the iterate). Measured at Ns=4096 nested light: double 2017→1887 MB (−1 N-double), float 1584→1387 MB (−1 N-double −1 N-float) → at Ns=16384: −2.1 GiB double, −3.2 GiB float on top of the zero-copy gap savings.

**Large-grid memory recipe** (memory-bound nodes, e.g. Ns=16384, N≈2.7×10⁸ on 32 GiB): `hc.solve_nested(Ns, gap, p_bar, coarsest=64, q=4, leaf_side=16, precond=True, single_precision=True, light_result=True)` → solve ≈ 100 B/DOF ≈ 27 GiB. On the Python side the surface generation (meshgrid + complex FFT temporaries) is often the real hog — build it in **float32 via broadcasting** (not `np.meshgrid`), `del` temporaries, and `ctypes.CDLL("libc.so.6").malloc_trim(0)` before the solve. See `example_rough_contact.py` (its `Ns==16384` branch). The FFT preconditioner (`fourier_precond.cpp`) stores only the kx ∈ [0, Ns/2] half spectrum with the Ns² round-trip scale folded into the symbol; scratch (one real Ns×Ns + one complex (Ns/2+1)×Ns buffer, ≈2 N reals) is object-owned and reused across iterations, and the engine is pocketfft (default, BSD) or FFTW3 plans (`-DASPHER_USE_FFTW=ON`). The mask/scatter/gather passes stay OpenMP. **Run 16384 alone** — the killer is co-tenancy (any other heavy process OOMs it).
- **Memory (measured, O(N) C++ solve):** `single_precision + light_result` cuts the solve by ~⅓ *with* the preconditioner on (Ns=4096: 2.31→1.57 GiB → Ns=16384 ≈ 25 GiB, fits a 32 GiB node); the float `|q|` FFT still has O(N) transients. Without the preconditioner the cut is ~½ (1.37→0.74) — but single precision **needs** the preconditioner to converge, so keep it on. At these sizes the Python **surface generation** (meshgrid + complex FFT temporaries) is often the real hog — build it in `float32` via broadcasting and free temporaries (see `example_rough_contact.py`).

Key step: **overlap correction** `p_i -= τ·g_i` for nodes where p=0 and gap<0.
This is in the 1999 paper but absent from informal pseudocode — omitting it breaks convergence on rough surfaces.
Full algorithm with theory in `doc/theory/pcg.tex` (compile with `pdflatex`; run `bibtex` once for the bibliography). Its §"Spectral Preconditioning and Finite-Precision Implementation" documents the 2026-07 performance pass: half-spectrum FFT preconditioner application, double accumulation of all grid-length reductions under `Real=float` (naive float summation loses all digits at N≳4×10⁶ — this is what stalled the old float path), the centred vs expanded line-search denominator (catastrophic cancellation analysis), and the allocation-free into-style iteration. `doc/theory/h2_fmm_detailed.tex` carries the matching operator-side notes (persistent M/L scratch, CSR interaction lists).

---

## Known Quirks and Bugs

### 1. pybind11 header order in `bindings.cpp`
pybind11 headers (`pybind11/pybind11.h`, `pybind11/eigen.h`) **must come before all project headers**.
If they come after, `<ctime>` from project headers triggers `timespec_get` not-declared error with conda gcc.

### 2. Tamaas dcfft non-periodic bug
`solver.registerNonPeriodic()` alone is **not enough** to activate non-periodic mode.
Must also call `solver.setIntegralOperator("dcfft")` **before** `solver.solve()`.
Without it, PKR silently solves the periodic problem even though `registerNonPeriodic` was called.

### 3. Tamaas dcfft effective modulus
With Tamaas `dcfft`, the effective modulus is `2E²` (not `E`).
To get `E* = 1`, use `E = 1/sqrt(2)` in `tamaas_reference.py`.
Testing with `E=1` gives `p_max/p0 ≈ 1.583` instead of `≈ 1`.

### 4. Tamaas dcfft Gibbs errors
Tamaas dcfft coefficients have ±2–8% near-field errors at r = 1–3h (Gibbs-like oscillations).
This is why `compare_tamaas.py` asserts `L2 diff < 5%` (not 2%).
The 3.3% observed difference is dominated by Tamaas error, not by our H-matrix approximation.

### 5. Beamer enumitem conflict
`\begin{enumerate}[leftmargin=*,label=\arabic*.]` causes `\beamer@parseitem` error.
Fix: use plain `\begin{enumerate}` and `\begin{itemize}` without optional arguments inside frames.

---

## Validated Numbers

| Benchmark | Value |
|-----------|-------|
| Hertz contact radius ratio `a_num/a_Hertz` | 1.016 |
| Hertz peak pressure ratio `p_max/p0` | 0.998 |
| Hertz convergence (PR+, Ns=64, tol=1e-8) | 24 iterations |
| Hertz convergence (FR,  Ns=64, tol=1e-8) | 28 iterations |
| Hertz convergence (PR+, Ns=128) | 35 iterations |
| Hertz convergence (PR+, Ns=256) | 44 iterations |
| Hertz convergence (PR+, Ns=512, tol=1e-6) | 27 iterations |
| H-matvec vs dense (rel L2) | < 1×10⁻⁵ |
| H-matrix compression at N=4096 (Ns=64, leaf=64) | 0.284× (36 MiB) |
| H-matrix compression at N=262144 (Ns=512, leaf=32) | 0.012× (6483 MiB) |
| Avg ACA rank (leaf=64, all Ns) | k ≈ 12 |
| Avg ACA-GP rank (leaf=64, Ns=64) | k ≈ 11.7 (5% lower, 2× slower assembly) |
| SVD recompression tol=0.01 (from ACA, leaf=64, Ns=64) | avg_k→2.3, 36→19 MiB, matvec err 1.2×10⁻⁴ |
| SVD recompression tol=0.5 (from ACA, leaf=64, Ns=64) | avg_k→1.0, 36→17 MiB, matvec err 6.9×10⁻⁴ |
| Rough contact fraction (Ns=64, H=0.8, p_bar=0.05) | Ac/A = 0.128 (leaf=64) |
| Rough convergence (no recompression) | 28 iterations |
| Rough convergence (SVD tol=0.01) | 30 iterations |
| Tamaas pressure L2 diff | 3.3% |
| Assembly time Ns=64 (20-core, OpenMP) | 10 ms |
| Matvec time Ns=64 | 0.6 ms |
| Assembly time Ns=128 | 82 ms |
| Matvec time Ns=128 | 11 ms |
| Assembly time Ns=256 | 457 ms |
| Assembly time Ns=512 | 9.3 s (6.5 GiB RAM) |
| H2 matvec accuracy vs dense (q=4 / q=6) | 1.3×10⁻⁴ / 3.2×10⁻⁶ rel L2 |
| H2 Hertz (Ns=64, q=6) | Ac/A = 0.1943 (== H-matrix), 22 iters |
| H2 memory Ns=512 (q=6) | 5.3 MiB (vs H-matrix 6194 MiB → 1169× less) |
| H2 build / matvec Ns=512 (q=6) | 0.03 s / 8.9 ms (vs 24 s / 144 ms H-matrix) |
| PCG iters, fixed-band rough Ns=1024 (none/fourier/nested) | 180 / 62 / 45 (4× fewer; wall 10.2→6.5 s) |
| Preconditioner solution match (fourier vs none) | ΔArea 0, pressure rel-L2 ~5×10⁻⁷ |
| Hertz Ns=64 iters (none/fourier; test_precond) | 27 / 16; warm-start from solution → 0 |
| Nested solve Ns=1024 double, rough p̄=0.005 (2026-07 perf pass) | 2.59 → 1.23 s (21 it, identical pressure to 9×10⁻¹⁵) |
| Nested solve Ns=2048 float, rough p̄=0.005 (2026-07 perf pass) | 52.5 s / 214 it → 3.4 s / 21 it (area now matches double to 3×10⁻⁴ rel) |
| f32 solve Ns=4096 (precond, light, 2026-07) | 16.3 s, 26 it, peak RSS 1.73 GiB (incl. Python surface) |
| FFT matvec vs dense (rel L2, double/float) | ~1×10⁻¹⁵ / ~1.4×10⁻⁷ (exact operator; test_fft) |
| FFT matvec time Ns=1024/2048/4096 (vs H2 q=6; bench_fft.py) | 17/73/337 ms vs 27/110/331 ms → 1.60×/1.50×/0.98× (measured under desktop co-tenancy; ratios more reliable than absolutes) |
| Nested solve Ns=4096 fft vs h2, double (p̄=0.002, seed 42) | 251 s/110 it vs 177 s/91 it, areas 0.005402 vs 0.005403 (different but both-valid PCG paths, agree to ~10⁻⁶ absolute; wall gap from BOTH the extra iterations at ~parity matvec AND load asymmetry — desktop co-tenancy, fft solves ran under heavier load; ratios indicative only) |
| Nested solve Ns=4096 fft vs h2, float (same case) | 62 s/53 it vs 51 s/52 it, area 0.005406 both (desktop co-tenancy, fft runs under heavier load; ratios indicative only) |

---

## Python Module Usage

```python
import sys
sys.path.insert(0, '/path/to/Hcontact/python')
import aspher as hc   # `import hmatrix_contact` = alias
import numpy as np

solver = hc.ContactSolver(
    grid_size=64,       # Ns
    domain_size=1.0,    # L
    E_star=1.0,         # reduced modulus
    eta=2.0,            # admissibility parameter
    aca_tol=1e-6,       # ACA stopping tolerance
    leaf_size=64,       # max cluster leaf size (default 64; optimal for memory)
    use_hmatrix=True,   # False → dense (for testing)
    use_acagp=False,    # True → ACA-GP geometric pivot (5% lower rank, 2× slower)
    central_fraction=0.3, # ACA-GP central subset radius fraction
    backend="",         # ""→hmatrix (or dense if use_hmatrix=False); "hmatrix"|"dense"|"h2"|"fft"
    q=4,                # H2 only: Chebyshev order (r=q²); q=6 for ~3e-6 accuracy
    near_radius=1,      # H2 only: direct near field within this many leaf boxes
    h2_leaf_side=8,     # H2 only: square leaf side (power of two)
)

# Matrix-free H2/FMM backend (O(N) memory; preferred for large Ns):
h2 = hc.ContactSolver(grid_size=512, backend="h2", q=6)
# same solve()/matvec() API; hmatrix_info() returns H2 stats when backend="h2"

gap0 = np.zeros(64*64)       # initial gap field (flattened Ns×Ns)
result = solver.solve(gap0, p_nominal=0.05)          # PR+ beta (default)
result = solver.solve(gap0, p_nominal=0.05, use_pr=False)  # Fletcher-Reeves
result = solver.solve(gap0, p_nominal=0.05, precond="fourier")  # |q| spectral preconditioner
result = solver.solve(gap0, p_nominal=0.05, precond="fourier", p_init=p_guess)  # + warm start

# Single-entry nested-grid (cascadic/FMG) solve — coarse->fine handled in C++:
result = hc.solve_nested(grid_size=1024, gap=gap0, p_nominal=0.05, coarsest=64, q=6)

print(result.contact_fraction)   # Ac/A
print(result.mean_pressure)      # should equal p_nominal
print(result.iterations)
print(result.converged)

info = solver.hmatrix_info()
print(f"compression: {info['compression']:.3f}x")

# Optional: post-ACA SVD recompression (47% memory reduction, <0.4% contact area change)
solver.recompress(svd_tol=0.01)  # tol=0.5 for 54% reduction (rank → 1)
info2 = solver.hmatrix_info()

# Block structure visualization
layout = solver.block_layout()  # (n_blocks, 5) array: row_begin, row_size, col_begin, col_size, is_dense
```

`ContactResult` fields: `pressure`, `displacement`, `gap`, `approach`, `objective`, `error`, `iterations`, `converged`, `contact_fraction`, `mean_pressure`.

### Runnable example: rough-surface contact (H2 backend)

`example_rough_contact.py` — end-to-end demo (imports → Ns → self-affine surface →
apply mean pressure → contact area → plot). Run in `fenicsx-env`:

```python
import numpy as np, aspher as hc
Ns = 128                                  # power of two for backend="h2"
surface = self_affine_surface(Ns, H=0.8, rms=0.02)        # height field
solver  = hc.ContactSolver(grid_size=Ns, backend="h2", q=6)
res     = solver.solve(gap=-surface, p_nominal=0.05)      # rigid flat: gap0 = -height
contact = np.asarray(res.pressure) > 0                     # in-contact mask
print(res.contact_area)                                   # Ac/A  (also contact.mean())
```

`python example_rough_contact.py` writes `example_rough_contact.png`
(surface | pressure | contact-area panels).

---

## What Is Left To Do

- **Larger grids (Ns > 512)**: ✅ largely solved by the `backend="h2"` operator — O(N) memory (5.3 MiB at Ns=512), so Ns=1024+ is now cheap. (H-matrix path still memory-bound; see below.)
- **H2 follow-ups**: active-domain/masking sparsity (skip near/far work outside the contact zone via PCG active set); FFT backend for full-rectangle matvec (often simplest/fastest); rectangular grids (nx≠ny); leaf/q auto-tuning; PCG convergence + timing sweep of H2 at Ns≥1024.
- ~~FFT preconditioner speed~~ ✅ done (2026-07): half-spectrum transforms on pocketfft (default, BSD) or FFTW3 plans (opt-in, GPL; ~16%/~5% faster double/float end-to-end at Ns=4096), object-owned scratch/plans. The FFT is now a small share of the iteration.
- ~~FFT-convolution matvec backend (`backend="fft"`)~~ ✅ done (2026-07): exact zero-padded Love-kernel convolution per `doc/specs/2026-07-09-fft-convolution-backend-design.md`, plumbed into `ContactSolver` and `solve_nested`. **Measured outcome**: exactness is the headline (matches dense to ~1e-15 double / ~1.4e-7 float — no interpolation, no Gibbs); performance is modestly better than H2 at Ns ≤ 2048 (~1.5–1.6× matvec), ≈parity at Ns=4096 (measured under desktop load) — the padded transforms are bandwidth-bound, so the spec's flop-count 2–3× estimate did not materialise; H2 remains preferred for very large Ns.
- **FFTW-engine pruned transforms (many-plan decomposition)** — the pocketfft path skips structurally-zero forward lines and unread inverse lines in the operator's padded transforms; the FFTW path still runs full 2-D plans there.
- ~~Preallocate solve buffers~~ ✅ done (2026-07): `matvec_into`/`apply_into` + into-style functors in `solve_contact_impl`; the steady-state PCG loop makes no large allocations.
- **Single-precision storage (H-matrix)**: Halves H-matrix (ACA) memory (replace `double` with `float` in HBlock.D/U/V). Not yet implemented. (The *H2* solve path already has a `single_precision` mode — see the PCG section.)
- **ACA-GP improvement**: Current implementation gives only 5% rank reduction for the smooth Boussinesq kernel. The central-subset radius and random trial selection could be tuned further.
- Tangential/adhesive contact (Mindlin, JKR/DMT)
- Non-conforming surface meshes
- GPU ACA (cuBLAS)
