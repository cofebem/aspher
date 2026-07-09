# FFT-convolution matvec backend (`backend="fft"`) — design

Status: agreed in discussion 2026-07-09, not yet implemented.
Goal: a third operator backend that applies `u = S p` as an **exact**
zero-padded FFT convolution, 2–3× faster per PCG iteration than H2 at
Ns ≤ 8192, complementing (not replacing) H2, which remains the choice for
very large Ns (memory) and future non-uniform grids.

## 1. Mathematical core

On the uniform Ns×Ns grid the influence matrix is translation-invariant:
`S[i,j] = T(ix−jx, iy−jy)` where `T` is the **exact Love (1929) element
integral** — precisely the Ns×Ns lookup table `BoussinesqKernel` already
holds (`entry_offset(dx, dy)`, including the self term at dx=dy=0). So

    u(x) = Σ_y T(x−y) p(y)          — a 2-D discrete *aperiodic* convolution.

An aperiodic convolution of an Ns-support signal with an (2Ns−1)-support
kernel equals a *circular* convolution on a grid padded to ≥ 2Ns−1 per axis.
We use 2Ns (power of two, FFT-friendly):

1. Build once: embed `T(dx,dy)`, dx,dy ∈ [−(Ns−1), Ns−1], into a (2Ns)²
   array in wrap-around order (negative offsets stored at the top end),
   unused row/col Ns zeroed; r2c-FFT it → kernel spectrum `K̂`.
2. Per matvec: zero-pad `p` into a (2Ns)² buffer → r2c FFT → pointwise
   multiply by `K̂` → c2r inverse FFT → read the Ns×Ns block at the origin.

This is the classical Hockney/Stanley–Kato approach and is what Tamaas's
"dcfft" *intends* to be.

## 2. Why there is no Gibbs noise here (the key point vs Tamaas)

Tamaas's dcfft builds its influence coefficients **spectrally**: it samples
the continuous half-space symbol (∝ 1/|q|) at grid wavenumbers and inverse
transforms. Truncating that continuous, slowly decaying spectrum at the grid
Nyquist produces ringing in real space — the measured ±2–8% oscillations of
its near-field coefficients at r = 1–3h (see `doc/tamaas_findings.md`).

We go in the **opposite direction**: the kernel is defined in *real space*
by the exact closed-form Love integrals — the same numbers the dense matrix
and the H2 near field use — and the DFT of that finite, exact spatial kernel
is taken as `K̂`. The forward+inverse transform pair then reproduces the
circular convolution with those exact coefficients to machine roundoff.
There is no continuous spectrum to truncate, hence nothing to ring:
**the FFT backend is exact — it equals the dense matvec to ~1e-13 (double)**,
strictly better than H2's interpolation error (~1e-4 at q=4, ~3e-6 at q=6).

Corollary on the "Love near field": in this backend the near field is not a
separate correction to plug in. The padded kernel *is* the Love table for
every offset including the self term, so near and far field are exact by
construction in one pass. (The "near-field stencil" concept exists only in
the H2 operator, where the far field is interpolated and the 3×3 leaf
neighbourhood is patched with exact Love stencils.)

Free-space (non-periodic) BCs are also exact: zero padding to 2Ns removes
all wrap-around aliasing for an Ns-support pressure — this is *not* the
periodic solver; it matches ASPHER's existing non-periodic formulation.

## 3. Cost model

Per matvec: 2 real FFTs of (2Ns)² = 4N points + O(N) pointwise work.
Estimated 60–120 ms at Ns=4096 double (pocketfft/FFTW) vs ~300 ms for the
H2 matvec → whole PCG iteration 2–3× faster (two matvecs dominate).

Memory per precision (N = Ns²):
- kernel spectrum `K̂`: (2Ns)(Ns+1) complex ≈ 8N reals — or ≈ 4N if the
  even symmetry of T makes K̂ real (verify numerically; T(dx,dy) is even in
  both axes, so the DFT should be real up to roundoff — then store real).
- padded work buffer: (2Ns)² = 4N reals, plus its half spectrum ≈ 4N reals
  (reusable in place if the engine allows).
Total ≈ 12–16N reals scratch. At Ns=4096 double ≈ 1.3–2.1 GiB (fine);
at Ns=16384 double it does not fit — float only, and H2 stays preferable
there. This is why FFT complements H2 rather than replacing it.

## 4. Implementation plan

1. **Shared FFT engine header** (`src/fft_engine.hpp`, internal): extract the
   pocketfft/FFTW dispatch currently private to `fourier_precond.cpp`
   (engine selection `HMC_USE_FFTW`, OpenMP-slab parallelisation for
   pocketfft, plan handling for FFTW, `POCKETFFT_CACHE_SIZE`) so the
   preconditioner and the new operator share one implementation.
2. **`FFTOperator`** (`include/fft_operator.hpp`, `src/fft_operator.cpp`):
   mirrors `H2Operator`'s interface — `build()` (kernel embed + spectrum),
   `matvec_into(x, y)`, `matvec_single_into`, `info()`; object-owned padded
   scratch per precision; not thread-reentrant (same contract as H2).
3. **Bindings / solver plumbing**: `backend="fft"` in `ContactSolver`;
   a `backend` parameter on `hc.solve_nested` (default `"h2"` initially;
   consider auto-select `fft` for Ns ≤ 8192 once validated).
4. **Tests** (`tests/test_fft_operator.cpp`): matvec vs dense at Ns=64
   (assert rel L2 < 1e-12 double / < 1e-5 float); Hertz solve equality vs
   dense/H2 backends; add to CTest.
5. **Benchmarks**: extend the Ns=4096 rough-solve comparison (double/float,
   both FFT engines); update README performance table and CLAUDE.md
   validated numbers.

Validation gate before merging: FFT-backend solve must reproduce the
dense-backend solve bit-comparably (same iterations, pressure to roundoff)
at small Ns, and the H2-backend contact area at large Ns to ≤ 1e-6.

## 5. Risks / notes

- Padding/wrap-around indexing is the classic bug source → the dense
  comparison test at small Ns is the safety net.
- Kernel-spectrum realness: if roundoff leaves a tiny imaginary part,
  either keep complex K̂ (costs memory) or symmetrise before the FFT.
- The per-iteration FFTs are 4× the preconditioner's (padded grid): the
  OpenMP-slab pocketfft path and FFTW plans both apply as-is.
- Licensing unchanged: pocketfft default keeps binaries BSD; FFTW opt-in.
