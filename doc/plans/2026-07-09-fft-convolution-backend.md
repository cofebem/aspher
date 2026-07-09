# FFT-Convolution Matvec Backend (`backend="fft"`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A third operator backend that applies `u = S p` as an exact zero-padded FFT convolution of the Love-kernel table — equal to the dense matvec to roundoff, 2–3× faster per PCG iteration than H2 at Ns ≤ 8192.

**Architecture:** Extract the pocketfft/FFTW dispatch from `fourier_precond.cpp` into a shared internal engine header (`src/fft_engine.hpp`, class `fft::SquareR2C<S>`); build `FFTOperator` on top of it (kernel table embedded in wrap-around order on a 2Ns×2Ns padded grid, r2c'd once into a **real** spectrum, two FFTs per matvec); plumb `backend="fft"` through `ContactSolver` and `solve_nested`. Spec: `doc/specs/2026-07-09-fft-convolution-backend-design.md`.

**Tech Stack:** C++17, Eigen, OpenMP, bundled pocketfft (default) / FFTW3 (`-DASPHER_USE_FFTW=ON`), pybind11, CMake + CTest.

## Global Constraints

- Build env: `source /home/users02/vyastrebov/DISTR/miniconda3/etc/profile.d/conda.sh && conda activate fenicsx-env`; compiler `/usr/bin/g++`; `pybind11_DIR` borrowed from `dolfinx-010` (see CLAUDE.md build commands).
- Both FFT engines must keep working: default pocketfft build (`build/`) **and** `-DASPHER_USE_FFTW=ON` (`build-fftw/`). pocketfft may only be included from `src/*.cpp` (its include dir is PRIVATE to `aspher_core`); public headers in `include/` must not include pocketfft or fftw3.
- Layout contract everywhere: grid `G(ix, iy)` with ix contiguous (flat index `i = iy*Ns + ix`); half spectrum `C(kx, ky)` with kx ∈ [0, n/2] contiguous. Transforms unnormalised (fwd+inv multiplies by n²).
- FFTW plans bind to buffer pointers and `FFTW_MEASURE` (env `HMC_FFTW_MEASURE=1`) scribbles on the arrays at plan time → always **allocate, bind, then fill** buffers; never reallocate a bound buffer.
- Operators are not thread-reentrant (same contract as `H2Operator`/`FourierPreconditioner`); scratch is object-owned, sized lazily per precision, reused across calls — the PCG loop must stay allocation-free in steady state.
- Validation gate (spec §4): FFT-backend solve reproduces the dense-backend solve (same iteration count, pressure to roundoff) at small Ns, and the H2-backend contact area at Ns=1024 to ≤ 1e-6.
- Benchmarks on this noisy workstation: min of ~30 reps for matvec timings; C++-side timing where ms-level precision matters (pybind copy ≈ 5 ms at Ns=1024). Commit after every task. Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

**Deviation from spec noted:** the test file is `tests/test_fft.cpp` (not `test_fft_operator.cpp`) so the existing CMake `foreach(t kernel hmatrix contact h2 precond fft)` pattern names it `test_fft`, consistent with `test_h2.cpp`.

---

### Task 1: Shared FFT engine header (`src/fft_engine.hpp`) + `fourier_precond` refactor

Pure refactor — no behavior change. The existing tests (especially `test_precond`) are the safety net; they must pass in **both** engine builds before and after.

**Files:**
- Create: `src/fft_engine.hpp`
- Modify: `include/fourier_precond.hpp`
- Modify: `src/fourier_precond.cpp`

**Interfaces:**
- Consumes: nothing new (moves code currently private to `src/fourier_precond.cpp`).
- Produces: `hmc::fft::SquareR2C<S>` (S = double/float) with `void bind(int n, S* g, std::complex<S>* c)`, `bool bound() const`, `void fwd()` (g→c), `void inv()` (c→g, destroys c); non-copyable, destructor releases FFTW plans. Task 2's `FFTOperator` builds on exactly this.

- [ ] **Step 1: Baseline — confirm both engine builds are green before touching anything**

```bash
source /home/users02/vyastrebov/DISTR/miniconda3/etc/profile.d/conda.sh && conda activate fenicsx-env
cmake --build build -j$(nproc) && ctest --test-dir build --output-on-failure
# FFTW build dir (create once if absent):
cmake -S . -B build-fftw -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DASPHER_USE_FFTW=ON \
  -Dpybind11_DIR=$(conda run -n dolfinx-010 python -m pybind11 --cmakedir)
cmake --build build-fftw -j$(nproc) && ctest --test-dir build-fftw --output-on-failure
```

Expected: all 5 tests (kernel, hmatrix, contact, h2, precond) PASS in both dirs.

- [ ] **Step 2: Create `src/fft_engine.hpp`**

The engine code moves out of `src/fourier_precond.cpp` (its anonymous namespace, lines 24–198) nearly verbatim; the only new thing is the `SquareR2C` wrapper class that owns the plan/size state. Full file:

```cpp
#pragma once

// Internal 2-D square r2c/c2r FFT engine, shared by the |q| spectral
// preconditioner (fourier_precond.cpp) and the FFT-convolution operator
// (fft_operator.cpp). Not installed: include only from src/*.cpp — the
// bundled pocketfft include dir is PRIVATE to aspher_core.
//
// Engine selection at build time:
//   * default: bundled pocketfft (BSD-3-Clause). The 2-D transform is
//     decomposed into a single-axis r2c/c2r along the contiguous axis plus an
//     in-place c2c along the other, parallelised over slabs inside ONE OpenMP
//     region (each slab transformed with nthreads=1). This deliberately
//     avoids (a) pocketfft's multi-axis c2r path, which copies the whole
//     (const) input spectrum into an internal temporary per call, and
//     (b) pocketfft's private std::thread pool, which contends with OpenMP's
//     spin-waiting workers when the solver's parallel regions surround it.
//   * -DASPHER_USE_FFTW=ON (HMC_USE_FFTW): FFTW3 2-D plans bound to the
//     caller's buffers.
//
// Layout contract: the real grid G(ix, iy) has ix contiguous; the half
// spectrum C(kx, ky) has kx ∈ [0, n/2] contiguous. Transforms are
// unnormalised: fwd followed by inv multiplies by n². inv destroys the
// spectrum buffer.

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HMC_USE_FFTW
#include <fftw3.h>
#include <mutex>
#else
// cache pocketfft's internal plans (twiddle tables): without it every call
// rebuilds them, which dominates when the transform is applied per CG
// iteration on slabs from many threads
#define POCKETFFT_CACHE_SIZE 16
#include "pocketfft_hdronly.h"
#endif

namespace hmc {
namespace fft {

inline int nthreads_max() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

#ifndef HMC_USE_FFTW
inline int nthreads_in_region() {
#ifdef _OPENMP
    return omp_get_num_threads();
#else
    return 1;
#endif
}
inline int thread_id() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}
#endif

#ifdef HMC_USE_FFTW
// ── FFTW engine: 2-D r2c/c2r plans bound to caller-owned buffers ────────────

// FFTW's threaded init must run exactly once per process (per precision).
inline void init_fftw_threads() {
#ifdef HMC_FFTW_THREADS
    static std::once_flag once;
    std::call_once(once, [] {
        fftw_init_threads();
        fftwf_init_threads();
    });
#endif
}

// FFTW_MEASURE searches for the fastest algorithm at plan time (seconds of
// one-off cost at large Ns); the default FFTW_ESTIMATE plans instantly and is
// within tens of percent for power-of-two sizes. Opt in via HMC_FFTW_MEASURE.
inline unsigned plan_flags() {
    return std::getenv("HMC_FFTW_MEASURE") ? FFTW_MEASURE : FFTW_ESTIMATE;
}

#ifndef HMC_FFTW_THREADS
// Without the *_omp libraries the *_plan_with_nthreads symbols don't exist;
// stub them out so the shims below compile (FFTW then runs single-threaded).
inline void fftw_plan_with_nthreads(int) {}
inline void fftwf_plan_with_nthreads(int) {}
#endif

// Precision-dispatch shims over the fftw_/fftwf_ APIs. The grid G(ix, iy) is
// column-major with ix contiguous, i.e. row-major (iy, ix) — FFTW's (n0, n1)
// with the last (contiguous) axis ix, which r2c halves: the output C(kx, ky)
// (kx contiguous, kx ∈ [0, n/2]) shares its flat layout with the callers'
// spectra.
template <class S> struct FFTW;

template <> struct FFTW<double> {
    using Plan = fftw_plan;
    static Plan plan_r2c(int n, double* in, std::complex<double>* out) {
        fftw_plan_with_nthreads(nthreads_max());
        return fftw_plan_dft_r2c_2d(n, n, in,
                                    reinterpret_cast<fftw_complex*>(out),
                                    plan_flags());
    }
    static Plan plan_c2r(int n, std::complex<double>* in, double* out) {
        fftw_plan_with_nthreads(nthreads_max());
        return fftw_plan_dft_c2r_2d(n, n,
                                    reinterpret_cast<fftw_complex*>(in), out,
                                    plan_flags() | FFTW_DESTROY_INPUT);
    }
    static void execute(Plan p) { fftw_execute(p); }
    static void destroy(Plan p) { if (p) fftw_destroy_plan(p); }
};

template <> struct FFTW<float> {
    using Plan = fftwf_plan;
    static Plan plan_r2c(int n, float* in, std::complex<float>* out) {
        fftwf_plan_with_nthreads(nthreads_max());
        return fftwf_plan_dft_r2c_2d(n, n, in,
                                     reinterpret_cast<fftwf_complex*>(out),
                                     plan_flags());
    }
    static Plan plan_c2r(int n, std::complex<float>* in, float* out) {
        fftwf_plan_with_nthreads(nthreads_max());
        return fftwf_plan_dft_c2r_2d(n, n,
                                     reinterpret_cast<fftwf_complex*>(in), out,
                                     plan_flags() | FFTW_DESTROY_INPUT);
    }
    static void execute(Plan p) { fftwf_execute(p); }
    static void destroy(Plan p) { if (p) fftwf_destroy_plan(p); }
};

#else
// ── pocketfft engine (default, BSD): slab-parallel decomposed transforms ────

template <class S>
inline void pfft_fwd(int n, int nh, const S* in, std::complex<S>* out) {
    using CS = std::complex<S>;
    const pocketfft::stride_t sg{static_cast<std::ptrdiff_t>(sizeof(S)),
                                 static_cast<std::ptrdiff_t>(sizeof(S)) * n};
    const pocketfft::stride_t sc{static_cast<std::ptrdiff_t>(sizeof(CS)),
                                 static_cast<std::ptrdiff_t>(sizeof(CS)) * nh};
#pragma omp parallel
    {
        const int nt = nthreads_in_region();
        const int tid = thread_id();
        // pass 1: r2c along ix, slab of iy lines per thread
        const int y0 = static_cast<int>(std::int64_t(n) * tid / nt);
        const int y1 = static_cast<int>(std::int64_t(n) * (tid + 1) / nt);
        if (y1 > y0)
            pocketfft::r2c({static_cast<std::size_t>(n),
                            static_cast<std::size_t>(y1 - y0)},
                           sg, sc, pocketfft::shape_t{0}, true,
                           in + std::size_t(y0) * n,
                           out + std::size_t(y0) * nh, S(1), 1);
#pragma omp barrier
        // pass 2: c2c along iy, slab of kx rows per thread, in place
        const int k0 = static_cast<int>(std::int64_t(nh) * tid / nt);
        const int k1 = static_cast<int>(std::int64_t(nh) * (tid + 1) / nt);
        if (k1 > k0)
            pocketfft::c2c({static_cast<std::size_t>(k1 - k0),
                            static_cast<std::size_t>(n)},
                           sc, sc, pocketfft::shape_t{1}, true,
                           out + k0, out + k0, S(1), 1);
    }
}

template <class S>
inline void pfft_inv(int n, int nh, std::complex<S>* in, S* out) {
    using CS = std::complex<S>;
    const pocketfft::stride_t sg{static_cast<std::ptrdiff_t>(sizeof(S)),
                                 static_cast<std::ptrdiff_t>(sizeof(S)) * n};
    const pocketfft::stride_t sc{static_cast<std::ptrdiff_t>(sizeof(CS)),
                                 static_cast<std::ptrdiff_t>(sizeof(CS)) * nh};
#pragma omp parallel
    {
        const int nt = nthreads_in_region();
        const int tid = thread_id();
        // pass 1: inverse c2c along iy, slab of kx rows, in place
        const int k0 = static_cast<int>(std::int64_t(nh) * tid / nt);
        const int k1 = static_cast<int>(std::int64_t(nh) * (tid + 1) / nt);
        if (k1 > k0)
            pocketfft::c2c({static_cast<std::size_t>(k1 - k0),
                            static_cast<std::size_t>(n)},
                           sc, sc, pocketfft::shape_t{1}, false,
                           in + k0, in + k0, S(1), 1);
#pragma omp barrier
        // pass 2: c2r along ix, slab of iy lines per thread
        const int y0 = static_cast<int>(std::int64_t(n) * tid / nt);
        const int y1 = static_cast<int>(std::int64_t(n) * (tid + 1) / nt);
        if (y1 > y0)
            pocketfft::c2r({static_cast<std::size_t>(n),
                            static_cast<std::size_t>(y1 - y0)},
                           sc, sg, pocketfft::shape_t{0}, false,
                           in + std::size_t(y0) * nh,
                           out + std::size_t(y0) * n, S(1), 1);
    }
}
#endif

// Square n×n r2c/c2r transform pair bound to caller-owned buffers: g is the
// n×n real grid, c the (n/2+1)×n complex half spectrum. bind() must be called
// after the buffers are allocated (FFTW plans bind to the pointers, and
// FFTW_MEASURE scribbles on the arrays during planning) and the buffers must
// not be reallocated afterwards. Not thread-reentrant.
template <class S>
class SquareR2C {
public:
    SquareR2C() = default;
    ~SquareR2C() {
#ifdef HMC_USE_FFTW
        FFTW<S>::destroy(fwd_);
        FFTW<S>::destroy(inv_);
#endif
    }
    SquareR2C(const SquareR2C&) = delete;
    SquareR2C& operator=(const SquareR2C&) = delete;

    void bind(int n, S* g, std::complex<S>* c) {
        n_ = n;
        nh_ = n / 2 + 1;
        g_ = g;
        c_ = c;
#ifdef HMC_USE_FFTW
        init_fftw_threads();
        fwd_ = FFTW<S>::plan_r2c(n_, g_, c_);
        inv_ = FFTW<S>::plan_c2r(n_, c_, g_);
#endif
    }
    bool bound() const { return n_ > 0; }

    void fwd() { // g -> c
#ifdef HMC_USE_FFTW
        FFTW<S>::execute(fwd_);
#else
        pfft_fwd<S>(n_, nh_, g_, c_);
#endif
    }
    void inv() { // c -> g (destroys c)
#ifdef HMC_USE_FFTW
        FFTW<S>::execute(inv_);
#else
        pfft_inv<S>(n_, nh_, c_, g_);
#endif
    }

private:
    int n_ = 0, nh_ = 0;
    S* g_ = nullptr;
    std::complex<S>* c_ = nullptr;
#ifdef HMC_USE_FFTW
    typename FFTW<S>::Plan fwd_ = nullptr, inv_ = nullptr;
#endif
};

} // namespace fft
} // namespace hmc
```

- [ ] **Step 3: Rework `include/fourier_precond.hpp` to hold the engine via forward declaration**

Replace the fftw3 include and the plan members. The header must not include fftw3.h or pocketfft anymore. Diffs:

Replace

```cpp
#include <Eigen/Dense>

#ifdef HMC_USE_FFTW
#include <fftw3.h>
#endif

#include <cstdint>
#include <vector>

namespace hmc {
```

with

```cpp
#include <Eigen/Dense>

#include <cstdint>
#include <memory>
#include <vector>

namespace hmc {

namespace fft { template <class S> class SquareR2C; }
```

and replace the plan members at the bottom of the class

```cpp
#ifdef HMC_USE_FFTW
    // FFTW plans, created on first use per precision and bound to the scratch
    // pointers (so the scratch is never reallocated afterwards)
    mutable fftw_plan fwd_d_ = nullptr, inv_d_ = nullptr;
    mutable fftwf_plan fwd_f_ = nullptr, inv_f_ = nullptr;
#endif
```

with

```cpp
    // FFT engine (FFTW plans / pocketfft size bookkeeping), created on first
    // use per precision and bound to the scratch pointers (so the scratch is
    // never reallocated afterwards)
    mutable std::unique_ptr<fft::SquareR2C<double>> fft_d_;
    mutable std::unique_ptr<fft::SquareR2C<float>> fft_f_;
```

(`~FourierPreconditioner()` stays declared — unique_ptr to an incomplete type needs its destructor out of line.) Also update the class doc comment sentence "through one of two engines, selected at build time" to mention the shared `src/fft_engine.hpp`.

- [ ] **Step 4: Rework `src/fourier_precond.cpp` on top of the engine**

Delete everything the engine header now owns (the includes of `<complex>`, `<cstdlib>`, omp.h, mutex/pocketfft, and the whole anonymous namespace lines 24–198, plus the `NoPlan` block at lines 315–318). The full new file:

```cpp
#include "fourier_precond.hpp"

#include "fft_engine.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace hmc {

FourierPreconditioner::FourierPreconditioner(int Ns)
    : Ns_(Ns), nh_(Ns / 2 + 1), wh_(Ns / 2 + 1, Ns) {
    // integer wavenumbers k = i (i < Ns/2) else i - Ns; symbol |k|, DC zeroed.
    // Absolute scale is irrelevant (cancels in CG), so 2π/L is dropped; the
    // unnormalised r2c+c2r round trip multiplies by Ns², so 1/Ns² is folded in
    // here (both engines run unnormalised). Only kx ∈ [0, Ns/2] is stored
    // (half spectrum; |kof(kx)| = kx on that range). The symbol is stored in
    // float: |k|/Ns² stays well inside float range and the float path
    // multiplies with it directly.
    const float norm = 1.0f / (static_cast<float>(Ns) * static_cast<float>(Ns));
    auto kof = [Ns](int i) { return (i < Ns / 2) ? i : i - Ns; };
    for (int ky = 0; ky < Ns; ++ky)
        for (int kx = 0; kx < nh_; ++kx)
            wh_(kx, ky) = norm * std::hypot(static_cast<float>(kx),
                                            static_cast<float>(kof(ky)));
    wh_(0, 0) = 0.0f;
}

FourierPreconditioner::~FourierPreconditioner() = default;

// Scalar-templated preconditioner apply. The grid field G is laid out
// G(ix, iy) with ix contiguous:
//   1. scatter the contact-masked residual into G
//   2. 2-D r2c: C(kx, ky), kx ∈ [0, Ns/2]  (half spectrum)
//   3. multiply by the |k|/Ns² symbol (real and even → consistent with the
//      omitted Hermitian half)
//   4. 2-D c2r back into G
//   5. gather on the contact set, remove the contact mean
// The mask passes are OpenMP-parallel; the transforms are engine-threaded.
// G and C are object-owned state reused across iterations, so the per-call
// footprint is one N-vector (the result).
template <class S>
static void
apply_t(int Ns, int nh, const Eigen::MatrixXf& wh,
        Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& G,
        Eigen::Matrix<std::complex<S>, Eigen::Dynamic, Eigen::Dynamic>& C,
        std::unique_ptr<fft::SquareR2C<S>>& eng,
        const Eigen::Matrix<S, Eigen::Dynamic, 1>& g,
        const std::vector<std::uint8_t>& contact,
        Eigen::Matrix<S, Eigen::Dynamic, 1>& z) {
    const int N = Ns * Ns;
    if (!eng) { // first use in this precision: allocate scratch, bind plans
        G.resize(Ns, Ns);
        C.resize(nh, Ns);
        eng = std::make_unique<fft::SquareR2C<S>>();
        eng->bind(Ns, G.data(), C.data());
    }
    if (z.size() != N) z.resize(N);

    // scatter the masked residual; column iy of G is the contiguous g-range
    // [iy*Ns, (iy+1)*Ns) (i = iy*Ns + ix).
#pragma omp parallel for schedule(static)
    for (int iy = 0; iy < Ns; ++iy) {
        S* col = G.data() + static_cast<std::ptrdiff_t>(iy) * Ns;
        const S* gi = g.data() + static_cast<std::ptrdiff_t>(iy) * Ns;
        const std::uint8_t* ci = contact.data() + static_cast<std::ptrdiff_t>(iy) * Ns;
        for (int ix = 0; ix < Ns; ++ix) col[ix] = ci[ix] ? gi[ix] : S(0);
    }

    eng->fwd();

    // real, even symbol on the half spectrum; wh and C share flat layout
#pragma omp parallel for schedule(static)
    for (int ky = 0; ky < Ns; ++ky) {
        std::complex<S>* c = C.data() + static_cast<std::ptrdiff_t>(ky) * nh;
        const float* w = wh.data() + static_cast<std::ptrdiff_t>(ky) * nh;
        for (int kx = 0; kx < nh; ++kx) c[kx] *= static_cast<S>(w[kx]);
    }

    eng->inv(); // consumes C, writes G

    // gather on the contact set, then remove the contact mean
    double zsum = 0.0;
    long nc = 0;
#pragma omp parallel for schedule(static) reduction(+ : zsum, nc)
    for (int iy = 0; iy < Ns; ++iy) {
        const S* col = G.data() + static_cast<std::ptrdiff_t>(iy) * Ns;
        const std::uint8_t* ci = contact.data() + static_cast<std::ptrdiff_t>(iy) * Ns;
        S* zi = z.data() + static_cast<std::ptrdiff_t>(iy) * Ns;
        for (int ix = 0; ix < Ns; ++ix) {
            if (ci[ix]) { zi[ix] = col[ix]; zsum += col[ix]; ++nc; }
            else zi[ix] = S(0);
        }
    }
    if (nc) {
        const S zmean = static_cast<S>(zsum / static_cast<double>(nc));
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i)
            if (contact[i]) z(i) -= zmean;
    }
}

Eigen::VectorXd
FourierPreconditioner::apply(const Eigen::VectorXd& g,
                             const std::vector<std::uint8_t>& contact) const {
    Eigen::VectorXd z;
    apply_into(g, contact, z);
    return z;
}

void FourierPreconditioner::apply_into(const Eigen::VectorXd& g,
                                       const std::vector<std::uint8_t>& contact,
                                       Eigen::VectorXd& z) const {
    apply_t<double>(Ns_, nh_, wh_, Gd_, Cd_, fft_d_, g, contact, z);
}

Eigen::VectorXf
FourierPreconditioner::apply_single(const Eigen::VectorXf& g,
                                    const std::vector<std::uint8_t>& contact) const {
    Eigen::VectorXf z;
    apply_single_into(g, contact, z);
    return z;
}

void FourierPreconditioner::apply_single_into(
    const Eigen::VectorXf& g, const std::vector<std::uint8_t>& contact,
    Eigen::VectorXf& z) const {
    apply_t<float>(Ns_, nh_, wh_, Gf_, Cf_, fft_f_, g, contact, z);
}

} // namespace hmc
```

- [ ] **Step 5: Rebuild and run all tests, both engines**

```bash
cmake --build build -j$(nproc) && ctest --test-dir build --output-on-failure
cmake --build build-fftw -j$(nproc) && ctest --test-dir build-fftw --output-on-failure
```

Expected: all 5 tests PASS in both dirs (identical numbers to Step 1 — this is a pure refactor).

- [ ] **Step 6: Commit**

```bash
git add src/fft_engine.hpp include/fourier_precond.hpp src/fourier_precond.cpp
git commit -m "refactor(fft): extract shared square r2c/c2r engine to src/fft_engine.hpp

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `FFTOperator` — exact zero-padded Love-kernel convolution (TDD)

**Files:**
- Create: `tests/test_fft.cpp`
- Create: `include/fft_operator.hpp`
- Create: `src/fft_operator.cpp`
- Modify: `CMakeLists.txt` (add `src/fft_operator.cpp` to `aspher_core`; add `fft` to the test foreach)

**Interfaces:**
- Consumes: `fft::SquareR2C<S>` from Task 1; `BoussinesqKernel::entry_offset(dx, dy)`, `grid_size()`; `hmc::solve_contact` (`include/contact_solver.hpp`).
- Produces: `hmc::FFTOperator` with `explicit FFTOperator(const BoussinesqKernel&)`, `void build()`, `Eigen::VectorXd matvec(const Eigen::VectorXd&) const`, `void matvec_into(const Eigen::VectorXd&, Eigen::VectorXd&) const`, `Eigen::VectorXf matvec_single(const Eigen::VectorXf&) const`, `void matvec_single_into(const Eigen::VectorXf&, Eigen::VectorXf&) const`, `void build_single_caches() const`, `FFTInfo info() const`, `void print_statistics() const`. Task 3 plugs these into the bindings and `nested_solve.cpp`.

- [ ] **Step 1: Write the failing test `tests/test_fft.cpp`**

```cpp
#include "boussinesq_kernel.hpp"
#include "contact_solver.hpp"
#include "fft_operator.hpp"

#include <cmath>
#include <cstdio>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAILED: %s (line %d)\n", #cond, __LINE__);        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

// The FFT backend is exact: it must match the dense matvec to roundoff
// (double) — strictly better than H2's interpolation error. Ns=24 guards the
// wrap-around indexing on a non-power-of-two grid (the operator itself has no
// power-of-two constraint).
static int test_matvec_exact() {
    for (int Ns : {24, 32, 64}) {
        hmc::BoussinesqKernel k(Ns, 1.0, 1.0);
        const Eigen::MatrixXd S = k.assemble_dense();
        hmc::FFTOperator A(k);
        A.build();
        A.print_statistics();
        Eigen::VectorXd x = Eigen::VectorXd::Random(Ns * Ns);
        const Eigen::VectorXd ref = S * x;
        const Eigen::VectorXd y = A.matvec(x);
        const double err = (y - ref).norm() / ref.norm();
        const Eigen::VectorXf yf = A.matvec_single(x.cast<float>());
        const double errf = (yf.cast<double>() - ref).norm() / ref.norm();
        std::printf("FFT matvec rel err (Ns=%d): double %.3e  float %.3e\n",
                    Ns, err, errf);
        CHECK(err < 1e-12);
        CHECK(errf < 1e-5);
    }
    return 0;
}

// matvec_into must reuse the object-owned scratch across calls and give the
// same answer on the second call (stale-padding guard: the previous inverse
// transform wrote the whole padded grid).
static int test_repeat_calls() {
    const int Ns = 32;
    hmc::BoussinesqKernel k(Ns, 1.0, 1.0);
    hmc::FFTOperator A(k);
    A.build();
    Eigen::VectorXd x = Eigen::VectorXd::Random(Ns * Ns);
    Eigen::VectorXd y1, y2;
    A.matvec_into(x, y1);
    Eigen::VectorXd other = Eigen::VectorXd::Random(Ns * Ns);
    A.matvec_into(other, y2); // scribble the scratch with a different field
    A.matvec_into(x, y2);
    CHECK((y1 - y2).norm() == 0.0);
    return 0;
}

// Validation gate (spec §4): the FFT-backend Hertz solve must reproduce the
// dense-backend solve — same iteration count, pressure to roundoff.
static int test_hertz_solve() {
    const int Ns = 64;
    const double R = 0.5, p_bar = 0.05, h = 1.0 / Ns;
    hmc::BoussinesqKernel k(Ns, 1.0, 1.0);
    Eigen::VectorXd g0(Ns * Ns);
    for (int iy = 0; iy < Ns; ++iy)
        for (int ix = 0; ix < Ns; ++ix) {
            const double x = (ix + 0.5) * h - 0.5, y = (iy + 0.5) * h - 0.5;
            g0(iy * Ns + ix) = (x * x + y * y) / (2.0 * R);
        }

    const Eigen::MatrixXd S = k.assemble_dense();
    auto mv_dense = [&S](const Eigen::VectorXd& v) -> Eigen::VectorXd {
        return S * v;
    };
    hmc::FFTOperator A(k);
    A.build();
    auto mv_fft = [&A](const Eigen::VectorXd& v) { return A.matvec(v); };

    const auto rd = hmc::solve_contact(mv_dense, g0, p_bar, 1e-8, 2000);
    const auto rf = hmc::solve_contact(mv_fft, g0, p_bar, 1e-8, 2000);
    CHECK(rd.converged);
    CHECK(rf.converged);
    std::printf("Hertz dense: %d it, Ac/A %.6f | fft: %d it, Ac/A %.6f\n",
                rd.iterations, rd.contact_fraction, rf.iterations,
                rf.contact_fraction);
    CHECK(rd.iterations == rf.iterations);
    CHECK(rd.contact_fraction == rf.contact_fraction);
    // PCG stops when the complementarity error crosses tol, so two operators
    // identical to machine roundoff still yield stopping iterates that agree
    // only to O(tol) (measured ~6e-8 at tol=1e-8; dp scales linearly with tol,
    // and even two dense-only matvec variants differ by more). 1e-6 still
    // discriminates: any *approximate* operator (H2 q=6 ~3e-6, dcfft Gibbs
    // %-level) fails it. [Calibrated 2026-07-09 with user approval.]
    const double dp = (rf.pressure - rd.pressure).norm() / rd.pressure.norm();
    std::printf("Hertz pressure rel diff (fft vs dense): %.3e\n", dp);
    CHECK(dp < 1e-6);
    return 0;
}

int main() {
    if (int rc = test_matvec_exact()) return rc;
    if (int rc = test_repeat_calls()) return rc;
    if (int rc = test_hertz_solve()) return rc;
    std::printf("test_fft: all passed\n");
    return 0;
}
```

- [ ] **Step 2: Wire it into CMake and verify it fails to build**

In `CMakeLists.txt`, add the source to the core library:

```cmake
add_library(aspher_core STATIC
  src/boussinesq_kernel.cpp
  src/cluster_tree.cpp
  src/hmatrix.cpp
  src/contact_solver.cpp
  src/cheb_basis.cpp
  src/uniform_quadtree.cpp
  src/h2_operator.cpp
  src/fourier_precond.cpp
  src/fft_operator.cpp
  src/nested_solve.cpp
)
```

and extend the test list:

```cmake
  foreach(t kernel hmatrix contact h2 precond fft)
```

Run: `cmake --build build -j$(nproc)`
Expected: FAIL — `src/fft_operator.cpp`/`fft_operator.hpp` do not exist yet.

- [ ] **Step 3: Write `include/fft_operator.hpp`**

```cpp
#pragma once

#include "boussinesq_kernel.hpp"

#include <Eigen/Dense>
#include <cstdint>
#include <memory>

namespace hmc {

namespace fft { template <class S> class SquareR2C; }

struct FFTInfo {
    int N = 0, Ns = 0, M = 0; // M = 2 Ns padded grid side
    std::int64_t bytes_spectrum = 0; // stored real kernel spectrum (per precision)
    std::int64_t bytes_scratch = 0;  // padded grid + half-spectrum work buffers
    std::int64_t bytes_total = 0;
};

// Exact FFT-convolution operator for the translation-invariant Boussinesq
// influence matrix on the uniform Ns x Ns grid: u = S p is the aperiodic
// convolution of p with the Love (1929) element table T(dx, dy) — the same
// numbers the dense matrix uses — evaluated as a circular convolution on a
// (2Ns)² zero-padded grid (Hockney). The kernel table is embedded once in
// wrap-around order and r2c-transformed; T is even in both axes, so its
// spectrum is real (verified at build) and stored as such, with the
// unnormalised fwd+inv round-trip scale 1/(2Ns)² folded in. Per matvec:
// zero-pad scatter -> r2c -> pointwise multiply -> c2r -> gather. Equals the
// dense matvec to roundoff (no interpolation, no spectral truncation — the
// near field IS the far field here), unlike the H2 operator's ~1e-4 (q=4)
// interpolation error. Free-space (non-periodic) BCs are exact: the padding
// removes all wrap-around aliasing for an Ns-support pressure.
//
// O(N log N) matvec, ~10 N reals of double scratch. Padded scratch and the
// FFT engine state are object-owned, sized lazily per precision, and reused
// across calls: a single FFTOperator must not be applied from two threads
// concurrently (same contract as H2Operator). Instances are non-copyable.
class FFTOperator {
public:
    explicit FFTOperator(const BoussinesqKernel& kernel);
    ~FFTOperator();
    FFTOperator(const FFTOperator&) = delete;
    FFTOperator& operator=(const FFTOperator&) = delete;

    void build();

    // u = S x, with x and u in natural flat order (global = iy*Ns + ix).
    Eigen::VectorXd matvec(const Eigen::VectorXd& x) const;
    // Allocation-free variant: writes into y (resized if needed).
    void matvec_into(const Eigen::VectorXd& x, Eigen::VectorXd& y) const;

    // Single-precision matvec: float copy of the kernel spectrum (built
    // lazily, idempotent) and float transforms, halving the O(N) working set.
    Eigen::VectorXf matvec_single(const Eigen::VectorXf& x) const;
    void matvec_single_into(const Eigen::VectorXf& x, Eigen::VectorXf& y) const;
    void build_single_caches() const;

    FFTInfo info() const;
    void print_statistics() const;

private:
    const BoussinesqKernel* kernel_;
    int Ns_, M_, nh_; // M_ = 2 Ns_, nh_ = M_/2 + 1 = Ns_ + 1

    // (nh x M) real kernel spectrum with the 1/M² round-trip scale folded in
    Eigen::MatrixXd Kh_;
    mutable bool have_single_ = false;
    mutable Eigen::MatrixXf Kh_f_;

    // padded scratch per precision, sized lazily on first matvec: G (M x M
    // real, laid out G(ix, iy)) and C (nh x M complex half spectrum)
    mutable Eigen::MatrixXd Gd_;
    mutable Eigen::MatrixXcd Cd_;
    mutable Eigen::MatrixXf Gf_;
    mutable Eigen::MatrixXcf Cf_;
    mutable std::unique_ptr<fft::SquareR2C<double>> fft_d_;
    mutable std::unique_ptr<fft::SquareR2C<float>> fft_f_;

    template <class S>
    void matvec_impl(const Eigen::Matrix<S, Eigen::Dynamic, 1>& x,
                     Eigen::Matrix<S, Eigen::Dynamic, 1>& y,
                     const Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& Kh,
                     Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& G,
                     Eigen::Matrix<std::complex<S>, Eigen::Dynamic,
                                   Eigen::Dynamic>& C,
                     std::unique_ptr<fft::SquareR2C<S>>& eng) const;
};

} // namespace hmc
```

- [ ] **Step 4: Write `src/fft_operator.cpp`**

```cpp
#include "fft_operator.hpp"

#include "fft_engine.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>

namespace hmc {

FFTOperator::FFTOperator(const BoussinesqKernel& kernel)
    : kernel_(&kernel), Ns_(kernel.grid_size()), M_(2 * kernel.grid_size()),
      nh_(kernel.grid_size() + 1) {}

FFTOperator::~FFTOperator() = default;

void FFTOperator::build() {
    // Embed the Love table T(dx, dy), dx, dy ∈ [-(Ns-1), Ns-1], into the
    // (M x M) padded grid in wrap-around order: offset d is stored at index d
    // (d >= 0) or M + d (d < 0); the unused row/column index Ns stays zero.
    // The embedding is even under index negation mod M (T is even in both
    // axes and row/col Ns is zero), so the DFT is real up to roundoff —
    // checked below, then stored as the real (nh x M) half spectrum with the
    // unnormalised fwd+inv round-trip scale 1/M² folded in.
    Eigen::MatrixXd K(M_, M_);
    Eigen::MatrixXcd Kc(nh_, M_);
    fft::SquareR2C<double> plan;
    // bind before filling: FFTW_MEASURE scribbles on the arrays at plan time
    plan.bind(M_, K.data(), Kc.data());

#pragma omp parallel for schedule(static)
    for (int iy = 0; iy < M_; ++iy) {
        double* col = K.data() + static_cast<std::ptrdiff_t>(iy) * M_;
        if (iy == Ns_) {
            for (int ix = 0; ix < M_; ++ix) col[ix] = 0.0;
            continue;
        }
        const int dy = (iy < Ns_) ? iy : iy - M_;
        for (int ix = 0; ix < M_; ++ix) {
            const int dx = (ix < Ns_) ? ix : ix - M_;
            col[ix] = (ix == Ns_) ? 0.0 : kernel_->entry_offset(dx, dy);
        }
    }

    plan.fwd();

    double max_im = 0.0, max_re = 0.0;
#pragma omp parallel for schedule(static) reduction(max : max_im, max_re)
    for (int ky = 0; ky < M_; ++ky) {
        const std::complex<double>* c =
            Kc.data() + static_cast<std::ptrdiff_t>(ky) * nh_;
        for (int kx = 0; kx < nh_; ++kx) {
            max_im = std::max(max_im, std::abs(c[kx].imag()));
            max_re = std::max(max_re, std::abs(c[kx].real()));
        }
    }
    if (max_im > 1e-10 * max_re)
        throw std::runtime_error(
            "FFTOperator: kernel spectrum has a non-negligible imaginary "
            "part; wrap-around embedding is broken");

    const double norm = 1.0 / (static_cast<double>(M_) * static_cast<double>(M_));
    Kh_.resize(nh_, M_);
#pragma omp parallel for schedule(static)
    for (int ky = 0; ky < M_; ++ky)
        for (int kx = 0; kx < nh_; ++kx)
            Kh_(kx, ky) = norm * Kc(kx, ky).real();
}

void FFTOperator::build_single_caches() const {
    if (have_single_) return;
    Kh_f_ = Kh_.cast<float>();
    have_single_ = true;
}

// Zero-pad scatter -> fwd FFT -> pointwise kernel spectrum -> inv FFT ->
// gather the Ns x Ns block at the origin. The previous call's inverse
// transform wrote the whole padded grid, so the scatter refreshes every
// entry (no stale padding).
template <class S>
void FFTOperator::matvec_impl(
    const Eigen::Matrix<S, Eigen::Dynamic, 1>& x,
    Eigen::Matrix<S, Eigen::Dynamic, 1>& y,
    const Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& Kh,
    Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& G,
    Eigen::Matrix<std::complex<S>, Eigen::Dynamic, Eigen::Dynamic>& C,
    std::unique_ptr<fft::SquareR2C<S>>& eng) const {
    const int N = Ns_ * Ns_;
    if (static_cast<int>(x.size()) != N)
        throw std::invalid_argument("FFTOperator::matvec: x size != Ns*Ns");
    if (!eng) { // first use in this precision: allocate scratch, bind plans
        G.resize(M_, M_);
        C.resize(nh_, M_);
        eng = std::make_unique<fft::SquareR2C<S>>();
        eng->bind(M_, G.data(), C.data());
    }
    if (y.size() != N) y.resize(N);

    // zero-pad scatter; column iy of G gets x-range [iy*Ns, (iy+1)*Ns)
#pragma omp parallel for schedule(static)
    for (int iy = 0; iy < M_; ++iy) {
        S* col = G.data() + static_cast<std::ptrdiff_t>(iy) * M_;
        if (iy < Ns_) {
            const S* xi = x.data() + static_cast<std::ptrdiff_t>(iy) * Ns_;
            for (int ix = 0; ix < Ns_; ++ix) col[ix] = xi[ix];
            for (int ix = Ns_; ix < M_; ++ix) col[ix] = S(0);
        } else {
            for (int ix = 0; ix < M_; ++ix) col[ix] = S(0);
        }
    }

    eng->fwd();

    // real, even kernel spectrum on the half spectrum; Kh and C share layout
#pragma omp parallel for schedule(static)
    for (int ky = 0; ky < M_; ++ky) {
        std::complex<S>* c = C.data() + static_cast<std::ptrdiff_t>(ky) * nh_;
        const S* w = Kh.data() + static_cast<std::ptrdiff_t>(ky) * nh_;
        for (int kx = 0; kx < nh_; ++kx) c[kx] *= w[kx];
    }

    eng->inv(); // consumes C, writes G

    // gather the Ns x Ns block at the origin
#pragma omp parallel for schedule(static)
    for (int iy = 0; iy < Ns_; ++iy) {
        const S* col = G.data() + static_cast<std::ptrdiff_t>(iy) * M_;
        S* yi = y.data() + static_cast<std::ptrdiff_t>(iy) * Ns_;
        for (int ix = 0; ix < Ns_; ++ix) yi[ix] = col[ix];
    }
}

Eigen::VectorXd FFTOperator::matvec(const Eigen::VectorXd& x) const {
    Eigen::VectorXd y;
    matvec_into(x, y);
    return y;
}

void FFTOperator::matvec_into(const Eigen::VectorXd& x,
                              Eigen::VectorXd& y) const {
    if (Kh_.size() == 0)
        throw std::logic_error("FFTOperator::matvec: build() not called");
    matvec_impl<double>(x, y, Kh_, Gd_, Cd_, fft_d_);
}

Eigen::VectorXf FFTOperator::matvec_single(const Eigen::VectorXf& x) const {
    Eigen::VectorXf y;
    matvec_single_into(x, y);
    return y;
}

void FFTOperator::matvec_single_into(const Eigen::VectorXf& x,
                                     Eigen::VectorXf& y) const {
    build_single_caches();
    matvec_impl<float>(x, y, Kh_f_, Gf_, Cf_, fft_f_);
}

FFTInfo FFTOperator::info() const {
    FFTInfo s;
    s.Ns = Ns_;
    s.N = Ns_ * Ns_;
    s.M = M_;
    s.bytes_spectrum =
        static_cast<std::int64_t>(sizeof(double)) * Kh_.size() +
        static_cast<std::int64_t>(sizeof(float)) * Kh_f_.size();
    s.bytes_scratch =
        static_cast<std::int64_t>(sizeof(double)) * Gd_.size() +
        static_cast<std::int64_t>(sizeof(std::complex<double>)) * Cd_.size() +
        static_cast<std::int64_t>(sizeof(float)) * Gf_.size() +
        static_cast<std::int64_t>(sizeof(std::complex<float>)) * Cf_.size();
    s.bytes_total = s.bytes_spectrum + s.bytes_scratch;
    return s;
}

void FFTOperator::print_statistics() const {
    const FFTInfo s = info();
    std::printf("FFTOperator: Ns=%d (N=%d), padded %dx%d, spectrum %.1f MiB, "
                "scratch %.1f MiB, total %.1f MiB\n",
                s.Ns, s.N, s.M, s.M, s.bytes_spectrum / 1048576.0,
                s.bytes_scratch / 1048576.0, s.bytes_total / 1048576.0);
}

} // namespace hmc
```

- [ ] **Step 5: Build and run the new test — both engines**

```bash
cmake --build build -j$(nproc) && build/test_fft
cmake --build build-fftw -j$(nproc) && build-fftw/test_fft
```

Expected: PASS with printed rel errors ~1e-14 (double) and ~1e-7 (float); Hertz iterations equal between dense and fft; pressure rel diff ≲ 1e-7 (O(tol) stopping-point agreement; see the comment in the test). If `rd.iterations == rf.iterations` or `contact_fraction` equality fails, do NOT loosen the assertion — this is the spec's validation gate; use superpowers:systematic-debugging (prime suspects: wrap-around embedding off by one, row/col Ns not zeroed, normalisation).

- [ ] **Step 6: Run the full suite, both engines**

```bash
ctest --test-dir build --output-on-failure
ctest --test-dir build-fftw --output-on-failure
```

Expected: all 6 tests PASS in both dirs.

- [ ] **Step 7: Commit**

```bash
git add include/fft_operator.hpp src/fft_operator.cpp tests/test_fft.cpp CMakeLists.txt
git commit -m "feat(fft): exact zero-padded Love-kernel FFT-convolution operator

Matches the dense matvec to roundoff (real even kernel spectrum, half-
spectrum storage, object-owned padded scratch, double+float paths).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Solver plumbing — `backend="fft"` in `ContactSolver`, `backend` in `solve_nested`

**Files:**
- Modify: `python/bindings.cpp`
- Modify: `include/nested_solve.hpp`
- Modify: `src/nested_solve.cpp`

**Interfaces:**
- Consumes: `FFTOperator` (Task 2 signatures), existing `MatVecIntoT<Real>` functors from `include/contact_solver.hpp`.
- Produces: Python `hc.ContactSolver(grid_size=…, backend="fft")` (same `matvec`/`solve`/`hmatrix_info` API) and `hc.solve_nested(…, backend="fft")`; C++ `NestedParams::backend` (`std::string`, default `"h2"`).

- [ ] **Step 1: Add `backend` to `NestedParams` in `include/nested_solve.hpp`**

Add as the LAST member (bindings brace-initialize the struct positionally; keep existing positions valid):

```cpp
struct NestedParams {
    int coarsest = 64;        // coarsest grid side (power of two, divides Ns)
    int q = 6;                // H2 Chebyshev order on every level
    int leaf_side = 8;        // H2 leaf side on every level
    bool precond = true;      // |q| spectral preconditioner per level
    double coarse_tol = 1e-4; // cascadic: looser tolerance on coarse levels
    bool single_precision = false; // run each level's solve in float (~half RAM)
    bool light_result = false;     // skip displacement/gap in the result (~2 N arrays)
    std::string backend = "h2";    // per-level operator: "h2" or "fft"
};
```

Add `#include <string>` to the header.

- [ ] **Step 2: Per-level operator selection in `src/nested_solve.cpp`**

Add includes `#include "fft_operator.hpp"` and `#include <memory>`. After the `levels.back() != Ns` check, fail fast:

```cpp
    if (np.backend != "h2" && np.backend != "fft")
        throw std::invalid_argument(
            "solve_contact_nested: backend must be 'h2' or 'fft'");
```

Replace the level-loop operator setup (currently the `BoussinesqKernel kernel…; H2Operator op…; MatVecIntoT<double> mv…` block and, below, the `op.build_single_caches(); MatVecIntoT<float> mvf…` block) with:

```cpp
    for (std::size_t li = 0; li < levels.size(); ++li) {
        const int n = levels[li];
        BoussinesqKernel kernel(n, L, E_star);
        std::unique_ptr<H2Operator> h2;
        std::unique_ptr<FFTOperator> fop;
        MatVecIntoT<double> mv;
        if (np.backend == "fft") {
            fop = std::make_unique<FFTOperator>(kernel);
            fop->build();
            mv = [&fop](const Eigen::VectorXd& v, Eigen::VectorXd& out) {
                fop->matvec_into(v, out);
            };
        } else {
            h2 = std::make_unique<H2Operator>(kernel,
                                              H2Params{np.leaf_side, np.q, 1});
            h2->build();
            mv = [&h2](const Eigen::VectorXd& v, Eigen::VectorXd& out) {
                h2->matvec_into(v, out);
            };
        }
```

and in the single-precision branch:

```cpp
        if (np.single_precision) {
            MatVecIntoT<float> mvf;
            if (fop) {
                fop->build_single_caches();
                mvf = [&fop](const Eigen::VectorXf& v, Eigen::VectorXf& out) {
                    fop->matvec_single_into(v, out);
                };
            } else {
                h2->build_single_caches();
                mvf = [&h2](const Eigen::VectorXf& v, Eigen::VectorXf& out) {
                    h2->matvec_single_into(v, out);
                };
            }
```

(rest of the branch unchanged). The double branch keeps using `mv`.

- [ ] **Step 3: Bindings — `backend="fft"` in `PyContactSolver` and `solve_nested`**

In `python/bindings.cpp`:

a. Add `#include "fft_operator.hpp"` to the project includes.

b. Constructor: after the `"h2"` branch add

```cpp
        } else if (backend_ == "fft") {
            fft_ = std::make_unique<hmc::FFTOperator>(kernel_);
            fft_->build();
```

and update the error message to `" (expected dense, hmatrix, h2, or fft)"`.

c. `apply()`:

```cpp
    Eigen::VectorXd apply(const Eigen::VectorXd& p) const {
        if (backend_ == "h2") return h2_->matvec(p);
        if (backend_ == "fft") return fft_->matvec(p);
        if (backend_ == "hmatrix") return hmat_->matvec(p);
        return dense_ * p;
    }
```

d. Member: `std::unique_ptr<hmc::FFTOperator> fft_;` next to `h2_`.

e. `hmatrix_info()`: add before the `"hmatrix"` handling

```cpp
        if (backend_ == "fft") {
            fft_->print_statistics();
            const auto s = fft_->info();
            d["backend"] = "fft";
            d["n"] = s.N;
            d["padded_side"] = s.M;
            d["bytes"] = s.bytes_total;
            d["bytes_spectrum"] = static_cast<long long>(s.bytes_spectrum);
            d["bytes_scratch"] = static_cast<long long>(s.bytes_scratch);
            d["compression"] =
                double(s.bytes_total) / (8.0 * double(s.N) * double(s.N));
            return d;
        }
```

f. `py_solve_nested`: add a `const std::string& backend` parameter (after `light_result`), set it on the params —

```cpp
    hmc::NestedParams np{coarsest, q, leaf_side, precond, coarse_tol,
                         single_precision, light_result, backend};
```

— and register `py::arg("backend") = "h2"` after `py::arg("light_result") = false` in the `m.def("solve_nested", …)` call. Mention `backend` in the docstring: `"backend='h2' (O(N) memory) or 'fft' (exact convolution, fastest at Ns<=8192)"`.

- [ ] **Step 4: Rebuild (dev build makes the Python module) and run C++ tests**

```bash
cmake --build build -j$(nproc) && ctest --test-dir build --output-on-failure
```

Expected: all 6 tests PASS (`python/aspher.cpython-312-*.so` refreshed).

- [ ] **Step 5: Python gate — exactness, solve equality, nested-area match at Ns=1024**

```bash
cd /home/vyastrebov/WORK/PROJECTS/Hcontact && conda run -n fenicsx-env python - <<'EOF'
import sys; sys.path.insert(0, 'python')
import numpy as np, aspher as hc

# 1. matvec exactness vs dense backend
Ns = 64
dense = hc.ContactSolver(grid_size=Ns, backend="dense")
fftb  = hc.ContactSolver(grid_size=Ns, backend="fft")
p = np.random.default_rng(0).random(Ns * Ns)
ud, uf = dense.matvec(p), fftb.matvec(p)
err = np.linalg.norm(uf - ud) / np.linalg.norm(ud)
print("matvec rel err (fft vs dense):", err)
assert err < 1e-12
fftb.hmatrix_info()

# 2. Hertz solve equality through the Python API
h = 1.0 / Ns
xs = (np.arange(Ns) + 0.5) * h - 0.5
X, Y = np.meshgrid(xs, xs, indexing="xy")
g0 = ((X**2 + Y**2) / (2 * 0.5)).ravel()
rd = dense.solve(g0, p_nominal=0.05)
rf = fftb.solve(g0, p_nominal=0.05)
print("Hertz iters:", rd.iterations, rf.iterations,
      "area:", rd.contact_area, rf.contact_area)
assert rd.iterations == rf.iterations and rd.contact_area == rf.contact_area

# 3. nested-solve area gate at Ns=1024: fft vs h2(q=6) to <= 1e-6 (spec §4)
Ns = 1024
rng = np.random.default_rng(42)
k = np.fft.fftfreq(Ns) * Ns
K = np.hypot(*np.meshgrid(k, k, indexing="ij")); K[0, 0] = 1.0
spec = np.fft.fft2(rng.standard_normal((Ns, Ns))) * K ** (-(1 + 0.8))
surf = np.real(np.fft.ifft2(spec)); surf -= surf.mean(); surf *= 0.02 / surf.std()
gap = (-surf).ravel()
r_h2  = hc.solve_nested(grid_size=Ns, gap=gap, p_nominal=0.005, coarsest=64, q=6)
r_fft = hc.solve_nested(grid_size=Ns, gap=gap, p_nominal=0.005, coarsest=64,
                        backend="fft")
d = abs(r_h2.contact_area - r_fft.contact_area)
print("nested Ns=1024: h2", r_h2.contact_area, r_h2.iterations, "it | fft",
      r_fft.contact_area, r_fft.iterations, "it | area diff", d)
assert d <= 1e-6

# 4. float nested path with fft backend converges (precond on)
r32 = hc.solve_nested(grid_size=Ns, gap=gap, p_nominal=0.005, coarsest=64,
                      backend="fft", single_precision=True, light_result=True)
print("float fft nested:", r32.contact_area, r32.iterations, "it,",
      "converged:", r32.converged)
assert r32.converged
assert abs(r32.contact_area - r_fft.contact_area) < 5e-4
print("ALL PYTHON GATES PASSED")
EOF
```

Expected: `ALL PYTHON GATES PASSED`. Note on gate 3: contact area at Ns=1024 is quantized in units of 1/N ≈ 9.5e-7, so ≤ 1e-6 means the two backends may differ by at most one cell (the residual H2 q=6 operator error ~3e-6 makes a 0–1 cell difference plausible; the FFT side is exact). If it fails at 2–3 cells, report the numbers to the user rather than silently loosening — this is the spec's acceptance threshold.

- [ ] **Step 6: Commit**

```bash
git add python/bindings.cpp include/nested_solve.hpp src/nested_solve.cpp
git commit -m "feat(fft): backend=\"fft\" in ContactSolver and solve_nested

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Benchmark + docs (README, CLAUDE.md, memory)

**Files:**
- Create: `bench_fft.py`
- Modify: `README.md` (performance section), `CLAUDE.md` (backend docs, file layout, validated numbers, What Is Left To Do)
- Modify (memory): `/home/vyastrebov/.claude/projects/-home-vyastrebov-WORK-PROJECTS-Hcontact/memory/aspher-project-state.md`

**Interfaces:**
- Consumes: Python API from Task 3 (`backend="fft"` on both entry points).
- Produces: measured numbers for the docs; no code consumed later.

- [ ] **Step 1: Write `bench_fft.py`**

```python
"""FFT-convolution backend benchmark: matvec timing vs H2, and the Ns=4096
rough nested solve (double/float, fft vs h2). Run alone on an idle machine
(fenicsx-env); matvec timings are min-of-30 (noisy workstation)."""
import sys, time
sys.path.insert(0, 'python')
import numpy as np
import aspher as hc


def surface(Ns, H=0.8, rms=0.02, seed=42):
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(Ns) * Ns
    K = np.hypot(*np.meshgrid(k, k, indexing="ij"))
    K[0, 0] = 1.0
    spec = np.fft.fft2(rng.standard_normal((Ns, Ns))) * K ** (-(1 + H))
    s = np.real(np.fft.ifft2(spec))
    s -= s.mean()
    s *= rms / s.std()
    return s


def bench_matvec(Ns, reps=30):
    p = np.random.default_rng(0).random(Ns * Ns)
    out = {}
    for backend, kw in (("h2", dict(q=6)), ("fft", {})):
        s = hc.ContactSolver(grid_size=Ns, backend=backend, **kw)
        s.matvec(p)  # warm-up: lazy scratch + FFT plans
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            s.matvec(p)
            best = min(best, time.perf_counter() - t0)
        out[backend] = best
    print(f"matvec Ns={Ns}: h2(q=6) {out['h2']*1e3:8.2f} ms | "
          f"fft {out['fft']*1e3:8.2f} ms | speedup {out['h2']/out['fft']:.2f}x")


def bench_solve(Ns=4096, p_bar=0.002):
    gap = (-surface(Ns)).ravel()
    for backend in ("h2", "fft"):
        for single in (False, True):
            t0 = time.perf_counter()
            r = hc.solve_nested(grid_size=Ns, gap=gap, p_nominal=p_bar,
                                coarsest=64, q=4, leaf_side=16,
                                backend=backend, single_precision=single,
                                light_result=True)
            dt = time.perf_counter() - t0
            print(f"nested Ns={Ns} {backend:3s} "
                  f"{'f32' if single else 'f64'}: {dt:7.1f} s, "
                  f"{r.iterations} it, area {r.contact_area:.6f}, "
                  f"converged {r.converged}")


if __name__ == "__main__":
    for Ns in (1024, 2048, 4096):
        bench_matvec(Ns)
    bench_solve()
```

Remove the stray placeholder block inside `bench_matvec` (the first `for backend…` loop with the broken `min(...)` expression) when writing the file — keep only the explicit min-of-reps loop. Note: `solver.matvec()` carries ~5 ms of pybind copy overhead at Ns=1024, identical for both backends, so the *ratio* is conservative and the absolute per-matvec numbers at Ns≥2048 are dominated by real work.

- [ ] **Step 2: Run it (idle machine, run alone)**

```bash
conda run -n fenicsx-env python bench_fft.py 2>&1 | tee /tmp/claude-1769/-home-vyastrebov-WORK-PROJECTS-Hcontact/f682e810-7461-46af-a101-5248f1a485f4/scratchpad/bench_fft.log
```

Expected (spec estimates, to be replaced by measured values): fft matvec 60–120 ms at Ns=4096 double vs ~300 ms H2 → nested solve ~1.5–2.5× faster; double h2 reference ≈ 47 s pocketfft (memory note, seed 42, p̄=0.002 — but that reference used q=4/leaf_side=16? it used the repro_4096.py defaults; compare like-for-like within this run only). The double fft and h2 areas must agree to ≤ a few cells (1/N ≈ 6e-8 at Ns=4096).

- [ ] **Step 3: Update CLAUDE.md**

With the measured numbers in hand:
- File layout: add `│   ├── fft_engine.hpp        # shared square r2c/c2r engine (pocketfft/FFTW dispatch)` under `src/` and `│   ├── fft_operator.hpp    # exact zero-padded Love-kernel FFT convolution (backend="fft")` under `include/`, plus `src/fft_operator.cpp` line; add `bench_fft.py` where the other bench scripts are listed.
- Theory summary: add a short `### FFT-convolution operator (backend="fft")` subsection after the H2 one: exact (matches dense to ~1e-13), O(N log N) matvec, ~10 N reals double scratch (spectrum 2N + padded grid 4N + half spectrum 4N), fastest at Ns ≤ 8192, H2 remains the choice for very large Ns; available in `ContactSolver(backend="fft")` and `solve_nested(..., backend="fft")`.
- Validated numbers table: add rows `FFT matvec vs dense (rel L2, double/float)`, `FFT matvec time Ns=4096 (vs H2 q=6)`, `Nested solve Ns=4096 fft vs h2 (double, float)` with the measured values from Step 2 and the test output.
- Python usage block: add `backend="fft"` to the backend comment line.
- What Is Left To Do: mark the FFT-convolution backend item ✅ done (2026-07) the way the preconditioner items were, and remove the "This is the next implementation task" sentence.

- [ ] **Step 4: Update README.md**

Read the README performance/backends section first, then: add `fft` to the backend list with one sentence (exact zero-padded Love-kernel convolution; equals the dense matvec to roundoff; fastest per iteration at Ns ≤ 8192) and add the measured Ns=4096 numbers to the performance table in the README's existing format.

- [ ] **Step 5: Update the project memory**

Edit `/home/vyastrebov/.claude/projects/-home-vyastrebov-WORK-PROJECTS-Hcontact/memory/aspher-project-state.md`: replace the "**Next task**: implement backend="fft" …" bullet with a "done 2026-07 (or actual date)" note carrying the measured Ns=4096 fft-vs-h2 numbers, and note the new shared `src/fft_engine.hpp`. Update the `MEMORY.md` index hook line for that file if the description no longer matches.

- [ ] **Step 6: Final full verification, both engines**

```bash
ctest --test-dir build --output-on-failure
ctest --test-dir build-fftw --output-on-failure
```

Expected: all 6 tests PASS in both dirs.

- [ ] **Step 7: Commit**

```bash
git add bench_fft.py README.md CLAUDE.md
git commit -m "docs+bench(fft): backend=\"fft\" benchmark and documentation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-Review Notes

- **Spec coverage:** §1–2 math/exactness → Task 2 (`build()` embedding + realness check, exactness tests); §3 cost/memory → `FFTInfo` + Task 4 bench; §4.1 shared engine → Task 1; §4.2 FFTOperator → Task 2; §4.3 plumbing → Task 3; §4.4 tests → Task 2 Step 1 (+ CTest wiring); §4.5 benchmarks/docs → Task 4; validation gate → Task 2 Step 5 (dense equality) + Task 3 Step 5 gate 3 (H2 area ≤ 1e-6); §5 risks → dense test at small Ns incl. non-power-of-two, realness symmetry check with throw, engine parity via `build-fftw`, licensing untouched (engine selection unchanged).
- **Spec deviations (deliberate):** test file named `tests/test_fft.cpp` to fit the CMake foreach; kernel-spectrum memory is 2N doubles (real half spectrum), smaller than the spec's §3 estimate — the spec over-counted; `solve_nested` gets `backend` now (spec left auto-select for later; default stays `"h2"`).
- **Type consistency:** `FFTOperator` method names mirror `H2Operator` exactly (`matvec_into`, `matvec_single_into`, `build_single_caches`, `info`, `print_statistics`) — used identically in Tasks 2–3; `NestedParams.backend` added last to keep positional brace-init in bindings valid, and the bindings init updated to pass it.

---

### Task 4b (inserted 2026-07-09 after benchmark review; human-approved)

The Ns=4096 benchmark (machine under desktop load) measured the fft matvec at
385–407 ms vs H2's 348–381 ms — bandwidth-bound, not the spec's 60–120 ms
estimate (which counted flops, not memory traffic). Human decision: optimize,
then re-bench. Full brief with complete code: `.superpowers/sdd/task-4b-brief.md`
(pruned padded transforms — `SquareR2C::bind()` gains an `ny_active` hint;
pocketfft skips the structurally-zero forward lines and the unread inverse
lines, ~20–30% of the matvec; FFTW keeps full 2-D plans, zeroing inactive
lines itself to honour the same contract). Task 4 (re-bench + docs) resumes
after 4b lands.
