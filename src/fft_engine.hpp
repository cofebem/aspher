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
