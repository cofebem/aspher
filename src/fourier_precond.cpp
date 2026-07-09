#include "fourier_precond.hpp"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HMC_USE_FFTW
#include <mutex>
#else
// cache pocketfft's internal plans (twiddle tables): without it every call
// rebuilds them, which dominates when the transform is applied per CG
// iteration on slabs from many threads
#define POCKETFFT_CACHE_SIZE 16
#include "pocketfft_hdronly.h"
#endif

namespace hmc {

namespace {

int fft_nthreads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

#ifndef HMC_USE_FFTW
int fft_nthreads_in_region() {
#ifdef _OPENMP
    return omp_get_num_threads();
#else
    return 1;
#endif
}
int fft_thread_id() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}
#endif

#ifdef HMC_USE_FFTW
// ── FFTW engine: 2-D r2c/c2r plans bound to the object-owned scratch ────────

// FFTW's threaded init must run exactly once per process (per precision).
void init_fftw_threads() {
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
unsigned plan_flags() {
    return std::getenv("HMC_FFTW_MEASURE") ? FFTW_MEASURE : FFTW_ESTIMATE;
}

#ifndef HMC_FFTW_THREADS
// Without the *_omp libraries the *_plan_with_nthreads symbols don't exist;
// stub them out so the shims below compile (FFTW then runs single-threaded).
void fftw_plan_with_nthreads(int) {}
void fftwf_plan_with_nthreads(int) {}
#endif

// Precision-dispatch shims over the fftw_/fftwf_ APIs. The grid G(ix, iy) is
// column-major with ix contiguous, i.e. row-major (iy, ix) — FFTW's (n0, n1)
// with the last (contiguous) axis ix, which r2c halves: the output C(kx, ky)
// (kx contiguous, kx ∈ [0, Ns/2]) shares its flat layout with the symbol wh_.
template <class S> struct FFTW;

template <> struct FFTW<double> {
    using Plan = fftw_plan;
    static Plan plan_r2c(int Ns, double* in, std::complex<double>* out) {
        fftw_plan_with_nthreads(fft_nthreads());
        return fftw_plan_dft_r2c_2d(Ns, Ns, in,
                                    reinterpret_cast<fftw_complex*>(out),
                                    plan_flags());
    }
    static Plan plan_c2r(int Ns, std::complex<double>* in, double* out) {
        fftw_plan_with_nthreads(fft_nthreads());
        return fftw_plan_dft_c2r_2d(Ns, Ns,
                                    reinterpret_cast<fftw_complex*>(in), out,
                                    plan_flags() | FFTW_DESTROY_INPUT);
    }
    static void execute(Plan p) { fftw_execute(p); }
    static void destroy(Plan p) { if (p) fftw_destroy_plan(p); }
};

template <> struct FFTW<float> {
    using Plan = fftwf_plan;
    static Plan plan_r2c(int Ns, float* in, std::complex<float>* out) {
        fftwf_plan_with_nthreads(fft_nthreads());
        return fftwf_plan_dft_r2c_2d(Ns, Ns, in,
                                     reinterpret_cast<fftwf_complex*>(out),
                                     plan_flags());
    }
    static Plan plan_c2r(int Ns, std::complex<float>* in, float* out) {
        fftwf_plan_with_nthreads(fft_nthreads());
        return fftwf_plan_dft_c2r_2d(Ns, Ns,
                                     reinterpret_cast<fftwf_complex*>(in), out,
                                     plan_flags() | FFTW_DESTROY_INPUT);
    }
    static void execute(Plan p) { fftwf_execute(p); }
    static void destroy(Plan p) { if (p) fftwf_destroy_plan(p); }
};

#else
// ── pocketfft engine (default, BSD) ─────────────────────────────────────────
// Layout mapping: G(ix, iy) column-major (ix contiguous), C(kx, ky) with
// kx ∈ [0, Ns/2] contiguous — matching the half-spectrum symbol layout.
//
// The 2-D transform is decomposed into a single-axis r2c/c2r along ix and an
// in-place c2c along iy, and both passes are parallelised over slabs from
// inside ONE OpenMP region (each slab transformed with nthreads=1). This
// deliberately avoids (a) pocketfft's multi-axis c2r path, which copies the
// whole (const) input spectrum into an internal temporary per call, and
// (b) pocketfft's private std::thread pool, which contends with OpenMP's
// spin-waiting workers when the solver's parallel regions surround the call.
template <class S>
void pfft_fwd(int Ns, int nh, const S* in, std::complex<S>* out) {
    using CS = std::complex<S>;
    const pocketfft::stride_t sg{static_cast<std::ptrdiff_t>(sizeof(S)),
                                 static_cast<std::ptrdiff_t>(sizeof(S)) * Ns};
    const pocketfft::stride_t sc{static_cast<std::ptrdiff_t>(sizeof(CS)),
                                 static_cast<std::ptrdiff_t>(sizeof(CS)) * nh};
#pragma omp parallel
    {
        const int nt = fft_nthreads_in_region();
        const int tid = fft_thread_id();
        // pass 1: r2c along ix, slab of iy lines per thread
        const int y0 = static_cast<int>(std::int64_t(Ns) * tid / nt);
        const int y1 = static_cast<int>(std::int64_t(Ns) * (tid + 1) / nt);
        if (y1 > y0)
            pocketfft::r2c({static_cast<std::size_t>(Ns),
                            static_cast<std::size_t>(y1 - y0)},
                           sg, sc, pocketfft::shape_t{0}, true,
                           in + std::size_t(y0) * Ns,
                           out + std::size_t(y0) * nh, S(1), 1);
#pragma omp barrier
        // pass 2: c2c along iy, slab of kx rows per thread, in place
        const int k0 = static_cast<int>(std::int64_t(nh) * tid / nt);
        const int k1 = static_cast<int>(std::int64_t(nh) * (tid + 1) / nt);
        if (k1 > k0)
            pocketfft::c2c({static_cast<std::size_t>(k1 - k0),
                            static_cast<std::size_t>(Ns)},
                           sc, sc, pocketfft::shape_t{1}, true,
                           out + k0, out + k0, S(1), 1);
    }
}

template <class S>
void pfft_inv(int Ns, int nh, std::complex<S>* in, S* out) {
    using CS = std::complex<S>;
    const pocketfft::stride_t sg{static_cast<std::ptrdiff_t>(sizeof(S)),
                                 static_cast<std::ptrdiff_t>(sizeof(S)) * Ns};
    const pocketfft::stride_t sc{static_cast<std::ptrdiff_t>(sizeof(CS)),
                                 static_cast<std::ptrdiff_t>(sizeof(CS)) * nh};
#pragma omp parallel
    {
        const int nt = fft_nthreads_in_region();
        const int tid = fft_thread_id();
        // pass 1: inverse c2c along iy, slab of kx rows, in place
        const int k0 = static_cast<int>(std::int64_t(nh) * tid / nt);
        const int k1 = static_cast<int>(std::int64_t(nh) * (tid + 1) / nt);
        if (k1 > k0)
            pocketfft::c2c({static_cast<std::size_t>(k1 - k0),
                            static_cast<std::size_t>(Ns)},
                           sc, sc, pocketfft::shape_t{1}, false,
                           in + k0, in + k0, S(1), 1);
#pragma omp barrier
        // pass 2: c2r along ix, slab of iy lines per thread
        const int y0 = static_cast<int>(std::int64_t(Ns) * tid / nt);
        const int y1 = static_cast<int>(std::int64_t(Ns) * (tid + 1) / nt);
        if (y1 > y0)
            pocketfft::c2r({static_cast<std::size_t>(Ns),
                            static_cast<std::size_t>(y1 - y0)},
                           sc, sg, pocketfft::shape_t{0}, false,
                           in + std::size_t(y0) * nh,
                           out + std::size_t(y0) * Ns, S(1), 1);
    }
}
#endif

} // namespace

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

FourierPreconditioner::~FourierPreconditioner() {
#ifdef HMC_USE_FFTW
    FFTW<double>::destroy(fwd_d_);
    FFTW<double>::destroy(inv_d_);
    FFTW<float>::destroy(fwd_f_);
    FFTW<float>::destroy(inv_f_);
#endif
}

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
template <class S, class PlanT>
static void
apply_t(int Ns, int nh, const Eigen::MatrixXf& wh,
        Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& G,
        Eigen::Matrix<std::complex<S>, Eigen::Dynamic, Eigen::Dynamic>& C,
        PlanT& fwd, PlanT& inv,
        const Eigen::Matrix<S, Eigen::Dynamic, 1>& g,
        const std::vector<std::uint8_t>& contact,
        Eigen::Matrix<S, Eigen::Dynamic, 1>& z) {
    const int N = Ns * Ns;
    if (G.cols() != Ns) { // first use in this precision: allocate scratch
        G.resize(Ns, Ns);
        C.resize(nh, Ns);
#ifdef HMC_USE_FFTW
        // plan after allocation (FFTW_MEASURE scribbles on the arrays during
        // planning); the plans bind to these pointers
        init_fftw_threads();
        fwd = FFTW<S>::plan_r2c(Ns, G.data(), C.data());
        inv = FFTW<S>::plan_c2r(Ns, C.data(), G.data());
#else
        (void)fwd;
        (void)inv;
#endif
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

#ifdef HMC_USE_FFTW
    FFTW<S>::execute(fwd);
#else
    pfft_fwd<S>(Ns, nh, G.data(), C.data());
#endif

    // real, even symbol on the half spectrum; wh and C share flat layout
#pragma omp parallel for schedule(static)
    for (int ky = 0; ky < Ns; ++ky) {
        std::complex<S>* c = C.data() + static_cast<std::ptrdiff_t>(ky) * nh;
        const float* w = wh.data() + static_cast<std::ptrdiff_t>(ky) * nh;
        for (int kx = 0; kx < nh; ++kx) c[kx] *= static_cast<S>(w[kx]);
    }

#ifdef HMC_USE_FFTW
    FFTW<S>::execute(inv); // consumes C, writes G
#else
    pfft_inv<S>(Ns, nh, C.data(), G.data()); // consumes C, writes G
#endif

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

#ifndef HMC_USE_FFTW
// planless engines still need a placeholder to thread through apply_t
namespace { struct NoPlan {}; }
#endif

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
#ifdef HMC_USE_FFTW
    apply_t<double>(Ns_, nh_, wh_, Gd_, Cd_, fwd_d_, inv_d_, g, contact, z);
#else
    NoPlan p0, p1;
    apply_t<double>(Ns_, nh_, wh_, Gd_, Cd_, p0, p1, g, contact, z);
#endif
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
#ifdef HMC_USE_FFTW
    apply_t<float>(Ns_, nh_, wh_, Gf_, Cf_, fwd_f_, inv_f_, g, contact, z);
#else
    NoPlan p0, p1;
    apply_t<float>(Ns_, nh_, wh_, Gf_, Cf_, p0, p1, g, contact, z);
#endif
}

} // namespace hmc
