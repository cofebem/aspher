#include "periodic_fft_operator.hpp"

#include "fft_engine.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace hmc {

PeriodicFFTOperator::PeriodicFFTOperator(int grid_size, double domain_size,
                                         double E_star)
    : Ns_(grid_size), nh_(grid_size / 2 + 1) {
    if (Ns_ < 2) throw std::invalid_argument("grid_size must be at least 2");
    if (!(domain_size > 0.0))
        throw std::invalid_argument("domain_size must be positive");
    if (!(E_star > 0.0))
        throw std::invalid_argument("E_star must be positive");

    // q = 2*pi*(kx,ky)/L and u_hat = 2*p_hat/(E*|q|). SquareR2C is an
    // unnormalised transform pair, hence the additional 1/Ns^2 factor.
    const double pi = std::acos(-1.0);
    scale_ = domain_size / (pi * E_star * static_cast<double>(Ns_) * Ns_);
}

PeriodicFFTOperator::~PeriodicFFTOperator() = default;

// Multiplies spectrum columns [k0, k1) (kx, contiguous) by the compliance
// symbol scale/|m|, m = (kx, signed ky); the zero mode is set to zero. The
// symbol is evaluated on the fly in double (a sqrt and a divide per mode,
// negligible next to the transforms) instead of being stored: that removes
// an N/2-real table from memory and from every apply's traffic.
template <class S>
static inline void apply_symbol(int Ns, int nh, double scale, int k0, int k1,
                                std::complex<S>* C) {
    for (int ky = 0; ky < Ns; ++ky) {
        const int my = (ky < (Ns + 1) / 2) ? ky : ky - Ns;
        const double my2 = static_cast<double>(my) * my;
        std::complex<S>* c = C + static_cast<std::ptrdiff_t>(ky) * nh;
        for (int kx = k0; kx < k1; ++kx) {
            const double m2 = static_cast<double>(kx) * kx + my2;
            const S w = (m2 > 0.0) ? static_cast<S>(scale / std::sqrt(m2))
                                   : S(0);
            c[kx] *= w;
        }
    }
}

#ifndef HMC_USE_FFTW
// One fused parallel region: r2c rows read straight from x, then per
// thread-owned block of kx columns the forward column c2c, the symbol and
// the inverse column c2c back to back while the block is cache resident,
// then c2r rows written straight into y. No N-real grid buffer, no copies,
// two barriers. x and y may alias (x is fully consumed before the first
// barrier, y is only written after the second).
constexpr int kColumnBlock = 16;

template <class S>
static void apply_periodic(int Ns, int nh, double scale,
                           Eigen::Matrix<std::complex<S>, Eigen::Dynamic,
                                         Eigen::Dynamic>& C,
                           const S* x, S* y) {
    using CS = std::complex<S>;
    if (C.rows() != nh || C.cols() != Ns) C.resize(nh, Ns);
    CS* c = C.data();
    const std::size_t n = static_cast<std::size_t>(Ns);
    const pocketfft::stride_t sg{static_cast<std::ptrdiff_t>(sizeof(S)),
                                 static_cast<std::ptrdiff_t>(sizeof(S)) * Ns};
    const pocketfft::stride_t sc{static_cast<std::ptrdiff_t>(sizeof(CS)),
                                 static_cast<std::ptrdiff_t>(sizeof(CS)) * nh};
#pragma omp parallel
    {
        const int nt = fft::nthreads_in_region();
        const int tid = fft::thread_id();
        const int y0 = static_cast<int>(std::int64_t(Ns) * tid / nt);
        const int y1 = static_cast<int>(std::int64_t(Ns) * (tid + 1) / nt);
        if (y1 > y0)
            pocketfft::r2c({n, static_cast<std::size_t>(y1 - y0)}, sg, sc,
                           pocketfft::shape_t{0}, true,
                           x + std::size_t(y0) * n, c + std::size_t(y0) * nh,
                           S(1), 1);
#pragma omp barrier
        const int k0 = static_cast<int>(std::int64_t(nh) * tid / nt);
        const int k1 = static_cast<int>(std::int64_t(nh) * (tid + 1) / nt);
        for (int kb = k0; kb < k1; kb += kColumnBlock) {
            const int ke = (kb + kColumnBlock < k1) ? kb + kColumnBlock : k1;
            const pocketfft::shape_t shp{static_cast<std::size_t>(ke - kb), n};
            pocketfft::c2c(shp, sc, sc, pocketfft::shape_t{1}, true, c + kb,
                           c + kb, S(1), 1);
            apply_symbol<S>(Ns, nh, scale, kb, ke, c);
            pocketfft::c2c(shp, sc, sc, pocketfft::shape_t{1}, false, c + kb,
                           c + kb, S(1), 1);
        }
#pragma omp barrier
        if (y1 > y0)
            pocketfft::c2r({n, static_cast<std::size_t>(y1 - y0)}, sc, sg,
                           pocketfft::shape_t{0}, false,
                           c + std::size_t(y0) * nh, y + std::size_t(y0) * n,
                           S(1), 1);
    }
}
#else
// FFTW engine: full 2-D plans bound to an owned grid buffer, so the input is
// copied in and the result copied out.
template <class S>
static void apply_periodic(
    int Ns, int nh, double scale,
    Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& G,
    Eigen::Matrix<std::complex<S>, Eigen::Dynamic, Eigen::Dynamic>& C,
    std::unique_ptr<fft::SquareR2C<S>>& engine, const S* x, S* y) {
    const std::int64_t N = static_cast<std::int64_t>(Ns) * Ns;
    if (!engine) {
        G.resize(Ns, Ns);
        C.resize(nh, Ns);
        engine = std::make_unique<fft::SquareR2C<S>>();
        engine->bind(Ns, G.data(), C.data());
    }
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < N; ++i) G.data()[i] = x[i];
    engine->fwd();
#pragma omp parallel for schedule(static)
    for (int kb = 0; kb < nh; kb += 16)
        apply_symbol<S>(Ns, nh, scale, kb, (kb + 16 < nh) ? kb + 16 : nh,
                        C.data());
    engine->inv();
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < N; ++i) y[i] = G.data()[i];
}
#endif

template <class S>
static void check_and_size(int Ns, const Eigen::Matrix<S, Eigen::Dynamic, 1>& x,
                           Eigen::Matrix<S, Eigen::Dynamic, 1>& y) {
    const std::int64_t N = static_cast<std::int64_t>(Ns) * Ns;
    if (x.size() != N)
        throw std::invalid_argument("PeriodicFFTOperator input size mismatch");
    if (y.size() != N) y.resize(N);
}

Eigen::VectorXd PeriodicFFTOperator::matvec(const Eigen::VectorXd& x) const {
    Eigen::VectorXd y;
    matvec_into(x, y);
    return y;
}

void PeriodicFFTOperator::matvec_into(const Eigen::VectorXd& x,
                                      Eigen::VectorXd& y) const {
    check_and_size<double>(Ns_, x, y);
#ifndef HMC_USE_FFTW
    apply_periodic<double>(Ns_, nh_, scale_, Cd_, x.data(), y.data());
#else
    apply_periodic<double>(Ns_, nh_, scale_, Gd_, Cd_, fft_d_, x.data(),
                           y.data());
#endif
}

// The symbol is computed on the fly, so the float path has no cache to
// build; kept for the common operator contract used by the nested driver.
void PeriodicFFTOperator::build_single_caches() const {}

Eigen::VectorXf
PeriodicFFTOperator::matvec_single(const Eigen::VectorXf& x) const {
    Eigen::VectorXf y;
    matvec_single_into(x, y);
    return y;
}

void PeriodicFFTOperator::matvec_single_into(const Eigen::VectorXf& x,
                                             Eigen::VectorXf& y) const {
    check_and_size<float>(Ns_, x, y);
#ifndef HMC_USE_FFTW
    apply_periodic<float>(Ns_, nh_, scale_, Cf_, x.data(), y.data());
#else
    apply_periodic<float>(Ns_, nh_, scale_, Gf_, Cf_, fft_f_, x.data(),
                          y.data());
#endif
}

void PeriodicFFTOperator::release_single_caches() const {
    fft_f_.reset();
    Gf_.resize(0, 0);
    Cf_.resize(0, 0);
}

std::size_t PeriodicFFTOperator::scratch_bytes() const {
    return sizeof(double) * static_cast<std::size_t>(Gd_.size()) +
           sizeof(std::complex<double>) * static_cast<std::size_t>(Cd_.size()) +
           sizeof(float) * static_cast<std::size_t>(Gf_.size()) +
           sizeof(std::complex<float>) * static_cast<std::size_t>(Cf_.size());
}

} // namespace hmc
