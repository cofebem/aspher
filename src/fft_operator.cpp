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
