#include "fourier_precond.hpp"

#include <unsupported/Eigen/FFT>

#include <cmath>
#include <complex>
#include <vector>

namespace hmc {

FourierPreconditioner::FourierPreconditioner(int Ns) : Ns_(Ns), w_(Ns, Ns) {
    // integer wavenumbers k = i (i < Ns/2) else i - Ns; symbol |k|, DC zeroed.
    // Absolute scale is irrelevant (cancels in CG), so 2π/L is dropped. The
    // symbol is stored in float: integer |k| < Ns is exact in float for the
    // grids used, and the float path multiplies with it directly.
    auto kof = [Ns](int i) { return (i < Ns / 2) ? i : i - Ns; };
    for (int ky = 0; ky < Ns; ++ky)
        for (int kx = 0; kx < Ns; ++kx)
            w_(ky, kx) = std::hypot(static_cast<float>(kof(kx)),
                                    static_cast<float>(kof(ky)));
    w_(0, 0) = 0.0f;
}

// 2D forward FFT by 1D transforms over rows then columns. Both passes are
// OpenMP-parallel (each thread owns its Eigen::FFT plan + scratch vectors); the
// column pass runs in place on `out`, so no Ns×Ns temporary is allocated. This
// is the per-iteration cost driver at large Ns, so it must use every core.
template <class S>
static void fft2_fwd(const Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& in,
                     Eigen::Matrix<std::complex<S>, Eigen::Dynamic, Eigen::Dynamic>& out) {
    const int n = static_cast<int>(in.rows());
#pragma omp parallel
    {
        Eigen::FFT<S> fft; // plan cached in the object after the first fwd()
        std::vector<S> rin(n);
        std::vector<std::complex<S>> rout(n);
#pragma omp for schedule(static)
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) rin[c] = in(r, c);
            fft.fwd(rout, rin);
            for (int c = 0; c < n; ++c) out(r, c) = rout[c];
        }
        std::vector<std::complex<S>> cin(n), cout(n);
#pragma omp for schedule(static)
        for (int c = 0; c < n; ++c) {
            for (int r = 0; r < n; ++r) cin[r] = out(r, c);
            fft.fwd(cout, cin);
            for (int r = 0; r < n; ++r) out(r, c) = cout[r];
        }
    }
}

// 2D inverse FFT. `in` is consumed as scratch (column inverse done in place),
// then the row inverse writes the real result to `out`. Eigen scales each 1D
// inverse by 1/n, so the pair gives the 1/n² of a 2D inverse. OpenMP-parallel.
template <class S>
static void fft2_inv(Eigen::Matrix<std::complex<S>, Eigen::Dynamic, Eigen::Dynamic>& in,
                     Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& out) {
    const int n = static_cast<int>(in.rows());
#pragma omp parallel
    {
        Eigen::FFT<S> fft;
        std::vector<std::complex<S>> cin(n), cout(n);
#pragma omp for schedule(static)
        for (int c = 0; c < n; ++c) {
            for (int r = 0; r < n; ++r) cin[r] = in(r, c);
            fft.inv(cout, cin);
            for (int r = 0; r < n; ++r) in(r, c) = cout[r];
        }
        std::vector<std::complex<S>> rin(n);
        std::vector<S> rout(n);
#pragma omp for schedule(static)
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) rin[c] = in(r, c);
            fft.inv(rout, rin);
            for (int c = 0; c < n; ++c) out(r, c) = rout[c];
        }
    }
}

// Scalar-templated preconditioner apply. w_ (float symbol) is used directly in
// float and cast up in double. Only three Ns×Ns buffers are live (R, C, Z);
// the FFTs run in place, avoiding extra Ns×Ns temporaries.
template <class S>
static Eigen::Matrix<S, Eigen::Dynamic, 1>
apply_t(int Ns, const Eigen::MatrixXf& wf,
        const Eigen::Matrix<S, Eigen::Dynamic, 1>& g,
        const std::vector<std::uint8_t>& contact) {
    using Vec = Eigen::Matrix<S, Eigen::Dynamic, 1>;
    using Mat = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;
    using CMat = Eigen::Matrix<std::complex<S>, Eigen::Dynamic, Eigen::Dynamic>;
    const int N = Ns * Ns;
    // Scatter the masked residual onto the grid (i = iy*Ns + ix). The O(N)
    // mask/extract/mean loops are parallelised too — at large Ns they are a
    // non-trivial share of the per-iteration cost next to the FFTs.
    Mat R = Mat::Zero(Ns, Ns);
#pragma omp parallel for schedule(static)
    for (int iy = 0; iy < Ns; ++iy)
        for (int ix = 0; ix < Ns; ++ix) {
            const int i = iy * Ns + ix;
            if (contact[i]) R(iy, ix) = g(i);
        }

    CMat C(Ns, Ns);
    fft2_fwd<S>(R, C);
    C.array() *= wf.template cast<S>().array(); // real symbol
    Mat Z(Ns, Ns);
    fft2_inv<S>(C, Z); // consumes C as scratch

    Vec z = Vec::Zero(N);
    double zsum = 0.0;
    long nc = 0;
#pragma omp parallel for schedule(static) reduction(+ : zsum, nc)
    for (int iy = 0; iy < Ns; ++iy)
        for (int ix = 0; ix < Ns; ++ix) {
            const int i = iy * Ns + ix;
            if (contact[i]) { const S v = Z(iy, ix); z(i) = v; zsum += v; ++nc; }
        }
    if (nc) {
        const S zmean = static_cast<S>(zsum / static_cast<double>(nc));
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i)
            if (contact[i]) z(i) -= zmean;
    }
    return z;
}

Eigen::VectorXd
FourierPreconditioner::apply(const Eigen::VectorXd& g,
                             const std::vector<std::uint8_t>& contact) const {
    return apply_t<double>(Ns_, w_, g, contact);
}

Eigen::VectorXf
FourierPreconditioner::apply_single(const Eigen::VectorXf& g,
                                    const std::vector<std::uint8_t>& contact) const {
    return apply_t<float>(Ns_, w_, g, contact);
}

} // namespace hmc
