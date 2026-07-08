#include "fourier_precond.hpp"

#include <unsupported/Eigen/FFT>

#include <cmath>
#include <complex>
#include <vector>

namespace hmc {

FourierPreconditioner::FourierPreconditioner(int Ns)
    : Ns_(Ns), nh_(Ns / 2 + 1), wh_(Ns / 2 + 1, Ns) {
    // integer wavenumbers k = i (i < Ns/2) else i - Ns; symbol |k|, DC zeroed.
    // Absolute scale is irrelevant (cancels in CG), so 2π/L is dropped. Only
    // kx ∈ [0, Ns/2] is stored (half spectrum; |kof(kx)| = kx on that range).
    // The symbol is stored in float: integer |k| < Ns is exact in float for
    // the grids used, and the float path multiplies with it directly.
    auto kof = [Ns](int i) { return (i < Ns / 2) ? i : i - Ns; };
    for (int ky = 0; ky < Ns; ++ky)
        for (int kx = 0; kx < nh_; ++kx)
            wh_(kx, ky) = std::hypot(static_cast<float>(kx),
                                     static_cast<float>(kof(ky)));
    wh_(0, 0) = 0.0f;
}

// Scalar-templated preconditioner apply. The grid field G is laid out
// G(ix, iy) so that the x-direction transforms run down contiguous columns:
//   1. scatter the contact-masked residual into G
//   2. r2c along x: C(kx, iy), kx ∈ [0, Ns/2]  (half spectrum)
//   3. per kx row: forward FFT along y, multiply by the |k| symbol, inverse
//      FFT along y (fused single pass over the strided rows)
//   4. c2r along x back into G
//   5. gather on the contact set, remove the contact mean
// All passes are OpenMP-parallel; G and C are caller-owned scratch reused
// across iterations, so the per-call footprint is one N-vector (the result).
template <class S>
static void
apply_t(int Ns, int nh, const Eigen::MatrixXf& wh,
        Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& G,
        Eigen::Matrix<std::complex<S>, Eigen::Dynamic, Eigen::Dynamic>& C,
        const Eigen::Matrix<S, Eigen::Dynamic, 1>& g,
        const std::vector<std::uint8_t>& contact,
        Eigen::Matrix<S, Eigen::Dynamic, 1>& z) {
    const int N = Ns * Ns;
    if (G.cols() != Ns) G.resize(Ns, Ns);
    if (C.cols() != Ns) C.resize(nh, Ns);
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

#pragma omp parallel
    {
        Eigen::FFT<S> fft; // plans cached in the object after the first call
        fft.SetFlag(Eigen::FFT<S>::HalfSpectrum);

        // r2c down the contiguous columns of G
#pragma omp for schedule(static)
        for (int iy = 0; iy < Ns; ++iy)
            fft.fwd(C.data() + static_cast<std::ptrdiff_t>(iy) * nh,
                    G.data() + static_cast<std::ptrdiff_t>(iy) * Ns, Ns);

        // per kx: gather the strided row, FFT along y, apply the symbol,
        // inverse FFT along y, scatter back — one fused pass
        std::vector<std::complex<S>> a(Ns), b(Ns);
#pragma omp for schedule(static)
        for (int kx = 0; kx < nh; ++kx) {
            for (int ky = 0; ky < Ns; ++ky) a[ky] = C(kx, ky);
            fft.fwd(b.data(), a.data(), Ns);
            for (int ky = 0; ky < Ns; ++ky) b[ky] *= static_cast<S>(wh(kx, ky));
            fft.inv(a.data(), b.data(), Ns); // scales by 1/Ns
            for (int ky = 0; ky < Ns; ++ky) C(kx, ky) = a[ky];
        }

        // c2r back down the columns, in place into G (scales by 1/Ns)
#pragma omp for schedule(static)
        for (int iy = 0; iy < Ns; ++iy)
            fft.inv(G.data() + static_cast<std::ptrdiff_t>(iy) * Ns,
                    C.data() + static_cast<std::ptrdiff_t>(iy) * nh, Ns);
    }

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
    apply_t<double>(Ns_, nh_, wh_, Gd_, Cd_, g, contact, z);
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
    apply_t<float>(Ns_, nh_, wh_, Gf_, Cf_, g, contact, z);
}

} // namespace hmc
