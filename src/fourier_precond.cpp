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
