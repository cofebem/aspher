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

// Indexed (compressed) apply: same transform pipeline as apply_t, but the
// scatter/gather run over the compressed entries through grid_index. The
// full grid G must be zeroed first (the compressed set is sparse in it).
template <class S>
static void
apply_indexed_t(int Ns, int nh, const Eigen::MatrixXf& wh,
                Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& G,
                Eigen::Matrix<std::complex<S>, Eigen::Dynamic, Eigen::Dynamic>& C,
                std::unique_ptr<fft::SquareR2C<S>>& eng,
                const Eigen::Matrix<S, Eigen::Dynamic, 1>& gc,
                const std::vector<std::uint8_t>& contact_c,
                const std::vector<int>& grid_index,
                Eigen::Matrix<S, Eigen::Dynamic, 1>& zc) {
    const std::ptrdiff_t nc = static_cast<std::ptrdiff_t>(grid_index.size());
    if (!eng) {
        G.resize(Ns, Ns);
        C.resize(nh, Ns);
        eng = std::make_unique<fft::SquareR2C<S>>();
        eng->bind(Ns, G.data(), C.data());
    }
    if (zc.size() != nc) zc.resize(nc);

    G.setZero();
    S* Gd = G.data(); // G(ix, iy) at flat grid index iy*Ns + ix
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t k = 0; k < nc; ++k)
        if (contact_c[k]) Gd[grid_index[k]] = gc(k);

    eng->fwd();
#pragma omp parallel for schedule(static)
    for (int ky = 0; ky < Ns; ++ky) {
        std::complex<S>* c = C.data() + static_cast<std::ptrdiff_t>(ky) * nh;
        const float* w = wh.data() + static_cast<std::ptrdiff_t>(ky) * nh;
        for (int kx = 0; kx < nh; ++kx) c[kx] *= static_cast<S>(w[kx]);
    }
    eng->inv();

    double zsum = 0.0;
    long ncontact = 0;
#pragma omp parallel for schedule(static) reduction(+ : zsum, ncontact)
    for (std::ptrdiff_t k = 0; k < nc; ++k) {
        if (contact_c[k]) {
            zc(k) = Gd[grid_index[k]];
            zsum += static_cast<double>(zc(k));
            ++ncontact;
        } else {
            zc(k) = S(0);
        }
    }
    if (ncontact) {
        const S zmean = static_cast<S>(zsum / static_cast<double>(ncontact));
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t k = 0; k < nc; ++k)
            if (contact_c[k]) zc(k) -= zmean;
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

void FourierPreconditioner::apply_into_indexed(
    const Eigen::VectorXd& gc, const std::vector<std::uint8_t>& contact_c,
    const std::vector<int>& grid_index, Eigen::VectorXd& zc) const {
    apply_indexed_t<double>(Ns_, nh_, wh_, Gd_, Cd_, fft_d_, gc, contact_c,
                            grid_index, zc);
}

void FourierPreconditioner::apply_single_into_indexed(
    const Eigen::VectorXf& gc, const std::vector<std::uint8_t>& contact_c,
    const std::vector<int>& grid_index, Eigen::VectorXf& zc) const {
    apply_indexed_t<float>(Ns_, nh_, wh_, Gf_, Cf_, fft_f_, gc, contact_c,
                           grid_index, zc);
}

void FourierPreconditioner::release_scratch() const {
    // engines first: they hold plans bound to the scratch pointers
    fft_d_.reset();
    fft_f_.reset();
    Gd_.resize(0, 0);
    Cd_.resize(0, 0);
    Gf_.resize(0, 0);
    Cf_.resize(0, 0);
}

TangentialFourierPreconditioner::TangentialFourierPreconditioner(int Ns,
                                                                 double nu)
    : Ns_(Ns), nh_(Ns / 2 + 1), wxx_(Ns / 2 + 1, Ns), wyy_(Ns / 2 + 1, Ns),
      wxy_(Ns / 2 + 1, Ns) {
    const double gamma = nu / (1.0 - nu);
    const double norm = 1.0 / (static_cast<double>(Ns) * static_cast<double>(Ns));
    auto kof = [Ns](int i) { return (i < Ns / 2) ? i : i - Ns; };
    for (int ky = 0; ky < Ns; ++ky) {
        const double fy = static_cast<double>(kof(ky));
        for (int kx = 0; kx < nh_; ++kx) {
            const double fx = static_cast<double>(kx);
            const double k2 = fx * fx + fy * fy;
            if (k2 == 0.0) {
                wxx_(kx, ky) = wyy_(kx, ky) = wxy_(kx, ky) = 0.0;
                continue;
            }
            const double kk = std::sqrt(k2);
            wxx_(kx, ky) = norm * kk * (1.0 + gamma * fx * fx / k2);
            wyy_(kx, ky) = norm * kk * (1.0 + gamma * fy * fy / k2);
            // The odd cross term is ill-defined at self-conjugate Nyquist
            // modes (kx == Ns/2 or ky == Ns/2): the half-spectrum mirror
            // reuses this entry, flipping the effective sign. Zero it there
            // (even terms are sign-independent and stay) — same treatment
            // as the odd xy kernel's zeroed self-conjugate row/col in
            // TangentialFFTOperator.
            wxy_(kx, ky) = (kx == Ns_ / 2 || ky == Ns_ / 2)
                               ? 0.0
                               : norm * kk * gamma * fx * fy / k2;
        }
    }
}

TangentialFourierPreconditioner::~TangentialFourierPreconditioner() = default;

void TangentialFourierPreconditioner::apply_into(
    const Eigen::VectorXd& g, const std::vector<std::uint8_t>& mask,
    bool remove_mean, Eigen::VectorXd& z) const {
    const int N = Ns_ * Ns_;
    if (!fft1_) {
        G1_.resize(Ns_, Ns_);
        G2_.resize(Ns_, Ns_);
        C1_.resize(nh_, Ns_);
        C2_.resize(nh_, Ns_);
        fft1_ = std::make_unique<fft::SquareR2C<double>>();
        fft2_ = std::make_unique<fft::SquareR2C<double>>();
        fft1_->bind(Ns_, G1_.data(), C1_.data());
        fft2_->bind(Ns_, G2_.data(), C2_.data());
    }
    if (z.size() != 2 * N) z.resize(2 * N);

    // masked scatter of both components
#pragma omp parallel for schedule(static)
    for (int iy = 0; iy < Ns_; ++iy) {
        double* c1 = G1_.data() + static_cast<std::ptrdiff_t>(iy) * Ns_;
        double* c2 = G2_.data() + static_cast<std::ptrdiff_t>(iy) * Ns_;
        const double* gx = g.data() + static_cast<std::ptrdiff_t>(iy) * Ns_;
        const double* gy =
            g.data() + N + static_cast<std::ptrdiff_t>(iy) * Ns_;
        const std::uint8_t* mi =
            mask.data() + static_cast<std::ptrdiff_t>(iy) * Ns_;
        for (int ix = 0; ix < Ns_; ++ix) {
            c1[ix] = mi[ix] ? gx[ix] : 0.0;
            c2[ix] = mi[ix] ? gy[ix] : 0.0;
        }
    }

    fft1_->fwd();
    fft2_->fwd();

    // per-mode symmetric 2×2 inverse-symbol multiply (in place, local temps)
#pragma omp parallel for schedule(static)
    for (int ky = 0; ky < Ns_; ++ky) {
        std::complex<double>* a =
            C1_.data() + static_cast<std::ptrdiff_t>(ky) * nh_;
        std::complex<double>* b =
            C2_.data() + static_cast<std::ptrdiff_t>(ky) * nh_;
        const double* xx = wxx_.data() + static_cast<std::ptrdiff_t>(ky) * nh_;
        const double* yy = wyy_.data() + static_cast<std::ptrdiff_t>(ky) * nh_;
        const double* xy = wxy_.data() + static_cast<std::ptrdiff_t>(ky) * nh_;
        for (int kx = 0; kx < nh_; ++kx) {
            const std::complex<double> ga = a[kx], gb = b[kx];
            a[kx] = xx[kx] * ga + xy[kx] * gb;
            b[kx] = xy[kx] * ga + yy[kx] * gb;
        }
    }

    fft1_->inv();
    fft2_->inv();

    // masked gather (+ per-component mask-mean removal for force control)
    double zsx = 0.0, zsy = 0.0;
    long nm = 0;
#pragma omp parallel for schedule(static) reduction(+ : zsx, zsy, nm)
    for (int iy = 0; iy < Ns_; ++iy) {
        const double* c1 = G1_.data() + static_cast<std::ptrdiff_t>(iy) * Ns_;
        const double* c2 = G2_.data() + static_cast<std::ptrdiff_t>(iy) * Ns_;
        const std::uint8_t* mi =
            mask.data() + static_cast<std::ptrdiff_t>(iy) * Ns_;
        double* zx = z.data() + static_cast<std::ptrdiff_t>(iy) * Ns_;
        double* zy = z.data() + N + static_cast<std::ptrdiff_t>(iy) * Ns_;
        for (int ix = 0; ix < Ns_; ++ix) {
            if (mi[ix]) {
                zx[ix] = c1[ix];
                zy[ix] = c2[ix];
                zsx += c1[ix];
                zsy += c2[ix];
                ++nm;
            } else {
                zx[ix] = 0.0;
                zy[ix] = 0.0;
            }
        }
    }
    if (remove_mean && nm) {
        const double mxv = zsx / static_cast<double>(nm);
        const double myv = zsy / static_cast<double>(nm);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i)
            if (mask[i]) {
                z(i) -= mxv;
                z(N + i) -= myv;
            }
    }
}

} // namespace hmc
