#include "tangential_operator.hpp"

#include "fft_engine.hpp"

#include <cmath>
#include <stdexcept>

namespace hmc {

TangentialFFTOperator::TangentialFFTOperator(const CerrutiKernel& kernel)
    : kernel_(&kernel), Ns_(kernel.grid_size()), M_(2 * kernel.grid_size()),
      nh_(kernel.grid_size() + 1) {}

TangentialFFTOperator::~TangentialFFTOperator() = default;

void TangentialFFTOperator::build() {
    // Embed each table in wrap-around order (offset d at index d >= 0, or
    // M + d for d < 0; row/col Ns zeroed), transform, and keep the half
    // spectrum. Each table is even under joint index negation mod M (xx/yy
    // even per axis; xy odd per axis = even jointly; row/col Ns zero), so
    // every spectrum must come out REAL — checked, imaginary part being
    // roundoff noise. The 1/M² fwd+inv scale is folded in.
    Eigen::MatrixXd K(M_, M_);
    Eigen::MatrixXcd Kc(nh_, M_);
    fft::SquareR2C<double> plan;
    plan.bind(M_, K.data(), Kc.data()); // bind before filling (FFTW_MEASURE)

    const double norm =
        1.0 / (static_cast<double>(M_) * static_cast<double>(M_));

    auto embed_transform_harvest = [&](auto entry, Eigen::MatrixXd& out,
                                       const char* name) {
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
                col[ix] = (ix == Ns_) ? 0.0 : entry(dx, dy);
            }
        }
        plan.fwd();

        double max_re = 0.0, max_im = 0.0;
#pragma omp parallel for schedule(static) reduction(max : max_re, max_im)
        for (int ky = 0; ky < M_; ++ky) {
            const std::complex<double>* c =
                Kc.data() + static_cast<std::ptrdiff_t>(ky) * nh_;
            for (int kx = 0; kx < nh_; ++kx) {
                max_re = std::max(max_re, std::abs(c[kx].real()));
                max_im = std::max(max_im, std::abs(c[kx].imag()));
            }
        }
        if (max_im > 1e-10 * max_re)
            throw std::runtime_error(
                std::string("TangentialFFTOperator: ") + name +
                " spectrum parity broken (wrap-around embedding)");

        out.resize(nh_, M_);
#pragma omp parallel for schedule(static)
        for (int ky = 0; ky < M_; ++ky)
            for (int kx = 0; kx < nh_; ++kx)
                out(kx, ky) = norm * Kc(kx, ky).real();
    };

    embed_transform_harvest(
        [this](int dx, int dy) { return kernel_->xx_offset(dx, dy); }, Kxx_,
        "xx");
    embed_transform_harvest(
        [this](int dx, int dy) { return kernel_->yy_offset(dx, dy); }, Kyy_,
        "yy");
    embed_transform_harvest(
        [this](int dx, int dy) { return kernel_->xy_offset(dx, dy); }, Kxy_,
        "xy");
}

Eigen::VectorXd TangentialFFTOperator::matvec(const Eigen::VectorXd& q) const {
    Eigen::VectorXd u;
    matvec_into(q, u);
    return u;
}

void TangentialFFTOperator::matvec_into(const Eigen::VectorXd& q,
                                        Eigen::VectorXd& u) const {
    if (Kxx_.size() == 0)
        throw std::logic_error("TangentialFFTOperator: build() not called");
    const int N = Ns_ * Ns_;
    if (static_cast<int>(q.size()) != 2 * N)
        throw std::invalid_argument(
            "TangentialFFTOperator::matvec: q size != 2*Ns*Ns");
    if (!fft1_) { // first use: allocate scratch, bind plans (ny_active hint
                  // as in FFTOperator: lines [Ns, M) never touched)
        G1_.resize(M_, M_);
        G2_.resize(M_, M_);
        C1_.resize(nh_, M_);
        C2_.resize(nh_, M_);
        fft1_ = std::make_unique<fft::SquareR2C<double>>();
        fft2_ = std::make_unique<fft::SquareR2C<double>>();
        fft1_->bind(M_, G1_.data(), C1_.data(), Ns_);
        fft2_->bind(M_, G2_.data(), C2_.data(), Ns_);
    }
    if (u.size() != 2 * N) u.resize(2 * N);

    // zero-pad scatter: q_x -> G1, q_y -> G2
#pragma omp parallel for schedule(static)
    for (int iy = 0; iy < Ns_; ++iy) {
        double* c1 = G1_.data() + static_cast<std::ptrdiff_t>(iy) * M_;
        double* c2 = G2_.data() + static_cast<std::ptrdiff_t>(iy) * M_;
        const double* qx = q.data() + static_cast<std::ptrdiff_t>(iy) * Ns_;
        const double* qy =
            q.data() + N + static_cast<std::ptrdiff_t>(iy) * Ns_;
        for (int ix = 0; ix < Ns_; ++ix) c1[ix] = qx[ix];
        for (int ix = Ns_; ix < M_; ++ix) c1[ix] = 0.0;
        for (int ix = 0; ix < Ns_; ++ix) c2[ix] = qy[ix];
        for (int ix = Ns_; ix < M_; ++ix) c2[ix] = 0.0;
    }

    fft1_->fwd();
    fft2_->fwd();

    // in-place all-real 2x2 spectral mix (local temps: both outputs need
    // both inputs)
#pragma omp parallel for schedule(static)
    for (int ky = 0; ky < M_; ++ky) {
        std::complex<double>* a =
            C1_.data() + static_cast<std::ptrdiff_t>(ky) * nh_;
        std::complex<double>* b =
            C2_.data() + static_cast<std::ptrdiff_t>(ky) * nh_;
        const double* wxx = Kxx_.data() + static_cast<std::ptrdiff_t>(ky) * nh_;
        const double* wyy = Kyy_.data() + static_cast<std::ptrdiff_t>(ky) * nh_;
        const double* wxy = Kxy_.data() + static_cast<std::ptrdiff_t>(ky) * nh_;
        for (int kx = 0; kx < nh_; ++kx) {
            const std::complex<double> qa = a[kx], qb = b[kx];
            a[kx] = wxx[kx] * qa + wxy[kx] * qb;
            b[kx] = wxy[kx] * qa + wyy[kx] * qb;
        }
    }

    fft1_->inv();
    fft2_->inv();

    // gather u_x <- G1, u_y <- G2
#pragma omp parallel for schedule(static)
    for (int iy = 0; iy < Ns_; ++iy) {
        const double* c1 = G1_.data() + static_cast<std::ptrdiff_t>(iy) * M_;
        const double* c2 = G2_.data() + static_cast<std::ptrdiff_t>(iy) * M_;
        double* ux = u.data() + static_cast<std::ptrdiff_t>(iy) * Ns_;
        double* uy = u.data() + N + static_cast<std::ptrdiff_t>(iy) * Ns_;
        for (int ix = 0; ix < Ns_; ++ix) ux[ix] = c1[ix];
        for (int ix = 0; ix < Ns_; ++ix) uy[ix] = c2[ix];
    }
}

TangentialH2Operator::TangentialH2Operator(const CerrutiKernel& kernel,
                                           H2Params params)
    : Ns_(kernel.grid_size()),
      Hxx_(kernel.grid_size(), kernel.element_size(),
           [k = &kernel](double dx, double dy) { return k->xx_far(dx, dy); },
           [k = &kernel](int di, int dj) { return k->xx_offset(di, dj); },
           params),
      Hyy_(kernel.grid_size(), kernel.element_size(),
           [k = &kernel](double dx, double dy) { return k->yy_far(dx, dy); },
           [k = &kernel](int di, int dj) { return k->yy_offset(di, dj); },
           params),
      Hxy_(kernel.grid_size(), kernel.element_size(),
           [k = &kernel](double dx, double dy) { return k->xy_far(dx, dy); },
           [k = &kernel](int di, int dj) { return k->xy_offset(di, dj); },
           params) {}

void TangentialH2Operator::build() {
    Hxx_.build();
    Hyy_.build();
    Hxy_.build();
}

Eigen::VectorXd TangentialH2Operator::matvec(const Eigen::VectorXd& q) const {
    Eigen::VectorXd u;
    matvec_into(q, u);
    return u;
}

void TangentialH2Operator::matvec_into(const Eigen::VectorXd& q,
                                       Eigen::VectorXd& u) const {
    const int N = Ns_ * Ns_;
    if (static_cast<int>(q.size()) != 2 * N)
        throw std::invalid_argument(
            "TangentialH2Operator::matvec: q size != 2*Ns*Ns");
    if (u.size() != 2 * N) u.resize(2 * N);

    qx_ = q.head(N); // both halves gathered before any write: u may alias q
    qy_ = q.tail(N);

    Hxx_.matvec_into(qx_, tc_);
    u.head(N) = tc_;
    Hxy_.matvec_into(qy_, tc_);
    u.head(N) += tc_;

    Hxy_.matvec_into(qx_, tc_);
    u.tail(N) = tc_;
    Hyy_.matvec_into(qy_, tc_);
    u.tail(N) += tc_;
}

} // namespace hmc
