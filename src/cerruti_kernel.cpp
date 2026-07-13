#include "cerruti_kernel.hpp"

namespace hmc {

// With k = |x|+a, l = |x|-a, m = |y|+b, n = |y|-b and R(u,v) = hypot(u,v)
// (Pohrt & Li eq. (13) shorthands, with absolute values to enforce parity):
//   integral of 1/rho            = xlog + ylog          (the Love bracket)
//   integral of (x-xi)^2/rho^3   = ylog
//   integral of (x-xi)(y-eta)/rho^3
//       = R(k,n) - R(k,m) + R(l,m) - R(l,n)
// where
//   xlog = k ln[(m+R(k,m))/(n+R(k,n))] + l ln[(n+R(l,n))/(m+R(l,m))]
//   ylog = m ln[(k+R(k,m))/(l+R(l,m))] + n ln[(l+R(l,n))/(k+R(k,n))]
// so the u_x bracket is (1-nu)*(xlog + ylog) + nu*ylog = (1-nu)*xlog + ylog.
double cerruti_uxx(double x, double y, double a, double b, double nu) {
    const double x_abs = std::abs(x);
    const double y_abs = std::abs(y);
    const double k = x_abs + a, l = x_abs - a;
    const double m = y_abs + b, n = y_abs - b;
    const double Rkm = std::hypot(k, m), Rkn = std::hypot(k, n);
    const double Rlm = std::hypot(l, m), Rln = std::hypot(l, n);
    const double xlog = k * std::log((m + Rkm) / (n + Rkn)) +
                        l * std::log((n + Rln) / (m + Rlm));
    const double ylog = m * std::log((k + Rkm) / (l + Rlm)) +
                        n * std::log((l + Rln) / (k + Rkn));
    return (1.0 - nu) * xlog + ylog;
}

double cerruti_uxy(double x, double y, double a, double b) {
    const double x_abs = std::abs(x);
    const double y_abs = std::abs(y);
    const double k = x_abs + a, l = x_abs - a;
    const double m = y_abs + b, n = y_abs - b;
    // uxy has odd parity in both x and y, encoded in the corners
    const int sign_x = (x >= 0.0) ? 1 : -1;
    const int sign_y = (y >= 0.0) ? 1 : -1;
    const double uxy_abs = std::hypot(k, n) - std::hypot(k, m) + std::hypot(l, m) -
                           std::hypot(l, n);
    return sign_x * sign_y * uxy_abs;
}

CerrutiKernel::CerrutiKernel(int Ns, double L, double E_star, double nu)
    : Ns_(Ns), L_(L), h_(L / Ns), a_(0.5 * L / Ns), E_star_(E_star), nu_(nu),
      pref_(1.0 / (M_PI * E_star * (1.0 - nu))), prefnu_(nu * pref_),
      xx_(static_cast<std::size_t>(Ns) * Ns),
      xy_(static_cast<std::size_t>(Ns) * Ns) {
    for (int dy = 0; dy < Ns_; ++dy)
        for (int dx = 0; dx < Ns_; ++dx) {
            const std::size_t at = static_cast<std::size_t>(dy) * Ns_ + dx;
            xx_[at] = pref_ * cerruti_uxx(dx * h_, dy * h_, a_, a_, nu_);
            xy_[at] = prefnu_ * cerruti_uxy(dx * h_, dy * h_, a_, a_);
        }
}

Eigen::MatrixXd CerrutiKernel::assemble_dense() const {
    const int N = size();
    Eigen::MatrixXd M(2 * N, 2 * N);
    for (int i = 0; i < N; ++i) {
        const int ixi = i % Ns_, iyi = i / Ns_;
        for (int j = 0; j < N; ++j) {
            const int dix = ixi - j % Ns_, diy = iyi - j / Ns_;
            const double cxy = xy_offset(dix, diy);
            M(i, j) = xx_offset(dix, diy);
            M(i, N + j) = cxy;
            M(N + i, j) = cxy;
            M(N + i, N + j) = yy_offset(dix, diy);
        }
    }
    return M;
}

Eigen::Matrix2d CerrutiKernel::symbol(double kx, double ky) const {
    // Continuum tangential symbol (spec section 3.2):
    //   Chat(k) = (2 / (E* (1-nu) |k|)) [ I - nu k k^T / |k|^2 ]
    // Longitudinal eigenvalue 2/(E*|k|) (== the normal Love symbol), transverse
    // 2/(E*(1-nu)|k|). Derived from the point kernels via
    // FT(1/s) = 2 pi/|k|, FT(x^2/s^3) = 2 pi ky^2/|k|^3,
    // FT(xy/s^3) = -2 pi kx ky/|k|^3.
    const double k2 = kx * kx + ky * ky;
    const double kk = std::sqrt(k2);
    const double c = 2.0 / (E_star_ * (1.0 - nu_) * kk);
    Eigen::Matrix2d S;
    S(0, 0) = c * (1.0 - nu_ * kx * kx / k2);
    S(1, 1) = c * (1.0 - nu_ * ky * ky / k2);
    S(0, 1) = S(1, 0) = c * (-nu_ * kx * ky / k2);
    return S;
}

} // namespace hmc
