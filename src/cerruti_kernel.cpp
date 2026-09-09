#include "cerruti_kernel.hpp"

#include "boussinesq_kernel.hpp" // asinh_diff (shared stable bracket)

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
// Stable evaluation (spec A05), the same audit as love_uz. The printed form
// subtracts logarithms of corner sums, which cancels when the field point is
// far (both arguments approach y/x) and is outright NaN at a cell corner,
// where l = n = 0 makes one argument log(0) against a zero multiplier. Using
// ln(t + sqrt(t^2+c^2)) = ln|c| + asinh(t/|c|), each pair collapses to
// t * [asinh(s1/|t|) - asinh(s2/|t|)] with the divergent ln|c| cancelling
// analytically and the removable t -> 0 limit taken explicitly. The exact
// difference s1 - s2 (2a or 2b) is passed to asinh_diff so no cancellation is
// reintroduced there either.
// ylog = int (x-xi)^2 / rho^3 dA'. Same switch as love_uz: far away the two
// surviving asinh terms are O(1) while the answer is O(A x^2 / r^3), so at
// (0, 32h) they cancelled to a relative error of 1.2e-11. The multipole
// expansion of (x-xi)^2/rho^3 to second moments (<xi^2> = a^2/3,
// <eta^2> = b^2/3) is
//   A [ x^2/r^3 + (a^2/6)(2/r^3 - 15 x^2/r^5 + 15 x^4/r^7)
//                + (b^2/6)(-3 x^2/r^5 + 15 x^2 y^2/r^7) ],
// with the same O((m/r)^4) remainder as the Love expansion.
static constexpr double CERRUTI_FAR_SWITCH = 1500.0; // in units of max(a, b)

static double cerruti_ylog(double x_abs, double y_abs, double a, double b) {
    const double mmax = (a > b) ? a : b;
    const double r2 = x_abs * x_abs + y_abs * y_abs;
    if (r2 > (CERRUTI_FAR_SWITCH * mmax) * (CERRUTI_FAR_SWITCH * mmax)) {
        const double r = std::sqrt(r2);
        const double r3 = r2 * r, r5 = r3 * r2, r7 = r5 * r2;
        const double A = 4.0 * a * b;
        const double x2 = x_abs * x_abs, y2 = y_abs * y_abs;
        return A * (x2 / r3 +
                    (a * a / 6.0) * (2.0 / r3 - 15.0 * x2 / r5 +
                                     15.0 * x2 * x2 / r7) +
                    (b * b / 6.0) * (-3.0 * x2 / r5 + 15.0 * x2 * y2 / r7));
    }
    const double k = x_abs + a, l = x_abs - a; // k > 0; l may vanish
    const double m = y_abs + b, n = y_abs - b; // m > 0; n may vanish
    const double da = 2.0 * a;
    auto term = [](double t, double s1, double s2, double ds) {
        if (t == 0.0) return 0.0; // removable: t ln(1/|t|) -> 0
        const double c = std::fabs(t);
        return t * asinh_diff(s1 / c, s2 / c, ds / c);
    };
    // ylog = m ln[(k+R(k,m))/(l+R(l,m))] + n ln[(l+R(l,n))/(k+R(k,n))]
    return term(m, k, l, da) + term(n, l, k, -da);
}

double cerruti_uxx(double x, double y, double a, double b, double nu) {
    // (1-nu) xlog + ylog with xlog = Love - ylog is written as
    //     (1-nu) Love + nu ylog,
    // which is the form WITHOUT cancellation: on the axis Love and ylog are
    // nearly equal, so forming xlog = Love - ylog explicitly would throw away
    // the digits the two stable branches just bought.
    const double x_abs = std::abs(x), y_abs = std::abs(y);
    return (1.0 - nu) * love_uz(x_abs, y_abs, a, b) +
           nu * cerruti_ylog(x_abs, y_abs, a, b);
}

double cerruti_uxy(double x, double y, double a, double b) {
    const double x_abs = std::abs(x);
    const double y_abs = std::abs(y);
    const double k = x_abs + a, l = x_abs - a;
    const double m = y_abs + b, n = y_abs - b;
    // uxy has odd parity in both x and y, encoded in the corners
    const int sign_x = (x >= 0.0) ? 1 : -1;
    const int sign_y = (y >= 0.0) ? 1 : -1;
    // R(k,n) - R(k,m) + R(l,m) - R(l,n) is a difference of nearly equal radii
    // far from the cell, where the true value is O(A x y / r^3) while each
    // radius is O(r). Rationalising BOTH differences turns it into a product
    // with no subtraction at all:
    //   R(k,n)-R(k,m) = -4b|y| / (R(k,n)+R(k,m)),
    //   R(l,m)-R(l,n) = +4b|y| / (R(l,m)+R(l,n)),
    //   and their sum, after rationalising R(k,.)-R(l,.) = 4a|x|/(R(k,.)+R(l,.)),
    //   = 16 a b |x| |y| [1/(R(k,n)+R(l,n)) + 1/(R(k,m)+R(l,m))]
    //     / [(R(l,m)+R(l,n)) (R(k,n)+R(k,m))].
    // Every factor is positive, so the result is exact to a few ulp and
    // correctly vanishes on the axes.
    const double Rkm = std::hypot(k, m), Rkn = std::hypot(k, n);
    const double Rlm = std::hypot(l, m), Rln = std::hypot(l, n);
    const double num = 16.0 * a * b * x_abs * y_abs *
                       (1.0 / (Rkn + Rln) + 1.0 / (Rkm + Rlm));
    const double den = (Rlm + Rln) * (Rkn + Rkm);
    const double uxy_abs = (den > 0.0) ? num / den : 0.0;
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
