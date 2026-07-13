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

} // namespace hmc
