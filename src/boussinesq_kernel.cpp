#include "boussinesq_kernel.hpp"

namespace hmc {

// ── Stable evaluation of the Love (1929) cell integral (spec A05) ──────────
//
// The textbook form subtracts logarithms of corner expressions:
//     (x+a) ln[(y+b+R++)/(y-b+R+-)] + ...
// Two things go wrong with it (review §6). Inside a logarithm, `t + sqrt(t^2 +
// c^2)` cancels catastrophically when t is negative and |t| >> c — which is
// exactly what the H2 far callback produces, because it passes SIGNED offsets
// straight through. And at a cell corner (x = a, y = b) two of those
// arguments are exactly zero, so the expression returns NaN although the
// integral is perfectly finite. Measured relative errors of the old form
// against 80-digit arithmetic: 1.1e-13 at (32,0)h, 4.6e-10 at (16384,0)h,
// 1.9e-7 at (16384,16384)h and ~1 at (-1e6,0)h.
//
// Two changes fix it exactly:
//
//  1. Parity folding. The integral over a rectangle centred at the origin is
//     even in x and even in y, so evaluate at |x|,|y| and every argument that
//     could cancel becomes non-negative.
//  2. The asinh rearrangement. Since ln(t + sqrt(t^2+c^2)) = ln|c| +
//     asinh(t/|c|), each pair of logarithms differing only in t collapses to a
//     difference of asinh, and the common ln|c| — the divergent part — cancels
//     analytically instead of numerically:
//         L = sum over four terms of  t * [asinh(s1/|t|) - asinh(s2/|t|)].
//     asinh is accurate for arguments of either sign and any magnitude, and
//     the t -> 0 limit is removable (t ln(1/|t|) -> 0), which is what makes
//     the edge and corner cases finite.
//
// At very large separation the answer is O(A/r) while the surviving terms are
// O(1), so the subtraction itself loses about log10(r/max(a,b)) digits. Past
// FAR_SWITCH cell half-widths the multipole expansion is both cheaper and more
// accurate:
//     int dA'/|r-r'| = A/r + (A/(6 r^5)) [a^2 (3x^2 - r^2) + b^2 (3y^2 - r^2)]
//                      + O(A m^4 / r^5),   A = 4ab,  m = max(a,b),
// which follows from <x'^2> = a^2/3, <y'^2> = b^2/3 and the in-plane identity
// for the second derivatives of 1/r. For a square this is the familiar
// (h^2/r)[1 + h^2/(24 r^2)]. The switch is placed where the two error sources
// cross: the asinh form costs ~eps*(r/m) and the expansion ~(m/r)^4, so both
// sit near 1e-13 at r/m = 1000. This is why replacing the far kernel by a
// bare point kernel is NOT exact — the h^2/(24 r^2) term is real.
static constexpr double LOVE_FAR_SWITCH = 1500.0; // in units of max(a, b)

// asinh(u) - asinh(v), evaluated without cancellation.
// The difference is itself catastrophic when u ~ v, which is exactly the far
// regime (both arguments approach y/x): at r/max(a,b) = 1000 on the diagonal
// the naive difference lost ~11 digits and drove the kernel's relative error
// to 9e-12. Use sinh(A-B) = sinh A cosh B - cosh A sinh B, i.e.
//     asinh(u) - asinh(v) = asinh( u sqrt(1+v^2) - v sqrt(1+u^2) ),
// and evaluate that inner expression by its conjugate when u and v share a
// sign (where it cancels):
//     u sqrt(1+v^2) - v sqrt(1+u^2) = (u-v)(u+v) / (u sqrt(1+v^2) + v sqrt(1+u^2)).
// `du` is the exactly known difference u - v (2a/|t| or 2b/|t|), so no
// cancellation is reintroduced by forming it. When the signs differ the two
// products add in magnitude and the direct form is already stable — and that
// branch is the one that must be taken when u = -v, where the conjugate form
// is 0/0.
double asinh_diff(double u, double v, double du) {
    constexpr double HUGE_ARG = 1e150; // keep u*sqrt(1+v^2) from overflowing
    if (std::fabs(u) > HUGE_ARG || std::fabs(v) > HUGE_ARG)
        return std::asinh(u) - std::asinh(v);
    const double su = std::sqrt(1.0 + u * u), sv = std::sqrt(1.0 + v * v);
    const double den = u * sv + v * su;
    const double w = (u * v > 0.0 && den != 0.0) ? du * (u + v) / den
                                                 : u * sv - v * su;
    return std::asinh(w);
}

double love_uz(double x, double y, double a, double b) {
    x = std::fabs(x);
    y = std::fabs(y);
    const double m = (a > b) ? a : b;
    const double r2 = x * x + y * y;
    if (r2 > (LOVE_FAR_SWITCH * m) * (LOVE_FAR_SWITCH * m)) {
        const double r = std::sqrt(r2);
        const double A = 4.0 * a * b;
        const double quad =
            (a * a * (3.0 * x * x - r2) + b * b * (3.0 * y * y - r2)) /
            (6.0 * r2 * r2);
        return A * (1.0 + quad) / r;
    }
    // t * [asinh(s1/|t|) - asinh(s2/|t|)], with the removable t -> 0 limit.
    // ds is s1 - s2, which is exactly 2a or 2b (never formed by subtraction).
    auto term = [](double t, double s1, double s2, double ds) {
        if (t == 0.0) return 0.0;
        const double c = std::fabs(t);
        return t * asinh_diff(s1 / c, s2 / c, ds / c);
    };
    const double xp = x + a, xm = x - a, yp = y + b, ym = y - b;
    const double da = 2.0 * a, db = 2.0 * b;
    return term(xp, yp, ym, db) + term(yp, xp, xm, da) +
           term(xm, ym, yp, -db) + term(ym, xm, xp, -da);
}

BoussinesqKernel::BoussinesqKernel(int Ns, double L, double E_star)
    : Ns_(Ns), L_(L), h_(L / Ns), E_star_(E_star),
      table_(static_cast<std::size_t>(Ns) * Ns) {
    const double a = 0.5 * h_;
    const double scale = 1.0 / (M_PI * E_star_);
    for (int dy = 0; dy < Ns_; ++dy)
        for (int dx = 0; dx < Ns_; ++dx)
            table_[static_cast<std::size_t>(dy) * Ns_ + dx] =
                scale * love_uz(dx * h_, dy * h_, a, a);
}

Eigen::MatrixXd BoussinesqKernel::assemble_dense() const {
    const int N = size();
    Eigen::MatrixXd S(N, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            S(i, j) = entry(i, j);
    return S;
}

} // namespace hmc
