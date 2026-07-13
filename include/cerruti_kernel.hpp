#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace hmc {

// Element-integrated Cerruti surface displacements at z = 0 due to uniform
// UNIT tangential traction q_x on the rectangle [-a,a] x [-b,b], as UNSCALED
// brackets (love_uz convention). Sources: Pohrt & Li, Phys. Mesomech. 17
// (2014) eqs. (17)-(18); Dydo & Busby, J. Elasticity 38 (1995). Note eq. (18)
// as printed carries spurious h^2 factors (dimensional inconsistency with
// their eq. (13)); the corner-evaluation form below is the correct rectangle
// integral of  x'y'/rho^3  (verified against quadrature in test_cerruti).
//
//   u_x = [1/(2 pi G)] * cerruti_uxx(x, y, a, b, nu),  1/(2 pi G) = 1/(pi E* (1-nu))
//   u_y = [nu/(2 pi G)] * cerruti_uxy(x, y, a, b)
//
// q_y loading by x<->y symmetry: u_y = [1/(2piG)] cerruti_uxx(y, x, b, a, nu),
// u_x = [nu/(2piG)] cerruti_uxy(x, y, a, b) (same mixed bracket).
double cerruti_uxx(double x, double y, double a, double b, double nu);
double cerruti_uxy(double x, double y, double a, double b);

// Tangential (Cerruti) influence tables for an Ns x Ns grid of square
// elements of side h = L/Ns carrying uniform tangential traction, mirror of
// BoussinesqKernel. Only the xx and xy tables are stored: yy is the x<->y
// transpose of xx, and xy's signs are restored from the odd parity
//   xx(dx,dy) even in dx and dy;  yy(dx,dy) = xx(dy,dx);
//   xy odd in dx and odd in dy (stored for |dx|,|dy|; zero on axes to within ~1 ULP).
// Prefactors: 1/(2 pi G) = 1/(pi E* (1-nu)) on xx/yy; extra nu on xy.
class CerrutiKernel {
public:
    CerrutiKernel(int Ns, double L, double E_star, double nu);

    int    grid_size() const { return Ns_; }
    int    size() const { return Ns_ * Ns_; }
    double element_size() const { return h_; }
    double domain_size() const { return L_; }
    double E_star() const { return E_star_; }
    double nu() const { return nu_; }

    double xx_offset(int dix, int diy) const {
        const int dx = std::abs(dix), dy = std::abs(diy);
        if (dx >= Ns_ || dy >= Ns_) return 0.0;
        return xx_[static_cast<std::size_t>(dy) * Ns_ + dx];
    }
    double yy_offset(int dix, int diy) const { return xx_offset(diy, dix); }
    double xy_offset(int dix, int diy) const {
        const int dx = std::abs(dix), dy = std::abs(diy);
        if (dx >= Ns_ || dy >= Ns_) return 0.0;
        const double v = xy_[static_cast<std::size_t>(dy) * Ns_ + dx];
        return ((dix > 0) == (diy > 0)) ? v : -v; // axis entries are zero to within ~1 ULP
    }

    double xx_far(double dx, double dy) const {
        return pref_ * cerruti_uxx(dx, dy, a_, a_, nu_);
    }
    double yy_far(double dx, double dy) const { return xx_far(dy, dx); }
    double xy_far(double dx, double dy) const {
        return prefnu_ * cerruti_uxy(dx, dy, a_, a_);
    }

    Eigen::MatrixXd assemble_dense() const;
    Eigen::Matrix2d symbol(double kx, double ky) const;

private:
    int Ns_;
    double L_, h_, a_, E_star_, nu_, pref_, prefnu_;
    std::vector<double> xx_, xy_;
};

} // namespace hmc
