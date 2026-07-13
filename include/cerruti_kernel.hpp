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

} // namespace hmc
