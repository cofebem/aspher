#include "bipotential.hpp"

#include <cmath>
#include <cstdio>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAILED: %s (line %d)\n", #cond, __LINE__);        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

// The analytic cone projection: verify the three sectors and the projection
// property (idempotence + distance minimization checked against a dense
// sampled search on the cone boundary).
static int test_cone_projection() {
    const double mu = 0.4;
    // separation sector: mu|t_t| < -t_n -> zero
    {
        double tx = 0.1, ty = 0.0, tn = -0.5;
        hmc::project_coulomb_cone(mu, tx, ty, tn);
        CHECK(tx == 0.0 && ty == 0.0 && tn == 0.0);
    }
    // stick sector: |t_t| <= mu t_n -> unchanged
    {
        double tx = 0.1, ty = 0.05, tn = 1.0;
        const double tx0 = tx, ty0 = ty, tn0 = tn;
        hmc::project_coulomb_cone(mu, tx, ty, tn);
        CHECK(tx == tx0 && ty == ty0 && tn == tn0);
    }
    // slip sector: radial return lands ON the cone, is idempotent, and is
    // closer to tau than any sampled cone-boundary point
    {
        const double tx0 = 1.0, ty0 = 0.3, tn0 = 0.8; // |t_t| > mu tn
        double tx = tx0, ty = ty0, tn = tn0;
        hmc::project_coulomb_cone(mu, tx, ty, tn);
        const double qt = std::hypot(tx, ty);
        CHECK(std::abs(qt - mu * tn) <= 1e-14 * std::max(1.0, qt));
        double tx2 = tx, ty2 = ty, tn2 = tn;
        hmc::project_coulomb_cone(mu, tx2, ty2, tn2);
        CHECK(tx2 == tx && ty2 == ty && tn2 == tn);
        const double d2 = (tx - tx0) * (tx - tx0) + (ty - ty0) * (ty - ty0) +
                          (tn - tn0) * (tn - tn0);
        for (int k = 0; k <= 200; ++k) { // sample the boundary generatrix
            const double pn = 0.01 * k;
            // boundary point along the tangential direction of tau
            const double tt = std::hypot(tx0, ty0);
            const double bx = mu * pn * tx0 / tt, by = mu * pn * ty0 / tt;
            const double dd = (bx - tx0) * (bx - tx0) +
                              (by - ty0) * (by - ty0) +
                              (pn - tn0) * (pn - tn0);
            CHECK(d2 <= dd + 1e-12);
        }
    }
    return 0;
}

int main() {
    if (int rc = test_cone_projection()) return rc;
    std::printf("test_uzawa: all checks passed\n");
    return 0;
}
