#include "friction_model.hpp"

#include <cmath>
#include <cstdio>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAILED: %s (line %d)\n", #cond, __LINE__);        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

static int test_models() {
    const int N = 6;
    Eigen::VectorXd p(N), v(N), T(N), s;
    p << 1.0, 0.5, 0.0, -0.1, 2.0, 0.0;
    v << 0.0, 1.0, 2.0, 3.0, 0.5, 0.0;
    T.setConstant(300.0);

    hmc::TrescaModel tresca(0.07);
    tresca.threshold(p, v, T, s);
    CHECK(s.size() == N);
    CHECK(s(0) == 0.07 && s(1) == 0.07 && s(4) == 0.07);
    CHECK(s(2) == 0.0 && s(3) == 0.0 && s(5) == 0.0); // p <= 0 -> s = 0
    CHECK(!tresca.velocity_dependent());

    hmc::CoulombModel coulomb(0.3);
    coulomb.threshold(p, v, T, s);
    CHECK(s(0) == 0.3 && s(1) == 0.15 && s(3) == 0.0);
    CHECK(!coulomb.velocity_dependent());

    // callback: rate-weakening law; wrapper must sanitize a sloppy callback
    // that ignores the p <= 0 rule and can return negatives
    hmc::CallbackModel user(
        [](const Eigen::VectorXd& pp, const Eigen::VectorXd& vv,
           const Eigen::VectorXd& TT, Eigen::VectorXd& ss) {
            ss = 0.3 * pp.array() / (1.0 + vv.array()) -
                 0.0 * TT.array(); // uses T to avoid unused warnings
        },
        /*velocity_dependent=*/true);
    user.threshold(p, v, T, s);
    CHECK(std::abs(s(0) - 0.3) < 1e-15);
    CHECK(std::abs(s(1) - 0.15 / 2.0) < 1e-15);
    CHECK(s(3) == 0.0); // raw callback returns -0.03/4: sanitized to 0
    CHECK(s(2) == 0.0 && s(5) == 0.0);
    CHECK(user.velocity_dependent());
    return 0;
}

int main() {
    if (int rc = test_models()) return rc;
    std::printf("test_driver: all checks passed\n");
    return 0;
}
