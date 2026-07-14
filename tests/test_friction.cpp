#include "fourier_precond.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAILED: %s (line %d)\n", #cond, __LINE__);        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

// The tangential preconditioner's symbol is the analytic inverse of the
// Cerruti symbol up to an irrelevant overall scale:
//   M⁻¹(k) = |k_int| (I + γ k̂k̂ᵀ),  γ = ν/(1−ν)
// (integer wavenumbers; the r2c+c2r round-trip scale is folded in). On a
// full mask a pure Fourier mode is an exact eigenfunction (the grid FFT is
// exact): longitudinal polarization (v ∥ k) responds with |k|(1+γ),
// transverse (v ⊥ k) with |k|.
static int test_precond_symbol() {
    const int Ns = 64, N = Ns * Ns;
    const double nu = 0.3, gamma = nu / (1.0 - nu);
    hmc::TangentialFourierPreconditioner M(Ns, nu);
    const int mx = 3, my = 5;
    const double knorm = std::hypot(double(mx), double(my));
    const double kx = mx / knorm, ky = my / knorm; // unit k̂
    std::vector<std::uint8_t> mask(N, 1);

    auto run = [&](double vx, double vy, double lambda) {
        Eigen::VectorXd g(2 * N), z;
        for (int iy = 0; iy < Ns; ++iy)
            for (int ix = 0; ix < Ns; ++ix) {
                const double ph =
                    2.0 * M_PI * (mx * ix + my * iy) / double(Ns);
                g(iy * Ns + ix) = std::cos(ph) * vx;
                g(N + iy * Ns + ix) = std::cos(ph) * vy;
            }
        M.apply_into(g, mask, false, z);
        const double err = (z - lambda * g).lpNorm<Eigen::Infinity>() /
                           (lambda * g.lpNorm<Eigen::Infinity>());
        std::printf("precond mode response err %.3e (lambda %.6f)\n", err,
                    lambda);
        CHECK(err < 1e-10);
        return 0;
    };
    if (run(kx, ky, knorm * (1.0 + gamma))) return 1;  // longitudinal
    if (run(-ky, kx, knorm)) return 1;                 // transverse
    return 0;
}

// mask and mean-removal behavior
static int test_precond_mask_mean() {
    const int Ns = 32, N = Ns * Ns;
    hmc::TangentialFourierPreconditioner M(Ns, 0.3);
    std::vector<std::uint8_t> mask(N, 0);
    for (int i = 0; i < N; ++i) mask[i] = (i % 3 == 0) ? 1 : 0;
    Eigen::VectorXd g = Eigen::VectorXd::Random(2 * N), z;
    M.apply_into(g, mask, true, z);
    double mx = 0.0, my = 0.0;
    int nm = 0;
    for (int i = 0; i < N; ++i) {
        if (!mask[i]) {
            CHECK(z(i) == 0.0 && z(N + i) == 0.0); // strictly zero off-mask
        } else {
            mx += z(i);
            my += z(N + i);
            ++nm;
        }
    }
    CHECK(std::abs(mx / nm) < 1e-12 && std::abs(my / nm) < 1e-12);
    return 0;
}

int main() {
    if (int rc = test_precond_symbol()) return rc;
    if (int rc = test_precond_mask_mean()) return rc;
    std::printf("test_friction: all checks passed\n");
    return 0;
}
