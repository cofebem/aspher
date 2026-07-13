#include "boussinesq_kernel.hpp"
#include "cheb_basis.hpp"
#include "h2_operator.hpp"
#include "uniform_quadtree.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAILED: %s (line %d)\n", #cond, __LINE__);        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

static int test_cheb() {
    hmc::ChebBasis b(5);
    CHECK(static_cast<int>(b.nodes.size()) == 5);

    // partition of unity: weights sum to 1 at any t
    for (double t : {-0.9, -0.3, 0.0, 0.4, 1.0}) {
        const double s = b.weights(t).sum();
        CHECK(std::abs(s - 1.0) < 1e-12);
    }

    // interpolation exactness for polynomials of degree < q
    auto f = [](double x) {
        return 3 - 2 * x + 0.5 * x * x - x * x * x + 0.25 * x * x * x * x;
    };
    Eigen::VectorXd fnodes(5);
    for (int m = 0; m < 5; ++m) fnodes(m) = f(b.nodes[m]);
    for (double t : {-0.7, 0.1, 0.85}) {
        const double interp = b.weights(t).dot(fnodes);
        CHECK(std::abs(interp - f(t)) < 1e-10);
    }
    return 0;
}

static int test_tree() {
    hmc::UniformQuadTree t(16, 4); // 16x16 grid, leaf side 4 -> leaf level 2
    CHECK(t.leaf_level() == 2);
    CHECK(t.nlevels() == 3);
    CHECK(t.boxes_per_side(0) == 1);
    CHECK(t.boxes_per_side(2) == 4);

    // leaf ranges partition the grid
    int covered = 0;
    for (const auto& b : t.boxes())
        if (b.leaf) covered += b.side * b.side;
    CHECK(covered == 16 * 16);
    for (const auto& b : t.boxes())
        if (b.leaf) CHECK(b.side == 4);

    // near-neighbor counts (radius 1)
    const int interior = t.box_id(2, 1, 1);
    CHECK(static_cast<int>(t.neighbors(interior, 1).size()) == 9);
    const int corner = t.box_id(2, 0, 0);
    CHECK(static_cast<int>(t.neighbors(corner, 1).size()) == 4);

    // interaction list excludes near neighbors and stays at same level
    const auto il = t.interaction_list(interior, 1);
    const auto nb = t.neighbors(interior, 1);
    for (int s : il) {
        CHECK(std::find(nb.begin(), nb.end(), s) == nb.end());
        CHECK(t.boxes()[s].level == t.boxes()[interior].level);
    }

    // parent/child consistency
    for (const auto& b : t.boxes())
        if (!b.leaf)
            for (int c = 0; c < 4; ++c) {
                CHECK(b.child[c] >= 0);
                CHECK(t.boxes()[b.child[c]].parent == t.box_id(b.level, b.bx, b.by));
            }
    return 0;
}

static int test_caches() {
    hmc::BoussinesqKernel k(16, 1.0, 1.0);
    hmc::H2Operator A(k, {4, 4, 1});
    A.build();
    A.print_statistics();
    CHECK(A.n_unique_couplings() > 0);
    CHECK(A.n_unique_couplings() < A.n_far_interactions()); // sharing happened
    return 0;
}

static int test_accuracy() {
    const int Ns = 32;
    hmc::BoussinesqKernel k(Ns, 1.0, 1.0);
    const Eigen::MatrixXd S = k.assemble_dense();
    Eigen::VectorXd x = Eigen::VectorXd::Random(Ns * Ns);
    const Eigen::VectorXd ref = S * x;
    auto relerr = [&](int q) {
        hmc::H2Operator A(k, {8, q, 1});
        A.build();
        const Eigen::VectorXd y = A.matvec(x);
        return (y - ref).norm() / ref.norm();
    };
    const double e4 = relerr(4), e6 = relerr(6);
    std::printf("H2 rel err: q=4 %.3e  q=6 %.3e\n", e4, e6);
    CHECK(e4 < 5e-3);
    CHECK(e6 < e4);
    CHECK(e6 < 1e-4);
    return 0;
}

// M1 gate: the generic-kernel constructor fed the Boussinesq far/near
// functions must reproduce the kernel-constructed operator bit-for-bit.
static int test_functor_ctor() {
    const int Ns = 64;
    const double L = 1.0, E = 1.3, h = L / Ns;
    hmc::BoussinesqKernel K(Ns, L, E);
    hmc::H2Params prm{8, 4, 1};

    hmc::H2Operator A(K, prm);
    A.build();

    hmc::H2Operator B(
        Ns, h,
        [scale = 1.0 / (M_PI * E), a = 0.5 * h](double dx, double dy) {
            return scale * hmc::love_uz(dx, dy, a, a);
        },
        [&K](int dix, int diy) { return K.entry_offset(dix, diy); }, prm);
    B.build();

    std::mt19937 rng(3);
    std::uniform_real_distribution<double> d(0.0, 1.0);
    Eigen::VectorXd x(Ns * Ns);
    for (int i = 0; i < x.size(); ++i) x(i) = d(rng);
    const Eigen::VectorXd ya = A.matvec(x), yb = B.matvec(x);
    CHECK((ya - yb).lpNorm<Eigen::Infinity>() == 0.0); // bit-for-bit
    return 0;
}

int main() {
    if (int rc = test_cheb()) return rc;
    if (int rc = test_tree()) return rc;
    if (int rc = test_caches()) return rc;
    if (int rc = test_accuracy()) return rc;
    if (int rc = test_functor_ctor()) return rc;
    return 0;
}
