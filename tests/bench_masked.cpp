// M1 gate benchmark: masked vs unmasked H2 matvec cost on a real contact
// mask. Not a ctest target — run via bench_masked_gate.py, which generates
// the candidate masks (quick solve, threshold + dilate) and tabulates the
// masked/unmasked ratios that gate milestone M2 (plan
// doc/plans/2026-07-10-active-set-solver.md: proceed if masked <~ 5% of full
// at N_c/N ~ 5e-3).
//
// usage: bench_masked Ns maskfile leaf_side q reps_full reps_masked
//   maskfile: Ns*Ns raw uint8, natural flat order (iy*Ns + ix)
#include "boussinesq_kernel.hpp"
#include "h2_operator.hpp"
#include "uniform_quadtree.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <vector>

template <class F>
static double best_of(F&& f, int reps) {
    double best = 1e300;
    for (int i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        const std::chrono::duration<double> dt =
            std::chrono::steady_clock::now() - t0;
        best = std::min(best, dt.count());
    }
    return best;
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(stderr,
                     "usage: %s Ns maskfile leaf_side q reps_full reps_masked\n",
                     argv[0]);
        return 2;
    }
    const int Ns = std::atoi(argv[1]);
    const char* maskfile = argv[2];
    const int ls = std::atoi(argv[3]);
    const int q = std::atoi(argv[4]);
    const int reps_full = std::atoi(argv[5]);
    const int reps_masked = std::atoi(argv[6]);
    const std::size_t N = static_cast<std::size_t>(Ns) * Ns;

    std::vector<std::uint8_t> mask(N);
    {
        std::ifstream f(maskfile, std::ios::binary);
        if (!f.read(reinterpret_cast<char*>(mask.data()),
                    static_cast<std::streamsize>(N))) {
            std::fprintf(stderr, "cannot read %zu bytes from %s\n", N, maskfile);
            return 2;
        }
    }
    std::size_t nc = 0;
    for (auto v : mask) nc += v;

    hmc::BoussinesqKernel kernel(Ns, 1.0, 1.0);
    hmc::H2Operator op(kernel, hmc::H2Params{ls, q, 1});
    op.build();
    const hmc::H2Mask src = op.build_mask(mask);

    // occupied-leaf fraction (drives the near-field + P2M/L2P cost)
    const hmc::UniformQuadTree tree(Ns, ls);
    const int nls = Ns / ls;
    std::size_t occ_leaves = 0;
    for (int by = 0; by < nls; ++by)
        for (int bx = 0; bx < nls; ++bx)
            occ_leaves += src.box_occ[tree.box_id(tree.leaf_level(), bx, by)];
    std::size_t occ_boxes = 0;
    for (auto v : src.box_occ) occ_boxes += v;

    // x supported on the mask
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> val(0.0, 1.0);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(N);
    for (std::size_t i = 0; i < N; ++i)
        if (mask[i]) x(i) = val(rng);
    const Eigen::VectorXf xf = x.cast<float>();

    Eigen::VectorXd y;
    Eigen::VectorXf yf;

    op.matvec_into(x, y); // warm-up: sizes the M/L scratch
    const double t_full = best_of([&] { op.matvec_into(x, y); }, reps_full);
    const double t_restr =
        best_of([&] { op.matvec_masked_into(x, y, src, &src); }, reps_masked);
    const double t_verif =
        best_of([&] { op.matvec_masked_into(x, y, src, nullptr); }, reps_masked);

    op.matvec_single_into(xf, yf); // warm-up: float caches + scratch
    const double t_full_f =
        best_of([&] { op.matvec_single_into(xf, yf); }, reps_full);
    const double t_restr_f = best_of(
        [&] { op.matvec_masked_single_into(xf, yf, src, &src); }, reps_masked);
    const double t_verif_f = best_of(
        [&] { op.matvec_masked_single_into(xf, yf, src, nullptr); }, reps_masked);

    std::printf("Ns=%d ls=%d q=%d  N_c/N=%.2e  leaf_occ=%.4f  box_occ=%.4f\n",
                Ns, ls, q, double(nc) / double(N),
                double(occ_leaves) / (double(nls) * nls),
                double(occ_boxes) / double(src.box_occ.size()));
    std::printf("  f64: full %9.3f ms | restricted %8.3f ms (%.2f%%) | "
                "verify %8.3f ms (%.2f%%)\n",
                1e3 * t_full, 1e3 * t_restr, 100.0 * t_restr / t_full,
                1e3 * t_verif, 100.0 * t_verif / t_full);
    std::printf("  f32: full %9.3f ms | restricted %8.3f ms (%.2f%%) | "
                "verify %8.3f ms (%.2f%%)\n",
                1e3 * t_full_f, 1e3 * t_restr_f, 100.0 * t_restr_f / t_full_f,
                1e3 * t_verif_f, 100.0 * t_verif_f / t_full_f);
    return 0;
}
