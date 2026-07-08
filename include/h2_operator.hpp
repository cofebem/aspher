#pragma once

#include "boussinesq_kernel.hpp"
#include "cheb_basis.hpp"
#include "uniform_quadtree.hpp"

#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <vector>

namespace hmc {

struct H2Params {
    int leaf_side = 8;   // square leaf side in elements (power of two)
    int q = 4;           // Chebyshev interpolation order (rank r = q*q)
    int near_radius = 1; // direct near field within this many leaf boxes
};

struct H2Info {
    int N = 0, Ns = 0, nlevels = 0, leaf_side = 0, q = 0, r = 0;
    int n_boxes = 0, n_leaves = 0;
    std::int64_t n_near_interactions = 0, n_far_interactions = 0;
    int n_unique_couplings = 0, n_near_stencils = 0;
    std::int64_t bytes_coupling = 0, bytes_near = 0, bytes_buffers = 0, bytes_total = 0;
};

// Matrix-free black-box FMM (Fong & Darve 2009) operator for the translation-
// invariant Boussinesq half-space kernel on a uniform Ns x Ns grid. Far field
// via tensor-product Chebyshev interpolation with cached, translation-invariant
// transfer (M2M/L2L) and coupling (M2L) operators; near field via exact Love
// stencils cached by relative leaf offset. O(N) memory and matvec.
class H2Operator {
public:
    H2Operator(const BoussinesqKernel& kernel, H2Params params);

    void build();

    // u = S x, with x and u in natural flat order (global = iy*Ns + ix).
    // Reuses internal multipole/local scratch across calls: concurrent matvec
    // calls on the same operator are not supported.
    Eigen::VectorXd matvec(const Eigen::VectorXd& x) const;

    // Allocation-free variant: writes into y (resized if needed). Lets the CG
    // loop reuse one output buffer instead of allocating N doubles per apply.
    void matvec_into(const Eigen::VectorXd& x, Eigen::VectorXd& y) const;

    // Single-precision matvec: builds float copies of the caches on first use
    // (idempotent) and runs the passes in float, halving the O(N) working set.
    Eigen::VectorXf matvec_single(const Eigen::VectorXf& x) const;
    void matvec_single_into(const Eigen::VectorXf& x, Eigen::VectorXf& y) const;
    void build_single_caches() const;

    H2Info info() const { return info_; }
    void print_statistics() const;

    int n_far_interactions() const { return static_cast<int>(info_.n_far_interactions); }
    int n_unique_couplings() const { return info_.n_unique_couplings; }

private:
    struct FarInter { int source_box; int coupling_id; };
    struct NearInter { int source_box; int stencil_id; };

    // 1D element-center coordinates of a box, normalized to [-1, 1] (independent
    // of the box origin; depends only on the side length).
    std::vector<double> centers_norm(int side) const;

    const BoussinesqKernel* kernel_;
    H2Params p_;
    int Ns_, q_, q2_, ls_, ls2_;
    double h_, scale_; // scale_ = 1 / (pi E*)

    ChebBasis cheb_;
    UniformQuadTree tree_;
    Eigen::MatrixXd Wleaf_;                 // (ls x q) leaf interpolation
    std::array<Eigen::MatrixXd, 4> R_;      // (q2 x q2) M2M per child quadrant

    std::vector<int> leaves_;
    // CSR interaction lists: entries of box t are far_[far_off_[t]..far_off_[t+1]),
    // entries of leaf li are near_[near_off_[li]..near_off_[li+1])
    std::vector<std::int64_t> far_off_;
    std::vector<FarInter> far_;
    std::vector<std::int64_t> near_off_;
    std::vector<NearInter> near_;

    std::vector<Eigen::MatrixXd> couplings_;             // q2 x q2, by id
    std::vector<Eigen::MatrixXd> near_stencils_;         // ls2 x ls2, by id

    // float copies of the numeric caches (built lazily by build_single_caches)
    mutable bool have_single_ = false;
    mutable Eigen::MatrixXf Wleaf_f_;
    mutable std::array<Eigen::MatrixXf, 4> R_f_;
    mutable std::vector<Eigen::MatrixXf> couplings_f_, near_stencils_f_;

    // multipole/local scratch (q2 x nbox), sized lazily on first matvec of each
    // precision and reused across calls — the per-call allocation of ~2*nbox
    // small vectors dominated the matvec setup cost at large Ns.
    mutable Eigen::MatrixXd Mbuf_, Lbuf_;
    mutable Eigen::MatrixXf Mbuf_f_, Lbuf_f_;

    H2Info info_;

    double far_kernel(double dx, double dy) const; // g(dx,dy)

    // Scalar-templated matvec: the passes are identical for double/float, only
    // the numeric caches differ. The tree/leaf/interaction indices are shared.
    template <class S>
    void matvec_impl(
        const Eigen::Matrix<S, Eigen::Dynamic, 1>& x,
        Eigen::Matrix<S, Eigen::Dynamic, 1>& y,
        const Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& Wleaf,
        const std::array<Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>, 4>& R,
        const std::vector<Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>>& couplings,
        const std::vector<Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>>& near_st,
        Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& M,
        Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& L) const;
};

// ── matvec_impl (header-defined member template) ──────────────────────────────
template <class S>
void H2Operator::matvec_impl(
    const Eigen::Matrix<S, Eigen::Dynamic, 1>& x,
    Eigen::Matrix<S, Eigen::Dynamic, 1>& y,
    const Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& Wleaf,
    const std::array<Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>, 4>& R,
    const std::vector<Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>>& couplings,
    const std::vector<Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>>& near_st,
    Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& M,
    Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& L) const {
    using Vec = Eigen::Matrix<S, Eigen::Dynamic, 1>;
    using Mat = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;
    const auto& boxes = tree_.boxes();
    const int nbox = static_cast<int>(boxes.size());
    const int nleaf = static_cast<int>(leaves_.size());
    if (M.cols() != nbox) M.resize(q2_, nbox);
    if (L.cols() != nbox) L.resize(q2_, nbox);
    if (y.size() != x.size()) y.resize(x.size());

    // P2M: with XsT(lx,ly) = x at (ix0+lx, iy0+ly), the moment vector
    // M(ax + q*ay) equals vec(Wleaf' * XsT * Wleaf) in column-major order.
#pragma omp parallel
    {
        Mat XsT(ls_, ls_), T(q_, ls_), B(q_, q_);
#pragma omp for schedule(static)
        for (int li = 0; li < nleaf; ++li) {
            const int t = leaves_[li];
            const int ix0 = boxes[t].ix0, iy0 = boxes[t].iy0;
            for (int ly = 0; ly < ls_; ++ly)
                XsT.col(ly) = x.segment((iy0 + ly) * Ns_ + ix0, ls_);
            T.noalias() = Wleaf.transpose() * XsT;
            B.noalias() = T * Wleaf;
            M.col(t) = Eigen::Map<const Vec>(B.data(), q2_);
        }
    }

    // M2M
    for (int l = tree_.leaf_level() - 1; l >= 0; --l) {
        const int b0 = tree_.level_begin(l);
        const int nb = (1 << l) * (1 << l);
#pragma omp parallel for schedule(static)
        for (int k = 0; k < nb; ++k) {
            const int b = b0 + k;
            auto Mb = M.col(b);
            Mb.setZero();
            for (int c = 0; c < 4; ++c) {
                const int cc = boxes[b].child[c];
                if (cc >= 0) Mb.noalias() += R[c] * M.col(cc);
            }
        }
    }

    // M2L
#pragma omp parallel for schedule(dynamic, 64)
    for (int t = 0; t < nbox; ++t) {
        auto Lt = L.col(t);
        Lt.setZero();
        for (std::int64_t k = far_off_[t]; k < far_off_[t + 1]; ++k)
            Lt.noalias() += couplings[far_[k].coupling_id] * M.col(far_[k].source_box);
    }

    // L2L
    for (int l = 1; l <= tree_.leaf_level(); ++l) {
        const int b0 = tree_.level_begin(l);
        const int nb = (1 << l) * (1 << l);
#pragma omp parallel for schedule(static)
        for (int k = 0; k < nb; ++k) {
            const int b = b0 + k;
            const int c = (boxes[b].bx & 1) + 2 * (boxes[b].by & 1);
            L.col(b).noalias() += R[c].transpose() * L.col(boxes[b].parent);
        }
    }

    // L2P + near. With LmT = Map(L.col(t)) as a (q x q) matrix (ax, ay), the
    // transposed leaf field YtT(lx,ly) = Wleaf * LmT * Wleaf'.
#pragma omp parallel
    {
        Mat T(ls_, q_), YtT(ls_, ls_);
        Vec yloc(ls2_), xs(ls2_);
#pragma omp for schedule(static)
        for (int li = 0; li < nleaf; ++li) {
            const int t = leaves_[li];
            const int ix0 = boxes[t].ix0, iy0 = boxes[t].iy0;
            Eigen::Map<const Mat> LmT(L.col(t).data(), q_, q_);
            T.noalias() = Wleaf * LmT;
            YtT.noalias() = T * Wleaf.transpose();

            yloc.setZero();
            for (std::int64_t k = near_off_[li]; k < near_off_[li + 1]; ++k) {
                const H2Box& sb = boxes[near_[k].source_box];
                for (int ly = 0; ly < ls_; ++ly)
                    xs.segment(ly * ls_, ls_) =
                        x.segment((sb.iy0 + ly) * Ns_ + sb.ix0, ls_);
                yloc.noalias() += near_st[near_[k].stencil_id] * xs;
            }
            for (int ly = 0; ly < ls_; ++ly)
                y.segment((iy0 + ly) * Ns_ + ix0, ls_) =
                    YtT.col(ly) + yloc.segment(ly * ls_, ls_);
        }
    }
}

} // namespace hmc
