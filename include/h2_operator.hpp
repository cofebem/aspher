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

// Per-box occupancy bitmap over the complete quad-tree: box_occ[b] != 0 iff
// box b's element range intersects the marked grid set. Built by
// H2Operator::build_mask; leaf occupancy is OR-propagated to all ancestors,
// so the occupied set is closed under taking parents (an occupied non-root
// box always has an occupied parent — L2L relies on this).
struct H2Mask {
    std::vector<std::uint8_t> box_occ; // size = number of tree boxes
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

    // Tree occupancy mask from a grid mask (size N, natural flat order,
    // nonzero = marked): marks the leaf boxes containing marked elements and
    // OR-propagates to ancestors. O(N + nbox).
    H2Mask build_mask(const std::vector<std::uint8_t>& grid_mask) const;
    // Same from a flat-index list (entries in [0, N)). O(N_c + nbox).
    H2Mask build_mask(const int* idx, std::size_t n) const;

    // Masked matvec for the active-set solver: every pass skips contributions
    // outside the masks.
    //   src — source occupancy; x MUST vanish outside src-occupied leaves.
    //   tgt — target occupancy, or nullptr for all targets (the verification
    //         mode: full-grid output from a masked source).
    // For x supported on src the output equals the unmasked matvec
    // BIT-FOR-BIT on tgt-occupied leaves: skipped terms are exact zeros and
    // kept terms keep their summation order. y is resized to N but entries
    // outside tgt-occupied leaves are unspecified (stale) — no O(N) zeroing
    // is performed per apply. Shares the M/L scratch with the unmasked path:
    // concurrent applies on one operator are not supported.
    void matvec_masked_into(const Eigen::VectorXd& x, Eigen::VectorXd& y,
                            const H2Mask& src,
                            const H2Mask* tgt = nullptr) const;
    void matvec_masked_single_into(const Eigen::VectorXf& x, Eigen::VectorXf& y,
                                   const H2Mask& src,
                                   const H2Mask* tgt = nullptr) const;

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

    // Masked variant of matvec_impl (same caches, same scratch): occupancy
    // guards on every pass, dynamic scheduling (occupied boxes are clustered,
    // static chunks would idle most threads). Guards only skip exact-zero
    // contributions and never reorder kept sums, so the tgt-leaf output is
    // bit-identical to the unmasked pass for x supported on src.
    template <class S>
    void matvec_masked_impl(
        const Eigen::Matrix<S, Eigen::Dynamic, 1>& x,
        Eigen::Matrix<S, Eigen::Dynamic, 1>& y,
        const H2Mask& src, const H2Mask* tgt,
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

// ── matvec_masked_impl (header-defined member template) ───────────────────────
template <class S>
void H2Operator::matvec_masked_impl(
    const Eigen::Matrix<S, Eigen::Dynamic, 1>& x,
    Eigen::Matrix<S, Eigen::Dynamic, 1>& y,
    const H2Mask& src, const H2Mask* tgt,
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
    const std::uint8_t* so = src.box_occ.data();
    const std::uint8_t* to = tgt ? tgt->box_occ.data() : nullptr;
    if (M.cols() != nbox) M.resize(q2_, nbox);
    if (L.cols() != nbox) L.resize(q2_, nbox);
    if (y.size() != x.size()) y.resize(x.size());

    // P2M on src-occupied leaves only. Skipped columns keep stale data, but
    // M2M/M2L read a column only when its box is src-occupied.
#pragma omp parallel
    {
        Mat XsT(ls_, ls_), T(q_, ls_), B(q_, q_);
#pragma omp for schedule(dynamic, 32)
        for (int li = 0; li < nleaf; ++li) {
            const int t = leaves_[li];
            if (!so[t]) continue;
            const int ix0 = boxes[t].ix0, iy0 = boxes[t].iy0;
            for (int ly = 0; ly < ls_; ++ly)
                XsT.col(ly) = x.segment((iy0 + ly) * Ns_ + ix0, ls_);
            T.noalias() = Wleaf.transpose() * XsT;
            B.noalias() = T * Wleaf;
            M.col(t) = Eigen::Map<const Vec>(B.data(), q2_);
        }
    }

    // M2M: occupied parents combine their occupied children only (occupancy
    // is the OR of the children's, so an occupied parent has one).
    for (int l = tree_.leaf_level() - 1; l >= 0; --l) {
        const int b0 = tree_.level_begin(l);
        const int nb = (1 << l) * (1 << l);
#pragma omp parallel for schedule(dynamic, 32)
        for (int k = 0; k < nb; ++k) {
            const int b = b0 + k;
            if (!so[b]) continue;
            auto Mb = M.col(b);
            Mb.setZero();
            for (int c = 0; c < 4; ++c) {
                const int cc = boxes[b].child[c];
                if (cc >= 0 && so[cc]) Mb.noalias() += R[c] * M.col(cc);
            }
        }
    }

    // M2L on tgt boxes; non-src sources are skipped inside the CSR walk (one
    // byte load per interaction — their moments are exact zeros for x
    // supported on src, so skipping them changes nothing).
#pragma omp parallel for schedule(dynamic, 64)
    for (int t = 0; t < nbox; ++t) {
        if (to && !to[t]) continue;
        auto Lt = L.col(t);
        Lt.setZero();
        for (std::int64_t k = far_off_[t]; k < far_off_[t + 1]; ++k)
            if (so[far_[k].source_box])
                Lt.noalias() += couplings[far_[k].coupling_id] * M.col(far_[k].source_box);
    }

    // L2L: the tgt set is closed under parents, so L.col(parent) is valid.
    for (int l = 1; l <= tree_.leaf_level(); ++l) {
        const int b0 = tree_.level_begin(l);
        const int nb = (1 << l) * (1 << l);
#pragma omp parallel for schedule(dynamic, 32)
        for (int k = 0; k < nb; ++k) {
            const int b = b0 + k;
            if (to && !to[b]) continue;
            const int c = (boxes[b].bx & 1) + 2 * (boxes[b].by & 1);
            L.col(b).noalias() += R[c].transpose() * L.col(boxes[b].parent);
        }
    }

    // L2P + near on tgt leaves; near sums skip non-src source leaves (their
    // x fields are zero).
#pragma omp parallel
    {
        Mat T(ls_, q_), YtT(ls_, ls_);
        Vec yloc(ls2_), xs(ls2_);
#pragma omp for schedule(dynamic, 32)
        for (int li = 0; li < nleaf; ++li) {
            const int t = leaves_[li];
            if (to && !to[t]) continue;
            const int ix0 = boxes[t].ix0, iy0 = boxes[t].iy0;
            Eigen::Map<const Mat> LmT(L.col(t).data(), q_, q_);
            T.noalias() = Wleaf * LmT;
            YtT.noalias() = T * Wleaf.transpose();

            yloc.setZero();
            for (std::int64_t k = near_off_[li]; k < near_off_[li + 1]; ++k) {
                if (!so[near_[k].source_box]) continue;
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
