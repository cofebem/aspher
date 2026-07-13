#pragma once

#include "boussinesq_kernel.hpp"
#include "cheb_basis.hpp"
#include "uniform_quadtree.hpp"

#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <functional>
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
//
// Slot map (the O(N_c) compressed representation): occupied leaves get
// consecutive slot ids in leaf order. A compressed vector has
// nslots()*ls² entries; slot s stores its leaf's elements contiguously in
// local row-major order, entry s*ls² + ly*ls + lx ↔ grid element
// (ix0 + lx, iy0 + ly). This is a superset of the marked set (whole leaves),
// ~2–4 N_c for contact-like masks.
struct H2Mask {
    std::vector<std::uint8_t> box_occ; // size = number of tree boxes
    std::vector<int> leaf_slot;        // per leaf li (leaf order): slot or -1
    std::vector<int> slot_leaf;        // per slot: leaf index li
    int nslots() const { return static_cast<int>(slot_leaf.size()); }
};

// Kernel plumbing (friction M1): the far kernel is the element-integrated
// influence evaluated at a continuous center offset (dx, dy); the near kernel
// is the exact element-offset entry. Both are FULLY SCALED (elastic constants
// included) and are called only during build() — couplings and stencils are
// cached — so the indirection never touches the matvec path.
using FarKernelFn = std::function<double(double, double)>;
using NearKernelFn = std::function<double(int, int)>;

// Matrix-free black-box FMM (Fong & Darve 2009) operator for the translation-
// invariant Boussinesq half-space kernel on a uniform Ns x Ns grid. Far field
// via tensor-product Chebyshev interpolation with cached, translation-invariant
// transfer (M2M/L2L) and coupling (M2L) operators; near field via exact Love
// stencils cached by relative leaf offset. O(N) memory and matvec.
class H2Operator {
public:
    H2Operator(const BoussinesqKernel& kernel, H2Params params);
    // Generic translation-invariant kernel. far/near must be fully scaled;
    // near's backing storage (e.g. a kernel table) must outlive the operator,
    // matching the kernel-reference contract of the other constructor.
    H2Operator(int Ns, double h, FarKernelFn far, NearKernelFn near,
               H2Params params);

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

    // ── O(N_c) compressed variants (active-set milestone M3) ──
    // Flat grid index of every compressed-vector entry, in slot order
    // (size nslots()*ls²): the driver's gather/scatter map.
    std::vector<int> slot_grid_indices(const H2Mask& mask) const;

    // Compressed masked matvec, src = tgt = mask: xc and yc are compressed
    // vectors (nslots()*ls² entries, see H2Mask). No N-sized array anywhere.
    // Same bit-for-bit contract as matvec_masked_into: identical values in
    // identical summation order, only the addressing differs.
    void matvec_masked_compressed_into(const Eigen::VectorXd& xc,
                                       Eigen::VectorXd& yc,
                                       const H2Mask& mask) const;
    void matvec_masked_compressed_single_into(const Eigen::VectorXf& xc,
                                              Eigen::VectorXf& yc,
                                              const H2Mask& mask) const;

    // Streamed full-target matvec from a compressed masked source (the M3
    // verification pass): for EVERY leaf of the grid, computes the ls² output
    // tile (tile[ly*ls+lx] = u at (ix0+lx, iy0+ly)) and hands it to sink,
    // then discards it — O(N) compute, O(ls² per thread) output memory.
    // sink is called concurrently from the OpenMP team: it must be
    // thread-safe (leaves are disjoint, so writes keyed by leaf are safe).
    void matvec_masked_stream(
        const Eigen::VectorXd& xc, const H2Mask& src,
        const std::function<void(int ix0, int iy0, const double* tile)>& sink)
        const;
    void matvec_masked_stream_single(
        const Eigen::VectorXf& xc, const H2Mask& src,
        const std::function<void(int ix0, int iy0, const float* tile)>& sink)
        const;

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

    H2Params p_;
    int Ns_, q_, q2_, ls_, ls2_;
    double h_;
    FarKernelFn far_fn_;   // fully scaled far kernel (build()-time only)
    NearKernelFn near_fn_; // fully scaled near element entry (build()-time only)

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

    // Shared masked tree passes (M2M / M2L / L2L), used by the grid,
    // compressed, and streamed masked matvecs. Extracted verbatim from
    // matvec_masked_impl — identical operations in identical order, so all
    // masked variants inherit the bit-for-bit contract.
    template <class S>
    void masked_upward(
        const H2Mask& src,
        const std::array<Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>, 4>& R,
        Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& M) const;
    template <class S>
    void masked_coupling(
        const H2Mask& src, const H2Mask* tgt,
        const std::vector<Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>>& couplings,
        const Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& M,
        Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& L) const;
    template <class S>
    void masked_downward(
        const H2Mask* tgt,
        const std::array<Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>, 4>& R,
        Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& L) const;

    // P2M from a compressed (slot-blocked) source vector.
    template <class S>
    void p2m_compressed(
        const Eigen::Matrix<S, Eigen::Dynamic, 1>& xc, const H2Mask& src,
        const Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& Wleaf,
        Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& M) const;

    template <class S>
    void matvec_masked_compressed_impl(
        const Eigen::Matrix<S, Eigen::Dynamic, 1>& xc,
        Eigen::Matrix<S, Eigen::Dynamic, 1>& yc, const H2Mask& mask,
        const Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& Wleaf,
        const std::array<Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>, 4>& R,
        const std::vector<Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>>& couplings,
        const std::vector<Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>>& near_st,
        Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& M,
        Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& L) const;

    template <class S>
    void matvec_masked_stream_impl(
        const Eigen::Matrix<S, Eigen::Dynamic, 1>& xc, const H2Mask& src,
        const std::function<void(int, int, const S*)>& sink,
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

    masked_upward<S>(src, R, M);
    masked_coupling<S>(src, tgt, couplings, M, L);
    masked_downward<S>(tgt, R, L);

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

// ── shared masked tree passes ─────────────────────────────────────────────────
// M2M: occupied parents combine their occupied children only (occupancy is
// the OR of the children's, so an occupied parent has one).
template <class S>
void H2Operator::masked_upward(
    const H2Mask& src,
    const std::array<Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>, 4>& R,
    Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& M) const {
    const auto& boxes = tree_.boxes();
    const std::uint8_t* so = src.box_occ.data();
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
}

// M2L on tgt boxes (nullptr → all); non-src sources are skipped inside the
// CSR walk (one byte load per interaction — their moments are exact zeros
// for x supported on src, so skipping them changes nothing).
template <class S>
void H2Operator::masked_coupling(
    const H2Mask& src, const H2Mask* tgt,
    const std::vector<Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>>& couplings,
    const Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& M,
    Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& L) const {
    const int nbox = static_cast<int>(tree_.boxes().size());
    const std::uint8_t* so = src.box_occ.data();
    const std::uint8_t* to = tgt ? tgt->box_occ.data() : nullptr;
#pragma omp parallel for schedule(dynamic, 64)
    for (int t = 0; t < nbox; ++t) {
        if (to && !to[t]) continue;
        auto Lt = L.col(t);
        Lt.setZero();
        for (std::int64_t k = far_off_[t]; k < far_off_[t + 1]; ++k)
            if (so[far_[k].source_box])
                Lt.noalias() += couplings[far_[k].coupling_id] * M.col(far_[k].source_box);
    }
}

// L2L: the tgt set is closed under parents, so L.col(parent) is valid.
template <class S>
void H2Operator::masked_downward(
    const H2Mask* tgt,
    const std::array<Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>, 4>& R,
    Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& L) const {
    const auto& boxes = tree_.boxes();
    const std::uint8_t* to = tgt ? tgt->box_occ.data() : nullptr;
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
}

// P2M from a compressed source: slot s IS the leaf's element block in local
// row-major order, so XsT is a direct Map — same values, same products as
// the grid gather, hence bit-identical moments.
template <class S>
void H2Operator::p2m_compressed(
    const Eigen::Matrix<S, Eigen::Dynamic, 1>& xc, const H2Mask& src,
    const Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& Wleaf,
    Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& M) const {
    using Vec = Eigen::Matrix<S, Eigen::Dynamic, 1>;
    using Mat = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;
    const int nslot = src.nslots();
#pragma omp parallel
    {
        Mat T(q_, ls_), B(q_, q_);
#pragma omp for schedule(static)
        for (int s = 0; s < nslot; ++s) {
            const int t = leaves_[src.slot_leaf[s]];
            Eigen::Map<const Mat> XsT(
                xc.data() + static_cast<std::ptrdiff_t>(s) * ls2_, ls_, ls_);
            T.noalias() = Wleaf.transpose() * XsT;
            B.noalias() = T * Wleaf;
            M.col(t) = Eigen::Map<const Vec>(B.data(), q2_);
        }
    }
}

// ── matvec_masked_compressed_impl ─────────────────────────────────────────────
template <class S>
void H2Operator::matvec_masked_compressed_impl(
    const Eigen::Matrix<S, Eigen::Dynamic, 1>& xc,
    Eigen::Matrix<S, Eigen::Dynamic, 1>& yc, const H2Mask& mask,
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
    const int nslot = mask.nslots();
    const int leaf_begin = tree_.level_begin(tree_.leaf_level());
    const std::uint8_t* so = mask.box_occ.data();
    if (M.cols() != nbox) M.resize(q2_, nbox);
    if (L.cols() != nbox) L.resize(q2_, nbox);
    if (yc.size() != xc.size()) yc.resize(xc.size());

    p2m_compressed<S>(xc, mask, Wleaf, M);
    masked_upward<S>(mask, R, M);
    masked_coupling<S>(mask, &mask, couplings, M, L);
    masked_downward<S>(&mask, R, L);

    // L2P + near over the slots; near sources are src-occupied by the mask
    // check, so their slot lookup always succeeds
#pragma omp parallel
    {
        Mat T(ls_, q_), YtT(ls_, ls_);
        Vec yloc(ls2_);
#pragma omp for schedule(static)
        for (int s = 0; s < nslot; ++s) {
            const int li = mask.slot_leaf[s];
            const int t = leaves_[li];
            Eigen::Map<const Mat> LmT(L.col(t).data(), q_, q_);
            T.noalias() = Wleaf * LmT;
            YtT.noalias() = T * Wleaf.transpose();

            yloc.setZero();
            for (std::int64_t k = near_off_[li]; k < near_off_[li + 1]; ++k) {
                const int sbox = near_[k].source_box;
                if (!so[sbox]) continue;
                const int ss = mask.leaf_slot[sbox - leaf_begin];
                Eigen::Map<const Vec> xs(
                    xc.data() + static_cast<std::ptrdiff_t>(ss) * ls2_, ls2_);
                yloc.noalias() += near_st[near_[k].stencil_id] * xs;
            }
            for (int ly = 0; ly < ls_; ++ly)
                yc.segment(static_cast<std::ptrdiff_t>(s) * ls2_ + ly * ls_, ls_) =
                    YtT.col(ly) + yloc.segment(ly * ls_, ls_);
        }
    }
}

// ── matvec_masked_stream_impl ─────────────────────────────────────────────────
template <class S>
void H2Operator::matvec_masked_stream_impl(
    const Eigen::Matrix<S, Eigen::Dynamic, 1>& xc, const H2Mask& src,
    const std::function<void(int, int, const S*)>& sink,
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
    const int leaf_begin = tree_.level_begin(tree_.leaf_level());
    const std::uint8_t* so = src.box_occ.data();
    if (M.cols() != nbox) M.resize(q2_, nbox);
    if (L.cols() != nbox) L.resize(q2_, nbox);

    p2m_compressed<S>(xc, src, Wleaf, M);
    masked_upward<S>(src, R, M);
    masked_coupling<S>(src, nullptr, couplings, M, L); // all targets
    masked_downward<S>(nullptr, R, L);

    // L2P + near tile per leaf, handed to the sink and discarded — no
    // N-sized output. The tile layout matches the compressed slot layout:
    // tile[ly*ls + lx] = u at (ix0 + lx, iy0 + ly).
#pragma omp parallel
    {
        Mat T(ls_, q_), YtT(ls_, ls_);
        Vec yloc(ls2_), tile(ls2_);
#pragma omp for schedule(dynamic, 32)
        for (int li = 0; li < nleaf; ++li) {
            const int t = leaves_[li];
            const int ix0 = boxes[t].ix0, iy0 = boxes[t].iy0;
            Eigen::Map<const Mat> LmT(L.col(t).data(), q_, q_);
            T.noalias() = Wleaf * LmT;
            YtT.noalias() = T * Wleaf.transpose();

            yloc.setZero();
            for (std::int64_t k = near_off_[li]; k < near_off_[li + 1]; ++k) {
                const int sbox = near_[k].source_box;
                if (!so[sbox]) continue;
                const int ss = src.leaf_slot[sbox - leaf_begin];
                Eigen::Map<const Vec> xs(
                    xc.data() + static_cast<std::ptrdiff_t>(ss) * ls2_, ls2_);
                yloc.noalias() += near_st[near_[k].stencil_id] * xs;
            }
            for (int ly = 0; ly < ls_; ++ly)
                tile.segment(ly * ls_, ls_) =
                    YtT.col(ly) + yloc.segment(ly * ls_, ls_);
            sink(ix0, iy0, tile.data());
        }
    }
}

} // namespace hmc
