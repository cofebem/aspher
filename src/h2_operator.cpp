#include "h2_operator.hpp"

#include <cmath>
#include <cstdio>
#include <unordered_map>

namespace hmc {

H2Operator::H2Operator(const BoussinesqKernel& kernel, H2Params params)
    : kernel_(&kernel), p_(params), Ns_(kernel.grid_size()), q_(params.q),
      q2_(params.q * params.q), ls_(params.leaf_side), ls2_(params.leaf_side * params.leaf_side),
      h_(kernel.element_size()), scale_(1.0 / (M_PI * kernel.E_star())),
      cheb_(params.q), tree_(kernel.grid_size(), params.leaf_side) {}

std::vector<double> H2Operator::centers_norm(int side) const {
    // element-center k (k = 0..side-1) normalized into the box geometric extent:
    //   (k - side/2 + 0.5) / (side/2)
    std::vector<double> c(side);
    const double half = side / 2.0;
    for (int k = 0; k < side; ++k) c[k] = (k - half + 0.5) / half;
    return c;
}

double H2Operator::far_kernel(double dx, double dy) const {
    return scale_ * love_uz(dx, dy, 0.5 * h_, 0.5 * h_);
}

// packed key for the coupling cache: (level, dbx, dby)
static inline std::int64_t far_key(int level, int dbx, int dby) {
    return (static_cast<std::int64_t>(level) << 40) ^
           ((static_cast<std::int64_t>(dbx + 512) & 0xFFFFF) << 20) ^
           (static_cast<std::int64_t>(dby + 512) & 0xFFFFF);
}
static inline std::int64_t near_key(int dx, int dy) {
    return (static_cast<std::int64_t>(dx + 64) << 16) ^ static_cast<std::int64_t>(dy + 64);
}

void H2Operator::build() {
    const auto& boxes = tree_.boxes();
    const int nbox = static_cast<int>(boxes.size());

    // ── Leaf interpolation matrix Wleaf (ls x q) ──
    Wleaf_ = cheb_.weights_at(centers_norm(ls_));

    // ── M2M transfer matrices R_[c] (q2 x q2): R[c][A,a] = Wx_c[ax](Ax) Wy_c[ay](Ay) ──
    for (int c = 0; c < 4; ++c) {
        const double signx = (c & 1) ? 0.5 : -0.5;
        const double signy = (c & 2) ? 0.5 : -0.5;
        // child node a's coordinate in parent-normalized space
        std::vector<Eigen::VectorXd> Wx(q_), Wy(q_);
        for (int a = 0; a < q_; ++a) {
            Wx[a] = cheb_.weights(signx + 0.5 * cheb_.nodes[a]);
            Wy[a] = cheb_.weights(signy + 0.5 * cheb_.nodes[a]);
        }
        Eigen::MatrixXd R(q2_, q2_);
        for (int Ay = 0; Ay < q_; ++Ay)
            for (int Ax = 0; Ax < q_; ++Ax) {
                const int A = Ax + q_ * Ay;
                for (int ay = 0; ay < q_; ++ay)
                    for (int ax = 0; ax < q_; ++ax)
                        R(A, ax + q_ * ay) = Wx[ax](Ax) * Wy[ay](Ay);
            }
        R_[c] = std::move(R);
    }

    // ── Far interactions + coupling cache (CSR by target box) ──
    far_off_.assign(nbox + 1, 0);
    std::unordered_map<std::int64_t, int> coupling_ids;
    for (int t = 0; t < nbox; ++t) {
        for (int s : tree_.interaction_list(t, p_.near_radius)) {
            const int level = boxes[t].level;
            const int dbx = boxes[t].bx - boxes[s].bx;
            const int dby = boxes[t].by - boxes[s].by;
            const std::int64_t key = far_key(level, dbx, dby);
            auto it = coupling_ids.find(key);
            int cid;
            if (it == coupling_ids.end()) {
                cid = static_cast<int>(couplings_.size());
                coupling_ids.emplace(key, cid);
                // K[A,B] = g(Xt_A - Xs_B); node coord = center + (side*h/2)*xi
                const int side = boxes[t].side;
                const double bh = 0.5 * side * h_;
                Eigen::MatrixXd K(q2_, q2_);
                for (int Ay = 0; Ay < q_; ++Ay)
                    for (int Ax = 0; Ax < q_; ++Ax) {
                        const int A = Ax + q_ * Ay;
                        for (int By = 0; By < q_; ++By)
                            for (int Bx = 0; Bx < q_; ++Bx) {
                                const double Dx = dbx * side * h_ +
                                    bh * (cheb_.nodes[Ax] - cheb_.nodes[Bx]);
                                const double Dy = dby * side * h_ +
                                    bh * (cheb_.nodes[Ay] - cheb_.nodes[By]);
                                K(A, Bx + q_ * By) = far_kernel(Dx, Dy);
                            }
                    }
                couplings_.push_back(std::move(K));
            } else {
                cid = it->second;
            }
            far_.push_back({s, cid});
            ++info_.n_far_interactions;
        }
        far_off_[t + 1] = static_cast<std::int64_t>(far_.size());
    }

    // ── Leaves + near interactions + near stencil cache (CSR by leaf) ──
    for (int b = 0; b < nbox; ++b)
        if (boxes[b].leaf) leaves_.push_back(b);
    near_off_.assign(leaves_.size() + 1, 0);
    std::unordered_map<std::int64_t, int> near_ids;
    for (int li = 0; li < static_cast<int>(leaves_.size()); ++li) {
        const int t = leaves_[li];
        for (int s : tree_.neighbors(t, p_.near_radius)) {
            const int dx = boxes[s].bx - boxes[t].bx; // source - target, in leaf units
            const int dy = boxes[s].by - boxes[t].by;
            const std::int64_t key = near_key(dx, dy);
            auto it = near_ids.find(key);
            int sid;
            if (it == near_ids.end()) {
                sid = static_cast<int>(near_stencils_.size());
                near_ids.emplace(key, sid);
                // A_near[lt, ls] = entry_offset(lx_t-lx_s-dx*ls, ly_t-ly_s-dy*ls)
                Eigen::MatrixXd A(ls2_, ls2_);
                for (int lyt = 0; lyt < ls_; ++lyt)
                    for (int lxt = 0; lxt < ls_; ++lxt) {
                        const int rt = lxt + ls_ * lyt;
                        for (int lys = 0; lys < ls_; ++lys)
                            for (int lxs = 0; lxs < ls_; ++lxs)
                                A(rt, lxs + ls_ * lys) = kernel_->entry_offset(
                                    lxt - lxs - dx * ls_, lyt - lys - dy * ls_);
                    }
                near_stencils_.push_back(std::move(A));
            } else {
                sid = it->second;
            }
            near_.push_back({s, sid});
            ++info_.n_near_interactions;
        }
        near_off_[li + 1] = static_cast<std::int64_t>(near_.size());
    }

    // ── Statistics ──
    info_.N = Ns_ * Ns_;
    info_.Ns = Ns_;
    info_.nlevels = tree_.nlevels();
    info_.leaf_side = ls_;
    info_.q = q_;
    info_.r = q2_;
    info_.n_boxes = nbox;
    info_.n_leaves = static_cast<int>(leaves_.size());
    info_.n_unique_couplings = static_cast<int>(couplings_.size());
    info_.n_near_stencils = static_cast<int>(near_stencils_.size());
    info_.bytes_coupling = 8LL * info_.n_unique_couplings * q2_ * q2_;
    info_.bytes_near = 8LL * info_.n_near_stencils * ls2_ * ls2_;
    info_.bytes_buffers = 8LL * 2 * nbox * q2_;
    info_.bytes_total = info_.bytes_coupling + info_.bytes_near + info_.bytes_buffers;
}

Eigen::VectorXd H2Operator::matvec(const Eigen::VectorXd& x) const {
    Eigen::VectorXd y;
    matvec_into(x, y);
    return y;
}

void H2Operator::matvec_into(const Eigen::VectorXd& x, Eigen::VectorXd& y) const {
    matvec_impl<double>(x, y, Wleaf_, R_, couplings_, near_stencils_, Mbuf_, Lbuf_);
}

// OR-propagate leaf occupancy to the ancestors. Boxes are stored level by
// level (children after their parent), so one reverse sweep suffices.
static void propagate_up(const std::vector<H2Box>& boxes,
                         std::vector<std::uint8_t>& occ) {
    for (int b = static_cast<int>(boxes.size()) - 1; b > 0; --b)
        if (occ[b]) occ[boxes[b].parent] = 1;
}

H2Mask H2Operator::build_mask(const std::vector<std::uint8_t>& grid_mask) const {
    const auto& boxes = tree_.boxes();
    H2Mask m;
    m.box_occ.assign(boxes.size(), 0);
    const int ll = tree_.leaf_level();
    const int nls = Ns_ / ls_; // leaf boxes per side
    // scan each leaf's elements with early exit (race-free; O(N) worst case)
#pragma omp parallel for schedule(dynamic, 8)
    for (int by = 0; by < nls; ++by)
        for (int bx = 0; bx < nls; ++bx) {
            const int ix0 = bx * ls_, iy0 = by * ls_;
            bool occ = false;
            for (int ly = 0; ly < ls_ && !occ; ++ly)
                for (int lx = 0; lx < ls_; ++lx)
                    if (grid_mask[(iy0 + ly) * Ns_ + ix0 + lx]) {
                        occ = true;
                        break;
                    }
            if (occ) m.box_occ[tree_.box_id(ll, bx, by)] = 1;
        }
    propagate_up(boxes, m.box_occ);
    return m;
}

H2Mask H2Operator::build_mask(const int* idx, std::size_t n) const {
    H2Mask m;
    m.box_occ.assign(tree_.boxes().size(), 0);
    const int ll = tree_.leaf_level();
    for (std::size_t k = 0; k < n; ++k) {
        const int ix = idx[k] % Ns_, iy = idx[k] / Ns_;
        m.box_occ[tree_.box_id(ll, ix / ls_, iy / ls_)] = 1;
    }
    propagate_up(tree_.boxes(), m.box_occ);
    return m;
}

void H2Operator::matvec_masked_into(const Eigen::VectorXd& x, Eigen::VectorXd& y,
                                    const H2Mask& src, const H2Mask* tgt) const {
    matvec_masked_impl<double>(x, y, src, tgt, Wleaf_, R_, couplings_,
                               near_stencils_, Mbuf_, Lbuf_);
}

void H2Operator::matvec_masked_single_into(const Eigen::VectorXf& x,
                                           Eigen::VectorXf& y,
                                           const H2Mask& src,
                                           const H2Mask* tgt) const {
    build_single_caches();
    matvec_masked_impl<float>(x, y, src, tgt, Wleaf_f_, R_f_, couplings_f_,
                              near_stencils_f_, Mbuf_f_, Lbuf_f_);
}

void H2Operator::build_single_caches() const {
    if (have_single_) return;
    Wleaf_f_ = Wleaf_.cast<float>();
    for (int c = 0; c < 4; ++c) R_f_[c] = R_[c].cast<float>();
    couplings_f_.resize(couplings_.size());
    for (std::size_t i = 0; i < couplings_.size(); ++i)
        couplings_f_[i] = couplings_[i].cast<float>();
    near_stencils_f_.resize(near_stencils_.size());
    for (std::size_t i = 0; i < near_stencils_.size(); ++i)
        near_stencils_f_[i] = near_stencils_[i].cast<float>();
    have_single_ = true;
}

Eigen::VectorXf H2Operator::matvec_single(const Eigen::VectorXf& x) const {
    Eigen::VectorXf y;
    matvec_single_into(x, y);
    return y;
}

void H2Operator::matvec_single_into(const Eigen::VectorXf& x,
                                    Eigen::VectorXf& y) const {
    build_single_caches();
    matvec_impl<float>(x, y, Wleaf_f_, R_f_, couplings_f_, near_stencils_f_,
                       Mbuf_f_, Lbuf_f_);
}

void H2Operator::print_statistics() const {
    const H2Info& s = info_;
    std::printf("── H2 operator ─────────────────────────────────\n");
    std::printf("  N=%d  Ns=%d  levels=%d  leaf_side=%d  q=%d  r=%d\n",
                s.N, s.Ns, s.nlevels, s.leaf_side, s.q, s.r);
    std::printf("  boxes=%d  leaves=%d\n", s.n_boxes, s.n_leaves);
    std::printf("  near interactions=%lld  far interactions=%lld\n",
                (long long)s.n_near_interactions, (long long)s.n_far_interactions);
    std::printf("  unique couplings=%d  near stencils=%d\n",
                s.n_unique_couplings, s.n_near_stencils);
    std::printf("  mem: coupling=%.2f MiB  near=%.2f MiB  buffers=%.2f MiB  total=%.2f MiB\n",
                s.bytes_coupling / 1048576.0, s.bytes_near / 1048576.0,
                s.bytes_buffers / 1048576.0, s.bytes_total / 1048576.0);
}

} // namespace hmc
