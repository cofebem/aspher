#include "hmatrix.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>

#ifdef __GLIBC__
#endif

namespace hmc {

// ──────────────────────────────────────────────────────────────
// ACA-GP geometry helpers (file-scope, not exported)
// ──────────────────────────────────────────────────────────────
namespace {

struct Circle { double cx, cy, r; };

// Circumcircle of three points; returns r=1e30 for degenerate (collinear) input.
Circle find_circle(double x1, double y1, double x2, double y2,
                   double x3, double y3) {
    const double A1 = x2 - x1, B1 = y2 - y1;
    const double C1 = 0.5 * ((x2*x2 - x1*x1) + (y2*y2 - y1*y1));
    const double A2 = x3 - x1, B2 = y3 - y1;
    const double C2 = 0.5 * ((x3*x3 - x1*x1) + (y3*y3 - y1*y1));
    const double D  = A1*B2 - A2*B1;
    if (std::abs(D) < 1e-14) return {0.0, 0.0, 1e30};
    const double cx = (C1*B2 - C2*B1) / D;
    const double cy = (A1*C2 - A2*C1) / D;
    const double r  = std::hypot(x1 - cx, y1 - cy);
    return {cx, cy, r};
}

double dist_to_circle(double x, double y, const Circle& c) {
    if (c.r > 1e29) return 0.0;
    return std::abs(std::hypot(x - c.cx, y - c.cy) - c.r);
}

} // anonymous namespace

// ──────────────────────────────────────────────────────────────
// Constructor
// ──────────────────────────────────────────────────────────────
HMatrix::HMatrix(const BoussinesqKernel& kernel, const ClusterTree& tree,
                 double eta, double aca_tol,
                 bool use_acagp, double central_fraction, double inline_svd_tol)
    : kernel_(&kernel), tree_(&tree), eta_(eta), tol_(aca_tol),
      use_acagp_(use_acagp), central_fraction_(central_fraction),
      inline_svd_tol_(inline_svd_tol) {
    build(tree.root(), tree.root());

    // NOTE: this constructor used to call mallopt() to force glibc to mmap
    // allocations above 128 KB, which keeps peak RSS at (compressed blocks +
    // one full-rank block per thread) rather than the whole uncompressed
    // H-matrix. That changed the policy of the entire host process from inside
    // a library call; it is now the explicit, application-level
    // hmc::configure_allocator() (include/runtime_policy.hpp), which callers
    // building large H-matrices should invoke once at start-up.

    const int nb = static_cast<int>(blocks_.size());
#pragma omp parallel for schedule(dynamic, 4)
    for (int bi = 0; bi < nb; ++bi) {
        HBlock& blk = blocks_[bi];
        if (blk.dense)
            fill_dense(blk);
        else if (use_acagp_)
            fill_aca_gp(blk);
        else
            fill_aca(blk);
        // Inline SVD recompression: truncate immediately after fill to keep
        // peak memory at O(one block) rather than O(all blocks at full rank).
        if (inline_svd_tol_ > 0.0 && !blk.dense)
            recompress_block(blk, inline_svd_tol_);
    }
}

// ──────────────────────────────────────────────────────────────
// Block partition (unchanged)
// ──────────────────────────────────────────────────────────────
void HMatrix::build(int t, int s) {
    const ClusterNode& nt = tree_->nodes()[t];
    const ClusterNode& ns = tree_->nodes()[s];
    const int d    = ClusterTree::dist(nt.box, ns.box);
    const int dmin = std::min(ClusterTree::diam(nt.box), ClusterTree::diam(ns.box));

    if (d > 0 && dmin <= eta_ * d) {
        blocks_.push_back({nt.begin, nt.end - nt.begin, ns.begin,
                           ns.end - ns.begin, false, {}, {}, {}});
        return;
    }
    if (nt.leaf && ns.leaf) {
        blocks_.push_back({nt.begin, nt.end - nt.begin, ns.begin,
                           ns.end - ns.begin, true, {}, {}, {}});
        return;
    }
    if (nt.leaf) {
        for (int q = 0; q < 4; ++q)
            if (ns.child[q] >= 0) build(t, ns.child[q]);
    } else if (ns.leaf) {
        for (int q = 0; q < 4; ++q)
            if (nt.child[q] >= 0) build(nt.child[q], s);
    } else {
        for (int qt = 0; qt < 4; ++qt)
            for (int qs = 0; qs < 4; ++qs)
                if (nt.child[qt] >= 0 && ns.child[qs] >= 0)
                    build(nt.child[qt], ns.child[qs]);
    }
}

// ──────────────────────────────────────────────────────────────
// Dense fill (unchanged)
// ──────────────────────────────────────────────────────────────
void HMatrix::fill_dense(HBlock& blk) const {
    blk.D.resize(blk.row_size, blk.col_size);
    for (int i = 0; i < blk.row_size; ++i)
        for (int j = 0; j < blk.col_size; ++j)
            blk.D(i, j) = entry_perm(blk.row_begin + i, blk.col_begin + j);
}

// ──────────────────────────────────────────────────────────────
// Classical ACA with partial pivoting (unchanged)
// ──────────────────────────────────────────────────────────────
void HMatrix::fill_aca(HBlock& blk) const {
    const int m = blk.row_size, n = blk.col_size;
    const int kmax = std::min(m, n);
    Eigen::MatrixXd U(m, kmax), V(kmax, n);
    std::vector<char> row_used(m, 0);
    double frob2 = 0.0;
    int k = 0, i_star = 0;

    while (k < kmax) {
        Eigen::RowVectorXd r(n);
        for (int j = 0; j < n; ++j)
            r(j) = entry_perm(blk.row_begin + i_star, blk.col_begin + j);
        if (k > 0) r.noalias() -= U.row(i_star).head(k) * V.topRows(k);
        row_used[i_star] = 1;

        int j_star = 0;
        const double rmax = r.cwiseAbs().maxCoeff(&j_star);
        const bool negligible =
            k > 0 && rmax * rmax * m * n <= tol_ * tol_ * frob2;
        if (rmax == 0.0 || negligible) {
            i_star = -1;
            for (int i = 0; i < m; ++i)
                if (!row_used[i]) { i_star = i; break; }
            if (i_star < 0) break;
            continue;
        }

        Eigen::VectorXd c(m);
        for (int i = 0; i < m; ++i)
            c(i) = entry_perm(blk.row_begin + i, blk.col_begin + j_star);
        if (k > 0) c.noalias() -= U.leftCols(k) * V.col(j_star).head(k);

        const Eigen::VectorXd u = c / r(j_star);
        const double u2 = u.squaredNorm(), v2 = r.squaredNorm();
        double cross = 0.0;
        for (int l = 0; l < k; ++l)
            cross += U.col(l).dot(u) * V.row(l).dot(r);
        frob2 += 2.0 * cross + u2 * v2;
        U.col(k) = u;
        V.row(k) = r;
        ++k;

        if (u2 * v2 <= tol_ * tol_ * frob2) break;

        double best = -1.0;
        i_star = -1;
        for (int i = 0; i < m; ++i)
            if (!row_used[i] && std::abs(u(i)) > best) {
                best = std::abs(u(i));
                i_star = i;
            }
        if (i_star < 0) break;
    }
    blk.U = U.leftCols(k);
    blk.V = V.topRows(k);
}

// ──────────────────────────────────────────────────────────────
// ACA-GP (Yastrebov 2025): geometric pivot selection
// Implements the full algorithm including rank-2 conjugate circle
// and rank-3 constructions, then higher_ranks_selection for k >= 4.
// ──────────────────────────────────────────────────────────────
void HMatrix::fill_aca_gp(HBlock& blk) const {
    const int m = blk.row_size, n = blk.col_size;
    if (m <= 0 || n <= 0) return;
    const int kmax = std::min(m, n);
    const int Ns   = tree_->grid_size();

    // ── 1. Extract 2D element coordinates (in grid-index units) ──
    // flat index k = iy*Ns + ix  →  ix = k%Ns, iy = k/Ns
    std::vector<double> rx(m), ry(m), cx(n), cy(n);
    for (int i = 0; i < m; ++i) {
        const int flat = tree_->perm()[blk.row_begin + i];
        rx[i] = flat % Ns;
        ry[i] = flat / Ns;
    }
    for (int j = 0; j < n; ++j) {
        const int flat = tree_->perm()[blk.col_begin + j];
        cx[j] = flat % Ns;
        cy[j] = flat / Ns;
    }

    // ── 2. Cluster centers and direction vector ──
    double rcx = 0, rcy = 0, ccx = 0, ccy = 0;
    for (int i = 0; i < m; ++i) { rcx += rx[i]; rcy += ry[i]; }
    rcx /= m; rcy /= m;
    for (int j = 0; j < n; ++j) { ccx += cx[j]; ccy += cy[j]; }
    ccx /= n; ccy /= n;
    const double dx = ccx - rcx, dy = ccy - rcy;

    // ── 3. Geometric first pivot ──
    // Row pivot: element closest to row-center on the side facing col cluster.
    // Col pivot: element closest to col-center on the side facing row cluster.
    auto geom_first = [](const std::vector<double>& px, const std::vector<double>& py,
                         double pcx, double pcy, double dpx, double dpy) -> int {
        bool has_pos = false;
        for (int i = 0; i < (int)px.size(); ++i)
            if ((px[i]-pcx)*dpx + (py[i]-pcy)*dpy > 0) { has_pos = true; break; }
        int best = 0;
        double min_d = 1e30;
        for (int i = 0; i < (int)px.size(); ++i) {
            const double proj = (px[i]-pcx)*dpx + (py[i]-pcy)*dpy;
            if (has_pos && proj <= 0) continue;
            const double d = (px[i]-pcx)*(px[i]-pcx) + (py[i]-pcy)*(py[i]-pcy);
            if (d < min_d) { min_d = d; best = i; }
        }
        return best;
    };
    const int i1 = geom_first(rx, ry, rcx, rcy,  dx,  dy);
    const int j1 = geom_first(cx, cy, ccx, ccy, -dx, -dy);

    // ── 4. Central subsets Ic, Jc ──
    // Points within central_fraction * diam from the first pivot.
    double diam_r = 0, diam_c = 0;
    for (int i = 0; i < m; ++i)
        diam_r = std::max(diam_r, std::hypot(rx[i]-rx[i1], ry[i]-ry[i1]));
    diam_r *= 2.0;
    for (int j = 0; j < n; ++j)
        diam_c = std::max(diam_c, std::hypot(cx[j]-cx[j1], cy[j]-cy[j1]));
    diam_c *= 2.0;

    auto build_central = [&](const std::vector<double>& px,
                              const std::vector<double>& py,
                              int anchor, double diam, int need) {
        std::vector<int> sub;
        double cf = central_fraction_;
        for (int attempt = 0; attempt < 50; ++attempt) {
            sub.clear();
            for (int i = 0; i < (int)px.size(); ++i) {
                if (i == anchor) continue;
                if (std::hypot(px[i]-px[anchor], py[i]-py[anchor]) <= cf * diam)
                    sub.push_back(i);
            }
            if ((int)sub.size() >= need || (int)sub.size() == (int)px.size() - 1) break;
            cf *= 1.1;
        }
        return sub;
    };

    const int need_r = std::min(kmax + 5, m - 1);
    const int need_c = std::min(kmax + 5, n - 1);
    std::vector<int> Ic = build_central(rx, ry, i1, diam_r, need_r);
    std::vector<int> Jc = build_central(cx, cy, j1, diam_c, need_c);

    // ── 5. Residual helper ──
    // R(i,j) = A(i,j) − Σ_{l<k} U(i,l) V(l,j)
    Eigen::MatrixXd U(m, kmax), V(kmax, n);
    U.setZero(); V.setZero();
    double frob2 = 0.0;
    int k = 0;

    auto residual = [&](int i, int j) -> double {
        double val = entry_perm(blk.row_begin + i, blk.col_begin + j);
        if (k > 0) val -= U.row(i).head(k).dot(V.col(j).head(k));
        return val;
    };

    // ── 6. First rank using geometric first pivot ──
    {
        const double pivot = residual(i1, j1);
        if (std::abs(pivot) < 1e-15) {
            // Degenerate first pivot: fall back to classical ACA
            fill_aca(blk);
            return;
        }
        Eigen::RowVectorXd r(n);
        for (int j = 0; j < n; ++j) r(j) = entry_perm(blk.row_begin + i1, blk.col_begin + j);
        Eigen::VectorXd c(m);
        for (int i = 0; i < m; ++i) c(i) = entry_perm(blk.row_begin + i, blk.col_begin + j1);
        U.col(0) = c / pivot;
        V.row(0) = r;
        const double u2 = U.col(0).squaredNorm(), v2 = V.row(0).squaredNorm();
        frob2 = u2 * v2;
        k = 1;

        // Remove first pivot from central subsets
        Ic.erase(std::remove(Ic.begin(), Ic.end(), i1), Ic.end());
        Jc.erase(std::remove(Jc.begin(), Jc.end(), j1), Jc.end());
    }

    // Track all selected pivot indices (needed for circle constructions)
    std::vector<int> Ik = {i1}, Jk = {j1};

    // Thread-local RNG (fill_aca_gp is called from OpenMP threads)
    thread_local std::mt19937 rng{std::random_device{}()};

    // ── 7. Main ACA-GP loop ──
    while (k < kmax) {
        int ir = -1, jr = -1;

        // ── 7a. Rank-2 selection: circle through first pivot and trial row ──
        if (k == 1 && !Ic.empty() && !Jc.empty()) {
            // Pick random trial row from Ic
            const int i2 = Ic[std::uniform_int_distribution<int>(
                                   0, (int)Ic.size() - 1)(rng)];
            // Circle through (rx[i1], ry[i1]), (cx[j1], cy[j1]), (rx[i2], ry[i2])
            const Circle circ = find_circle(rx[i1], ry[i1],
                                            cx[j1], cy[j1],
                                            rx[i2], ry[i2]);
            // Iteratively find best j in Jc by proximity to circle
            std::vector<int> Jcl = Jc;
            int j2_best = Jcl[0];
            double best_pivot = 0.0;
            while (!Jcl.empty()) {
                // Find j in Jcl closest to circle
                int j2_l = Jcl[0];
                double min_dist = 1e30;
                for (int j : Jcl) {
                    const double d = dist_to_circle(cx[j], cy[j], circ);
                    if (d < min_dist) { min_dist = d; j2_l = j; }
                }
                const double piv = residual(i2, j2_l);
                if (std::abs(piv) <= std::abs(best_pivot)) break;
                best_pivot = piv;
                j2_best = j2_l;
                Jcl.erase(std::find(Jcl.begin(), Jcl.end(), j2_l));
            }
            ir = i2;
            jr = j2_best;
        }
        // ── 7b. Rank-3 selection: conjugate circle construction ──
        else if (k == 2 && !Ic.empty() && !Jc.empty()
                 && (int)Ik.size() >= 2 && (int)Jk.size() >= 2) {
            // Circle through i1, j1, i2
            const Circle circ = find_circle(rx[Ik[0]], ry[Ik[0]],
                                            cx[Jk[0]], cy[Jk[0]],
                                            rx[Ik[1]], ry[Ik[1]]);
            // Conjugate circle at (rx[Ik[0]], ry[Ik[0]]): tangent-orthogonal
            Circle circ_r, circ_c;
            if (circ.r < 1e29) {
                auto conjugate = [&](double px, double py) -> Circle {
                    double rad_x = px - circ.cx, rad_y = py - circ.cy;
                    double len = std::hypot(rad_x, rad_y);
                    if (len < 1e-14) return {px, py, circ.r};
                    double tan_x = -rad_y / len, tan_y = rad_x / len;
                    // Orient tangent toward the other cluster center
                    double ocx = (px == rx[Ik[0]]) ? ccx : rcx;
                    double ocy = (px == rx[Ik[0]]) ? ccy : rcy;
                    if (tan_x * (ocx - px) + tan_y * (ocy - py) < 0) {
                        tan_x = -tan_x; tan_y = -tan_y;
                    }
                    return {px + circ.r * tan_x, py + circ.r * tan_y, circ.r};
                };
                circ_r = conjugate(rx[Ik[0]], ry[Ik[0]]);  // for row selection
                circ_c = conjugate(cx[Jk[0]], cy[Jk[0]]);  // for col selection
            } else {
                circ_r = circ_c = circ;
            }
            // i3: row element closest to conjugate circle circ_r
            int i3 = Ic[0];
            double min_d = 1e30;
            for (int i : Ic) {
                const double d = dist_to_circle(rx[i], ry[i], circ_r);
                if (d < min_d) { min_d = d; i3 = i; }
            }
            // j3: best in Jc by proximity to circ_c, increasing pivot
            std::vector<int> Jcl = Jc;
            int j3_best = Jcl[0];
            double best_pivot = 0.0;
            while (!Jcl.empty()) {
                int j3_l = Jcl[0];
                double min_dist2 = 1e30;
                for (int j : Jcl) {
                    const double d = dist_to_circle(cx[j], cy[j], circ_c);
                    if (d < min_dist2) { min_dist2 = d; j3_l = j; }
                }
                const double piv = residual(i3, j3_l);
                if (std::abs(piv) <= std::abs(best_pivot)) break;
                best_pivot = piv;
                j3_best = j3_l;
                Jcl.erase(std::find(Jcl.begin(), Jcl.end(), j3_l));
            }
            ir = i3;
            jr = j3_best;
        }
        // ── 7c. Higher ranks: central-subset algebraic search ──
        else if (!Ic.empty() && !Jc.empty()) {
            // Random trial row from Ic
            const int i_t = Ic[std::uniform_int_distribution<int>(
                                    0, (int)Ic.size() - 1)(rng)];
            // Best col in Jc for this trial row
            double best_abs = -1.0;
            for (int j : Jc) {
                const double abs_r = std::abs(residual(i_t, j));
                if (abs_r > best_abs) { best_abs = abs_r; jr = j; }
            }
            // Best row in Ic for the chosen col
            best_abs = -1.0;
            for (int i : Ic) {
                const double abs_r = std::abs(residual(i, jr));
                if (abs_r > best_abs) { best_abs = abs_r; ir = i; }
            }
        }
        // ── 7d. Fallback: classical pivot selection ──
        else {
            // No central subset remaining: pick row with max residual column
            double best = -1.0;
            for (int i = 0; i < m; ++i) {
                if (std::find(Ik.begin(), Ik.end(), i) != Ik.end()) continue;
                for (int j = 0; j < n; ++j) {
                    const double a = std::abs(residual(i, j));
                    if (a > best) { best = a; ir = i; jr = j; }
                }
            }
            if (ir < 0) break;
        }

        if (ir < 0 || jr < 0) break;

        // ── 8. Compute full row at ir and full col at jr ──
        Eigen::RowVectorXd r(n);
        for (int j = 0; j < n; ++j)
            r(j) = entry_perm(blk.row_begin + ir, blk.col_begin + j);
        if (k > 0) r.noalias() -= U.row(ir).head(k) * V.topRows(k);

        Eigen::VectorXd c(m);
        for (int i = 0; i < m; ++i)
            c(i) = entry_perm(blk.row_begin + i, blk.col_begin + jr);
        if (k > 0) c.noalias() -= U.leftCols(k) * V.col(jr).head(k);

        const double pivot = r(jr);
        if (std::abs(pivot) < 1e-15) {
            // Skip degenerate pivot; remove from subsets and try again
            Ic.erase(std::remove(Ic.begin(), Ic.end(), ir), Ic.end());
            Jc.erase(std::remove(Jc.begin(), Jc.end(), jr), Jc.end());
            continue;
        }

        // ── 9. Update U, V and Frobenius norm ──
        const Eigen::VectorXd u = c / pivot;
        const double u2 = u.squaredNorm(), v2 = r.squaredNorm();
        double cross = 0.0;
        for (int l = 0; l < k; ++l) cross += U.col(l).dot(u) * V.row(l).dot(r);
        frob2 += 2.0 * cross + u2 * v2;
        U.col(k) = u;
        V.row(k) = r;
        ++k;

        // ── 10. Stopping criterion ──
        if (u2 * v2 <= tol_ * tol_ * frob2) break;

        // Update bookkeeping
        Ik.push_back(ir); Jk.push_back(jr);
        Ic.erase(std::remove(Ic.begin(), Ic.end(), ir), Ic.end());
        Jc.erase(std::remove(Jc.begin(), Jc.end(), jr), Jc.end());
    }

    blk.U = U.leftCols(k);
    blk.V = V.topRows(k);
}

// ──────────────────────────────────────────────────────────────
// Matvec (unchanged)
// ──────────────────────────────────────────────────────────────
Eigen::VectorXd HMatrix::matvec(const Eigen::VectorXd& p) const {
    const int N = static_cast<int>(tree_->perm().size());
    Eigen::VectorXd pt(N), ut = Eigen::VectorXd::Zero(N);
    for (int pos = 0; pos < N; ++pos) pt(pos) = p(tree_->perm()[pos]);

    const int nb = static_cast<int>(blocks_.size());
#pragma omp parallel
    {
        Eigen::VectorXd local = Eigen::VectorXd::Zero(N);
#pragma omp for schedule(dynamic, 8) nowait
        for (int bi = 0; bi < nb; ++bi) {
            const HBlock& b = blocks_[bi];
            const auto x = pt.segment(b.col_begin, b.col_size);
            auto y = local.segment(b.row_begin, b.row_size);
            if (b.dense)
                y.noalias() += b.D * x;
            else
                y.noalias() += b.U * (b.V * x);
        }
#pragma omp critical
        ut += local;
    }

    Eigen::VectorXd u(N);
    for (int pos = 0; pos < N; ++pos) u(tree_->perm()[pos]) = ut(pos);
    return u;
}

// ──────────────────────────────────────────────────────────────
// Truncated SVD recompression of a single low-rank block U (m×k), V (k×n):
//   QR(U) = Qu Ru,  QR(V^T) = Qv Rv
//   SVD(Ru Rv^T) = P Σ W^T  (k×k)
//   Keep r terms where σ_r >= svd_tol * σ_0
//   U_new = Qu P_r diag(σ_r)^{1/2},  V_new = diag(σ_r)^{1/2} W_r^T Qv^T
// Cost: O(k²(m+n)).
// ──────────────────────────────────────────────────────────────
void HMatrix::recompress_block(HBlock& blk, double svd_tol) {
    if (blk.dense) return;
    const int k0 = static_cast<int>(blk.U.cols());
    if (k0 <= 1) return;
    const int m = blk.row_size, n = blk.col_size;

    // Economy QR of U (m×k0): U = Qu Ru
    Eigen::HouseholderQR<Eigen::MatrixXd> qru(blk.U);
    Eigen::MatrixXd Qu = (qru.householderQ() *
                          Eigen::MatrixXd::Identity(m, k0)).eval();
    Eigen::MatrixXd Ru = qru.matrixQR().topRows(k0).eval();
    Ru.template triangularView<Eigen::StrictlyLower>().setZero();

    // Economy QR of V^T (n×k0): V^T = Qv Rv
    Eigen::MatrixXd Vt = blk.V.transpose();  // n×k0
    Eigen::HouseholderQR<Eigen::MatrixXd> qrv(Vt);
    Eigen::MatrixXd Qv = (qrv.householderQ() *
                          Eigen::MatrixXd::Identity(n, k0)).eval();
    Eigen::MatrixXd Rv = qrv.matrixQR().topRows(k0).eval();
    Rv.template triangularView<Eigen::StrictlyLower>().setZero();

    // SVD of k0×k0 product Ru Rv^T
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        (Ru * Rv.transpose()).eval(),
        Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd& sigma = svd.singularValues();

    // Determine truncation rank
    int r = k0;
    if (sigma(0) > 0.0)
        for (int i = 1; i < k0; ++i)
            if (sigma(i) < svd_tol * sigma(0)) { r = i; break; }
    if (r >= k0) return;  // no compression needed

    // New factors: U_new = Qu * P_r * diag(sqrt σ), V_new = diag(sqrt σ) * W_r^T * Qv^T
    const Eigen::VectorXd sq = sigma.head(r).cwiseSqrt();
    blk.U = Qu * svd.matrixU().leftCols(r) * sq.asDiagonal();
    blk.V = sq.asDiagonal() * svd.matrixV().leftCols(r).transpose() * Qv.transpose();
}

void HMatrix::recompress(double svd_tol) {
    for (auto& blk : blocks_)
        recompress_block(blk, svd_tol);
}

// ──────────────────────────────────────────────────────────────
// Info (unchanged)
// ──────────────────────────────────────────────────────────────
HMatrixInfo HMatrix::info() const {
    HMatrixInfo s;
    s.n = static_cast<int>(tree_->perm().size());
    std::int64_t rank_sum = 0;
    for (const auto& b : blocks_) {
        if (b.dense) {
            ++s.n_dense;
            s.bytes += 8LL * b.row_size * b.col_size;
        } else {
            ++s.n_lowrank;
            const int k = static_cast<int>(b.U.cols());
            rank_sum += k;
            s.max_rank = std::max(s.max_rank, k);
            s.bytes += 8LL * k * (b.row_size + b.col_size);
        }
    }
    s.avg_rank = s.n_lowrank ? double(rank_sum) / s.n_lowrank : 0.0;
    s.compression = double(s.bytes) / (8.0 * double(s.n) * double(s.n));
    return s;
}

std::string HMatrixInfo::to_string() const {
    std::ostringstream os;
    os << "H-matrix " << n << " x " << n << ": " << n_dense << " dense + "
       << n_lowrank << " low-rank blocks, max rank " << max_rank
       << ", avg rank " << avg_rank << ", storage "
       << double(bytes) / (1 << 20) << " MiB, compression "
       << compression << " of dense";
    return os.str();
}

} // namespace hmc
