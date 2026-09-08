#pragma once
// Independent contact-QP oracle and diagnostics for the certification tests
// (plan doc/plans/2026-09-08-accuracy-efficiency-validation.md §1.2).
//
// NOTHING in this header may call the production solver, its certificate
// helper or its projector: the whole point is to recompute every acceptance
// quantity with separate code. Only Eigen and the standard library are used.

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace oracle {

// ── Independent diagnostics ────────────────────────────────────────────────
// Recomputed from the returned pressure and a dense operator, following the
// declared physical normalisation but with separate arithmetic.
struct Diagnostics {
    double approach = 0.0;      // mean of v over strictly positive pressure
    double min_gap = 0.0;       // min_i (v_i - approach)
    double fw_gap = 0.0;        // sum_i p_i (v_i - min_j v_j)
    double fw_error = 0.0;      // fw_gap / (P g_ref)
    double penetration = 0.0;   // max(0, -min_gap) / g_ref
    double load_error = 0.0;    // |sum p - P| / P
    double min_pressure = 0.0;
    double pk_error = 0.0;      // legacy sum p |g| / (P g_ref)
    double objective = 0.0;     // 1/2 p'Sp + p'g0
    int n_contact = 0;
};

inline Diagnostics diagnose(const Eigen::MatrixXd& S, const Eigen::VectorXd& g0,
                            const Eigen::VectorXd& p, double P_total,
                            double g_ref) {
    Diagnostics d;
    const Eigen::VectorXd v = S * p + g0;
    double s = 0.0;
    int nc = 0;
    for (int i = 0; i < p.size(); ++i)
        if (p(i) > 0.0) { s += v(i); ++nc; }
    d.n_contact = nc;
    d.approach = nc ? s / nc : v.minCoeff();
    d.min_gap = v.minCoeff() - d.approach;
    double G = 0.0, pk = 0.0;
    const double vmin = v.minCoeff();
    for (int i = 0; i < p.size(); ++i) {
        G += p(i) * (v(i) - vmin);
        pk += p(i) * std::abs(v(i) - d.approach);
    }
    d.fw_gap = G;
    d.fw_error = G / (P_total * g_ref);
    d.penetration = std::max(0.0, -d.min_gap) / g_ref;
    d.load_error = std::abs(p.sum() - P_total) / P_total;
    d.min_pressure = p.minCoeff();
    d.pk_error = pk / (P_total * g_ref);
    d.objective = 0.5 * p.dot(S * p) + p.dot(g0);
    return d;
}

// ── Independent projection onto {p >= 0, sum p = P} (sorting reference) ────
inline Eigen::VectorXd sorting_projection(const Eigen::VectorXd& y,
                                          double P_total) {
    const int n = static_cast<int>(y.size());
    std::vector<double> u(y.data(), y.data() + n);
    std::sort(u.begin(), u.end(), std::greater<double>());
    double css = 0.0, theta = 0.0;
    int k = 0;
    for (int i = 0; i < n; ++i) {
        css += u[i];
        const double t = (css - P_total) / double(i + 1);
        if (u[i] > t) { theta = t; k = i + 1; }
    }
    (void)k;
    Eigen::VectorXd p(n);
    for (int i = 0; i < n; ++i) p(i) = std::max(y(i) - theta, 0.0);
    return p;
}

// ── Dense QP oracle by support enumeration (n <= 12) ────────────────────────
// For each nonempty support A solve
//   [S_AA  -1; 1' 0] [p_A; alpha] = [-g0_A; P]
// and keep the candidate with p_A >= 0 and nonnegative gaps outside A.
// Returns the best (lowest objective) admissible candidate; `found` is false
// when the enumeration found none (should not happen for SPD S).
struct OracleSolution {
    Eigen::VectorXd p;
    double alpha = 0.0;
    double objective = 0.0;
    bool found = false;
};

inline OracleSolution enumerate_qp(const Eigen::MatrixXd& S,
                                   const Eigen::VectorXd& g0, double P_total,
                                   double tol = 1e-10) {
    const int n = static_cast<int>(g0.size());
    OracleSolution best;
    best.objective = std::numeric_limits<double>::infinity();
    const long long nsets = 1LL << n;
    for (long long m = 1; m < nsets; ++m) {
        std::vector<int> A;
        for (int i = 0; i < n; ++i)
            if (m & (1LL << i)) A.push_back(i);
        const int k = static_cast<int>(A.size());
        Eigen::MatrixXd K(k + 1, k + 1);
        K.setZero();
        Eigen::VectorXd rhs(k + 1);
        for (int a = 0; a < k; ++a) {
            for (int b = 0; b < k; ++b) K(a, b) = S(A[a], A[b]);
            K(a, k) = -1.0;
            K(k, a) = 1.0;
            rhs(a) = -g0(A[a]);
        }
        rhs(k) = P_total;
        Eigen::FullPivLU<Eigen::MatrixXd> lu(K);
        if (!lu.isInvertible()) continue;
        const Eigen::VectorXd sol = lu.solve(rhs);
        if ((K * sol - rhs).norm() > 1e-8 * (1.0 + rhs.norm())) continue;
        Eigen::VectorXd p = Eigen::VectorXd::Zero(n);
        bool ok = true;
        for (int a = 0; a < k && ok; ++a) {
            if (sol(a) < -tol) ok = false;
            p(A[a]) = std::max(sol(a), 0.0);
        }
        if (!ok) continue;
        const double alpha = sol(k);
        const Eigen::VectorXd v = S * p + g0;
        for (int i = 0; i < n && ok; ++i)
            if (v(i) - alpha < -tol * (1.0 + std::abs(alpha))) ok = false;
        if (!ok) continue;
        const double obj = 0.5 * p.dot(S * p) + p.dot(g0);
        if (obj < best.objective) {
            best.objective = obj;
            best.p = p;
            best.alpha = alpha;
            best.found = true;
        }
    }
    return best;
}

// ── Manufactured KKT fixture ───────────────────────────────────────────────
// Given p* >= 0 and g* >= 0 with complementary support, choose
//   g0 = alpha* 1 - S p* + g*
// so that (p*, alpha*) satisfies the KKT system exactly: v = S p* + g0
// = alpha* 1 + g*, hence v - alpha* = g* >= 0 and p*_i g*_i = 0.
struct Fixture {
    Eigen::VectorXd g0;
    Eigen::VectorXd p_star;
    double P_total = 0.0;
    double alpha_star = 0.0;
};

inline Fixture manufacture(const Eigen::MatrixXd& S,
                           const Eigen::VectorXd& p_star,
                           const Eigen::VectorXd& g_star, double alpha_star) {
    Fixture f;
    f.p_star = p_star;
    f.alpha_star = alpha_star;
    f.P_total = p_star.sum();
    f.g0 = Eigen::VectorXd::Constant(p_star.size(), alpha_star) - S * p_star +
           g_star;
    return f;
}

// Small SPD test matrix: a smooth decaying kernel plus a diagonal shift.
inline Eigen::MatrixXd spd_kernel(int n, double decay = 1.0) {
    Eigen::MatrixXd S(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            S(i, j) = std::exp(-decay * std::abs(double(i - j))) /
                      (1.0 + 0.25 * std::abs(double(i - j)));
    S += 0.1 * Eigen::MatrixXd::Identity(n, n);
    return 0.5 * (S + S.transpose());
}

} // namespace oracle
