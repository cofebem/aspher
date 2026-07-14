#include "bipotential.hpp"

#include <cmath>
#include <stdexcept>

namespace hmc {

void project_coulomb_cone(double mu, double& tx, double& ty, double& tn) {
    const double tt = std::hypot(tx, ty);
    if (mu * tt < -tn) { // inside the polar cone: separation
        tx = 0.0;
        ty = 0.0;
        tn = 0.0;
    } else if (tt > mu * tn) { // outside the cone: radial return (slip)
        const double k = (tn + mu * tt) / (1.0 + mu * mu);
        const double f = (tt > 0.0) ? k * mu / tt : 0.0;
        tx *= f;
        ty *= f;
        tn = k;
    } // else: inside the cone (stick) — unchanged
}

BipotentialResult solve_bipotential(
    const std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>& S,
    const std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>& C,
    const Eigen::VectorXd& g0, double mu, double alpha,
    const Eigen::Vector2d& delta_t, double tol, int max_iter,
    double rho_scale, const Eigen::VectorXd* s_frozen, int npower) {
    const int N = static_cast<int>(g0.size());
    if (N == 0) throw std::invalid_argument("solve_bipotential: empty g0");
    if (s_frozen && static_cast<int>(s_frozen->size()) != N)
        throw std::invalid_argument("solve_bipotential: s_frozen size");

    // rho from the stacked operator norm: power iteration on diag(C, S)
    // (lambda_max = max of the two norms; both symbols peak at the lowest
    // wavenumber, power iteration converges in a few sweeps)
    double lam = 0.0;
    {
        Eigen::VectorXd v = Eigen::VectorXd::Ones(N), Sv(N);
        Eigen::VectorXd w2 = Eigen::VectorXd::Ones(2 * N), Cw(2 * N);
        v(0) += 0.5; // break symmetry a little
        w2(1) += 0.5;
        double ls = 0.0, lc = 0.0;
        for (int k = 0; k < npower; ++k) {
            S(v, Sv);
            ls = Sv.norm() / v.norm();
            v = Sv / Sv.norm();
            C(w2, Cw);
            lc = Cw.norm() / w2.norm();
            w2 = Cw / Cw.norm();
        }
        lam = std::max(ls, lc);
    }
    if (!(lam > 0.0))
        throw std::runtime_error("solve_bipotential: operator norm estimate");
    const double rho = rho_scale / lam;

    Eigen::VectorXd p = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd q = Eigen::VectorXd::Zero(2 * N);
    Eigen::VectorXd uz(N), ut(2 * N);

    BipotentialResult res;
    int it = 0;
    for (it = 0; it < max_iter; ++it) {
        S(p, uz);
        C(q, ut);
        double upd2 = 0.0, cur2 = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : upd2, cur2)
        for (int i = 0; i < N; ++i) {
            const double g = uz(i) + g0(i) - alpha;
            const double wx = delta_t(0) - ut(i);
            const double wy = delta_t(1) - ut(N + i);
            double tx = q(i) + rho * wx;
            double ty = q(N + i) + rho * wy;
            double tn;
            if (!s_frozen) { // Coulomb cone
                tn = p(i) - rho * (g + mu * std::hypot(wx, wy));
                project_coulomb_cone(mu, tx, ty, tn);
            } else { // frozen-threshold cylinder: p and q decouple
                tn = std::max(0.0, p(i) - rho * g);
                const double si = (*s_frozen)(i);
                const double tt = std::hypot(tx, ty);
                if (tt > si) {
                    const double f = (tt > 0.0) ? si / tt : 0.0;
                    tx *= f;
                    ty *= f;
                }
                if (tn == 0.0) {  // no shear without contact pressure: for an
                    tx = 0.0;       // externally-supplied s_frozen this is NOT
                    ty = 0.0;       // redundant with the disk clamp (s may be > 0
                }                   // where p transiently vanishes mid-iteration)
            }
            upd2 += (tx - q(i)) * (tx - q(i)) +
                    (ty - q(N + i)) * (ty - q(N + i)) +
                    (tn - p(i)) * (tn - p(i));
            cur2 += tx * tx + ty * ty + tn * tn;
            q(i) = tx;
            q(N + i) = ty;
            p(i) = tn;
        }
        res.error = (cur2 > 0.0) ? std::sqrt(upd2 / cur2) : 1.0;
        if (res.error < tol) {
            res.converged = true;
            break;
        }
    }

    res.iterations = it;
    int no = 0, nst = 0, nsl = 0;
    for (int i = 0; i < N; ++i) {
        const double qt = std::hypot(q(i), q(N + i));
        const double si = s_frozen ? (*s_frozen)(i) : mu * p(i);
        if (p(i) <= 0.0)
            ++no;
        else if (qt < si * (1.0 - 1e-9))
            ++nst;
        else
            ++nsl;
    }
    res.n_open = no;
    res.n_stick = nst;
    res.n_slip = nsl;
    res.p = std::move(p);
    res.q = std::move(q);
    return res;
}

} // namespace hmc
