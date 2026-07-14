#include "boussinesq_kernel.hpp"
#include "fft_operator.hpp"
#include "contact_solver.hpp"
#include "friction_driver.hpp"
#include "friction_model.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAILED: %s (line %d)\n", #cond, __LINE__);        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

static int test_models() {
    const int N = 6;
    Eigen::VectorXd p(N), v(N), T(N), s;
    p << 1.0, 0.5, 0.0, -0.1, 2.0, 0.0;
    v << 0.0, 1.0, 2.0, 3.0, 0.5, 0.0;
    T.setConstant(300.0);

    hmc::TrescaModel tresca(0.07);
    tresca.threshold(p, v, T, s);
    CHECK(s.size() == N);
    CHECK(s(0) == 0.07 && s(1) == 0.07 && s(4) == 0.07);
    CHECK(s(2) == 0.0 && s(3) == 0.0 && s(5) == 0.0); // p <= 0 -> s = 0
    CHECK(!tresca.velocity_dependent());

    hmc::CoulombModel coulomb(0.3);
    coulomb.threshold(p, v, T, s);
    CHECK(s(0) == 0.3 && s(1) == 0.15 && s(3) == 0.0);
    CHECK(!coulomb.velocity_dependent());

    // callback: rate-weakening law; wrapper must sanitize a sloppy callback
    // that ignores the p <= 0 rule and can return negatives
    hmc::CallbackModel user(
        [](const Eigen::VectorXd& pp, const Eigen::VectorXd& vv,
           const Eigen::VectorXd& TT, Eigen::VectorXd& ss) {
            ss = 0.3 * pp.array() / (1.0 + vv.array()) -
                 0.0 * TT.array(); // uses T to avoid unused warnings
        },
        /*velocity_dependent=*/true);
    user.threshold(p, v, T, s);
    CHECK(std::abs(s(0) - 0.3) < 1e-15);
    CHECK(std::abs(s(1) - 0.15 / 2.0) < 1e-15);
    CHECK(s(3) == 0.0); // raw callback returns -0.03/4: sanitized to 0
    CHECK(s(2) == 0.0 && s(5) == 0.0);
    CHECK(user.velocity_dependent());

    // Test case 1: callback returning NaN at p > 0
    hmc::CallbackModel nan_callback(
        [](const Eigen::VectorXd& pp, const Eigen::VectorXd&,
           const Eigen::VectorXd&, Eigen::VectorXd& ss) {
            ss = pp.array().isNaN().select(pp.array() * 0.0 / 0.0, pp * 0.1);
            ss(0) = std::nan(""); // inject NaN at first point
        },
        false);
    nan_callback.threshold(p, v, T, s);
    CHECK(s(0) == 0.0); // NaN sanitized to 0
    CHECK(std::abs(s(1) - 0.05) < 1e-15);

    // Test case 2: callback returning wrong-sized threshold
    hmc::CallbackModel bad_size_callback(
        [](const Eigen::VectorXd&, const Eigen::VectorXd&,
           const Eigen::VectorXd&, Eigen::VectorXd& ss) {
            ss.resize(3); // wrong size
        },
        false);
    bool bad_size_threw = false;
    try {
        bad_size_callback.threshold(p, v, T, s);
    } catch (const std::invalid_argument&) {
        bad_size_threw = true;
    }
    CHECK(bad_size_threw);

    return 0;
}

static Eigen::VectorXd wavy_gap(int Ns, double L) {
    Eigen::VectorXd g0(Ns * Ns);
    const double h = L / Ns;
    const int mj[6][2] = {{1, 2}, {3, 1}, {2, 4}, {5, 3}, {4, 6}, {7, 2}};
    const double Aj[6] = {0.8, 0.45, 0.3, 0.18, 0.12, 0.08};
    const double ph[6] = {0.3, 1.1, 2.0, 4.2, 5.1, 0.7};
    for (int iy = 0; iy < Ns; ++iy)
        for (int ix = 0; ix < Ns; ++ix) {
            const double x = (ix + 0.5) * h / L, y = (iy + 0.5) * h / L;
            double z = 0.0;
            for (int j = 0; j < 6; ++j)
                z += Aj[j] * std::cos(2.0 * M_PI * (mj[j][0] * x +
                                                    mj[j][1] * y) + ph[j]);
            g0(iy * Ns + ix) = 0.02 * z;
        }
    g0.array() -= g0.minCoeff();
    return g0;
}

// Two-step Coulomb sequence at nu=0 on a rough surface: normal load, then
// tangential force. Checks the full staggered pipeline and dissipation.
static int test_driver_two_step() {
    const int Ns = 64, N = Ns * Ns;
    const double L = 1.0, E = 1.0, nu = 0.0, mu = 0.4, p_bar = 0.01;
    hmc::CoulombModel model(mu);
    hmc::FrictionDriver drv(Ns, L, E, nu, model);
    drv.set_gap(wavy_gap(Ns, L));

    // step 1: normal only
    hmc::FrictionStepSpec s1;
    s1.p_bar = p_bar;
    hmc::FrictionStepResult r1 = drv.step(s1);
    CHECK(r1.converged);
    CHECK(r1.normal.converged);
    CHECK(r1.threshold_iters == 0); // no tangential target
    CHECK(std::abs(r1.normal.mean_pressure - p_bar) < 1e-12);

    // step 2: tangential force at fixed normal load
    hmc::FrictionStepSpec s2;
    s2.has_q_bar = true;
    s2.q_bar = Eigen::Vector2d(0.5 * mu * p_bar, 0.0);
    hmc::FrictionStepResult r2 = drv.step(s2);
    CHECK(r2.converged);
    CHECK(r2.threshold_iters == 1); // Coulomb: single threshold pass
    CHECK((r2.tangential.q_mean - s2.q_bar).norm() <= 1e-8 * s2.q_bar.norm());
    CHECK(r2.dissipation >= -1e-12); // slip work is non-negative
    CHECK(r2.tangential.n_slip > 0 && r2.tangential.n_stick > 0);

    // discrete-exact C-J reference for the first tangential step (nu = 0):
    // q_x = mu (p - p*), p* = normal solve at p_bar - q_bar_x / mu
    hmc::BoussinesqKernel BK(Ns, L, E);
    hmc::FFTOperator S(BK);
    S.build();
    hmc::MatVec Sop = [&S](const Eigen::VectorXd& x) { return S.matvec(x); };
    hmc::ContactResult red = hmc::solve_contact(
        Sop, wavy_gap(Ns, L), p_bar - s2.q_bar(0) / mu, 1e-12, 20000);
    CHECK(red.converged);
    const Eigen::VectorXd q_ref = mu * (drv.pressure() - red.pressure);
    const double rel =
        (r2.tangential.q.head(N) - q_ref).norm() / q_ref.norm();
    std::printf("driver two-step: C-J rel %.3e  dissipation %.3e\n", rel,
                r2.dissipation);
    CHECK(rel < 1e-3); // driver pipeline floor (warm paths, g_floor)

    // transactionality: a step whose target is beyond the gross-slip limit
    // makes solve_tangential throw std::invalid_argument (|q_bar| >= mean(s))
    // rather than return unconverged; the driver must still leave its
    // persistent state untouched.
    const Eigen::VectorXd q_before = drv.q();
    const Eigen::Vector2d delta_t_before = drv.delta_t();
    hmc::FrictionStepSpec s3;
    s3.has_q_bar = true;
    s3.q_bar = Eigen::Vector2d(10.0 * mu * p_bar, 0.0);
    bool threw = false;
    try {
        drv.step(s3);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    CHECK(drv.q() == q_before);
    CHECK(drv.delta_t() == delta_t_before);
    return 0;
}

// Monotonic proportional tangential loading is path-independent (Jäger):
// loading to Q2 in two increments must equal loading in one, and both equal
// the discrete-exact C-J superposition at nu = 0.
static int test_path_independence() {
    const int Ns = 64, N = Ns * Ns;
    const double L = 1.0, E = 1.0, nu = 0.0, mu = 0.4, p_bar = 0.01;
    const double q1 = 0.25 * mu * p_bar, q2 = 0.55 * mu * p_bar;
    hmc::CoulombModel model(mu);

    hmc::FrictionDriver two(Ns, L, E, nu, model);
    two.set_gap(wavy_gap(Ns, L));
    hmc::FrictionStepSpec sn;
    sn.p_bar = p_bar;
    CHECK(two.step(sn).converged);
    hmc::FrictionStepSpec st;
    st.has_q_bar = true;
    st.q_bar = Eigen::Vector2d(q1, 0.0);
    CHECK(two.step(st).converged);
    st.q_bar = Eigen::Vector2d(q2, 0.0);
    hmc::FrictionStepResult r_two = two.step(st);
    CHECK(r_two.converged);
    CHECK(r_two.dissipation >= -1e-12);

    hmc::FrictionDriver one(Ns, L, E, nu, model);
    one.set_gap(wavy_gap(Ns, L));
    CHECK(one.step(sn).converged);
    st.q_bar = Eigen::Vector2d(q2, 0.0);
    hmc::FrictionStepResult r_one = one.step(st);
    CHECK(r_one.converged);

    const double d12 = (r_two.tangential.q - r_one.tangential.q).norm() /
                       r_one.tangential.q.norm();
    // discrete-exact reference
    hmc::BoussinesqKernel BK(Ns, L, E);
    hmc::FFTOperator S(BK);
    S.build();
    hmc::MatVec Sop = [&S](const Eigen::VectorXd& x) { return S.matvec(x); };
    hmc::ContactResult red =
        hmc::solve_contact(Sop, wavy_gap(Ns, L), p_bar - q2 / mu, 1e-12, 20000);
    const Eigen::VectorXd q_ref = mu * (one.pressure() - red.pressure);
    const double e_one = (r_one.tangential.q.head(N) - q_ref).norm() / q_ref.norm();
    const double e_two = (r_two.tangential.q.head(N) - q_ref).norm() / q_ref.norm();
    std::printf("path independence: |two-one| %.3e, vs C-J: one %.3e two %.3e\n",
                d12, e_one, e_two);
    CHECK(e_one < 1e-3);
    CHECK(e_two < 2e-3); // two warm-started solves accumulate two floors
    CHECK(d12 < 3e-3);
    return 0;
}

// Mindlin unloading at nu = 0, discrete-exact: unloading from q1 to q2
// superposes a doubled counter-slip corrective distribution:
//   q_unl = mu(p - p*(p̄ - q1/mu)) - 2 mu(p - p*(p̄ - (q1-q2)/(2 mu)))
// (means: q1 - 2(q1-q2)/2 = q2 ✓). This exercises the u_hist path hard —
// the counter-slip annulus exists ONLY because history is carried.
static int test_mindlin_unloading() {
    const int Ns = 64, N = Ns * Ns;
    const double L = 1.0, E = 1.0, nu = 0.0, mu = 0.4, p_bar = 0.01;
    const double q1 = 0.6 * mu * p_bar, q2 = 0.2 * mu * p_bar;
    hmc::CoulombModel model(mu);
    hmc::FrictionDriver drv(Ns, L, E, nu, model);
    drv.set_gap(wavy_gap(Ns, L));
    hmc::FrictionStepSpec sn;
    sn.p_bar = p_bar;
    CHECK(drv.step(sn).converged);
    hmc::FrictionStepSpec st;
    st.has_q_bar = true;
    st.q_bar = Eigen::Vector2d(q1, 0.0);
    CHECK(drv.step(st).converged);
    st.q_bar = Eigen::Vector2d(q2, 0.0); // UNLOAD
    hmc::FrictionStepResult r = drv.step(st);
    CHECK(r.converged);
    CHECK(r.dissipation >= -1e-12); // counter-slip still dissipates

    hmc::BoussinesqKernel BK(Ns, L, E);
    hmc::FFTOperator S(BK);
    S.build();
    hmc::MatVec Sop = [&S](const Eigen::VectorXd& x) { return S.matvec(x); };
    hmc::ContactResult red1 =
        hmc::solve_contact(Sop, wavy_gap(Ns, L), p_bar - q1 / mu, 1e-12, 20000);
    hmc::ContactResult redr = hmc::solve_contact(
        Sop, wavy_gap(Ns, L), p_bar - (q1 - q2) / (2.0 * mu), 1e-12, 20000);
    CHECK(red1.converged && redr.converged);
    const Eigen::VectorXd q_ref =
        mu * (drv.pressure() - red1.pressure) -
        2.0 * mu * (drv.pressure() - redr.pressure);
    const double rel =
        (r.tangential.q.head(N) - q_ref).norm() / q_ref.norm();
    const double qy_rel = r.tangential.q.tail(N).norm() / q_ref.norm();
    std::printf("mindlin unloading: rel %.3e qy %.3e  D %.3e\n", rel, qy_rel,
                r.dissipation);
    CHECK(rel < 3e-3); // two accumulated solver floors + reversal
    CHECK(qy_rel < 1e-6);
    return 0;
}

// Velocity-dependent smoke: rate-weakening Coulomb converges in few
// threshold passes and the final threshold is self-consistent.
static int test_velocity_dependent() {
    const int Ns = 32;
    const double L = 1.0, E = 1.0, nu = 0.0, mu0 = 0.4, p_bar = 0.01,
                 v0 = 1e-3;
    hmc::CallbackModel model(
        [mu0, v0](const Eigen::VectorXd& p, const Eigen::VectorXd& v,
                  const Eigen::VectorXd&, Eigen::VectorXd& s) {
            s = mu0 * p.array().max(0.0) / (1.0 + v.array() / v0);
        },
        true);
    hmc::FrictionDriver drv(Ns, L, E, nu, model);
    drv.set_gap(wavy_gap(Ns, L));
    hmc::FrictionStepSpec sn;
    sn.p_bar = p_bar;
    CHECK(drv.step(sn).converged);
    hmc::FrictionStepSpec st;
    st.has_q_bar = true;
    st.q_bar = Eigen::Vector2d(0.3 * mu0 * p_bar, 0.0);
    st.dt = 1.0;
    hmc::FrictionStepResult r = drv.step(st);
    std::printf("velocity-dependent: threshold passes %d, D %.3e, conv %d\n",
                r.threshold_iters, r.dissipation, int(r.converged));
    CHECK(r.converged);
    CHECK(r.threshold_iters >= 2 && r.threshold_iters <= 15);
    CHECK(r.dissipation >= -1e-12);
    return 0;
}

int main() {
    if (int rc = test_models()) return rc;
    if (int rc = test_driver_two_step()) return rc;
    if (int rc = test_path_independence()) return rc;
    if (int rc = test_mindlin_unloading()) return rc;
    if (int rc = test_velocity_dependent()) return rc;
    std::printf("test_driver: all checks passed\n");
    return 0;
}
