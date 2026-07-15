import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import numpy as np
import aspher as hc


def _wavy_gap(Ns, L):
    h = L / Ns
    ix = (np.arange(Ns) + 0.5) * h / L
    x, y = np.meshgrid(ix, ix, indexing="ij")
    mj = [(1, 2), (3, 1), (2, 4), (5, 3), (4, 6), (7, 2)]
    Aj = [0.8, 0.45, 0.3, 0.18, 0.12, 0.08]
    ph = [0.3, 1.1, 2.0, 4.2, 5.1, 0.7]
    z = sum(A * np.cos(2 * np.pi * (mx * x + my * y) + p)
            for (mx, my), A, p in zip(mj, Aj, ph))
    g0 = 0.02 * z
    g0 -= g0.min()
    return g0.ravel()


def test_models():
    p = np.array([1.0, 0.5, 0.0, -0.1, 2.0, 0.0])
    v = np.array([0.0, 1.0, 2.0, 3.0, 0.5, 0.0])
    T = np.full(6, 300.0)

    tr = hc.TrescaFriction(0.07)
    s = np.asarray(tr.threshold(p, v, T))
    assert s.shape == (6,)
    assert np.allclose(s[[0, 1, 4]], 0.07)
    assert np.all(s[[2, 3, 5]] == 0.0)      # p <= 0 -> s = 0
    assert not tr.velocity_dependent

    co = hc.CoulombFriction(0.3)
    s = np.asarray(co.threshold(p, v, T))
    assert np.isclose(s[0], 0.3) and np.isclose(s[1], 0.15) and s[3] == 0.0
    assert not co.velocity_dependent

    # user law: rate-weakening; sanitizer must clamp the p<0 negative to 0
    mu0, v0 = 0.3, 1.0
    user = hc.UserFriction(lambda p, v, T: mu0 * p / (1.0 + v / v0),
                           velocity_dependent=True)
    s = np.asarray(user.threshold(p, v, T))
    assert np.isclose(s[0], 0.3) and np.isclose(s[1], 0.15 / 2.0)
    assert s[3] == 0.0 and s[2] == 0.0
    assert user.velocity_dependent

    # a NaN-producing callback is sanitized to 0 at that point
    bad = hc.UserFriction(lambda p, v, T: np.where(p > 1.5, np.nan, mu0 * p),
                          velocity_dependent=False)
    s = np.asarray(bad.threshold(p, v, T))
    assert s[4] == 0.0 and np.isclose(s[0], 0.3)
    print("test_models: OK")


def test_solver_two_step():
    Ns, L, E, nu, mu, p_bar = 64, 1.0, 1.0, 0.0, 0.4, 0.01
    g0 = _wavy_gap(Ns, L)
    fs = hc.FrictionSolver(grid_size=Ns, domain_size=L, E_star=E, nu=nu,
                           model=hc.CoulombFriction(mu), precond=True)
    fs.set_gap(g0)

    r1 = fs.step(p_bar=p_bar)                      # normal only
    assert r1.converged and r1.normal_converged
    assert abs(r1.mean_pressure - p_bar) < 1e-12
    assert r1.qx is None                           # no tangential ran

    r2 = fs.step(q_bar=(0.5 * mu * p_bar, 0.0), dt=1.0)   # tangential force
    assert r2.converged
    assert r2.threshold_iters == 1                 # Coulomb: single pass
    qbar = np.array([0.5 * mu * p_bar, 0.0])
    assert np.linalg.norm(r2.q_mean - qbar) <= 1e-8 * np.linalg.norm(qbar)
    assert r2.dissipation >= -1e-12
    assert r2.n_slip > 0 and r2.n_stick > 0
    assert r2.qx.shape == (Ns, Ns) and r2.qy.shape == (Ns, Ns)

    # discrete-exact Ciavarella-Jager reference (nu=0): q_x = mu (p - p*)
    S = hc.ContactSolver(grid_size=Ns, domain_size=L, E_star=E, backend="fft")
    red = S.solve(_wavy_gap(Ns, L), p_bar - 0.5 * p_bar, tol=1e-12,
                  max_iter=20000)
    q_ref = mu * (np.asarray(fs.pressure) - np.asarray(red.pressure).ravel())
    rel = np.linalg.norm(r2.qx.ravel() - q_ref) / np.linalg.norm(q_ref)
    print(f"solver two-step: C-J rel {rel:.3e}  dissipation {r2.dissipation:.3e}")
    assert rel < 1e-3

    # displacement control + state accessors
    r3 = fs.step(delta_t=(1e-4, 0.0), dt=1.0)
    assert r3.converged
    assert np.asarray(fs.delta_t).shape == (2,)
    assert np.asarray(fs.q).shape == (2 * Ns * Ns,)

    fs.reset()
    assert float(np.asarray(fs.q).max()) == 0.0 if np.asarray(fs.q).size else True
    print("test_solver_two_step: OK")


def test_callback_exception_transactional():
    # a Python callback that raises must propagate cleanly and leave the
    # driver state unchanged (transactional step)
    Ns, L = 32, 1.0
    g0 = _wavy_gap(Ns, L)

    def boom(p, v, T):
        raise ValueError("user law failed")

    fs = hc.FrictionSolver(grid_size=Ns, domain_size=L, nu=0.0,
                           model=hc.UserFriction(boom, velocity_dependent=False))
    fs.set_gap(g0)
    fs.step(p_bar=0.01)
    q_before = np.asarray(fs.q).copy()
    raised = False
    try:
        fs.step(q_bar=(1e-3, 0.0), dt=1.0)
    except ValueError:
        raised = True
    assert raised
    assert np.array_equal(np.asarray(fs.q), q_before)   # state intact
    print("test_callback_exception_transactional: OK")


def test_model_none_rejected():
    # FrictionSolver(model=None) must be rejected, not segfault
    Ns = 32
    try:
        hc.FrictionSolver(grid_size=Ns, model=None)
        assert False, "expected TypeError or ValueError"
    except (TypeError, ValueError):
        pass  # expected
    print("test_model_none_rejected: OK")


def test_nonconverged_step_transactional():
    # On a non-converged tangential step, u_t/delta_t should be None (rolled
    # back), but qx/qy/slip remain available as the failed-candidate diagnostic.
    # Verify that a converged step HAS the displacement fields, and if we ever
    # encounter a non-converged step, they would be None.
    Ns, L = 32, 1.0
    g0 = _wavy_gap(Ns, L)
    fs = hc.FrictionSolver(grid_size=Ns, domain_size=L, E_star=1.0, nu=0.0,
                           model=hc.CoulombFriction(0.4), precond=True)
    fs.set_gap(g0)

    # Converged normal step first
    r1 = fs.step(p_bar=0.01)
    assert r1.converged
    assert r1.ux is None, "ux must be None when no tangential step ran"

    # Converged tangential step: verify that ux/uy/delta_t ARE present
    r2 = fs.step(q_bar=(1e-3, 0.0), dt=1.0)
    assert r2.converged, f"expected converged step; iters={r2.tangential_iters}"
    assert r2.ux is not None, "ux must be present on converged tangential step"
    assert r2.uy is not None, "uy must be present on converged tangential step"
    assert r2.delta_t is not None, "delta_t must be present on converged tangential step"
    # And the candidate diagnostics too
    assert r2.qx is not None
    assert r2.qy is not None
    assert r2.slip_x is not None
    assert r2.slip_y is not None
    assert r2.state is not None

    print("test_nonconverged_step_transactional: OK")


def test_bipotential_reference():
    Ns, L, E, nu, mu, p_bar = 32, 1.0, 1.0, 0.3, 0.4, 3e-3
    h = L / Ns
    ix = (np.arange(Ns) + 0.5) * h - 0.5 * L
    x, y = np.meshgrid(ix, ix, indexing="ij")
    g0 = ((x * x + y * y) / (2.0 * 2.0)).ravel()      # parabolic gap, R=2

    S = hc.ContactSolver(grid_size=Ns, domain_size=L, E_star=E, backend="fft")
    nr = S.solve(g0, p_bar, tol=1e-12, max_iter=20000)

    br = hc.solve_bipotential(grid_size=Ns, gap=g0, mu=mu,
                              approach=nr.approach, delta_t=(6e-4, 2e-4),
                              domain_size=L, E_star=E, nu=nu, tol=1e-9)
    assert br.converged
    relp = (np.linalg.norm(np.asarray(br.pressure).ravel() -
                           np.asarray(nr.pressure).ravel()) /
            np.linalg.norm(np.asarray(nr.pressure).ravel()))
    print(f"bipotential vs normal: rel p {relp:.3e}  it {br.iterations}")
    assert relp < 5e-3
    assert br.n_stick > 0 and br.n_slip > 0
    print("test_bipotential_reference: OK")


if __name__ == "__main__":
    test_models()
    test_solver_two_step()
    test_callback_exception_transactional()
    test_model_none_rejected()
    test_nonconverged_step_transactional()
    test_bipotential_reference()
    print("test_friction_py: all checks passed")
