"""Doubly periodic backend through the Python API: consistency with the
free-space backend for a localised contact, equality of the single-level
and nested drivers, and the precision policy on the periodic operator.
(The Tamaas cross-check lives in compare_tamaas_periodic.py: it needs the
fluidpaper env.)"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import numpy as np
import pytest
import aspher as hc


def _periodic_rough(ns, seed=3, rms=0.01):
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(ns, 1.0 / ns)
    kk = np.hypot(k[:, None], k[None, :])
    kk[0, 0] = 1.0
    amp = np.where((kk >= 2) & (kk <= ns // 8), kk ** (-1.8), 0.0)
    h = np.real(np.fft.ifft2(amp * np.exp(2j * np.pi * rng.random((ns, ns)))))
    h *= rms / h.std()
    return np.ascontiguousarray((h.max() - h).ravel())  # gap >= 0, min 0


def _hertz_gap(ns, L, R=1.0):
    x = (np.arange(ns) + 0.5) * (L / ns) - 0.5 * L
    return np.ascontiguousarray((x[:, None] ** 2 + x[None, :] ** 2).ravel() / (2 * R))


def test_localised_contact_approaches_free_space():
    """Small central Hertz contact: periodic images vanish as a/L -> 0, so the
    periodic solution approaches the free-space one, down to the O(h) floor
    between the spectral symbol and the Love element integral."""
    h, P = 1.0 / 128, 1e-3
    diffs = []
    for ns in (64, 256):
        L = ns * h
        g = _hertz_gap(ns, L)
        res = {}
        for be in ("fft", "periodic"):
            s = hc.ContactSolver(ns, domain_size=L, E_star=1.0, backend=be)
            r = s.solve(g, P / L**2, tol=1e-11, precond="fourier-fft")
            assert r.status == "converged"
            res[be] = np.asarray(r.pressure).ravel()
        assert np.array_equal(res["fft"] > 0, res["periodic"] > 0)
        diffs.append(np.linalg.norm(res["periodic"] - res["fft"])
                     / np.linalg.norm(res["fft"]))
    assert diffs[1] < 5e-3          # measured 4.05e-3 (the O(h) floor)
    assert diffs[1] < diffs[0]      # images recede (8.9e-3 at a/L = 0.18)


def test_nested_matches_single_level_and_precisions():
    ns = 256
    g = _periodic_rough(ns)
    pbar = 0.01
    s = hc.ContactSolver(ns, domain_size=1.0, E_star=1.0, backend="periodic")
    ref = np.asarray(s.solve(g, pbar, tol=1e-12, precond="fourier-fft").pressure).ravel()
    for prec, tol, bound in (("double", 1e-10, 1e-7),
                             ("float_then_double", 1e-10, 1e-7),
                             ("float", 2e-6, 1e-4)):
        r = hc.solve_nested(grid_size=ns, gap=g, p_nominal=pbar, coarsest=32,
                            backend="periodic", tol=tol, precision=prec)
        assert r.status == "converged", (prec, r.status, r.status_reason)
        p = np.asarray(r.pressure).ravel()
        assert abs(p.mean() - pbar) < 1e-6 * pbar
        assert np.linalg.norm(p - ref) / np.linalg.norm(ref) < bound, prec


def test_matvec_is_the_periodic_symbol():
    ns, L, Es = 32, 2.0, 1.7
    s = hc.ContactSolver(ns, domain_size=L, E_star=Es, backend="periodic")
    x = np.arange(ns) * L / ns
    p = np.cos(2 * np.pi * 3 * x / L)[None, :] * np.ones((ns, 1))
    u = np.asarray(s.matvec(p.ravel())).reshape(ns, ns)
    q = 2 * np.pi * 3 / L
    np.testing.assert_allclose(u, 2.0 / (Es * q) * p, atol=1e-14)
    assert s.hmatrix_info()["backend"] == "periodic"


def test_active_set_rejects_periodic():
    g = _periodic_rough(128)
    with pytest.raises(Exception):
        hc.solve_nested(grid_size=128, gap=g, p_nominal=0.01, coarsest=32,
                        backend="periodic", active_set=True)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(name, "ok")
