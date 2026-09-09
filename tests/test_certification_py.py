"""Certified-termination and gap-datum regressions through the Python API
(spec A01/A02/A04; tests T02, T07 of the 2026-09-08 validation plan)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import numpy as np
import aspher as hc


def _parabolic(ns):
    x = (np.arange(ns) + 0.5) / ns - 0.5
    return (x[:, None] ** 2 + x[None, :] ** 2).ravel()


def _rough(ns, seed=7):
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(ns, 1.0 / ns)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    kk = np.hypot(kx, ky)
    kk[0, 0] = 1.0
    amp = np.where((kk >= 2) & (kk <= 16), kk ** (-1.8), 0.0)
    h = np.real(np.fft.ifft2(amp * np.exp(1j * rng.uniform(0, 2 * np.pi, (ns, ns)))))
    h = 0.02 * h / h.std()
    g = (-h).ravel().copy()
    return g - g.min()


def _certificate(op, p, g0):
    """Independent recomputation of the acceptance quantities."""
    p = np.asarray(p).ravel()
    v = op.matvec(p) + g0
    alpha = v[p > 0].mean()
    g = v - alpha
    P = p.sum()
    g_ref = max(np.ptp(g0), 1e-300)
    return dict(fw=float(p @ (v - v.min()) / (P * g_ref)),
                pen=float(max(0.0, -g.min()) / g_ref),
                pmin=float(p.min()))


def test_singleton_warm_start_is_repaired():
    """review §2: all load on one corner used to 'converge' at iteration 0."""
    ns = 8
    g0 = _parabolic(ns)
    op = hc.ContactSolver(grid_size=ns, backend="dense")
    p_bar = 0.01
    warm = np.zeros(ns * ns)
    warm[0] = p_bar * ns * ns

    # the historical answer must be rejected by the independent checker
    bad = _certificate(op, warm, g0)
    assert bad["fw"] > 1e-3 and bad["pen"] > 1e-3

    cold = op.solve(g0, p_bar, tol=1e-10)
    got = op.solve(g0, p_bar, tol=1e-10, p_init=warm)
    c = _certificate(op, got.pressure, g0)
    print("warm:", got.status, got.iterations, c)
    assert got.status == "converged"
    assert got.converged is True
    assert got.iterations > 0
    assert c["fw"] <= 1e-10 and c["pen"] <= 1e-10 and c["pmin"] >= 0.0
    rel = (np.linalg.norm(np.asarray(got.pressure) - np.asarray(cold.pressure))
           / np.linalg.norm(np.asarray(cold.pressure)))
    assert rel < 1e-6, rel


def test_status_fields_are_consistent():
    ns = 16
    g0 = _parabolic(ns)
    op = hc.ContactSolver(grid_size=ns, backend="dense")
    r = op.solve(g0, 0.02, tol=1e-10)
    assert r.status == "converged" and r.converged
    assert r.fw_error <= r.effective_tol and r.penetration_error <= r.effective_tol
    assert r.load_error < 1e-12
    assert r.requested_tol == 1e-10 and r.effective_tol == 1e-10
    assert r.matvec_count > 0
    assert r.validation_scope == "solve_operator"
    assert r.operator_error_kind == "unavailable"
    assert r.operator_error is None  # unavailable is not zero

    # An unreachable target must not be reported as success — unless it
    # genuinely is reached: on a small grid the certificate can underflow to
    # exactly zero, which satisfies any tolerance. Accept that only when the
    # INDEPENDENT check confirms it is at the roundoff floor (the same
    # calibration as the C++ T03 gate); otherwise require an honest failure.
    bad = op.solve(g0, 0.02, tol=1e-25, max_iter=300)
    c = _certificate(op, bad.pressure, g0)
    print("tol=1e-25:", bad.status, bad.status_reason, bad.iterations, c)
    if bad.converged:
        assert c["fw"] <= 1e-15 and c["pen"] <= 1e-15, c
    else:
        assert bad.status in ("stagnated", "max_iterations")


def test_gap_datum_invariance_nested():
    """review §4: a 1e6 gap offset changed the float solution by rel err 1.08."""
    ns = 128
    g0 = _rough(ns)
    p_bar = 0.02
    ref = hc.solve_nested(ns, g0, p_bar, coarsest=32, backend="fft", tol=1e-10)
    assert ref.status == "converged"
    pref = np.asarray(ref.pressure).ravel()

    for shift in (0.0, 1e3, 1e6):
        for single in (False, True):
            # the float arm deliberately stays float-only (that is the path
            # whose datum handling is under test) and accepts its documented
            # floor explicitly, rather than hiding it behind a double polish
            r = hc.solve_nested(ns, g0 + shift, p_bar, coarsest=32,
                                backend="fft", single_precision=single,
                                allow_tolerance_relaxation=single,
                                tol=1e-8)
            rel = (np.linalg.norm(np.asarray(r.pressure).ravel() - pref)
                   / np.linalg.norm(pref))
            print(f"shift={shift:.0e} single={single} {r.status} "
                  f"it={r.iterations} rel={rel:.2e} approach={r.approach:.6f}")
            assert r.status == "converged"
            # the approach carries the datum; the physical solution does not
            assert abs(r.approach - ref.approach - shift) < 1e-6 * max(1.0, shift)
            assert rel < (1e-4 if single else 1e-5), (shift, single, rel)
