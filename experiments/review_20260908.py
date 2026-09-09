"""Reproducible review probes; production sources are not modified.

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 conda run -n fenicsx-env \
    python experiments/review_20260908.py
"""
import json
import os
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.linalg import eigvalsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import aspher as hc


def emit(label, **values):
    print(json.dumps(dict(probe=label, **values)), flush=True)


def certificate(op, p, g0):
    p = np.asarray(p).ravel()
    v = op.matvec(p) + g0
    alpha = v[p > 0].mean()
    g = v - alpha
    return dict(fw_gap=float(p @ v - p.sum() * v.min()),
                penetration=float(max(0, -g.min())),
                old_error=float(p @ np.abs(g) / (p.sum() * max(np.ptp(g0), 1e-300))))


def simplex_projection(x, total):
    u = np.sort(x)[::-1]
    t = (np.cumsum(u) - total) / np.arange(1, len(x) + 1)
    k = np.flatnonzero(u > t)[-1]
    return np.maximum(x - t[k], 0)


def warm_start_probe():
    ns = 8
    x = (np.arange(ns) + 0.5) / ns - 0.5
    g0 = (x[:, None] ** 2 + x[None, :] ** 2).ravel()
    op = hc.ContactSolver(grid_size=ns, backend="dense")
    pbar = 0.01
    warm = np.zeros(ns * ns)
    warm[0] = pbar * ns * ns
    bad = op.solve(g0, pbar, tol=1e-10, p_init=warm)
    cold = op.solve(g0, pbar, tol=1e-10)
    emit("warm_start", converged=bad.converged, iterations=bad.iterations,
         pressure_rel_error=float(np.linalg.norm(bad.pressure-cold.pressure) /
                                  np.linalg.norm(cold.pressure)),
         **certificate(op, bad.pressure, g0))
    # Independent convex-QP solution: projected gradient on the load simplex.
    eye = np.eye(ns * ns)
    S = np.column_stack([op.matvec(e) for e in eye])
    ev = eigvalsh(S)
    p = warm.copy()
    for it in range(20000):
        v = S @ p + g0
        fw = float(p @ v - p.sum() * v.min())
        if fw < 1e-12:
            break
        p = simplex_projection(p - v / ev[-1], warm.sum())
    emit("simplex_repair", iterations=it, smallest_eigenvalue=float(ev[0]),
         pressure_error_bound=float(np.sqrt(2 * max(0, fw) / ev[0])),
         error_vs_cold=float(np.linalg.norm(p-cold.pressure.ravel())),
         **certificate(op, p, g0))


def love_np(x, y, a=0.5):
    xp, xm, yp, ym = x+a, x-a, y+a, y-a
    pp, pm, mp_, mm = np.hypot(xp, yp), np.hypot(xp, ym), np.hypot(xm, yp), np.hypot(xm, ym)
    with np.errstate(all="ignore"):
        return (xp*np.log((yp+pp)/(ym+pm)) + yp*np.log((xp+pp)/(xm+mp_)) +
                xm*np.log((ym+mm)/(yp+mp_)) + ym*np.log((xm+mm)/(xp+pm)))


def love_mp(x, y):
    x, y, a = mp.mpf(x), mp.mpf(y), mp.mpf("0.5")
    xp, xm, yp, ym = x+a, x-a, y+a, y-a
    pp, pm, mm_, mm = mp.hypot(xp, yp), mp.hypot(xp, ym), mp.hypot(xm, yp), mp.hypot(xm, ym)
    return (xp*mp.log((yp+pp)/(ym+pm)) + yp*mp.log((xp+pp)/(xm+mm_)) +
            xm*mp.log((ym+mm)/(yp+mm_)) + ym*mp.log((xm+mm)/(xp+pm)))


def quadrature(x, y, n=4):
    z, w = np.polynomial.legendre.leggauss(n)
    return np.sum(w[:, None]*w[None, :] /
                  np.hypot(abs(x)-0.5*z[:, None], abs(y)-0.5*z[None, :])) / 4


def kernel_probe():
    mp.mp.dps = 80
    for x, y in [(8., 0.), (32., 0.), (1024., 0.), (16384., 0.),
                 (-16384., 0.), (16384., 16384.), (-16384., -16384.),
                 (1e6, 0.), (-1e6, 0.)]:
        ref = float(love_mp(x, y))
        emit("kernel", x=x, y=y, love_relative=float(abs(love_np(x,y)/ref-1)),
             gauss4_relative=float(abs(quadrature(x,y)/ref-1)))
    emit("kernel_edge", value=str(love_np(0.5, 0.5)),
         explanation="finite integral, analytic expression has removable singularities")


def rough(ns):
    rng = np.random.default_rng(20260908)
    k = np.fft.fftfreq(ns)*ns
    kk = np.hypot(k[:, None], k[None, :])
    spectrum = np.fft.fft2(rng.standard_normal((ns, ns)))
    spectrum *= np.maximum(kk, 1.)**-1.8 * ((kk >= 2) & (kk <= 16))
    h = np.fft.ifft2(spectrum).real
    return (-0.02*h/h.std()).ravel()


def accuracy_probe():
    ns = 128
    g0, pbar = rough(ns), 0.005
    exact = hc.ContactSolver(grid_size=ns, backend="fft")
    ref = exact.solve(g0, pbar, tol=1e-10, precond="fourier")
    emit("fft_contact", iterations=ref.iterations, **certificate(exact, ref.pressure, g0))
    rng = np.random.default_rng(5)
    a, b = rng.normal(size=(2, ns*ns))
    for q in (4, 6, 8):
        op = hc.ContactSolver(grid_size=ns, backend="h2", q=q)
        r = op.solve(g0, pbar, tol=1e-10, precond="fourier")
        sa, sb = op.matvec(a), op.matvec(b)
        p = np.asarray(r.pressure).ravel()
        exact_u = exact.matvec(p)
        emit("h2_contact", q=q, iterations=r.iterations,
             pressure_rel_error=float(np.linalg.norm(p-ref.pressure.ravel()) / np.linalg.norm(ref.pressure)),
             area_difference=float(r.contact_area-ref.contact_area),
             displacement_rel_error=float(np.linalg.norm(op.matvec(p)-exact_u) / np.linalg.norm(exact_u)),
             symmetry_bilinear=float(abs(a@sb-b@sa)/(np.linalg.norm(a)*np.linalg.norm(sb))),
             **certificate(exact, p, g0))
    # A uniform rescaling of geometry gaps and pressure leaves contact invariant.
    for c in (1., 1e3, 1e6):
        r = hc.solve_nested(ns, c*g0, c*pbar, coarsest=32, backend="fft",
                            single_precision=True, tol=1e-8)
        emit("float_scale", scale=c, converged=r.converged, iterations=r.iterations,
             reported_error=r.error,
             pressure_rel_error=float(np.linalg.norm(r.pressure/c-ref.pressure) / np.linalg.norm(ref.pressure)),
             area=float(r.contact_area))
    for shift in (1e3, 1e6):
        shifted = g0 + shift
        for centered in (False, True):
            g = shifted-shift if centered else shifted
            r = hc.solve_nested(ns, g, pbar, coarsest=32, backend="fft",
                                single_precision=True, tol=1e-8)
            emit("float_gap_offset", shift=shift, centered=centered,
                 converged=r.converged, iterations=r.iterations, reported_error=r.error,
                 pressure_rel_error=float(np.linalg.norm(r.pressure-ref.pressure) / np.linalg.norm(ref.pressure)))
    r = exact.solve(g0, pbar, tol=1e-25, max_iter=600, precond="fourier")
    emit("stagnation", requested_tol=1e-25, converged=r.converged,
         iterations=r.iterations, reported_error=r.error)


def m2l_probe():
    for q in (4, 6, 8):
        z = np.cos((2*np.arange(q)+1)*np.pi/(2*q))
        xx, yy = np.meshgrid(z, z)
        xx, yy = xx.ravel(), yy.ravel()
        matrices = []
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if max(abs(dx), abs(dy)) <= 1:
                    continue
                matrices.append(love_np(dx*8+4*(xx[:, None]-xx[None, :]),
                                        dy*8+4*(yy[:, None]-yy[None, :])))
        for tol in (1e-6, 1e-8):
            ranks = []
            for K in matrices:
                sv = np.linalg.svd(K, compute_uv=False)
                ranks.append(int(np.count_nonzero(sv >= tol*sv[0])))
            emit("m2l_rank", q=q, relative_spectral_cutoff=tol,
                 min=min(ranks), median=float(np.median(ranks)), max=max(ranks),
                 average_factored_dense_flop_ratio=float(2*np.mean(ranks)/(q*q)))


def recurrence_probe():
    ns = 128
    gap, pb = rough(ns), 0.005
    op = hc.ContactSolver(grid_size=ns, backend="fft")
    k = np.fft.fftfreq(ns)*ns
    wh = np.hypot(k[:, None], k[None, :])

    def solve(reuse):
        p = np.full(ns*ns, pb)
        t, prev = np.zeros_like(p), np.zeros_like(p)
        u, available = None, False
        old, delta, calls, reused = 1., 0., 0, 0
        start = time.perf_counter()
        for it in range(2000):
            if not available:
                u = op.matvec(p)
                calls += 1
            mask = p > 0
            g = u+gap
            g -= g[mask].mean()
            if p@np.abs(g)/(p.sum()*np.ptp(gap)) < 1e-10:
                break
            z = np.fft.ifft2(wh*np.fft.fft2(np.where(mask,g,0).reshape(ns,ns))).real.ravel()
            z[~mask] = 0
            z[mask] -= z[mask].mean()
            G = z@g
            beta = delta*max(0., z@(g-prev)/old)
            t = np.where(mask,z+beta*t,0)
            prev, old = g.copy(), G
            r = op.matvec(t)
            calls += 1
            den = (r[mask]-r[mask].mean())@t[mask]
            if den <= 0:
                delta, available = 0., False
                continue
            tau = (g[mask]@t[mask])/den
            raw = p-tau*t
            p = np.maximum(raw,0)
            overlap = (p == 0) & (g < 0)
            p[overlap] -= tau*g[overlap]
            delta = 0. if overlap.any() else 1.
            factor = (pb*len(p))/p.sum()
            p *= factor
            # Only reuse when the update is linear; refresh periodically.
            available = reuse and not overlap.any() and np.all(raw >= 0) and (it+1)%20 != 0
            if available:
                u = factor*(u-tau*r)
                reused += 1
        return p, dict(iterations=it, matvecs=calls, recurrences=reused,
                       seconds=time.perf_counter()-start)
    base, binfo = solve(False)
    rec, rinfo = solve(True)
    emit("residual_recurrence", baseline=binfo, reuse=rinfo,
         pressure_rel_difference=float(np.linalg.norm(rec-base)/np.linalg.norm(base)),
         **certificate(op,rec,gap))


def friction_probe():
    ns, mu = 32, 0.3
    x = (np.arange(ns)+0.5)/ns-0.5
    gap = (x[:, None]**2+x[None, :]**2).ravel()
    fs = hc.FrictionSolver(grid_size=ns, model=hc.CoulombFriction(mu))
    fs.set_gap(gap)
    r = fs.step(p_bar=0.01, q_bar=(0.001, 0.0))
    emit("friction_load", converged=r.converged)
    if not r.converged:
        return
    r = fs.step(p_bar=0.0001)
    p = np.asarray(fs.pressure).ravel()
    q = np.asarray(fs.q).ravel()
    qn = np.hypot(q[:ns*ns],q[ns*ns:])
    emit("friction_normal_unload", converged=r.converged,
         cone_excess=float(np.max(qn-mu*p)),
         shear_on_open=int(np.count_nonzero((p==0)&(qn>1e-14))))
    before = p.copy()
    r = fs.step(p_bar=0.0)
    emit("friction_zero_normal_load", converged=r.converged,
         pressure_unchanged=bool(np.array_equal(before,np.asarray(fs.pressure).ravel())),
         actual_mean=float(np.mean(fs.pressure)))


if __name__ == "__main__":
    emit("environment", omp_threads=os.environ.get("OMP_NUM_THREADS"), module=hc.__file__)
    warm_start_probe()
    kernel_probe()
    accuracy_probe()
    m2l_probe()
    recurrence_probe()
    friction_probe()
