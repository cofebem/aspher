"""FETI-style block decomposition experiment for the Boussinesq contact problem.

Question: can we tear the Ns x Ns contact problem into B x B tiles, solve the
tiles independently (each tile's diagonal operator S_ii is the same small Love
operator by translation invariance), and glue them by iterating on the frozen
far-field displacement (block Jacobi / damped Jacobi)?

Formulation: prescribed-approach. The reference load-controlled solve gives
(p*, alpha*); with alpha = alpha* fixed, p* is also the unique solution of the
approach-controlled problem, so the block iteration's only job is to recover
the inter-tile elastic coupling — the scalar load-balance loop that a real
solver would wrap around this is deliberately excluded from the study.

Per sweep:
    u      = S_global p                (one global FFT matvec)
    ufar_i = u|_i - S_loc p_i          (exact: same Love kernel, same h, E*)
    p_i    <- LCP(S_loc, g0|_i + ufar_i - alpha*)   for every tile (parallel)
    p      <- (1-omega) p + omega p_new

The local LCP solver is the Polonsky-Keer projected CG stripped of the load
constraint and rigid-body shift (alpha is prescribed): p >= 0, w = Sp + g >= 0,
p.w = 0.

Outputs per (pbar, blocks) config: gluing error of fully independent tiles
(sweep 0 with ufar = 0), then per-sweep rel-L2 pressure error vs p*, contact
area error, complementarity residual, and local-iteration totals for cost
accounting (sum of local matvecs ~ one global matvec-equivalent per local
iteration when tiles cover the grid).

Run in fenicsx-env:  python experiments/block_gs_feti.py [--ns 1024]
"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "/home/vyastrebov/WORK/PROJECTS/Hcontact/python")
import aspher as hc


def make_surface(Ns, seed=42, hurst=0.8, rms=0.02):
    """Self-affine surface via spectral filtering (same recipe as ref_solve)."""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(Ns) * Ns
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)
    with np.errstate(divide="ignore"):
        psd = np.where(K > 0, K ** (-2.0 * (1.0 + hurst)), 0.0)
    noise = rng.standard_normal((Ns, Ns))
    surf = np.fft.ifft2(np.fft.fft2(noise) * np.sqrt(psd)).real
    surf *= rms / surf.std()
    return surf


def lcp_solve(matvec, g, p0=None, tol=1e-10, max_iter=400, seed_scale=1.0):
    """Projected CG for p >= 0, w = S p + g >= 0, p.w = 0 (alpha prescribed).

    Polonsky-Keer structure without the mean-pressure constraint: PR+ beta,
    exact line search on the contact set, overlap correction for penetrating
    zero-pressure points. seed_scale converts penetration depth to a pressure
    of the right magnitude for cold starts (~ pi E*/(4 h ln(1+sqrt2)), the
    inverse Love self-term). Returns (p, iterations, converged).
    """
    N = g.size
    g_scale = g.max() - g.min()
    if g_scale <= 0.0:
        g_scale = 1.0
    p = np.zeros(N) if p0 is None else np.maximum(p0, 0.0)
    if p.max() == 0.0:
        # cold start: seed pressure on penetrating points
        p = np.maximum(-g, 0.0) * seed_scale
        if p.max() == 0.0:
            return p, 0, True  # no penetration anywhere: p = 0 is the solution
    t = np.zeros(N)
    g_prev = np.zeros(N)
    G_old = 1.0
    delta = 0.0
    for it in range(1, max_iter + 1):
        w = matvec(p) + g
        contact = p > 0.0
        pscale = p.sum() * g_scale
        err = float(np.abs(w[contact]).dot(p[contact])) / pscale if pscale > 0 else 0.0
        pen = max(0.0, float(-w[~contact].min()) if (~contact).any() else 0.0)
        if err < tol and pen < tol * g_scale:
            return p, it, True
        if not contact.any():
            # projection emptied the contact set but penetration remains:
            # re-seed on the penetrating points and restart the CG
            p = np.maximum(-w, 0.0) * seed_scale
            t[:] = 0.0
            delta = 0.0
            continue
        z = np.where(contact, w, 0.0)
        G = float(z[contact].dot(w[contact]))
        G_pr = float(z[contact].dot(w[contact] - g_prev[contact]))
        beta = delta * max(0.0, G_pr / G_old)
        t = np.where(contact, z + beta * t, 0.0)
        g_prev = w.copy()
        G_old = G
        St = matvec(t)
        num = float(w[contact].dot(t[contact]))
        den = float(St[contact].dot(t[contact]))
        if den <= 0.0:
            delta = 0.0
            continue
        tau = num / den
        p = np.where(contact, np.maximum(p - tau * t, 0.0), p)
        overlap = (p == 0.0) & (w < 0.0)
        if overlap.any():
            p = np.where(overlap, p - tau * w, p)
            delta = 0.0
        else:
            delta = 1.0
    return p, max_iter, False


def run_config(Ns, gap0, pref, alpha, nb, omega, sweeps, S_glob, tol_loc):
    """Block-Jacobi iteration with nb x nb tiles. Returns per-sweep records."""
    B = Ns // nb  # tile side
    S_loc = hc.ContactSolver(grid_size=B, domain_size=1.0 * B / Ns,
                             E_star=1.0, backend="fft")
    mv_loc = lambda x: np.asarray(S_loc.matvec(x))
    # penetration -> pressure conversion for cold starts (inverse Love self-term)
    seed = np.pi / (4.0 * (1.0 / Ns) * np.log(1.0 + np.sqrt(2.0)))

    g0b = gap0.reshape(nb, B, nb, B)
    pref_n = np.linalg.norm(pref)
    area_ref = float((pref > 0).mean())

    def metrics(p, tag, local_iters, dt):
        relp = np.linalg.norm(p - pref) / pref_n
        area = float((p > 0).mean())
        load = p.mean()
        print(f"    {tag:>10s}: rel-L2(p) {relp:9.3e}  area {area:.5f} "
              f"(ref {area_ref:.5f})  mean(p)/pbar {load/pref.mean():6.3f}  "
              f"loc-iters {local_iters:5d}  [{dt:5.1f}s]", flush=True)
        return relp

    # ---- sweep 0: fully independent tiles (interaction OFF) ----
    t0 = time.time()
    p = np.zeros((nb, B, nb, B))
    tot_it = 0
    for i in range(nb):
        for j in range(nb):
            geff = (g0b[i, :, j, :] - alpha).ravel()
            pij, its, _ = lcp_solve(mv_loc, geff, tol=tol_loc, seed_scale=seed)
            p[i, :, j, :] = pij.reshape(B, B)
            tot_it += its
    p = p.reshape(Ns * Ns)
    relp0 = metrics(p, "indep", tot_it, time.time() - t0)

    # ---- block-Jacobi sweeps with frozen far field ----
    hist = [relp0]
    for s in range(1, sweeps + 1):
        t0 = time.time()
        u = np.asarray(S_glob.matvec(p))
        pb = p.reshape(nb, B, nb, B)
        ub = u.reshape(nb, B, nb, B)
        pnew = np.empty_like(pb)
        tot_it = 0
        for i in range(nb):
            for j in range(nb):
                pij = pb[i, :, j, :].ravel()
                ufar = ub[i, :, j, :].ravel() - mv_loc(pij)
                geff = g0b[i, :, j, :].ravel() + ufar - alpha
                pn, its, _ = lcp_solve(mv_loc, geff, p0=pij, tol=tol_loc,
                                       seed_scale=seed)
                pnew[i, :, j, :] = pn.reshape(B, B)
                tot_it += its
        p = ((1.0 - omega) * pb + omega * pnew).reshape(Ns * Ns)
        relp = metrics(p, f"sweep {s}", tot_it, time.time() - t0)
        hist.append(relp)
        if relp < 1e-8 or relp > 10.0:
            break
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, default=1024)
    ap.add_argument("--pbar", type=float, nargs="+", default=[0.002, 0.02])
    ap.add_argument("--blocks", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--sweeps", type=int, default=10)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--tol-loc", type=float, default=1e-10)
    args = ap.parse_args()

    Ns = args.ns
    surf = make_surface(Ns)
    gap0 = (-surf).ravel().astype(np.float64)
    S_glob = hc.ContactSolver(grid_size=Ns, backend="fft")

    for pbar in args.pbar:
        t0 = time.time()
        ref = hc.solve_nested(grid_size=Ns, gap=gap0, p_nominal=pbar,
                              coarsest=64, q=6, tol=1e-10)
        pref = np.asarray(ref.pressure).ravel()
        alpha = ref.approach
        print(f"\n== pbar={pbar}: reference {ref.iterations} finest-level "
              f"PCG iters, area {ref.contact_area:.5f}, alpha* {alpha:.6e} "
              f"[{time.time()-t0:.1f}s] ==", flush=True)
        for nb in args.blocks:
            print(f"  -- {nb}x{nb} tiles (tile {Ns//nb}x{Ns//nb}), "
                  f"omega={args.omega} --", flush=True)
            run_config(Ns, gap0, pref, alpha, nb, args.omega, args.sweeps,
                       S_glob, args.tol_loc)


if __name__ == "__main__":
    main()
