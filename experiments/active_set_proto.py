"""Active-set localization prototype for the Boussinesq contact solver.

Three questions (see plan / active_set_results.md):
  Q1  Conditioning: does the PK PCG restricted to (active set + halo) converge
      without the global |q| preconditioner, with it, or with a REDUCED-
      FREQUENCY |q| preconditioner (capped symbol min(|q|,qc) -- the
      Kochmann/Gierden-Reese reduced-wave-vector idea, arXiv:2103.10203,
      which becomes O(|R| N_c) with no O(N) arrays when input and output
      live on the candidate set)?
  Q2  Discovery: can candidate sets built from the coarse level -- dilated
      coarse contact UNION {prolonged coarse gap < delta} (gap-proximity
      halos; fine-scale asperities invisible to the coarse grid must be
      caught by the gap threshold, not geometry) -- find the fine active set
      in 1-2 verify-and-extend rounds? Counts "orphan islands": fine contact
      patches with zero coarse-contact precursor.
  Q3  Cost model inputs: iteration counts, |C|/N_c, rounds.

Iteration counts and correctness are the measurables; Python wall time is
not representative (matvecs are emulated with the full-grid C++ FFT backend).

Run in fenicsx-env:  python experiments/active_set_proto.py [--ns 1024]
"""
import argparse
import sys
import time

import numpy as np
from scipy import ndimage

sys.path.insert(0, "/home/vyastrebov/WORK/PROJECTS/Hcontact/python")
import aspher as hc


def make_surface(Ns, seed=42, hurst=0.8, rms=0.02):
    """Self-affine surface with spectral content from k=1 to the fine-grid
    Nyquist, so coarser grids genuinely cannot see the smallest asperities."""
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


class QPrecond:
    """|q| spectral preconditioner (mirrors fourier_precond.cpp), full or
    reduced. symbol = |q| (full) or min(|q>, qc) (reduced/capped: exact on the
    smooth band |q| < qc, identity-like at patch scales). qc=None -> full."""

    def __init__(self, Ns, qc=None):
        kx = np.fft.rfftfreq(Ns) * Ns
        ky = np.fft.fftfreq(Ns) * Ns
        Q = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
        sym = Q if qc is None else np.minimum(Q, float(qc))
        sym[0, 0] = 0.0  # zero DC
        self.sym = sym
        self.Ns = Ns

    def apply(self, r, contact):
        Ns = self.Ns
        g = np.where(contact, r, 0.0).reshape(Ns, Ns)
        z = np.fft.irfft2(np.fft.rfft2(g) * self.sym, s=(Ns, Ns)).ravel()
        z = np.where(contact, z, 0.0)
        nc = contact.sum()
        if nc:
            z[contact] -= z[contact].mean()
        return z


def restricted_pk(matvec, g0, p_bar, C, tol=1e-8, max_iter=2000,
                  precond=None, p_init=None):
    """Polonsky-Keer PCG (PR+, overlap correction, load constraint, implicit
    alpha) with the working set restricted to the candidate mask C: pressure,
    directions, and the overlap correction live on C only; points outside C
    are the verification loop's responsibility. Mirrors solve_contact_impl.
    Returns (p, alpha, iterations, converged)."""
    N = g0.size
    P_total = p_bar * N
    g_scale = g0.max() - g0.min() or 1.0
    if p_init is not None:
        p = np.where(C, np.maximum(p_init, 0.0), 0.0)
        s = p.sum()
        p = p * (P_total / s) if s > 0 else np.where(C, P_total / C.sum(), 0.0)
    else:
        p = np.where(C, P_total / C.sum(), 0.0)
    t = np.zeros(N)
    g_prev = np.zeros(N)
    G_old, delta, alpha = 1.0, 0.0, 0.0
    it = 0
    for it in range(1, max_iter + 1):
        u = matvec(p)
        g = u + g0
        contact = p > 0.0
        nc = contact.sum()
        alpha = g[contact].mean() if nc else 0.0
        g -= alpha
        err = float(p[contact].dot(np.abs(g[contact]))) / (P_total * g_scale)
        if err < tol:
            return p, alpha, it, True
        z = precond.apply(g, contact) if precond else np.where(contact, g, 0.0)
        G = float(z[contact].dot(g[contact]))
        G_pr = float(z[contact].dot(g[contact] - g_prev[contact]))
        beta = delta * max(0.0, G_pr / G_old)
        t = np.where(contact, z + beta * t, 0.0)
        g_prev = g.copy()
        G_old = G
        r = matvec(t)
        rmean = r[contact].mean() if nc else 0.0
        num = float(g[contact].dot(t[contact]))
        den = float((r[contact] - rmean).dot(t[contact]))
        if den <= 0.0:
            delta = 0.0
            continue
        tau = num / den
        p = np.where(contact, np.maximum(p - tau * t, 0.0), p)
        overlap = C & (p == 0.0) & (g < 0.0)  # overlap correction on C only
        if overlap.any():
            p = np.where(overlap, p - tau * g, p)
            delta = 0.0
        else:
            delta = 1.0
        total = p.sum()
        p = p * (P_total / total) if total > 0 else np.where(C, p_bar, 0.0)
    return p, alpha, it, False


def dilate(mask2d, k):
    return ndimage.binary_dilation(mask2d, iterations=k) if k > 0 else mask2d


def prolong(f2d):
    return np.repeat(np.repeat(f2d, 2, axis=0), 2, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, default=1024)
    ap.add_argument("--pbar", type=float, nargs="+", default=[0.002, 0.01])
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--halos", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--qc", type=int, nargs="+", default=[8, 16, 32],
                    help="reduced-precond cutoffs; empty list to skip")
    ap.add_argument("--skip-q2", action="store_true")
    args = ap.parse_args()
    Ns = args.ns
    N = Ns * Ns

    surf = make_surface(Ns)
    gap0 = (-surf).ravel().astype(np.float64)
    S = hc.ContactSolver(grid_size=Ns, backend="fft")
    mv = lambda x: np.asarray(S.matvec(x))
    pc_full = QPrecond(Ns)

    for pbar in args.pbar:
        # ---- reference + global-arm iteration counts (C++ solver) ----
        ref = hc.solve_nested(grid_size=Ns, gap=gap0, p_nominal=pbar,
                              coarsest=64, q=6, tol=1e-10)
        pref = np.asarray(ref.pressure).ravel()
        Astar = (pref > 0.0)
        Nc = int(Astar.sum())
        nisl = ndimage.label(Astar.reshape(Ns, Ns))[1]
        g_none = S.solve(gap0, p_nominal=pbar, tol=args.tol, precond="none")
        g_four = S.solve(gap0, p_nominal=pbar, tol=args.tol, precond="fourier")
        print(f"\n== pbar={pbar}: N_c={Nc} ({Nc/N:.2%} of N), {nisl} islands; "
              f"global PK iters none/fourier = {g_none.iterations}/"
              f"{g_four.iterations} (tol {args.tol:g}) ==", flush=True)

        # ---- Q1: oracle candidate sets, three preconditioner arms ----
        for k in args.halos:
            C = dilate(Astar.reshape(Ns, Ns), k).ravel()
            arms = [("none", None), ("fourier", pc_full)]
            arms += [(f"reduced qc={qc}", QPrecond(Ns, qc=qc))
                     for qc in args.qc]
            line = [f"  Q1 halo k={k} (|C|/N_c={C.sum()/Nc:.2f}):"]
            for name, pc in arms:
                t0 = time.time()
                p, a, its, ok = restricted_pk(mv, gap0, pbar, C, tol=args.tol,
                                              precond=pc)
                rel = np.linalg.norm(p - pref) / np.linalg.norm(pref)
                line.append(f"{name} {its}it"
                            f"{'' if ok else '(!)'} relL2={rel:.1e}")
                _ = time.time() - t0
            print("  " + "  |  ".join(line), flush=True)

        # ---- Q2: discovery from the coarse level (gap-proximity halos) ----
        if args.skip_q2:
            continue
        Nc2 = Ns // 2
        gapc = gap0.reshape(Ns // 2, 2, Ns // 2, 2).mean(axis=(1, 3)).ravel()
        refc = hc.solve_nested(grid_size=Nc2, gap=gapc, p_nominal=pbar,
                               coarsest=64, q=6, tol=1e-9)
        pc2 = np.asarray(refc.pressure).ravel()
        gc2 = np.asarray(refc.gap).ravel()  # coarse gap field (>=0)
        pinit = prolong(pc2.reshape(Nc2, Nc2)).ravel()
        gfine_pred = prolong(gc2.reshape(Nc2, Nc2)).ravel()
        coarse_supp = (pinit > 0.0).reshape(Ns, Ns)

        # orphan islands: fine contact patches with no coarse precursor
        lab, nlab = ndimage.label(Astar.reshape(Ns, Ns))
        orphan = sum(1 for i in range(1, nlab + 1)
                     if not coarse_supp[lab == i].any())
        rms = 0.02
        print(f"  Q2 coarse {Nc2}: area {refc.contact_area:.5f} vs fine "
              f"{ref.contact_area:.5f}; orphan islands {orphan}/{nlab}",
              flush=True)
        for ddelta in (0.01, 0.03, 0.1):
            delta = ddelta * rms
            C = (dilate(coarse_supp, 2) | (gfine_pred < delta).reshape(Ns, Ns)
                 ).ravel()
            missed0 = int((Astar & ~C).sum())
            rounds, its_tot = 0, 0
            p_in = pinit.copy()
            while True:
                p, a, its, ok = restricted_pk(mv, gap0, pbar, C, tol=args.tol,
                                              p_init=p_in)
                its_tot += its
                rounds += 1
                gfull = mv(p) + gap0 - a          # full-grid verification
                viol = (~C) & (gfull < -args.tol * rms)
                if not viol.any() or rounds >= 5:
                    break
                C = C | dilate(viol.reshape(Ns, Ns), 1).ravel()
                p_in = p
            rel = np.linalg.norm(p - pref) / np.linalg.norm(pref)
            print(f"    delta={ddelta:.2f}*rms: |C|={C.sum()} "
                  f"({C.sum()/Nc:.2f}*N_c) missed@start={missed0} "
                  f"rounds={rounds} iters_total={its_tot} relL2={rel:.1e}",
                  flush=True)


if __name__ == "__main__":
    main()
