"""Mesh-convergence study of the true contact area with the area-correction of

  Yastrebov, Anciaux & Molinari, "On the accurate computation of the true
  contact-area in mechanical contact of random rough surfaces", Tribology
  International 114 (2017) 161-171   (doc/convergence_study/2017_TI.pdf)

Reproduces the central result: the *raw* discrete area A_d/A_0 = N_a/N^2 over-
estimates the true area and drifts with the grid, while the *corrected* area

        A_*/A_0 = N_a/N^2 - (pi - 1 + ln2)/24 * M/N^2                    (Eq. 15)

collapses onto a single grid-independent master curve.  N_a is the number of
contacting nodes and M the number of contact<->non-contact switches along all
grid lines (the discrete perimeter S_d = M*dx, S_d*dx = M*dx^2).

Method (as in the paper):
  * one periodic self-affine master surface at N = Nmaster (band-limited to the
    short-wavelength cut-off k_s), coarser grids obtained by *subsampling* it
    (exact, no aliasing since k_s << N/2 on every grid);
  * pressure normalised by E* sqrt(<|grad z|^2>) with the rms gradient measured
    SPECTRALLY (2nd spectral moment) so it is discretisation-independent;
  * frictionless non-adhesive normal contact solved with the H2/FMM Boussinesq
    BEM (this repo) + |q| spectral preconditioner.

Note on the kernel: the 2017 paper used a *periodic* spectral (Westergaard/FFT)
method; here we use the repo's *non-periodic* half-space Boussinesq BEM.  The
area correction is a purely geometric post-processing, independent of the
elastic kernel, so the collapse is reproduced either way; only the absolute
level relative to Persson/asperity references carries a mild edge effect.

Run (fenicsx-env), e.g.:
  OMP_NUM_THREADS=20 python convergence_study.py --grids 128 256 512 1024 2048
  OMP_NUM_THREADS=20 python convergence_study.py --grids 4096 8192   # resumes
"""
import argparse
import math
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))
import numpy as np
import hmatrix_contact as hc
import rfgen as rf

# --- area-correction coefficient  (pi - 1 + ln2)/24  ---------------------------
BETA_PI4 = (np.pi - 1.0 + np.log(2.0)) / 24.0  # = 0.11811416...


# ------------------------------------------------------------------ geometry ---
def contact_mask(pressure):
    """Boolean N x N mask of contacting nodes (p > 0)."""
    return np.asarray(pressure) > 0.0


def count_switches(mask):
    """M = number of contact<->non-contact transitions along all horizontal and
    vertical grid lines (the discrete perimeter is S_d = M*dx)."""
    mh = np.count_nonzero(mask[:, :-1] != mask[:, 1:])
    mv = np.count_nonzero(mask[:-1, :] != mask[1:, :])
    return int(mh + mv)


def corrected_area_fraction(mask):
    """Return (A_d/A_0, A_*/A_0, N_a, M) for a contact mask on an N x N grid."""
    n = mask.shape[0]
    na = int(np.count_nonzero(mask))
    m = count_switches(mask)
    ad = na / (n * n)
    astar = ad - BETA_PI4 * m / (n * n)
    return ad, astar, na, m


def rms_gradient_spectral(z, L):
    """sqrt(<|grad z|^2>) = sqrt(2 m2) measured spectrally (discretisation-
    independent for a band-limited surface)."""
    n = z.shape[0]
    zh = np.fft.fft2(z)
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)  # 2*pi*m/L
    kx, ky = np.meshgrid(k, k, indexing="ij")
    k2 = kx * kx + ky * ky
    grad2 = np.sum(k2 * (np.abs(zh) ** 2)) / (n ** 4)
    return float(np.sqrt(grad2))


def prolong2(p_flat, nc):
    """Pixel-replication prolongation coarse (nc x nc) -> fine (2nc x 2nc), used
    to seed a finer grid's cold solve from the coarse solution."""
    pc = np.asarray(p_flat).reshape(nc, nc)
    return np.repeat(np.repeat(pc, 2, axis=0), 2, axis=1).ravel()


# ------------------------------------------------------------------- surface ---
def make_master_surface(nmaster, H, kl, ks, L, target_gradient, seed, use_noise):
    """Periodic self-affine surface at N = nmaster, band-limited to [kl, ks]
    (integer waves-per-box), rescaled so its spectral rms gradient equals
    target_gradient (keeps the small-slope BEM assumption valid)."""
    rng = np.random.default_rng(seed)
    z = rf.selfaffine_field(
        dim=2, N=nmaster, Hurst=H,
        k_low=kl / nmaster, k_high=ks / nmaster,   # k_tilde/N = cycles/sample
        plateau=(kl > 1), noise=use_noise, rng=rng, verbose=False,
    ).astype(np.float64)
    z -= z.mean()
    g = rms_gradient_spectral(z, L)
    z *= target_gradient / g
    return z


# --------------------------------------------------------------------- solve ---
def solve_grid(N, surface_N, pprimes, rms_grad, L, E_star, q, tol, max_iter,
               seed_field, seed_N):
    """Load-step a single grid over the list of normalised pressures pprimes.

    Warm-starts each pressure step from the previous one, and cold-starts the
    lowest pressure from `seed_field` (a coarser-grid solution, prolonged).
    Returns a dict of arrays plus the lowest-pressure field (to seed the next
    finer grid)."""
    solver = hc.ContactSolver(grid_size=N, domain_size=L, E_star=E_star,
                              backend="h2", q=q)
    gap = (-surface_N).ravel()

    p_init = None
    if seed_field is not None:
        f = seed_field
        while seed_N < N:
            f = prolong2(f, seed_N)
            seed_N *= 2
        if seed_N == N:
            p_init = f

    ad_l, astar_l, na_l, m_l, it_l = [], [], [], [], []
    seed_out = None
    for j, pp in enumerate(pprimes):
        p0 = pp * E_star * rms_grad
        res = solver.solve(gap, p0, tol=tol, max_iter=max_iter,
                           precond="fourier", p_init=p_init)
        p_field = np.asarray(res.pressure)
        p_init = p_field.ravel()          # warm-start next (higher) pressure
        if j == 0:
            seed_out = p_field.ravel()     # seed for next finer grid

        ad, astar, na, m = corrected_area_fraction(contact_mask(p_field))
        ad_l.append(ad); astar_l.append(astar); na_l.append(na)
        m_l.append(m); it_l.append(int(res.iterations))

    return {
        "N": N, "pprime": np.asarray(pprimes),
        "area_raw": np.asarray(ad_l), "area_corr": np.asarray(astar_l),
        "Na": np.asarray(na_l), "M": np.asarray(m_l),
        "iters": np.asarray(it_l),
    }, seed_out


# ---------------------------------------------------------------------- plots ---
def plot_area(cases, pprimes, rms_grad, outpath, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grids = sorted(cases)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(grids)))
    pref = np.linspace(1e-4, pprimes.max() * 1.02, 200)
    asper = np.sqrt(2 * np.pi) * pref          # asperity asymptote
    persson = np.array([math.erf(math.sqrt(2) * p) for p in pref])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)
    for a, key, ttl in ((ax[0], "area_raw", "(a) raw  $A_d/A_0$"),
                        (ax[1], "area_corr", "(b) corrected  $A_*/A_0$  (Eq. 15)")):
        for c, N in zip(cmap, grids):
            d = cases[N]
            a.plot(d["pprime"], d[key], "-o", ms=3, color=c, lw=1.2,
                   label=f"N={N}")
        a.plot(pref, asper, ":", color="0.35", lw=1.3,
               label=r"asperity $\sqrt{2\pi}\,p'$")
        a.plot(pref, persson, "-.", color="0.15", lw=1.3,
               label=r"Persson $\mathrm{erf}(\sqrt{2}\, p')$")
        a.set_title(ttl); a.set_xlabel(r"$p'=p_0/E^*\sqrt{\langle|\nabla z|^2\rangle}$")
        a.grid(alpha=0.3)
    ax[0].set_ylabel(r"$A/A_0$")
    ax[0].legend(fontsize=7, ncol=2, loc="upper left")
    ax[0].set_ylim(0, min(0.6, max(cases[grids[-1]]["area_raw"].max() * 1.25, 0.3)))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_spread(cases, outpath, title):
    """Spread across grids at each pressure: raw vs corrected (max-min)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grids = sorted(cases)
    pp = cases[grids[0]]["pprime"]
    raw = np.array([cases[N]["area_raw"] for N in grids])
    cor = np.array([cases[N]["area_corr"] for N in grids])
    spread_raw = raw.max(0) - raw.min(0)
    spread_cor = cor.max(0) - cor.min(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pp, spread_raw, "-o", ms=3, label="raw  $A_d/A_0$")
    ax.plot(pp, spread_cor, "-s", ms=3, label="corrected  $A_*/A_0$")
    ax.set_xlabel(r"$p'$"); ax.set_ylabel("grid spread  (max$-$min over $N$)")
    ax.set_yscale("log"); ax.grid(alpha=0.3, which="both"); ax.legend()
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(outpath, dpi=200); plt.close(fig)


# ----------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--H", type=float, default=0.8)
    ap.add_argument("--kl", type=int, default=1, help="k_tilde_low (waves/box)")
    ap.add_argument("--ks", type=int, default=32, help="k_tilde_short (cut-off)")
    ap.add_argument("--nmaster", type=int, default=8192)
    ap.add_argument("--grids", type=int, nargs="+",
                    default=[128, 256, 512, 1024, 2048])
    ap.add_argument("--npress", type=int, default=12)
    ap.add_argument("--pmax", type=float, default=0.35, help="max p'")
    ap.add_argument("--gradient", type=float, default=0.1,
                    help="target spectral rms gradient (small-slope)")
    ap.add_argument("--L", type=float, default=1.0)
    ap.add_argument("--Estar", type=float, default=1.0)
    ap.add_argument("--q", type=int, default=6)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--max-iter", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--noise", action="store_true", default=True)
    ap.add_argument("--outdir", default="doc/convergence_study/repro")
    args = ap.parse_args()

    outdir = os.path.join(os.path.dirname(__file__), args.outdir)
    cache_dir = os.path.join(outdir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    ftag = f"H{args.H}_kl{args.kl}_ks{args.ks}"          # clean, for figures
    tag = ftag + f"_np{args.npress}_pm{args.pmax}_g{args.gradient}"  # cache key

    # normalised-pressure sampling (skip 0; linear from small to pmax)
    pprimes = np.linspace(args.pmax / args.npress, args.pmax, args.npress)

    print(f"[master] N={args.nmaster}, H={args.H}, k=[{args.kl},{args.ks}]")
    master = make_master_surface(args.nmaster, args.H, args.kl, args.ks,
                                 args.L, args.gradient, args.seed, args.noise)
    rms_grad = rms_gradient_spectral(master, args.L)
    print(f"[master] spectral rms gradient = {rms_grad:.5f}")

    grids = sorted(args.grids)
    for N in grids:
        if N > args.nmaster or args.nmaster % N != 0:
            raise SystemExit(f"grid {N} must divide nmaster {args.nmaster}")

    # solve coarse->fine, seeding each finer cold-start from the coarser result
    cases = {}
    seed_field, seed_N = None, None
    for N in grids:
        cpath = os.path.join(cache_dir, f"{tag}_N{N}.pkl")
        if os.path.exists(cpath):
            with open(cpath, "rb") as fh:
                cases[N] = pickle.load(fh)
            print(f"[N={N:5d}] cached  "
                  f"(area_raw[-1]={cases[N]['area_raw'][-1]:.4f}, "
                  f"area_corr[-1]={cases[N]['area_corr'][-1]:.4f})")
            seed_field, seed_N = cases[N]["Na"] * 0.0, N  # not reused across runs
            continue
        s = args.nmaster // N
        surf_N = master[::s, ::s].copy()
        t0 = time.time()
        res, seed_out = solve_grid(N, surf_N, pprimes, rms_grad, args.L,
                                   args.Estar, args.q, args.tol, args.max_iter,
                                   seed_field, seed_N)
        dt = time.time() - t0
        res["seconds"] = dt
        cases[N] = res
        with open(cpath, "wb") as fh:
            pickle.dump(res, fh)
        seed_field, seed_N = seed_out, N
        print(f"[N={N:5d}] {dt:7.1f}s  iters(sum)={int(res['iters'].sum()):5d}  "
              f"area_raw[-1]={res['area_raw'][-1]:.4f}  "
              f"area_corr[-1]={res['area_corr'][-1]:.4f}")

    # figures
    have = sorted(cases)
    title = (f"H={args.H}, $\\tilde k_l$={args.kl}, $\\tilde k_s$={args.ks}"
             f"  (master N={args.nmaster}, non-periodic Boussinesq BEM)")
    plot_area(cases, pprimes, rms_grad,
              os.path.join(outdir, f"area_{ftag}.png"), title)
    plot_spread(cases, os.path.join(outdir, f"spread_{ftag}.png"), title)
    print(f"[figures] wrote area_{ftag}.png and spread_{ftag}.png in {outdir}")

    # console summary of the collapse
    raw = np.array([cases[N]["area_raw"] for N in have])
    cor = np.array([cases[N]["area_corr"] for N in have])
    sr = (raw.max(0) - raw.min(0)).mean()
    sc = (cor.max(0) - cor.min(0)).mean()
    print(f"[collapse] mean grid-spread  raw={sr:.4e}  corrected={sc:.4e}  "
          f"(x{sr / max(sc, 1e-12):.1f} tighter)")


if __name__ == "__main__":
    main()
