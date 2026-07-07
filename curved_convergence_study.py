"""Contact-area mesh-convergence study with a GLOBAL CURVATURE (rough Hertz).

Unlike `convergence_study.py` (flat mean plane, periodic-style roughness), here a
paraboloid indenter of radius R is pressed onto a rough half-space, producing a
*localized* Hertzian contact patch of radius ~a that must stay well inside the
1x1 window (the H2/FMM Boussinesq BEM is non-periodic, so the patch may not touch
the boundary). The area correction of Yastrebov, Anciaux & Molinari (Trib. Int.
114, 2017) is then applied exactly as in the flat case:

    A_*/A_0 = N_a/N^2 - (pi - 1 + ln2)/24 * M/N^2 .

This first stage is an EXPLORATION (`--mode explore`, default): on a coarse
512x512 grid it sweeps several loads bracketing the target radius `a` and writes
one contact-map PNG per load (target circle + measured a_eff overlaid), plus a
summary `a_eff(F)` vs the smooth-Hertz reference. Pick a load, then the
convergence stage (added later) subsamples a fine master at that load.

Hertz (sphere on elastic half-space, radius R, modulus E*):
    F = 4 E* a^3 / (3 R),   a = (3 F R / (4 E*))^{1/3},   p0 = 3F/(2 pi a^2).
With L=1 the solver's mean-pressure constraint is p_bar = F / L^2 = F.

Run (fenicsx-env):
    OMP_NUM_THREADS=20 python curved_convergence_study.py            # explore
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

BETA_PI4 = (np.pi - 1.0 + np.log(2.0)) / 24.0  # area-correction coefficient


# ------------------------------------------------------------------- Hertz -----
def hertz_force(a, R, Estar):
    return 4.0 * Estar * a ** 3 / (3.0 * R)


def hertz_radius(F, R, Estar):
    return (3.0 * F * R / (4.0 * Estar)) ** (1.0 / 3.0)


# ---------------------------------------------------------------- geometry -----
def build_geometry(N, L, R, rms_rough, H, kl, ks, seed):
    """Return the initial gap field (N x N, >=0) for a paraboloid of radius R
    centred in the [−L/2, L/2]^2 window plus a self-affine roughness of rms
    height `rms_rough`. Also returns the (zero-mean, scaled) roughness so a
    finer grid can reuse the *same* surface later."""
    x = (np.arange(N) + 0.5) / N * L - 0.5 * L          # cell centres, centred
    X, Y = np.meshgrid(x, x, indexing="ij")
    curv = (X ** 2 + Y ** 2) / (2.0 * R)                # paraboloid gap to a flat

    rough = np.zeros((N, N))
    if rms_rough > 0.0:
        rng = np.random.default_rng(seed)
        rough = rf.selfaffine_field(dim=2, N=N, Hurst=H,
                                    k_low=kl / N, k_high=ks / N,
                                    plateau=(kl > 1), noise=True, rng=rng,
                                    verbose=False).astype(np.float64)
        rough -= rough.mean()
        rough *= rms_rough / rough.std()

    gap = curv - rough
    gap -= gap.min()                                    # shift so min gap = 0
    return gap, rough


# ------------------------------------------------------------- measurement -----
def count_switches(mask):
    mh = np.count_nonzero(mask[:, :-1] != mask[:, 1:])
    mv = np.count_nonzero(mask[:-1, :] != mask[1:, :])
    return int(mh + mv)


def measure(pressure, L):
    """Contact metrics from a pressure field on an N x N grid over [-L/2,L/2]^2.

    a_eff  = radius of a disc with the same *real* contact area sqrt(A/pi).
    a_foot = *footprint* radius: distance from the centre (0,0) containing 90%
             of the contacting nodes — the extent of the contact cloud."""
    N = pressure.shape[0]
    mask = np.asarray(pressure) > 0.0
    na = int(np.count_nonzero(mask))
    m = count_switches(mask)
    A0 = L * L
    ad = na / (N * N)                       # A_d / A_0
    astar = ad - BETA_PI4 * m / (N * N)     # A_* / A_0
    a_eff = math.sqrt(max(ad, 0.0) * A0 / math.pi)      # equivalent disc radius

    a_foot = 0.0
    if na:
        x = (np.arange(N) + 0.5) / N * L - 0.5 * L
        X, Y = np.meshgrid(x, x, indexing="ij")
        r = np.hypot(X[mask], Y[mask])
        a_foot = float(np.percentile(r, 90.0))          # 90% of spots within
    # boundary contact: any contacting node within a 3-cell frame of the edge
    b = 3
    touches = bool(mask[:b, :].any() or mask[-b:, :].any()
                   or mask[:, :b].any() or mask[:, -b:].any())
    return dict(mask=mask, Na=na, M=m, area_raw=ad, area_corr=astar,
                a_eff=a_eff, a_foot=a_foot, boundary=touches)


# --------------------------------------------------------------- plotting ------
def plot_map(pressure, met, F, a_target, L, R, Estar, outpath, idx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = pressure.shape[0]
    ext = [-0.5 * L, 0.5 * L, -0.5 * L, 0.5 * L]
    aH = hertz_radius(F, R, Estar)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))

    ax[0].imshow(met["mask"].T, origin="lower", extent=ext, cmap="Greys",
                 vmin=0, vmax=1)
    for a_val, col, lab in ((a_target, "tab:blue", f"target a={a_target:g}"),
                            (met["a_foot"], "tab:orange",
                             f"footprint a90={met['a_foot']:.3f}"),
                            (met["a_eff"], "tab:red", f"a_eff={met['a_eff']:.3f}"),
                            (aH, "tab:green", f"Hertz a={aH:.3f}")):
        ax[0].add_patch(plt.Circle((0, 0), a_val, fill=False, lw=1.4,
                                   color=col, label=lab))
    ax[0].legend(fontsize=7, loc="upper right")
    ax[0].set_title(f"contact mask  (A/A0={met['area_raw']*100:.2f}%"
                    f"{'  EDGE!' if met['boundary'] else ''})")
    ax[0].set_xlim(ext[:2]); ax[0].set_ylim(ext[2:])

    p = np.asarray(pressure).T
    z = 3.0 * max(a_target, met["a_eff"])               # zoom window on centre
    im = ax[1].imshow(p, origin="lower", extent=ext, cmap="inferno")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    ax[1].set_xlim(-z, z); ax[1].set_ylim(-z, z)
    ax[1].set_title("pressure (zoom)")

    fig.suptitle(f"[{idx}] F={F:.3e}  p_bar={F/(L*L):.3e}  "
                 f"a_eff={met['a_eff']:.4f}  N_a={met['Na']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_summary(records, a_target, R, Estar, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F = np.array([r["F"] for r in records])
    a_eff = np.array([r["a_eff"] for r in records])
    a_foot = np.array([r["a_foot"] for r in records])
    Fs = np.geomspace(F.min() * 0.8, F.max() * 1.2, 100)
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.plot(Fs, [hertz_radius(f, R, Estar) for f in Fs], "-", color="0.5",
            label="smooth Hertz  $a=(3FR/4E^*)^{1/3}$")
    ax.plot(F, a_foot, "s-", color="tab:orange", label="footprint  $a_{90}$")
    ax.plot(F, a_eff, "o-", color="tab:red", label="real  $a_{\\mathrm{eff}}$")
    ax.axhline(a_target, ls="--", color="tab:blue", label=f"target a={a_target:g}")
    for r in records:
        ax.annotate(str(r["idx"]), (r["F"], r["a_eff"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("normal force $F$")
    ax.set_ylabel("contact radius $a$"); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ---------------------------------------------------- convergence (curved) -----
def save_contact_svg(mask, met, F, a_target, L, outpath, N, step):
    """Vector (SVG) contact-area map showing the raw contact pixels: the binary
    mask is embedded with interpolation='none', so it is stored at *native*
    resolution with no resampling — one image pixel = one grid cell."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = [-0.5 * L, 0.5 * L, -0.5 * L, 0.5 * L]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mask.T, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1,
              interpolation="none")  # native pixels, no smoothing/resampling
    ax.set_aspect("equal"); ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:])
    ax.set_title(f"N={N}  step {step}  F={F:.3e}\n"
                 f"$A_d/A_0$={met['area_raw']*100:.2f}%   "
                 f"$A_*/A_0$={met['area_corr']*100:.2f}%   $N_a$={met['Na']}",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, format="svg")
    plt.close(fig)


def plot_converge(cases, loads, tag, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grids = sorted(cases)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(grids)))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)
    for a, key, ttl in ((ax[0], "area_raw", "(a) raw  $A_d/A_0$"),
                        (ax[1], "area_corr", "(b) corrected  $A_*/A_0$")):
        for c, N in zip(cmap, grids):
            a.plot(loads, np.asarray(cases[N][key]) * 100, "-o", ms=3, color=c,
                   lw=1.3, label=f"N={N}")
        a.set_title(ttl); a.set_xlabel("normal force $F$ (= $p_0$, since $L=1$)")
        a.grid(alpha=0.3)
    ax[0].set_ylabel(r"$A/A_0$  [%]"); ax[0].legend(fontsize=8)
    fig.suptitle("Rough Hertz contact " + tag.replace("_", "  "), fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, format="svg")                 # vector deliverable
    fig.savefig(os.path.splitext(outpath)[0] + ".png", dpi=150)  # for embedding
    plt.close(fig)


def converge(args):
    """Mesh-convergence of the corrected contact area for the curved (rough
    Hertz) geometry: one roughness master at N=`master`, subsampled to each grid;
    fixed load sweep; raw vs corrected area per grid; an SVG per (grid, step)."""
    outdir = os.path.join(os.path.dirname(__file__), args.outdir)
    svgdir = os.path.join(outdir, "svg")
    cachedir = os.path.join(outdir, "cache")
    os.makedirs(svgdir, exist_ok=True)
    os.makedirs(cachedir, exist_ok=True)

    m = args.master
    print(f"[master] roughness N={m}, H={args.H}, k=[{args.kl},{args.ks}], "
          f"rms={args.rms_rough:.1e}  (subsampled to {args.grids})")
    rng = np.random.default_rng(args.seed)
    rough_m = rf.selfaffine_field(dim=2, N=m, Hurst=args.H,
                                  k_low=args.kl / m, k_high=args.ks / m,
                                  plateau=(args.kl > 1), noise=True, rng=rng,
                                  verbose=False).astype(np.float64)
    rough_m -= rough_m.mean()
    rough_m *= args.rms_rough / rough_m.std()

    loads = np.linspace(args.fmax / args.nloads, args.fmax, args.nloads)
    tag = (f"R{args.R:g}_rms{args.rms_rough:g}_kl{args.kl}_ks{args.ks}"
           f"_fmax{args.fmax:g}_nl{args.nloads}")
    print(f"[loads] {args.nloads} steps: {loads[0]:.3e} .. {loads[-1]:.3e}")

    grids = sorted(args.grids)
    for N in grids:
        if m % N != 0:
            raise SystemExit(f"grid {N} must divide master {m}")

    cases = {}
    for N in grids:
        cpath = os.path.join(cachedir, f"{tag}_N{N}.pkl")
        if os.path.exists(cpath):
            with open(cpath, "rb") as fh:
                cases[N] = pickle.load(fh)
            print(f"[N={N:5d}] cached  "
                  f"(A_d[-1]={cases[N]['area_raw'][-1]*100:.2f}%, "
                  f"A_*[-1]={cases[N]['area_corr'][-1]*100:.2f}%)")
            continue

        s = m // N
        rough_N = rough_m[::s, ::s]
        x = (np.arange(N) + 0.5) / N * args.L - 0.5 * args.L
        X, Y = np.meshgrid(x, x, indexing="ij")
        gap = (X ** 2 + Y ** 2) / (2.0 * args.R) - rough_N
        gap -= gap.min()
        gap_flat = gap.ravel()

        solver = hc.ContactSolver(grid_size=N, domain_size=args.L,
                                  E_star=args.Estar, backend="h2", q=args.q)
        rec = {k: [] for k in ("area_raw", "area_corr", "a_foot", "a_eff",
                               "Na", "M", "iters")}
        p_init = None
        t0 = time.time()
        for j, F in enumerate(loads):
            res = solver.solve(gap_flat, F, tol=args.tol, max_iter=args.max_iter,
                               precond="fourier", p_init=p_init)
            p = np.asarray(res.pressure)
            p_init = p.ravel()
            met = measure(p, args.L)
            svg = os.path.join(svgdir, f"contact_N{N}_step{j}_F{F:.3e}.svg")
            save_contact_svg(met["mask"], met, F, args.a_target, args.L, svg, N, j)
            for k in ("area_raw", "area_corr", "a_foot", "a_eff", "Na", "M"):
                rec[k].append(met[k])
            rec["iters"].append(int(res.iterations))
            print(f"  N={N:5d} step {j:2d} F={F:.3e} it={res.iterations:4d} "
                  f"A_d={met['area_raw']*100:6.3f}%  A_*={met['area_corr']*100:6.3f}%")
        rec = {k: np.asarray(v) for k, v in rec.items()}
        rec["seconds"] = time.time() - t0
        cases[N] = rec
        with open(cpath, "wb") as fh:
            pickle.dump(rec, fh)
        print(f"[N={N:5d}] {rec['seconds']:.1f}s")

    plot_converge(cases, loads, tag,
                  os.path.join(outdir, f"converge_area_{tag}.svg"))
    # convergence summary: spread across grids at each load
    raw = np.array([cases[N]["area_raw"] for N in grids])
    cor = np.array([cases[N]["area_corr"] for N in grids])
    sr = (raw.max(0) - raw.min(0)).mean()
    sc = (cor.max(0) - cor.min(0)).mean()
    print(f"\n[done] {len(grids)} grids x {args.nloads} loads; "
          f"{len(grids)*args.nloads} SVGs in {svgdir}")
    print(f"[collapse] mean grid-spread  raw={sr*100:.3f}%  "
          f"corrected={sc*100:.3f}%  (x{sr/max(sc,1e-12):.1f} tighter)")


# ------------------------------------------------------------------- main ------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["explore", "converge"], default="explore")
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--grids", type=int, nargs="+",
                    default=[512, 1024, 2048, 4096], help="converge: grid sizes")
    ap.add_argument("--master", type=int, default=8192,
                    help="converge: fine grid the roughness is subsampled from")
    ap.add_argument("--fmax", type=float, default=4.439e-3,
                    help="converge: max load (10-step sweep ends here)")
    ap.add_argument("--L", type=float, default=1.0)
    ap.add_argument("--R", type=float, default=10.0, help="indenter radius")
    ap.add_argument("--Estar", type=float, default=1.0)
    ap.add_argument("--a-target", type=float, default=0.1)
    ap.add_argument("--rms-rough", type=float, default=2e-4,
                    help="rms roughness height (~0.2x Hertz penetration)")
    ap.add_argument("--H", type=float, default=0.8)
    ap.add_argument("--kl", type=int, default=4)
    ap.add_argument("--ks", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--nloads", type=int, default=7)
    ap.add_argument("--frange", type=float, nargs=2, default=[0.3, 2.5],
                    help="load sweep as multiples of the target Hertz force")
    ap.add_argument("--q", type=int, default=6)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--max-iter", type=int, default=20000)
    ap.add_argument("--outdir", default="doc/convergence_study/curved")
    args = ap.parse_args()

    if args.mode == "converge":
        return converge(args)

    outdir = os.path.join(os.path.dirname(__file__), args.outdir)
    os.makedirs(outdir, exist_ok=True)

    F_target = hertz_force(args.a_target, args.R, args.Estar)
    delta = args.a_target ** 2 / args.R
    print(f"[Hertz] a_target={args.a_target}  R={args.R}  ->  F_target="
          f"{F_target:.4e}, penetration delta={delta:.4e}, "
          f"rms_rough={args.rms_rough:.2e} ({args.rms_rough/delta:.2f} delta)")

    gap, _ = build_geometry(args.N, args.L, args.R, args.rms_rough,
                            args.H, args.kl, args.ks, args.seed)
    gap_flat = gap.ravel()
    print(f"[geom] N={args.N}, gap range [0, {gap.max():.4e}], "
          f"curvature gap at a_target r: {args.a_target**2/(2*args.R):.4e}")

    solver = hc.ContactSolver(grid_size=args.N, domain_size=args.L,
                              E_star=args.Estar, backend="h2", q=args.q)

    loads = np.geomspace(args.frange[0], args.frange[1], args.nloads) * F_target
    records, p_init = [], None
    print(f"\n idx      F        p_bar     iters conv   A/A0%  a_foot  a_eff   edge?")
    for idx, F in enumerate(loads):
        p_bar = F / (args.L * args.L)
        res = solver.solve(gap_flat, p_bar, tol=args.tol, max_iter=args.max_iter,
                           precond="fourier", p_init=p_init)
        p_field = np.asarray(res.pressure)
        p_init = p_field.ravel()
        met = measure(p_field, args.L)
        png = os.path.join(outdir, f"load_{idx}_F{F:.3e}.png")
        plot_map(p_field, met, F, args.a_target, args.L, args.R, args.Estar,
                 png, idx)
        records.append(dict(idx=idx, F=float(F), **{k: met[k] for k in
                       ("a_eff", "a_foot", "area_raw", "area_corr", "Na", "M",
                        "boundary")}))
        print(f" {idx:3d}  {F:.3e}  {p_bar:.3e}  {res.iterations:5d} "
              f"{str(res.converged):5s} {met['area_raw']*100:6.2f} "
              f"{met['a_foot']:.4f}  {met['a_eff']:.4f}  "
              f"{'YES' if met['boundary'] else 'no'}")

    plot_summary(records, args.a_target, args.R, args.Estar,
                 os.path.join(outdir, "summary_a_vs_F.png"))
    print(f"\n[done] {len(records)} contact maps + summary_a_vs_F.png in {outdir}")
    # which loads land near the target FOOTPRINT with an interior patch?
    ok = [r for r in records if not r["boundary"]]
    if ok:
        best = min(ok, key=lambda r: abs(r["a_foot"] - args.a_target))
        print(f"[hint] closest interior load to footprint a={args.a_target}: "
              f"idx={best['idx']}  F={best['F']:.3e}  a_foot={best['a_foot']:.4f}"
              f"  a_eff={best['a_eff']:.4f}")


if __name__ == "__main__":
    main()
