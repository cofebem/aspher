#!/usr/bin/env python3
"""Generate all PDF figures for the Beamer slides.

Run from doc/slides/:
    conda activate fenicsx-env
    python generate_figures.py
"""
import sys, json, time
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

HERE    = Path(__file__).resolve().parent
ROOT    = HERE.parent.parent          # repository root
sys.path.insert(0, str(ROOT / 'python'))
import aspher as hmc

FIGDIR   = HERE / 'figures'
CACHEDIR = HERE / 'cache'
DATADIR  = ROOT / 'data'
for d in (FIGDIR, CACHEDIR): d.mkdir(exist_ok=True)

# ── style ────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'text.usetex'        : True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'font.family'        : 'serif',
    'font.size'          : 9,
    'axes.linewidth'     : 0.8,
    'axes.spines.top'    : False,
    'axes.spines.right'  : False,
    'xtick.major.size'   : 3,
    'ytick.major.size'   : 3,
    'xtick.direction'    : 'in',
    'ytick.direction'    : 'in',
    'lines.linewidth'    : 1.4,
    'legend.frameon'     : False,
    'legend.fontsize'    : 8,
    'savefig.bbox'       : 'tight',
    'savefig.pad_inches' : 0.05,
})

C = dict(blue='#2c7bb6', red='#d73027', green='#1a9850',
         orange='#f46d43', gray='#aaaaaa', dblue='#08519c',
         lblue='#9ecae1', ly='#ffffb2')


def savefig(name, fig=None):
    (fig or plt).savefig(str(FIGDIR / name))
    plt.close('all')
    print(f'  {name}')


# ── data helpers ─────────────────────────────────────────────────────────────
def load_or_run_hertz():
    cache = CACHEDIR / 'hertz.npz'
    if cache.exists():
        d = np.load(str(cache))
        return d['gap'], d['pressure'], d['gap_field']
    Ns, L, E_star, R, p_bar = 64, 1.0, 1.0, 0.5, 0.0123
    h = L / Ns
    x = (np.arange(Ns) + 0.5) * h - 0.5 * L
    X, Y = np.meshgrid(x, x)
    gap = (X**2 + Y**2) / (2 * R)
    solver = hmc.ContactSolver(grid_size=Ns, domain_size=L, E_star=E_star)
    res = solver.solve(gap=gap, p_nominal=p_bar, tol=1e-10, max_iter=4000)
    np.savez(str(cache), gap=gap, pressure=res.pressure, gap_field=res.gap)
    return gap, res.pressure, res.gap


def load_or_run_rough():
    cache = CACHEDIR / 'rough_pressure.npy'
    surface = np.load(str(DATADIR / 'surface.npy'))
    if cache.exists():
        return surface, np.load(str(cache))
    n = surface.shape[0]
    solver = hmc.ContactSolver(grid_size=n)
    res = solver.solve(gap=-surface, p_nominal=0.05, tol=1e-8, max_iter=5000)
    np.save(str(cache), res.pressure)
    return surface, res.pressure


def timing_sweep():
    cache = CACHEDIR / 'timing.json'
    if cache.exists():
        with open(cache) as f:
            return json.load(f)
    ns_vals = [16, 24, 32, 48, 64, 96, 128, 192, 256]
    out = dict(ns=ns_vals, N=[], assembly_s=[], matvec_s=[], hmat_bytes=[])
    for Ns in ns_vals:
        print(f'    sweep Ns={Ns}...', end='', flush=True)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            solver = hmc.ContactSolver(grid_size=Ns)
            times.append(time.perf_counter() - t0)
        t_asm = min(times)

        p = np.ones(Ns * Ns)
        solver.matvec(p)  # warmup
        t0 = time.perf_counter()
        for _ in range(20): solver.matvec(p)
        t_mv = (time.perf_counter() - t0) / 20

        info = solver.hmatrix_info()
        out['N'].append(Ns * Ns)
        out['assembly_s'].append(t_asm)
        out['matvec_s'].append(t_mv)
        out['hmat_bytes'].append(info['bytes'])
        print(f' asm={t_asm:.3f}s mv={t_mv*1e3:.2f}ms')
    with open(cache, 'w') as f:
        json.dump(out, f, indent=2)
    return out


# ── block partition (pure Python, mirrors C++) ────────────────────────────────
def hmatrix_blocks_py(Ns, leaf_size=8, eta=2.0):
    N = Ns * Ns
    perm = list(range(N))
    nodes = []

    def build(begin, end, box):
        nid = len(nodes)
        nodes.append({'b': begin, 'e': end, 'box': box, 'ch': []})
        if end - begin <= leaf_size:
            return nid
        ix0, ix1, iy0, iy1 = box
        mx, my = (ix0 + ix1) // 2, (iy0 + iy1) // 2
        qb = [(ix0,mx,iy0,my),(mx,ix1,iy0,my),(ix0,mx,my,iy1),(mx,ix1,my,iy1)]
        bkts = [[], [], [], []]
        for p in perm[begin:end]:
            ix, iy = p % Ns, p // Ns
            bkts[(1 if ix >= mx else 0) + (2 if iy >= my else 0)].append(p)
        b = begin
        for q in range(4):
            if bkts[q]:
                e = b + len(bkts[q]); perm[b:e] = bkts[q]
                nodes[nid]['ch'].append(build(b, e, qb[q])); b = e
        return nid

    build(0, N, (0, Ns, 0, Ns))

    def diam(box): return max(box[1]-box[0], box[3]-box[2])
    def dist_(a, b):
        return max(max(0, b[0]-a[1], a[0]-b[1]),
                   max(0, b[2]-a[3], a[2]-b[3]))

    blocks = []
    def collect(t, s):
        nt, ns_ = nodes[t], nodes[s]
        d  = dist_(nt['box'], ns_['box'])
        dm = min(diam(nt['box']), diam(ns_['box']))
        if d > 0 and dm <= eta * d:
            blocks.append((nt['b'], nt['e']-nt['b'], ns_['b'], ns_['e']-ns_['b'], False))
        elif not nt['ch'] and not ns_['ch']:
            blocks.append((nt['b'], nt['e']-nt['b'], ns_['b'], ns_['e']-ns_['b'], True))
        elif not nt['ch']:
            for c in ns_['ch']: collect(t, c)
        elif not ns_['ch']:
            for c in nt['ch']: collect(c, s)
        else:
            for ct in nt['ch']:
                for cs in ns_['ch']: collect(ct, cs)
    collect(0, 0)
    return blocks, N


# ════════════════════════════════════════════════════════════════════════════
# Figure functions
# ════════════════════════════════════════════════════════════════════════════

def fig_kernel_decay():
    Ns, L, E_star = 64, 1.0, 1.0
    h = L / Ns
    K = hmc.ContactSolver(grid_size=Ns, E_star=E_star, use_hmatrix=False)
    # S(dx,0) for dx = 0..40
    dx = np.arange(0, 42)
    s_love = np.array([K.matvec(
        (np.arange(Ns*Ns) == int(Ns//2)*Ns + int(Ns//2) + d).astype(float)
    )[int(Ns//2)*Ns + int(Ns//2)] for d in dx])
    # normalise: pi E* S / h
    s_norm = s_love * (np.pi * E_star) / h

    r_cont = np.linspace(0.01, 41, 400) * h
    s_ps = h / r_cont          # point source pi E* S / h = h / (r/h * h) = 1/(r/h)
    r_nd = np.linspace(0.5, 41, 400)

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.plot(dx + 0.5, s_norm, 'o', ms=3.5, color=C['blue'], label=r'Love formula $\mathcal{L}$')
    ax.plot(r_nd, 1.0 / r_nd, '--', color=C['red'], label=r'$h^2/(\pi E^* r)$ asymptote')
    ax.axvline(5, color=C['gray'], lw=0.8, ls=':')
    ax.annotate(r'$r = 5h$', xy=(5, 0.05), xytext=(7, 0.09),
                arrowprops=dict(arrowstyle='->', color=C['gray'], lw=0.8),
                fontsize=8, color=C['gray'])
    ax.set_xlabel(r'$r/h$')
    ax.set_ylabel(r'$\pi E^* S_{ij} / h$')
    ax.set_xlim(0, 42); ax.set_ylim(0, 0.5)
    ax.legend(loc='upper right')
    savefig('fig_kernel_decay.pdf')


def fig_hmatrix_blocks():
    blocks, N = hmatrix_blocks_py(Ns=32, leaf_size=8, eta=2.0)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    for rb, rs, cb, cs, dense in blocks:
        color = C['gray'] if dense else C['lblue']
        ec    = 'white'
        rect = mpatches.Rectangle((cb, N-rb-rs), cs, rs,
                                   lw=0.15, edgecolor=ec, facecolor=color)
        ax.add_patch(rect)
    ax.set_xlim(0, N); ax.set_ylim(0, N)
    ax.set_aspect('equal')
    ax.set_xlabel('column index (permuted)')
    ax.set_ylabel('row index (permuted)')
    ax.set_title(r'H-matrix block structure, $N_s=32$', fontsize=9)
    patches = [mpatches.Patch(facecolor=C['gray'], label='dense'),
               mpatches.Patch(facecolor=C['lblue'], label='low-rank (ACA)')]
    ax.legend(handles=patches, loc='lower right', fontsize=8)
    savefig('fig_hmatrix_blocks.pdf')


def fig_hertz_geometry():
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.set_xlim(-2.8, 2.8); ax.set_ylim(-0.7, 3.4)
    ax.set_aspect('equal'); ax.axis('off')

    # Half-space
    ax.fill_between([-2.8, 2.8], -0.7, 0, color='#e8e8e8', zorder=1)
    ax.axhline(0, color='#555555', lw=2, zorder=2)
    ax.text(2.5, -0.45, 'Elastic\nhalf-space', ha='right', va='center',
            fontsize=8, color='#555555')

    # Sphere
    theta = np.linspace(0, 2*np.pi, 300)
    R_sph = 1.6
    cx, cy = 0, R_sph
    ax.fill(cx + R_sph*np.cos(theta), cy + R_sph*np.sin(theta),
            color='#d4e8f8', zorder=3)
    ax.plot(cx + R_sph*np.cos(theta), cy + R_sph*np.sin(theta),
            color='#2c7bb6', lw=1.5, zorder=4)
    ax.text(2.1, 1.6, 'Rigid\nsphere', ha='left', va='center',
            fontsize=8, color='#2c7bb6')

    # Radius line
    ax.annotate('', xy=(0, 0), xytext=(0, R_sph),
                arrowprops=dict(arrowstyle='<->', color='#2c7bb6',
                                lw=1.2, shrinkA=0, shrinkB=0))
    ax.text(0.12, R_sph*0.55, r'$R$', fontsize=10, color='#2c7bb6')

    # Contact zone (red)
    a_val = 0.55
    ax.plot([-a_val, a_val], [0, 0], color=C['red'], lw=4, zorder=5, solid_capstyle='round')
    ax.annotate('', xy=(-a_val, -0.35), xytext=(a_val, -0.35),
                arrowprops=dict(arrowstyle='<->', color=C['red'], lw=1.2,
                                shrinkA=0, shrinkB=0))
    ax.text(0, -0.55, r'$2a$', ha='center', fontsize=10, color=C['red'])

    # Force arrow
    ax.annotate('', xy=(0, R_sph*2+0.1), xytext=(0, R_sph*2+1.0),
                arrowprops=dict(arrowstyle='->', color='#333333',
                                lw=2, mutation_scale=14))
    ax.text(0.15, R_sph*2+0.6, r'$F$', fontsize=11)

    # Pressure profile sketch (red arch under sphere)
    xp = np.linspace(-a_val, a_val, 100)
    yp = -0.28 * np.sqrt(np.maximum(0, 1 - (xp/a_val)**2))
    ax.fill_between(xp, 0, yp, color=C['red'], alpha=0.25, zorder=6)
    ax.plot(xp, yp, color=C['red'], lw=1, zorder=7)
    ax.text(a_val+0.1, -0.18, r'$p(r)$', fontsize=9, color=C['red'])

    savefig('fig_hertz_geometry.pdf')


def fig_hertz_validation():
    gap, pressure, gap_field = load_or_run_hertz()
    Ns, L, E_star, R, p_bar = 64, 1.0, 1.0, 0.5, 0.0123
    h = L / Ns
    x = (np.arange(Ns) + 0.5) * h - 0.5 * L
    X, Y = np.meshgrid(x, x)
    r_flat = np.sqrt(X**2 + Y**2).flatten()
    p_flat = pressure.flatten()

    F = p_bar * L * L
    a = (3*F*R / (4*E_star))**(1/3)
    p0 = 3*F / (2*np.pi*a**2)
    r_fine = np.linspace(0, 1.5*a, 400)
    p_hertz = p0 * np.sqrt(np.maximum(0, 1 - (r_fine/a)**2))

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.6), gridspec_kw={'wspace': 0.35})

    # Left: radial profile
    ax = axes[0]
    mask = r_flat < 1.5 * a
    ax.scatter(r_flat[mask]/a, p_flat[mask]/p0, s=4, color=C['blue'],
               alpha=0.6, zorder=2, label='H-matrix BEM')
    ax.plot(r_fine/a, p_hertz/p0, '-', color=C['red'], lw=1.8,
            label='Hertz analytic', zorder=3)
    ax.set_xlabel(r'$r / a$')
    ax.set_ylabel(r'$p / p_0$')
    ax.set_xlim(0, 1.4); ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=7.5)
    ax.text(0.6, 0.85, fr'$a_\mathrm{{num}}/a = {np.sqrt((p_flat>1e-10).mean()*L*L/np.pi)/a:.4f}$',
            transform=ax.transAxes, fontsize=8)

    # Right: 2D pressure map
    ax2 = axes[1]
    im = ax2.imshow(pressure, origin='lower', cmap='YlOrRd',
                    extent=[0, L, 0, L], interpolation='bilinear')
    theta = np.linspace(0, 2*np.pi, 200)
    ax2.plot(0.5+a*np.cos(theta), 0.5+a*np.sin(theta),
             '--', color='white', lw=1.2, label=r'Hertz $r=a$')
    cb = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cb.set_label(r'pressure $p$', fontsize=8)
    ax2.set_xlabel(r'$x$'); ax2.set_ylabel(r'$y$')
    ax2.set_title(r'Pressure field', fontsize=9)
    ax2.legend(fontsize=7.5)
    savefig('fig_hertz_validation.pdf')


def fig_surface():
    surface = np.load(str(DATADIR / 'surface.npy'))
    fig, ax = plt.subplots(figsize=(2.8, 2.4))
    im = ax.imshow(surface, origin='lower', cmap='viridis',
                   interpolation='bilinear')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r'height $h$ (a.u.)', fontsize=8)
    ax.set_xlabel(r'$i_x$'); ax.set_ylabel(r'$i_y$')
    ax.set_title(r'Self-affine surface ($H = 0.8$)', fontsize=9)
    savefig('fig_surface.pdf')


def fig_rough_result():
    surface, pressure = load_or_run_rough()
    gap_field = np.load(str(CACHEDIR / 'rough_pressure.npy'))  # reuse pressure
    # recompute gap: approximate as pressure==0 → not in contact
    contact = pressure > 1e-10 * pressure.max()

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5),
                             gridspec_kw={'wspace': 0.38})

    im0 = axes[0].imshow(surface, origin='lower', cmap='viridis',
                          interpolation='bilinear')
    cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.set_label(r'height', fontsize=7)
    axes[0].set_title(r'Surface $g_0$', fontsize=9)

    im1 = axes[1].imshow(pressure, origin='lower', cmap='YlOrRd',
                          interpolation='bilinear')
    cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.set_label(r'pressure $p$', fontsize=7)
    axes[1].set_title(r'Pressure $p$ (H-matrix BEM)', fontsize=9)
    n = surface.shape[0]
    axes[1].text(0.02, 0.96, fr'$A_c/A = {contact.mean():.3f}$',
                 transform=axes[1].transAxes, fontsize=7.5,
                 va='top', color='white', fontweight='bold')

    cmap_c = mpl.colors.ListedColormap(['#f0f0f0', C['blue']])
    im2 = axes[2].imshow(contact.astype(float), origin='lower',
                          cmap=cmap_c, vmin=0, vmax=1, interpolation='nearest')
    axes[2].set_title(r'Contact zone', fontsize=9)
    patches = [mpatches.Patch(facecolor='#f0f0f0', label='non-contact', ec='gray', lw=0.5),
               mpatches.Patch(facecolor=C['blue'], label='contact')]
    axes[2].legend(handles=patches, loc='lower right', fontsize=6.5)

    for ax in axes:
        ax.set_xlabel(r'$i_x$', fontsize=8); ax.set_ylabel(r'$i_y$', fontsize=8)
        ax.tick_params(labelsize=7)
    savefig('fig_rough_result.pdf')


def fig_memory_scaling():
    td = timing_sweep()
    N  = np.array(td['N'],          dtype=float)
    hm = np.array(td['hmat_bytes'], dtype=float) / 1024**2  # MiB
    dm = 8 * N**2 / 1024**2

    # fit H-matrix: log(hm) ~ alpha * log(N) + beta
    coeffs = np.polyfit(np.log(N), np.log(hm), 1)
    N_fit  = np.geomspace(N[0], N[-1]*1.5, 200)
    hm_fit = np.exp(np.polyval(coeffs, np.log(N_fit)))

    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ax.loglog(N, dm, '--', color=C['gray'], lw=1.4, label=r'Dense $8N^2$')
    ax.loglog(N, hm, 'o', color=C['blue'], ms=5, zorder=3)
    ax.loglog(N_fit, hm_fit, '-', color=C['blue'],
              label=rf'H-matrix ($N^{{{coeffs[0]:.2f}}}$)')
    # Annotate compression at N=4096
    idx = list(td['N']).index(4096)
    comp = td['hmat_bytes'][idx] / (8 * 4096**2)
    ax.annotate(fr'$\times {comp:.2f}$ at $N=4096$',
                xy=(4096, hm[idx]), xytext=(1200, 60),
                arrowprops=dict(arrowstyle='->', lw=0.9, color=C['dblue']),
                fontsize=7.5, color=C['dblue'])
    ax.set_xlabel(r'$N$ (number of elements)')
    ax.set_ylabel(r'Memory (MiB)')
    ax.legend(fontsize=8)
    savefig('fig_memory_scaling.pdf')


def fig_timing_scaling():
    td     = timing_sweep()
    N      = np.array(td['N'],         dtype=float)
    t_asm  = np.array(td['assembly_s'], dtype=float)
    t_mv   = np.array(td['matvec_s'],   dtype=float)

    # reference N log²N
    ref = N * np.log2(N)**2
    ref_asm = ref * (t_asm[-3] / ref[-3])
    ref_mv  = ref * (t_mv[-3]  / ref[-3])

    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ax.loglog(N, ref_asm, ':', color=C['gray'], lw=1, label=r'$O(N\log^2 N)$')
    ax.loglog(N, t_asm,   'o-', color=C['blue'], ms=4.5, label='assembly')
    ax.loglog(N, t_mv,    's-', color=C['red'],  ms=4.5, label='matvec')
    ax.set_xlabel(r'$N$ (number of elements)')
    ax.set_ylabel(r'Wall time (s)')
    ax.legend(fontsize=8)
    savefig('fig_timing_scaling.pdf')


def fig_tamaas_comparison():
    meta = json.loads((DATADIR / 'tamaas_meta.json').read_text())
    surface, hmc_p = load_or_run_rough()
    tamaas_p = np.load(str(DATADIR / 'tamaas_pressure.npy'))

    frac_t = meta['contact_fraction']
    frac_h = float((hmc_p > 1e-10 * hmc_p.max()).mean())
    time_t = meta['time_solve_s']
    # time our solver
    n = surface.shape[0]
    solver = hmc.ContactSolver(grid_size=n)
    t0 = time.perf_counter()
    res = solver.solve(gap=-surface, p_nominal=0.05, tol=1e-8, max_iter=5000)
    time_h = time.perf_counter() - t0
    l2 = float(np.linalg.norm(hmc_p - tamaas_p) / np.linalg.norm(tamaas_p))

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.6),
                             gridspec_kw={'wspace': 0.42})

    # Left: contact fraction bar chart
    ax = axes[0]
    labels = ['Tamaas\n(dcfft)', 'H-matrix\nBEM']
    fracs  = [frac_t, frac_h]
    bars = ax.bar(labels, fracs, color=[C['orange'], C['blue']], width=0.45,
                  edgecolor='none')
    for bar, v in zip(bars, fracs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.002,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel(r'Contact fraction $A_c/A$')
    ax.set_ylim(0, max(fracs)*1.25)
    ax.set_title(r'Contact fraction ($\bar{p}=0.05$)', fontsize=9)
    ax.text(0.5, 0.75, fr'$\Delta = {abs(frac_h-frac_t):.4f}$',
            transform=ax.transAxes, ha='center', fontsize=8.5, color=C['dblue'])

    # Right: timing comparison
    ax2 = axes[1]
    times = [time_t, time_h]
    bars2 = ax2.bar(labels, times, color=[C['orange'], C['blue']], width=0.45,
                    edgecolor='none')
    for bar, v in zip(bars2, times):
        ax2.text(bar.get_x() + bar.get_width()/2, v * 1.05,
                 f'{v:.3f} s', ha='center', va='bottom', fontsize=8)
    ax2.set_ylabel(r'Solve time (s)')
    ax2.set_title(r'Wall time (33 vs 26 iterations)', fontsize=9)
    ax2.text(0.5, 0.78, fr'$\|p_\mathrm{{BEM}} - p_\mathrm{{T}}\|_2 / \|p_\mathrm{{T}}\|_2 = {l2*100:.1f}\%$',
             transform=ax2.transAxes, ha='center', fontsize=8, color=C['dblue'])

    savefig('fig_tamaas_comparison.pdf')


# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    fns = [
        fig_kernel_decay,
        fig_hmatrix_blocks,
        fig_hertz_geometry,
        fig_hertz_validation,
        fig_surface,
        fig_rough_result,
        fig_memory_scaling,
        fig_timing_scaling,
        fig_tamaas_comparison,
    ]
    print('Generating figures:')
    for fn in fns:
        print(f'  {fn.__name__}...', end=' ', flush=True)
        fn()
