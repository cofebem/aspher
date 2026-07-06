"""Minimal rough-surface contact example (matrix-free H2 backend).

Flow:
  1. import what's needed
  2. define the size Ns
  3. define the rough surface
  4. apply pressure (solve the normal contact problem)
  5. get the contact area and plot it

Run (fenicsx-env):  python example_rough_contact.py
"""
# ── 1. imports ────────────────────────────────────────────────────────────────
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))
import numpy as np
import matplotlib.pyplot as plt
import hmatrix_contact as hc
import rfgen as rf
import time

# ── 2. size ───────────────────────────────────────────────────────────────────
Nmax = 4096 
sampling = 1
Ns = Nmax // sampling  # grid is Ns x Ns; must be a power of two for backend="h2"
L = 1.0           # domain side length
p_bar = 0.002      # applied (nominal) mean pressure, in units of E*

rng = np.random.default_rng(seed=42)  # for reproducibility
roughness = rf.selfaffine_field(
    dim=2,           # Dimension (1, 2, or 3)
    N=Nmax,           # Grid size per dimension
    Hurst=0.8,       # Hurst exponent ∈ [0, 1]
    k_low=12/Nmax,      # Lower wavenumber cutoff
    k_high=500/Nmax,      # Upper wavenumber cutoff (≤ 0.5 Nyquist)
    plateau=False,   # Flat spectrum for k < k_low
    noise=True,      # True: filtered noise, False: ideal spectrum
    rng=rng,        # numpy.random.Generator for reproducibility
    verbose=False    # Print parameters
)
rms = 0.002
roughness *= rms / np.std(roughness)

# Build the surface memory-leanly: a single Ns x Ns array via broadcasting
# (no full meshgrid), in float32, and free the roughness field once folded in.
# At Ns=16384 the meshgrid alone would be ~4 GiB; broadcasting avoids it.
import gc
xr = (np.linspace(0, 1, Ns, dtype=np.float32) - 0.5)
surface = -(xr[None, :]**2 + xr[:, None]**2)          # paraboloid, one Ns x Ns
surface += roughness[::sampling, ::sampling].astype(np.float32)
surface -= np.max(surface)
del roughness, xr
gc.collect()
# return the surface-generation temporaries to the OS before the (large) solve,
# so their transient peak does not stack on top of the solver's footprint.
try:
    import ctypes
    ctypes.CDLL("libc.so.6").malloc_trim(0)
except Exception:
    pass

N_load_steps = 2
plot_every = 1

pressures = np.linspace(0,p_bar,N_load_steps,endpoint=True)
# ── 4. apply pressure: solve the contact problem ──────────────────────────────
# A rigid flat is pressed onto the rough surface, so the initial gap is -height.
# (This solver object is only needed for the commented single-grid solve below;
#  skip building it at Ns=16384 where it would waste ~14 GiB unused.)
if Ns != 16384:
    solver = hc.ContactSolver(grid_size=Ns, domain_size=L, E_star=1.0,
                              backend="h2", q=6)

# result = hc.solve_nested(grid_size=1024, gap=g0, p_nominal=0.05, coarsest=64, q=6)
for inc,p in enumerate(pressures[1:]):
    print("Pressure = ", p)
    start = time.time()
    # res = solver.solve(gap=-surface, p_nominal=p, tol=1e-8, max_iter=5000)
    if Ns == 16384:
        # memory-lean settings for the largest grid (~2.7e8 DOFs):
        #  - single_precision: run the matvec + PCG in float (~half the RAM)
        #  - light_result: skip the displacement/gap result arrays (2 x Ns^2)
        #  - larger leaves (leaf_side=16), q=4
        # The |q| preconditioner stays ON: single precision needs it to
        # converge (the float solve stalls without it). It runs in float too,
        # so its FFT stays memory-lean. The solve reaches a ~2e-6 float floor —
        # plenty for the contact area.
        res = hc.solve_nested(grid_size=Ns, gap=-surface, p_nominal=p,
                              coarsest=64, q=4, leaf_side=16, precond=True,
                              single_precision=True, light_result=True)
    else:
        res = hc.solve_nested(grid_size=Ns, gap=-surface, p_nominal=p,
                              coarsest=64, q=6)
    print("CPU time = ", time.time() - start," seconds")
    
    # ── 5. contact area + plot ────────────────────────────────────────────────────
    if inc % plot_every == 0 or inc == pressures.shape[0]-2:
        pressure = np.asarray(res.pressure)          # (Ns, Ns)
        contact = pressure > 0.0                      # in-contact mask
        area_fraction = res.contact_area              # = Ac / A (also contact.mean())

        print(f"Ns={Ns}  applied mean pressure={p_bar}")
        print(f"converged={res.converged} in {res.iterations} iters, "
            f"mean_p={res.mean_pressure:.4f}")
        print(f"contact area fraction Ac/A = {area_fraction:.4f}")

        fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
        im0 = ax[0].imshow(surface, origin="lower", cmap="terrain", extent=[0, L, 0, L])
        ax[0].set_title("rough surface (height)")
        fig.colorbar(im0, ax=ax[0], fraction=0.046)

        im1 = ax[1].imshow(pressure, origin="lower", cmap="inferno", extent=[0, L, 0, L])
        ax[1].set_title("contact pressure")
        fig.colorbar(im1, ax=ax[1], fraction=0.046)

        ax[2].imshow(contact, origin="lower", cmap="Greys", extent=[0, L, 0, L])
        ax[2].set_title(f"contact area  (Ac/A = {area_fraction:.3f})")

        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        fig.tight_layout()
        out = os.path.join(os.path.dirname(__file__), f"Rough_contact_nested_{inc}_Ns_{Ns}_Nmax_{Nmax}_p0_{p}.png")
        fig.savefig(out, dpi=600)
        print(f"saved figure -> {out}")
