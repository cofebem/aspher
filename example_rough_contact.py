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
import aspher as hc
import rfgen as rf
import time

# ── 2. size ───────────────────────────────────────────────────────────────────
Nmax = 16384 #* 4
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
# Per-size H2 accuracy knobs: coarser interpolation (q=4) + bigger leaves at the
# largest grids to keep memory down; sharper (q=6) for smaller grids.
if Ns >= 16384:
    q, leaf_side = 4, 16
else:
    q, leaf_side = 6, 8

for inc, p in enumerate(pressures[1:]):
    print("Pressure = ", p)
    start = time.time()
    # Active-set nested solve (h2 backend; requires Ns > coarsest):
    #  - active_set        : the finest level runs restricted Polonsky-Keer on a
    #                        candidate set (dilated coarse contact + near-contact)
    #                        through the masked H2 matvec -> ~4x faster and
    #                        ~1.7x less RAM at large Ns, with identical area.
    #                        Knobs: active_delta=0.05, active_halo=2,
    #                        active_max_rounds=5 (defaults; fine at Ns=4096).
    #  - single_precision  : float matvec + PCG (~half the RAM). Needs precond ON.
    #  - light_result      : skip displacement/gap arrays; pressure still filled.
    #  - precond stays ON  : the float solve stalls without the |q| preconditioner,
    #                        and the restriction does not improve conditioning.
    res = hc.solve_nested(grid_size=Ns, gap=-surface, p_nominal=p,
                          coarsest=64, q=q, leaf_side=leaf_side, precond=True,
                          single_precision=True, light_result=True,
                          active_set=True)
    print("CPU time = ", time.time() - start," seconds")
    print(f"active_rounds={res.active_rounds}  active_fallback={res.active_fallback}")

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
