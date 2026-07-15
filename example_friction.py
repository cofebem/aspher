"""Cattaneo-Mindlin partial slip demo: a parabolic indenter pressed onto an
elastic half-space, then loaded tangentially below the gross-slip limit. The
contact splits into a central stick zone and an outer slip annulus. Writes
example_friction.png (pressure | shear |q| | stick/slip map).

Run in fenicsx-env:  python example_friction.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))
import numpy as np
import aspher as hc

Ns, L, E, nu, mu, R = 128, 1.0, 1.0, 0.3, 0.4, 2.0
p_bar = 3e-3

h = L / Ns
ix = (np.arange(Ns) + 0.5) * h - 0.5 * L
x, y = np.meshgrid(ix, ix, indexing="ij")
gap = ((x * x + y * y) / (2.0 * R)).ravel()

fs = hc.FrictionSolver(grid_size=Ns, domain_size=L, E_star=E, nu=nu,
                       model=hc.CoulombFriction(mu))
fs.set_gap(gap)

rn = fs.step(p_bar=p_bar)                    # normal load
P = p_bar * L * L
# tangential force at ~55% of the gross-slip limit mu*P -> partial slip
rt = fs.step(q_bar=(0.55 * mu * p_bar, 0.0), dt=1.0)

print(f"normal:  contact area {rn.contact_area:.4f}, {rn.normal_iters} iters")
print(f"tangent: stick {rt.n_stick}, slip {rt.n_slip}, "
      f"Q/muP {rt.q_mean[0] / (mu * p_bar):.3f}, dissipation {rt.dissipation:.3e}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = np.asarray(fs.pressure).reshape(Ns, Ns)
    qmag = np.hypot(np.asarray(rt.qx), np.asarray(rt.qy))
    state = np.asarray(rt.state)  # 0 open, 1 stick, 2 slip
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, img, title in zip(
            ax, [p, qmag, state],
            ["pressure p", "shear |q|", "0 open / 1 stick / 2 slip"]):
        im = a.imshow(img.T, origin="lower", cmap="viridis")
        a.set_title(title)
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout()
    fig.savefig("example_friction.png", dpi=110)
    print("wrote example_friction.png")
except ImportError:
    print("(matplotlib not available; skipped the figure)")
