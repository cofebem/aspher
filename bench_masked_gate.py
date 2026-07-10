"""M1 gate benchmark (active-set solver plan): masked vs unmasked H2 matvec
on real contact candidate masks.

Generates the seed-42 rough surface, runs a quick nested solve at p_bar=0.002,
takes the contact set (p > 0), dilates it by 2 (the candidate-set halo), and
times build/bench_masked on it. The Ns=16384 mask is the 4096 contact mask
upsampled 4x (same island geometry, no 24 GiB solve needed).

Gate (doc/plans/2026-07-10-active-set-solver.md): proceed to M2 if the
restricted masked matvec costs <~ 5% of the full matvec at N_c/N ~ 5e-3.

Run in fenicsx-env after building bench_masked:
    python bench_masked_gate.py [--only 2048,4096]
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, 'python')
import numpy as np
import aspher as hc

OUTDIR = "build/masks"
# (Ns, leaf_side, q, reps_full, reps_masked); ls/q pairs mirror the solve
# defaults at that size (q=6/ls=8 moderate Ns, q=4/ls=16 = the 16k recipe)
CASES = [
    (2048, 8, 6, 20, 60),
    (2048, 16, 4, 20, 60),
    (4096, 8, 6, 10, 40),
    (4096, 16, 4, 10, 40),
    (16384, 16, 4, 3, 12),
]


def surface(Ns, H=0.8, rms=0.02, seed=42):
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(Ns) * Ns
    K = np.hypot(*np.meshgrid(k, k, indexing="ij"))
    K[0, 0] = 1.0
    spec = np.fft.fft2(rng.standard_normal((Ns, Ns))) * K ** (-(1 + H))
    s = np.real(np.fft.ifft2(spec))
    s -= s.mean()
    s *= rms / s.std()
    return s


def dilate(m, k=2):
    out = m.copy()
    for _ in range(k):
        d = out.copy()
        for ax in (0, 1):
            for sh in (1, -1):
                d |= np.roll(out, sh, axis=ax)
        out = d
    return out


def contact_mask(Ns):
    gap = (-surface(Ns)).ravel()
    t0 = time.perf_counter()
    r = hc.solve_nested(grid_size=Ns, gap=gap, p_nominal=0.002, coarsest=64,
                        q=4, leaf_side=16, single_precision=True,
                        light_result=True)
    m = np.asarray(r.pressure) > 0
    print(f"  solve Ns={Ns}: {time.perf_counter()-t0:.1f} s, "
          f"{r.iterations} it, contact {m.mean():.2e}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="",
                    help="comma-separated Ns values to run (default: all)")
    args = ap.parse_args()
    only = {int(v) for v in args.only.split(",") if v} or None

    os.makedirs(OUTDIR, exist_ok=True)
    sizes = sorted({c[0] for c in CASES if only is None or c[0] in only})

    masks = {}
    for Ns in sizes:
        path = os.path.join(OUTDIR, f"cand_{Ns}.bin")
        if os.path.exists(path):
            print(f"mask Ns={Ns}: reusing {path}")
            masks[Ns] = path
            continue
        if Ns == 16384:
            base = contact_mask(4096)
            m = np.kron(base, np.ones((4, 4), dtype=bool))
        else:
            m = contact_mask(Ns)
        cand = dilate(m, 2)
        print(f"mask Ns={Ns}: contact {m.mean():.2e} -> candidate "
              f"{cand.mean():.2e}")
        cand.astype(np.uint8).tofile(path)
        masks[Ns] = path

    for Ns, ls, q, rf, rm in CASES:
        if only is not None and Ns not in only:
            continue
        subprocess.run(["build/bench_masked", str(Ns), masks[Ns], str(ls),
                        str(q), str(rf), str(rm)], check=True)


if __name__ == "__main__":
    main()
