"""Default-path identity check: the standard nested solve must stay
unchanged while shared solver files are modified (the protocol used
throughout the active-set work, doc/plans/2026-07-10-active-set-solver.md).

Protocol: BEFORE starting risky changes, build the known-good code and
`save` a reference; after each change + rebuild, `check`. The reference is
build-specific (compiler, environment) and regenerable, so it lives
untracked in data/ — always regenerate it from the pre-change build, never
reuse one across machines or toolchains.

Criteria (calibrated on this workstation, 2026-07-10): the f32 pressure is
bit-identical run-to-run; f64 carries OpenMP reduction-order scatter of
~5e-15 rel-L2, so the check is same iteration count + same contact area +
pressure rel-L2 <= 1e-13.

Run from the repo root in fenicsx-env:
    python tests/ref_solve.py save    # write data/ref_512.npz from this build
    python tests/ref_solve.py check   # compare this build against it
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))
import numpy as np
import aspher as hc

REF = os.path.join(ROOT, "data", "ref_512.npz")


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


def run():
    Ns = 512
    gap = (-surface(Ns)).ravel()
    out = {}
    r = hc.solve_nested(grid_size=Ns, gap=gap, p_nominal=0.005, coarsest=64,
                        q=4, leaf_side=16)
    out["p_f64"] = np.asarray(r.pressure).ravel()
    out["it_f64"] = r.iterations
    out["area_f64"] = r.contact_area
    rf = hc.solve_nested(grid_size=Ns, gap=gap, p_nominal=0.005, coarsest=64,
                         q=4, leaf_side=16, single_precision=True)
    out["p_f32"] = np.asarray(rf.pressure).ravel()
    out["it_f32"] = rf.iterations
    out["area_f32"] = rf.contact_area
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    cur = run()
    if mode == "save":
        os.makedirs(os.path.dirname(REF), exist_ok=True)
        np.savez(REF, **cur)
        print(f"saved {REF}: f64 {cur['it_f64']} it area {cur['area_f64']:.8f}"
              f" | f32 {cur['it_f32']} it area {cur['area_f32']:.8f}")
    else:
        if not os.path.exists(REF):
            sys.exit(f"no reference at {REF} — run 'save' on the known-good "
                     f"build first")
        ref = np.load(REF)
        ok = True
        for prec in ("f64", "f32"):
            same_it = int(ref[f"it_{prec}"]) == cur[f"it_{prec}"]
            same_area = float(ref[f"area_{prec}"]) == cur[f"area_{prec}"]
            rel = (np.linalg.norm(ref[f"p_{prec}"] - cur[f"p_{prec}"]) /
                   np.linalg.norm(ref[f"p_{prec}"]))
            bitident = np.array_equal(ref[f"p_{prec}"], cur[f"p_{prec}"])
            print(f"{prec}: iters {int(ref[f'it_{prec}'])}=={cur[f'it_{prec}']}"
                  f" {same_it}, area equal {same_area}, pressure rel-L2 "
                  f"{rel:.2e}, bit-identical {bitident}")
            ok &= same_it and same_area and rel <= 1e-13
        print("IDENTITY CHECK:", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
