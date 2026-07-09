"""FFT-convolution backend benchmark: matvec timing vs H2, and the Ns=4096
rough nested solve (double/float, fft vs h2). Run alone on an idle machine
(fenicsx-env); matvec timings are min-of-30 (noisy workstation)."""
import sys, time
sys.path.insert(0, 'python')
import numpy as np
import aspher as hc


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


def bench_matvec(Ns, reps=30):
    p = np.random.default_rng(0).random(Ns * Ns)
    out = {}
    for backend, kw in (("h2", dict(q=6)), ("fft", {})):
        s = hc.ContactSolver(grid_size=Ns, backend=backend, **kw)
        s.matvec(p)  # warm-up: lazy scratch + FFT plans
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            s.matvec(p)
            best = min(best, time.perf_counter() - t0)
        out[backend] = best
    print(f"matvec Ns={Ns}: h2(q=6) {out['h2']*1e3:8.2f} ms | "
          f"fft {out['fft']*1e3:8.2f} ms | speedup {out['h2']/out['fft']:.2f}x")


def bench_solve(Ns=4096, p_bar=0.002):
    gap = (-surface(Ns)).ravel()
    for backend in ("h2", "fft"):
        for single in (False, True):
            t0 = time.perf_counter()
            r = hc.solve_nested(grid_size=Ns, gap=gap, p_nominal=p_bar,
                                coarsest=64, q=4, leaf_side=16,
                                backend=backend, single_precision=single,
                                light_result=True)
            dt = time.perf_counter() - t0
            print(f"nested Ns={Ns} {backend:3s} "
                  f"{'f32' if single else 'f64'}: {dt:7.1f} s, "
                  f"{r.iterations} it, area {r.contact_area:.6f}, "
                  f"converged {r.converged}")


if __name__ == "__main__":
    for Ns in (1024, 2048, 4096):
        bench_matvec(Ns)
    bench_solve()
