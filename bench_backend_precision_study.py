"""Backend x precision x Ns benchmark/convergence sweep for the rough-contact
problem (see doc/specs/2026-07-10-backend-precision-benchmark-design.md).

Worker mode (runs ONE case in-process, prints one JSON line):
    python bench_backend_precision_study.py --backend h2 --precision double --ns 1024

Orchestrator mode (default; spawns one subprocess per case, resumable):
    python bench_backend_precision_study.py [--max-ns 16384] [--timeout 1800] [--force]
"""
import argparse
import ctypes
import gc
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NS_ALL = [256, 512, 1024, 2048, 4096, 8192, 16384]
BACKENDS = ["h2", "fft"]
PRECISIONS = ["double", "float"]
RESULTS_PATH = os.path.join(HERE, "data", "backend_precision_study.jsonl")

SEED = 42
HURST = 0.8
RMS = 0.02
K_HIGH = 0.33
P_BAR = 0.002
COARSEST = 64


def h2_settings(Ns):
    return (6, 8) if Ns <= 4096 else (4, 16)


def build_gap(Ns):
    """Rough surface via rfgen (see design spec Sec.1); returns -height,
    ravelled. Frees intermediates and returns large allocations to the OS
    before the solve at big Ns, mirroring example_rough_contact.py."""
    import numpy as np
    sys.path.insert(0, os.path.join(HERE, "python"))
    import rfgen as rf

    rng = np.random.default_rng(SEED)
    roughness = rf.selfaffine_field(
        dim=2, N=Ns, Hurst=HURST, k_low=12.0 / Ns, k_high=K_HIGH,
        plateau=False, noise=True, rng=rng, verbose=False,
    )
    roughness *= RMS / np.std(roughness)
    gap0 = (-roughness).astype(np.float64).ravel()
    del roughness
    gc.collect()
    if Ns >= 8192:
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass
    return gap0


def worker(backend, precision, Ns):
    import resource
    import numpy as np
    sys.path.insert(0, os.path.join(HERE, "python"))
    import aspher as hc

    gap0 = build_gap(Ns)
    q, leaf_side = h2_settings(Ns) if backend == "h2" else (None, None)
    single = (precision == "float")
    reps = 3 if Ns <= 512 else 1

    kwargs = dict(
        grid_size=Ns, gap=gap0, p_nominal=P_BAR, coarsest=COARSEST,
        precond=True, tol=1e-8, coarse_tol=1e-4, max_iter=20000,
        single_precision=single, light_result=True, backend=backend,
        record_error_history=True,
    )
    if backend == "h2":
        kwargs["q"] = q
        kwargs["leaf_side"] = leaf_side

    times = []
    res = None
    for _ in range(reps):
        t0 = time.perf_counter()
        res = hc.solve_nested(**kwargs)
        times.append(time.perf_counter() - t0)

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out = {
        "backend": backend, "precision": precision, "Ns": Ns, "N": Ns * Ns,
        "q": q, "leaf_side": leaf_side,
        "wall_time_s": min(times), "wall_time_all_s": times,
        "rss_gib": rss_kb / 1048576.0,
        "iterations": int(res.iterations), "converged": bool(res.converged),
        "final_error": float(res.error), "contact_area": float(res.contact_area),
        "mean_pressure": float(res.mean_pressure),
        "error_history": [float(v) for v in np.asarray(res.error_history)],
        "status": "ok",
        "seed": SEED, "p_bar": P_BAR, "rms": RMS, "Hurst": HURST,
        "k_low": 12.0 / Ns, "k_high": K_HIGH,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print("JSON " + json.dumps(out), flush=True)


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((d.get("backend"), d.get("precision"), d.get("Ns")))
    return done


def classify_failure(returncode, stderr):
    if returncode == -9:
        return "oom"
    if "bad_alloc" in stderr or "MemoryError" in stderr or "Killed" in stderr:
        return "oom"
    return "error"


def sweep(timeout, force, max_ns):
    done = set() if force else load_done(RESULTS_PATH)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    cases = [(b, p, n) for n in NS_ALL if n <= max_ns
             for p in PRECISIONS for b in BACKENDS]

    for backend, precision, Ns in cases:
        if (backend, precision, Ns) in done:
            print(f"skip {backend:3s} {precision:6s} Ns={Ns:6d} (already recorded)",
                  flush=True)
            continue
        print(f"running {backend:3s} {precision:6s} Ns={Ns:6d} ...", flush=True)
        cmd = [sys.executable, os.path.abspath(__file__),
               "--backend", backend, "--precision", precision, "--ns", str(Ns)]
        t0 = time.time()
        record = None
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            line = next((l for l in r.stdout.splitlines() if l.startswith("JSON ")),
                        None)
            if line is not None and r.returncode == 0:
                record = json.loads(line[5:])
            else:
                record = {
                    "backend": backend, "precision": precision, "Ns": Ns,
                    "N": Ns * Ns, "status": classify_failure(r.returncode, r.stderr),
                    "stderr_tail": r.stderr[-2000:],
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
        except subprocess.TimeoutExpired as e:
            stderr = e.stderr.decode(errors="replace") if e.stderr else ""
            record = {
                "backend": backend, "precision": precision, "Ns": Ns, "N": Ns * Ns,
                "status": "timeout", "stderr_tail": stderr[-2000:],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        dt = time.time() - t0
        with open(RESULTS_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  -> status={record.get('status')} wall={dt:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=BACKENDS, help="single-case worker mode")
    ap.add_argument("--precision", choices=PRECISIONS)
    ap.add_argument("--ns", type=int)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-ns", type=int, default=16384)
    args = ap.parse_args()
    if args.backend:
        worker(args.backend, args.precision, args.ns)
    else:
        sweep(args.timeout, args.force, args.max_ns)
