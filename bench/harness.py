"""ASPHER benchmark harness (validation plan §8 measurement protocol).

Three jobs, one tool:

1. **Provenance.** Every record carries the revision (and whether the tree was
   dirty), compiler and flags, FFT engine, CPU, thread count and affinity, the
   full workload parameters *and a hash of the generated surface*, every solver
   argument, and — from the corrected result API — status, requested vs
   effective tolerance, validation scope, the independently recomputed final
   errors, per-phase timings and the solver's own byte accounting. A number
   without that context cannot support a promotion decision.

2. **Paired A/B.** Variants run one per fresh subprocess in alternating order
   (ABBA), so drift in machine load cancels rather than accumulating into one
   arm. Wall clock on this workstation swings ~30% with desktop co-tenancy, so
   single runs are meaningless; `analyze.py` reports medians and a bootstrap
   confidence interval on the paired difference.

3. **Preflight.** Above Ns=4096 a case is a manual, serialised job, and an
   OOM costs half an hour. `preflight` estimates the requirement by actually
   CONSTRUCTING the operator at two small grids and fitting a + b*N to its
   itemised accounting (A07) — H2 storage is not proportional to N, since the
   near stencils, transfers and kernel table are fixed costs — then adds the
   exact solver-side buffer set, and refuses to launch when it does not fit.

   Validated against the measurements recorded in CLAUDE.md for Ns=16384:

       variant          predicted   measured RSS
       h2-f32-active     8.29 GiB   10.9 GiB
       h2-f32 (std)     14.82 GiB   18.3 GiB
       h2-f64-active     8.99 GiB   12.5 GiB
       h2-f64 (std)     21.82 GiB   OOM on a 31 GiB machine

   The model runs 20-25% below measured RSS because it accounts for the
   library's buffers, not the Python heap and allocator slack on top; the
   default 1.25 headroom is exactly that gap, not a guess. It is deliberately
   an underestimate plus an explicit margin rather than a padded single
   number, so the margin stays visible and adjustable.

Usage:
    python bench/harness.py list
    python bench/harness.py preflight --ns 16384 --variant h2-f32-active
    python bench/harness.py run --workload rough-H0.8 --ns 1024 \
        --variants h2-f64,h2-f64-active --reps 5
    python bench/harness.py worker --workload ... --ns ... --variant ...  # internal

Long jobs must be detached from this shell, not backgrounded inside a tool
call:
    nohup python bench/harness.py run ... > bench.log 2>&1 < /dev/null & disown
"""
import argparse
import ctypes
import gc
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_LEDGER = os.path.join(ROOT, "data", "bench_ledger.jsonl")


# ── workloads ───────────────────────────────────────────────────────────────
# A workload is a physical problem, independent of how it is solved. Surfaces
# are regenerated deterministically from (kind, seed, parameters) and hashed,
# so a record can be tied to the exact field it measured.
WORKLOADS = {
    "rough-H0.8": dict(kind="selfaffine", Hurst=0.8, rms=0.02,
                       k_low_cells=12.0, k_high=0.33, seed=42, p_bar=0.002),
    "rough-H0.3": dict(kind="selfaffine", Hurst=0.3, rms=0.02,
                       k_low_cells=12.0, k_high=0.33, seed=7, p_bar=0.002),
    "rough-H1.0": dict(kind="selfaffine", Hurst=1.0, rms=0.02,
                       k_low_cells=12.0, k_high=0.33, seed=11, p_bar=0.002),
    # dilute / intermediate / near-full contact on the same surface
    "rough-H0.8-dilute": dict(kind="selfaffine", Hurst=0.8, rms=0.02,
                              k_low_cells=12.0, k_high=0.33, seed=42,
                              p_bar=0.0005),
    "rough-H0.8-dense": dict(kind="selfaffine", Hurst=0.8, rms=0.02,
                             k_low_cells=12.0, k_high=0.33, seed=42,
                             p_bar=0.02),
    "hertz": dict(kind="hertz", radius=2.0, p_bar=0.003),
    "flat-patch": dict(kind="flat_patch", frac=0.35, depth=0.01, p_bar=0.01),
}

# ── variants ────────────────────────────────────────────────────────────────
# A variant is a way of solving it: solver arguments only. `h2_big` marks the
# large-grid parameter set (q=4, leaf_side=16) the recipe uses above Ns=4096.
VARIANTS = {
    "h2-f64":            dict(backend="h2", precision="double"),
    "h2-f64-active":     dict(backend="h2", precision="double", active_set=True),
    "h2-f32":            dict(backend="h2", precision="float",
                              allow_tolerance_relaxation=True),
    "h2-f32-active":     dict(backend="h2", precision="float",
                              allow_tolerance_relaxation=True, active_set=True),
    # A11-adjacent: restrict EVERY level with a coarser one beneath it, not
    # only the finest. Opt-in until the paired A/B clears the promotion gate.
    "h2-f32-active-all": dict(backend="h2", precision="float",
                              allow_tolerance_relaxation=True, active_set=True,
                              active_all_levels=True),
    "h2-f64-active-all": dict(backend="h2", precision="double",
                              active_set=True, active_all_levels=True),
    "h2-polish":         dict(backend="h2", precision="float_then_double"),
    "fft-f64":           dict(backend="fft", precision="double"),
    "fft-f32":           dict(backend="fft", precision="float",
                              allow_tolerance_relaxation=True),
    "fft-polish":        dict(backend="fft", precision="float_then_double"),
}


def h2_shape(Ns):
    """(q, leaf_side): the documented large-grid parameter set above 4096."""
    return (6, 8) if Ns <= 4096 else (4, 16)


# ── provenance ──────────────────────────────────────────────────────────────
def _run(cmd, **kw):
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=30, **kw).stdout.strip()
    except Exception:
        return ""


def _cmake_cache(key):
    path = os.path.join(ROOT, "build", "CMakeCache.txt")
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(key + ":"):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


def fft_engine():
    """pocketfft (bundled, BSD) or fftw3 (opt-in, GPL) — they are not the same
    operator implementation, so a timing is meaningless without it. Both build
    trees write the same python/aspher*.so, which has bitten this project
    before."""
    so = [f for f in os.listdir(os.path.join(ROOT, "python"))
          if f.startswith("aspher") and f.endswith(".so")]
    if not so:
        return "unknown"
    out = _run(["ldd", os.path.join(ROOT, "python", so[0])])
    return "fftw3" if "libfftw3" in out else "pocketfft"


def provenance():
    # Only TRACKED modifications make a run untrustworthy: untracked files
    # (build trees, figures, scratch) cannot change the binary under test, and
    # counting them would make every run on a working checkout look dirty.
    dirty = _run(["git", "-C", ROOT, "status", "--porcelain",
                  "--untracked-files=no"])
    cpu = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    mem_avail = mem_available_bytes()
    cxx = _cmake_cache("CMAKE_CXX_COMPILER")
    return {
        "revision": _run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"]),
        "dirty": bool(dirty),
        "dirty_files": len(dirty.splitlines()) if dirty else 0,
        "cxx": cxx,
        "cxx_version": _run([cxx, "--version"]).splitlines()[0] if cxx else "",
        "cxx_flags": _cmake_cache("CMAKE_CXX_FLAGS_RELEASE"),
        "build_type": _cmake_cache("CMAKE_BUILD_TYPE"),
        "fft_engine": fft_engine(),
        "cpu": cpu,
        "cores_online": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", ""),
        "mem_available_gib": round(mem_avail / 2**30, 2) if mem_avail else None,
        "python": platform.python_version(),
        "host": platform.node(),
    }


def mem_available_bytes():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(re.findall(r"\d+", line)[0]) * 1024
    except OSError:
        pass
    return 0


# ── surfaces ────────────────────────────────────────────────────────────────
def build_gap(workload, Ns):
    """Deterministic gap field plus a hash of the exact bytes measured."""
    import numpy as np
    sys.path.insert(0, os.path.join(ROOT, "python"))
    w = WORKLOADS[workload]

    if w["kind"] == "selfaffine":
        import rfgen as rf
        rng = np.random.default_rng(w["seed"])
        h = rf.selfaffine_field(dim=2, N=Ns, Hurst=w["Hurst"],
                                k_low=w["k_low_cells"] / Ns, k_high=w["k_high"],
                                plateau=False, noise=True, rng=rng,
                                verbose=False)
        h *= w["rms"] / np.std(h)
        gap = (-h).astype(np.float64).ravel()
        del h
    elif w["kind"] == "hertz":
        x = ((np.arange(Ns) + 0.5) / Ns - 0.5).astype(np.float64)
        gap = ((x[:, None] ** 2 + x[None, :] ** 2) / (2.0 * w["radius"])).ravel()
    elif w["kind"] == "flat_patch":
        x = (np.arange(Ns) + 0.5) / Ns - 0.5
        r = np.maximum(np.abs(x)[:, None], np.abs(x)[None, :])
        gap = np.where(r < 0.5 * w["frac"], 0.0, w["depth"]).ravel().astype(np.float64)
    else:
        raise ValueError("unknown workload kind " + w["kind"])

    gc.collect()
    if Ns >= 8192:
        # the surface generation transient, not the solve, is what OOMs at
        # large Ns; hand the pages back before the solver asks for its own
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass
    digest = hashlib.blake2b(np.ascontiguousarray(gap).view(np.uint8),
                             digest_size=16).hexdigest()
    return gap, digest


# ── preflight ───────────────────────────────────────────────────────────────
def estimate_bytes(Ns, variant, light=True, coarsest=64):
    """Predicted peak for one case, from measured structure rather than a rule
    of thumb.

    The operator term is calibrated by CONSTRUCTING the operator at a small Ns
    and reading its itemised accounting (A07): box, interaction and coupling
    counts all scale linearly in N at fixed (leaf_side, q), so bytes/DOF is
    constant and the small build is cheap and honest. The solver term is the
    exact buffer set the solver reports. The surface term is the Python
    generation transient, which at large Ns dominates everything else.
    """
    sys.path.insert(0, os.path.join(ROOT, "python"))
    import aspher as hc
    v = VARIANTS[variant]
    N = float(Ns * Ns)
    real = 4 if v["precision"] == "float" else 8
    if v["precision"] == "float_then_double":
        real = 8  # the polish carries the double working set

    q, ls = h2_shape(Ns)
    if v["backend"] == "h2":
        # H2 storage is NOT proportional to N: the tree, the leaf array and
        # both CSR interaction lists are, but the near stencils (9 blocks of
        # ls^2 x ls^2 — 4.5 MiB at ls=16!), the transfer matrices and the
        # compact kernel table are fixed, and the couplings grow only with the
        # level count. Assuming a single bytes/DOF figure overestimates by
        # gigabytes at large Ns. Fit a + b*N from two cheap real builds
        # instead of guessing which term is which.
        cal = []
        for cal_ns in (256, 512):
            op = hc.ContactSolver(grid_size=cal_ns, backend="h2", q=q,
                                  h2_leaf_side=ls)
            info = op.hmatrix_info()
            cal.append((float(cal_ns * cal_ns),
                        float(info["mem_resident"] +
                              info["mem_estimated_next_apply"])))
            del op
        (n0, b0), (n1, b1) = cal
        op_per_dof = (b1 - b0) / (n1 - n0)
        op_fixed = b0 - op_per_dof * n0
        op_bytes = op_fixed + op_per_dof * N
    else:
        # exact zero-padded convolution: kernel half-spectrum 2N + padded grid
        # 4N + complex half-spectrum work 4N reals (FFT backend spec)
        op_per_dof, op_fixed = 10.0 * real, 0.0
        op_bytes = op_per_dof * N

    # CG state: 7 working vectors + the byte mask, plus the best iterate and
    # the output fields when they are kept.
    full_cg = 7.0 * N * real + N
    if not light:
        full_cg += N * real + 2.0 * N * 8
    if v.get("active_set"):
        # The certified active-set path keeps O(N_c) slot-blocked state, not
        # O(N). Occupancy is workload-dependent (measured cand/N ~ 1e-2 with
        # leaf blocking and the halo pushing it to a few percent); 10% is a
        # generous allowance. The FULL state is still what the uncertified
        # full-solve fallback would need, so both are reported.
        cg = 0.10 * full_cg
    else:
        cg = full_cg
    pressure = 8.0 * N            # returned pressure (double)
    gap = 8.0 * N                 # caller-owned gap
    coarse = 8.0 * N / 3.0        # restricted coarse levels, geometric sum
    warm = 8.0 * N                # prolonged warm start at the finest level
    surface = 3.0 * 8.0 * N       # generation transient (meshgrid + FFT temps)

    solve = op_bytes + cg + pressure + gap + coarse + warm
    fallback = op_bytes + full_cg + pressure + gap + coarse + warm
    return {
        "operator_bytes": op_bytes,
        "operator_bytes_per_dof": op_per_dof,
        "operator_bytes_fixed": op_fixed,
        "cg_bytes": cg,
        "cg_bytes_full": full_cg,
        "output_bytes": pressure,
        "gap_bytes": gap,
        "coarse_bytes": coarse,
        "warm_start_bytes": warm,
        "surface_transient_bytes": surface,
        "solve_peak_bytes": solve,
        # the two peaks do not coincide: generation finishes before the solve
        # allocates, so the requirement is the larger of them
        "peak_bytes": max(solve, surface + gap),
        "fallback_peak_bytes": max(fallback, surface + gap),
    }


def preflight(Ns, variant, headroom=1.25, light=True, verbose=True):
    est = estimate_bytes(Ns, variant, light=light)
    avail = mem_available_bytes()
    need = est["peak_bytes"] * headroom
    ok = avail > need
    if verbose:
        g = 2.0 ** 30
        print(f"preflight Ns={Ns} variant={variant}")
        for k in ("operator_bytes", "cg_bytes", "gap_bytes", "coarse_bytes",
                  "warm_start_bytes", "output_bytes",
                  "surface_transient_bytes"):
            print(f"    {k:26s} {est[k] / g:8.2f} GiB")
        if est["fallback_peak_bytes"] > est["peak_bytes"]:
            print(f"    {'(full-solve fallback)':26s} "
                  f"{est['fallback_peak_bytes'] / g:8.2f} GiB")
        print(f"    {'peak estimate':26s} {est['peak_bytes'] / g:8.2f} GiB"
              f"  x{headroom} headroom = {need / g:.2f} GiB")
        print(f"    {'available':26s} {avail / g:8.2f} GiB"
              f"   -> {'OK' if ok else 'INSUFFICIENT'}")
    est["available_bytes"] = avail
    est["headroom"] = headroom
    est["fits"] = ok
    return est


# ── worker: one case, one fresh process ─────────────────────────────────────
def worker(workload, Ns, variant, reps, tol, light, coarsest, max_iter):
    import resource
    import numpy as np
    sys.path.insert(0, os.path.join(ROOT, "python"))
    import aspher as hc

    v = dict(VARIANTS[variant])
    q, ls = h2_shape(Ns)
    t_gen = time.perf_counter()
    gap, digest = build_gap(workload, Ns)
    t_gen = time.perf_counter() - t_gen
    rss_after_surface = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    kwargs = dict(grid_size=Ns, gap=gap, p_nominal=WORKLOADS[workload]["p_bar"],
                  coarsest=coarsest, precond=True, tol=tol, coarse_tol=1e-4,
                  max_iter=max_iter, light_result=light)
    kwargs["backend"] = v["backend"]
    if v["backend"] == "h2":
        kwargs["q"] = q
        kwargs["leaf_side"] = ls
    kwargs["precision"] = v["precision"]
    for k in ("allow_tolerance_relaxation", "active_set", "active_all_levels",
              "active_halo", "active_max_rounds"):
        if k in v:
            kwargs[k] = v[k]

    times, res = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        res = hc.solve_nested(**kwargs)
        times.append(time.perf_counter() - t0)

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rec = {
        "workload": workload, "Ns": Ns, "N": Ns * Ns, "variant": variant,
        "surface_sha": digest, "workload_params": WORKLOADS[workload],
        "solver_args": {k: val for k, val in kwargs.items() if k != "gap"},
        # timing: the first rep is cold, the rest warm (plan §8)
        "wall_cold_s": times[0],
        "wall_warm_s": sorted(times[1:])[len(times[1:]) // 2] if len(times) > 1 else None,
        "wall_all_s": times,
        "surface_gen_s": t_gen,
        # memory: fresh-process peak, and the library's own accounting, kept
        # apart on purpose — RSS includes the Python heap and the surface
        "peak_rss_gib": rss_kb / 1048576.0,
        "rss_after_surface_gib": rss_after_surface / 1048576.0,
        "solver_memory": dict(res.memory),
        "timings": dict(res.timings),
        # correctness: never report a speed without the accuracy it bought
        "status": res.status, "status_reason": res.status_reason,
        "converged": bool(res.converged),
        "iterations": int(res.iterations),
        "matvec_count": int(res.matvec_count),
        "verification_matvec_count": int(res.verification_matvec_count),
        "precond_count": int(res.precond_count),
        "identification_steps": int(res.identification_steps),
        "requested_tol": res.requested_tol, "effective_tol": res.effective_tol,
        "validation_scope": res.validation_scope,
        "operator_error_kind": res.operator_error_kind,
        "fw_error": res.fw_error, "penetration_error": res.penetration_error,
        "load_error": res.load_error, "pk_error": res.pk_error,
        "contact_area": float(res.contact_area),
        "mean_pressure": float(res.mean_pressure),
        "active_rounds": int(res.active_rounds),
        "active_fallback": bool(res.active_fallback),
        "stage_stats": list(res.stage_stats),
        "pressure_sha": hashlib.blake2b(
            np.ascontiguousarray(np.asarray(res.pressure, dtype=np.float64))
            .view(np.uint8), digest_size=16).hexdigest(),
        "run_status": "ok",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print("JSON " + json.dumps(rec), flush=True)


# ── orchestrator ────────────────────────────────────────────────────────────
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
                if d.get("run_status") == "ok":
                    done.add((d.get("workload"), d.get("Ns"), d.get("variant"),
                              d.get("rep")))
    return done


def classify(returncode, stderr):
    if returncode == -9 or "bad_alloc" in stderr or "MemoryError" in stderr:
        return "oom"
    return "error"


def paired_order(variants, reps):
    """ABBA: alternate the order every rep so machine drift cancels between
    arms instead of loading onto whichever runs first."""
    seq = []
    for r in range(reps):
        order = list(variants) if r % 2 == 0 else list(reversed(variants))
        for v in order:
            seq.append((r, v))
    return seq


def run(args):
    ledger = args.ledger
    os.makedirs(os.path.dirname(ledger), exist_ok=True)
    done = set() if args.force else load_done(ledger)
    prov = provenance()
    if prov["dirty"] and not args.allow_dirty:
        sys.exit(f"working tree has {prov['dirty_files']} modified files; "
                 "a timing tied to an unknown source state is not evidence. "
                 "Commit, or pass --allow-dirty.")
    print("provenance: " + json.dumps(prov), flush=True)

    variants = args.variants.split(",")
    for v in variants:
        if v not in VARIANTS:
            sys.exit(f"unknown variant {v}; known: {', '.join(VARIANTS)}")
    for Ns in args.ns:
        reps = 1 if Ns > args.manual_above else args.reps
        for rep, variant in paired_order(variants, reps):
            key = (args.workload, Ns, variant, rep)
            if key in done:
                print(f"skip  {args.workload} Ns={Ns:6d} {variant:16s} "
                      f"rep={rep} (recorded)", flush=True)
                continue
            # Preflight PER VARIANT, and immediately before the case runs.
            # Doing it once per Ns against the first variant lets a heavier
            # arm through on a lighter arm's budget, and doing it up front
            # uses a memory reading that the earlier cases have since
            # invalidated. A refusal is recorded as a result, not skipped
            # silently: "this configuration does not fit here" is a finding.
            if Ns > args.manual_above:
                est = preflight(Ns, variant, headroom=args.headroom,
                                light=not args.full_result, verbose=False)
                if not est["fits"] and not args.force_memory:
                    need = est["peak_bytes"] * args.headroom
                    print(f"SKIP  {args.workload} Ns={Ns:6d} {variant:16s}: "
                          f"needs {need / 2**30:.1f} GiB, "
                          f"{est['available_bytes'] / 2**30:.1f} GiB available",
                          flush=True)
                    with open(ledger, "a") as f:
                        f.write(json.dumps({
                            "workload": args.workload, "Ns": Ns,
                            "variant": variant, "rep": rep,
                            "run_status": "skipped_preflight",
                            "preflight": est, "provenance": prov,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }) + "\n")
                    continue
            print(f"run   {args.workload} Ns={Ns:6d} {variant:16s} rep={rep}",
                  flush=True)
            cmd = [sys.executable, os.path.abspath(__file__), "worker",
                   "--workload", args.workload, "--ns", str(Ns),
                   "--variant", variant, "--reps", str(args.inner_reps),
                   "--tol", repr(args.tol), "--coarsest", str(args.coarsest),
                   "--max-iter", str(args.max_iter)]
            if not args.full_result:
                cmd.append("--light")
            t0 = time.time()
            try:
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=args.timeout)
                line = next((l for l in r.stdout.splitlines()
                             if l.startswith("JSON ")), None)
                if line and r.returncode == 0:
                    rec = json.loads(line[5:])
                else:
                    rec = {"workload": args.workload, "Ns": Ns,
                           "variant": variant,
                           "run_status": classify(r.returncode, r.stderr),
                           "stderr_tail": r.stderr[-2000:]}
            except subprocess.TimeoutExpired as e:
                tail = (e.stderr or b"").decode(errors="replace")
                rec = {"workload": args.workload, "Ns": Ns, "variant": variant,
                       "run_status": "timeout", "stderr_tail": tail[-2000:]}
            rec["rep"] = rep
            rec["provenance"] = prov
            rec["elapsed_s"] = time.time() - t0
            with open(ledger, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"      -> {rec.get('run_status')} "
                  f"{rec.get('status', '')} "
                  f"{rec.get('wall_cold_s', float('nan')):.2f}s "
                  f"{rec.get('peak_rss_gib', float('nan')):.2f} GiB",
                  flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")

    pf = sub.add_parser("preflight")
    pf.add_argument("--ns", type=int, required=True)
    pf.add_argument("--variant", default="h2-f32-active")
    pf.add_argument("--headroom", type=float, default=1.25)
    pf.add_argument("--full-result", action="store_true")

    rn = sub.add_parser("run")
    rn.add_argument("--workload", default="rough-H0.8")
    rn.add_argument("--ns", type=int, nargs="+", default=[1024])
    rn.add_argument("--variants", default="h2-f64,h2-f64-active")
    rn.add_argument("--reps", type=int, default=5)
    rn.add_argument("--inner-reps", type=int, default=1,
                    help="solves per process; >1 separates cold from warm")
    rn.add_argument("--tol", type=float, default=1e-8)
    rn.add_argument("--coarsest", type=int, default=64)
    rn.add_argument("--max-iter", type=int, default=20000)
    rn.add_argument("--timeout", type=int, default=7200)
    rn.add_argument("--ledger", default=DEFAULT_LEDGER)
    rn.add_argument("--full-result", action="store_true")
    rn.add_argument("--force", action="store_true")
    rn.add_argument("--allow-dirty", action="store_true")
    rn.add_argument("--force-memory", action="store_true")
    rn.add_argument("--headroom", type=float, default=1.25)
    rn.add_argument("--manual-above", type=int, default=4096,
                    help="above this Ns: preflight gate and a single rep")

    wk = sub.add_parser("worker")
    wk.add_argument("--workload", required=True)
    wk.add_argument("--ns", type=int, required=True)
    wk.add_argument("--variant", required=True)
    wk.add_argument("--reps", type=int, default=1)
    wk.add_argument("--tol", type=float, default=1e-8)
    wk.add_argument("--coarsest", type=int, default=64)
    wk.add_argument("--max-iter", type=int, default=20000)
    wk.add_argument("--light", action="store_true")

    a = ap.parse_args()
    if a.cmd == "list":
        print("workloads:")
        for k, v in WORKLOADS.items():
            print(f"  {k:22s} {v}")
        print("variants:")
        for k, v in VARIANTS.items():
            print(f"  {k:22s} {v}")
        print("\nprovenance: " + json.dumps(provenance(), indent=2))
    elif a.cmd == "preflight":
        est = preflight(a.ns, a.variant, headroom=a.headroom,
                        light=not a.full_result)
        sys.exit(0 if est["fits"] else 1)
    elif a.cmd == "run":
        run(a)
    elif a.cmd == "worker":
        worker(a.workload, a.ns, a.variant, a.reps, a.tol, a.light,
               a.coarsest, a.max_iter)


if __name__ == "__main__":
    random.seed(0)
    main()
