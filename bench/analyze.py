"""Paired analysis and promotion decision for the ASPHER benchmark ledger.

Applies the validation plan's promotion rule (§8) rather than eyeballing two
numbers:

  * medians over >=5 paired samples, with a bootstrap confidence interval on
    the paired *difference* (the pairing is what makes ~30% machine drift
    cancel; comparing two independent medians would not);
  * a change is promotable only when the CI excludes zero in its favour,
    the median full-solve improvement reaches the target (10% for a new
    default), and no case on the suite regresses by more than 5% unexplained;
  * **equal accuracy is a precondition, not a footnote.** A variant that ran
    at a different effective tolerance, a different validation scope, or that
    did not converge, is disqualified before any timing is compared.

Usage:
    python bench/analyze.py --pair h2-f64,h2-f64-active
    python bench/analyze.py --pair a,b --metric peak_rss_gib --target 0.0
    python bench/analyze.py --summary
"""
import argparse
import json
import os
import random
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_LEDGER = os.path.join(ROOT, "data", "bench_ledger.jsonl")


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def ok(r):
    return r.get("run_status") == "ok"


def accuracy_key(r):
    """What must match before two timings may be compared at all."""
    return (r.get("status"), r.get("validation_scope"),
            r.get("effective_tol"), r.get("requested_tol"))


def bootstrap_ci(diffs, alpha=0.05, n=20000, seed=0):
    """Percentile bootstrap on the median paired difference."""
    if len(diffs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    meds = []
    k = len(diffs)
    for _ in range(n):
        meds.append(st.median(diffs[rng.randrange(k)] for _ in range(k)))
    meds.sort()
    lo = meds[int(alpha / 2 * n)]
    hi = meds[min(n - 1, int((1 - alpha / 2) * n))]
    return (lo, hi)


def compare(rows, a, b, metric, workload, target, lower_is_better=True):
    sel = [r for r in rows if ok(r) and r.get("workload") == workload
           and r.get("variant") in (a, b)]
    ns_values = sorted({r["Ns"] for r in sel})
    print(f"\n=== {workload}: {b} vs {a}   metric={metric} ===")
    verdicts = []
    for Ns in ns_values:
        ra = sorted([r for r in sel if r["Ns"] == Ns and r["variant"] == a],
                    key=lambda r: r.get("rep", 0))
        rb = sorted([r for r in sel if r["Ns"] == Ns and r["variant"] == b],
                    key=lambda r: r.get("rep", 0))
        if not ra or not rb:
            continue

        # accuracy gate first
        acc_a = {accuracy_key(r) for r in ra}
        acc_b = {accuracy_key(r) for r in rb}
        conv = all(r.get("converged") for r in ra + rb)
        area = [r.get("contact_area") for r in ra + rb]
        area_spread = (max(area) - min(area)) / max(abs(max(area)), 1e-300)
        if not conv:
            print(f"  Ns={Ns:6d}  DISQUALIFIED: not every run converged "
                  f"({[r.get('status') for r in ra + rb]})")
            verdicts.append(False)
            continue
        if acc_a != acc_b:
            print(f"  Ns={Ns:6d}  DISQUALIFIED: different accuracy contract\n"
                  f"            {a}: {acc_a}\n            {b}: {acc_b}")
            verdicts.append(False)
            continue

        n = min(len(ra), len(rb))
        va = [ra[i][metric] for i in range(n)]
        vb = [rb[i][metric] for i in range(n)]
        # paired relative change of b against a
        diffs = [(vb[i] - va[i]) / va[i] for i in range(n)]
        med = st.median(diffs)
        lo, hi = bootstrap_ci(diffs)
        gain = -med if lower_is_better else med
        gain_lo = -hi if lower_is_better else lo
        favourable = gain_lo > 0.0
        meets = gain >= target
        print(f"  Ns={Ns:6d}  n={n}  {a}: {st.median(va):9.4g}   "
              f"{b}: {st.median(vb):9.4g}   "
              f"change {100 * med:+7.2f}%  CI[{100 * lo:+.2f},{100 * hi:+.2f}]%"
              f"  area spread {area_spread:.1e}")
        if n < 5:
            print(f"            (only {n} paired samples; the plan asks for "
                  ">=5 — treat the interval as indicative)")
        verdicts.append(favourable and meets)
        if not favourable:
            print("            CI does not exclude zero in its favour")
        elif not meets:
            print(f"            improvement {100 * gain:.1f}% below the "
                  f"{100 * target:.0f}% target for a new default")
        if gain < -0.05:
            print(f"            REGRESSION beyond 5%: {100 * gain:.1f}%")
    print(f"  -> promote as default: "
          f"{'YES' if verdicts and all(verdicts) else 'NO'}")
    return verdicts


def summary(rows):
    print(f"{'workload':20s} {'Ns':>6s} {'variant':16s} {'rep':>3s} "
          f"{'status':10s} {'wall_s':>9s} {'RSS GiB':>8s} {'it':>5s} "
          f"{'fw_error':>10s}")
    for r in sorted(rows, key=lambda r: (r.get("workload", ""), r.get("Ns", 0),
                                         r.get("variant", ""),
                                         r.get("rep", 0))):
        if not ok(r):
            print(f"{r.get('workload','?'):20s} {r.get('Ns',0):6d} "
                  f"{r.get('variant','?'):16s} {r.get('rep',0):3d} "
                  f"{r.get('run_status','?'):10s}")
            continue
        print(f"{r['workload']:20s} {r['Ns']:6d} {r['variant']:16s} "
              f"{r.get('rep',0):3d} {r['status']:10s} "
              f"{r['wall_cold_s']:9.3f} {r['peak_rss_gib']:8.2f} "
              f"{r['iterations']:5d} {r['fw_error']:10.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default=DEFAULT_LEDGER)
    ap.add_argument("--pair", help="baseline,candidate")
    ap.add_argument("--workload", default="rough-H0.8")
    ap.add_argument("--metric", default="wall_cold_s")
    ap.add_argument("--target", type=float, default=0.10,
                    help="median improvement required to become a default")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    rows = load(a.ledger)
    if a.summary or not a.pair:
        summary(rows)
        return
    base, cand = a.pair.split(",")
    compare(rows, base, cand, a.metric, a.workload, a.target)


if __name__ == "__main__":
    main()
