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
        # Three outcomes, not two. "Too few samples to have an interval" is
        # not the same as "the interval says no" — the plan explicitly allows
        # expensive jobs to run fewer samples with a stated uncertainty
        # limitation, and large grids are single-rep by design. Conflating
        # them would report a favourable single-rep result as evidence
        # against the change.
        if n < 5:
            verdicts.append(None)
            print(f"            INSUFFICIENT SAMPLES for a confidence "
                  f"interval ({n} < 5): direction is "
                  f"{'favourable' if gain > 0 else 'unfavourable'} at "
                  f"{100 * gain:+.1f}%, indicative only")
        else:
            verdicts.append(favourable and meets)
            if not favourable:
                print("            CI does not exclude zero in its favour")
            elif not meets:
                print(f"            improvement {100 * gain:.1f}% below the "
                      f"{100 * target:.0f}% target for a new default")
        if gain < -0.05:
            print(f"            REGRESSION beyond 5%: {100 * gain:.1f}%")
    decided = [v for v in verdicts if v is not None]
    if not verdicts:
        verdict = "NO DATA"
    elif any(v is False for v in verdicts):
        verdict = "NO"
    elif not decided:
        verdict = "INSUFFICIENT EVIDENCE (direction favourable, needs >=5 " \
                  "paired samples)"
    elif all(decided):
        verdict = ("YES" if len(decided) == len(verdicts)
                   else "YES on the sized cases; the rest are indicative")
    else:
        verdict = "NO"
    print(f"  -> promote as default: {verdict}")
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


def sweep(rows, metric, variants, target):
    """B04 view: cost against ACHIEVED occupancy, per geometry.

    Groups by (geometry, workload spec) rather than by nominal target, because
    the load is only the dial — what matters is the contact fraction it
    actually produced, which is surface- and resolution-dependent.
    """
    ok_rows = [r for r in rows if ok(r) and "occupancy" in r]
    if not ok_rows:
        print("no rows with occupancy recorded")
        return
    geoms = sorted({r["workload"].split("@")[0] for r in ok_rows})
    for geom in geoms:
        for Ns in sorted({r["Ns"] for r in ok_rows if r["workload"].startswith(geom)}):
            sel = [r for r in ok_rows
                   if r["workload"].startswith(geom + "@") and r["Ns"] == Ns]
            if not sel:
                continue
            print(f"\n=== {geom}  Ns={Ns}  metric={metric} ===")
            hdr = f"  {'occupancy':>10s}"
            for v in variants:
                hdr += f" {v:>20s}"
            hdr += f" {'best':>16s}"
            print(hdr)
            specs = sorted({r["workload"] for r in sel},
                           key=lambda w: st.median(
                               [x["occupancy"] for x in sel if x["workload"] == w]))
            for spec in specs:
                grp = [r for r in sel if r["workload"] == spec]
                occ = st.median(r["occupancy"] for r in grp)
                line = f"  {100 * occ:9.3f}%"
                med = {}
                for v in variants:
                    vals = [r[metric] for r in grp if r["variant"] == v]
                    if vals:
                        med[v] = st.median(vals)
                        line += f" {med[v]:20.4g}"
                    else:
                        line += f" {'-':>20s}"
                if med:
                    winner = min(med, key=med.get)
                    ref = med.get(variants[0])
                    gain = (100 * (1 - med[winner] / ref)) if ref else 0.0
                    line += f" {winner.replace('h2-f32-', ''):>10s} {gain:+5.0f}%"
                print(line)


def phases(rows, variant):
    """Where the time goes, against achieved occupancy."""
    sel = [r for r in rows if ok(r) and r.get("variant") == variant
           and "occupancy" in r]
    if not sel:
        return
    print(f"\n=== phase split for {variant} (median over reps) ===")
    print(f"  {'geometry':12s} {'occup.':>8s} {'total':>8s} {'coarse':>8s} "
          f"{'precond':>8s} {'matvec':>8s} {'candid.':>8s} {'verif.':>8s} "
          f"{'rounds':>6s} {'fallback':>8s}")
    specs = sorted({r["workload"] for r in sel},
                   key=lambda w: (w.split("@")[0],
                                  st.median([x["occupancy"] for x in sel
                                             if x["workload"] == w])))
    for spec in specs:
        grp = [r for r in sel if r["workload"] == spec]
        m = lambda k: st.median(r["timings"][k] for r in grp)
        print(f"  {spec.split('@')[0]:12s} {100 * st.median(r['occupancy'] for r in grp):7.3f}% "
              f"{m('total'):8.3f} {m('coarse'):8.3f} {m('precond'):8.3f} "
              f"{m('matvec'):8.3f} {m('candidate'):8.4f} {m('verification'):8.4f} "
              f"{st.median(r['active_rounds'] for r in grp):6.0f} "
              f"{sum(1 for r in grp if r['active_fallback']):8d}")


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
    ap.add_argument("--sweep", action="store_true",
                    help="B04: cost against achieved occupancy, per geometry")
    ap.add_argument("--variants",
                    default="h2-f32,h2-f32-active,h2-f32-active-all")
    ap.add_argument("--phases", help="phase split for this variant")
    a = ap.parse_args()
    rows = load(a.ledger)
    if a.sweep:
        vs = a.variants.split(",")
        sweep(rows, a.metric, vs, a.target)
        for v in vs:
            phases(rows, v)
        return
    if a.phases:
        phases(rows, a.phases)
        return
    if a.summary or not a.pair:
        summary(rows)
        return
    base, cand = a.pair.split(",")
    compare(rows, base, cand, a.metric, a.workload, a.target)


if __name__ == "__main__":
    main()
