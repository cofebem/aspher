"""Reads data/backend_precision_study.jsonl and writes plots + a summary
to doc/backend_precision_study/. See
doc/specs/2026-07-10-backend-precision-benchmark-design.md Sec.4/6."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "backend_precision_study.jsonl")
OUT_DIR = os.path.join(HERE, "doc", "backend_precision_study")

COMBOS = [("h2", "double"), ("h2", "float"), ("fft", "double"), ("fft", "float")]
STYLE = {
    ("h2", "double"): dict(color="C0", marker="o", label="H2 double"),
    ("h2", "float"): dict(color="C0", marker="o", linestyle="--", label="H2 float"),
    ("fft", "double"): dict(color="C1", marker="s", label="FFT double"),
    ("fft", "float"): dict(color="C1", marker="s", linestyle="--", label="FFT float"),
}


def load_records():
    records = []
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def group(records):
    by_combo = {c: {} for c in COMBOS}
    failed = []
    for r in records:
        combo = (r.get("backend"), r.get("precision"))
        if r.get("status") != "ok":
            failed.append(r)
            continue
        if combo in by_combo:
            by_combo[combo][r["Ns"]] = r
    return by_combo, failed


def plot_metric(by_combo, key, ylabel, title, fname, logy=True):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for combo in COMBOS:
        pts = sorted(by_combo[combo].items())
        if not pts:
            continue
        xs = [ns for ns, _ in pts]
        ys = [rec[key] for _, rec in pts]
        ax.plot(xs, ys, **STYLE[combo])
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Ns")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close(fig)


def plot_convergence(by_combo, ns_list=(1024, 4096)):
    ns_list = [n for n in ns_list if any(by_combo[c].get(n) for c in COMBOS)]
    if not ns_list:
        return
    fig, axes = plt.subplots(1, len(ns_list), figsize=(5.5 * len(ns_list), 4.2),
                             squeeze=False)
    for ax, ns in zip(axes[0], ns_list):
        for combo in COMBOS:
            rec = by_combo[combo].get(ns)
            if rec is None or not rec.get("error_history"):
                continue
            hist = rec["error_history"]
            ax.plot(range(1, len(hist) + 1), hist, **STYLE[combo])
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("complementarity error")
        ax.set_title(f"Ns={ns}")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_convergence_curves.png"), dpi=150)
    plt.close(fig)


def write_summary(by_combo, failed):
    lines = ["# Backend x precision x Ns benchmark study — summary", ""]
    lines.append("| backend | precision | Ns | wall[s] | RSS[GiB] | iters | "
                 "converged | contact_area |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for combo in COMBOS:
        for ns, rec in sorted(by_combo[combo].items()):
            lines.append(
                f"| {combo[0]} | {combo[1]} | {ns} | {rec['wall_time_s']:.2f} | "
                f"{rec['rss_gib']:.2f} | {rec['iterations']} | "
                f"{rec['converged']} | {rec['contact_area']:.5f} |")
    if failed:
        lines.append("")
        lines.append("## Failed / incomplete cases")
        for r in failed:
            lines.append(f"- {r.get('backend')} {r.get('precision')} "
                         f"Ns={r.get('Ns')}: status={r.get('status')}")
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    records = load_records()
    by_combo, failed = group(records)
    plot_metric(by_combo, "wall_time_s", "wall time [s]",
               "Solve wall time vs Ns", "fig_walltime_vs_ns.png")
    plot_metric(by_combo, "rss_gib", "peak RSS [GiB]",
               "Peak memory vs Ns", "fig_memory_vs_ns.png")
    plot_metric(by_combo, "iterations", "PCG iterations",
               "Iteration count vs Ns", "fig_iterations_vs_ns.png", logy=False)
    plot_convergence(by_combo)
    write_summary(by_combo, failed)
    print(f"wrote plots + summary.md to {OUT_DIR}")
    if failed:
        print(f"{len(failed)} case(s) did not complete:")
        for r in failed:
            print(f"  {r.get('backend')} {r.get('precision')} Ns={r.get('Ns')}: "
                  f"{r.get('status')}")


if __name__ == "__main__":
    main()
