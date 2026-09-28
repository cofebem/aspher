"""Validate Aspher's periodic backend against Tamaas on one contact case.

Run in an environment containing Tamaas, with Aspher's development module on
PYTHONPATH, for example:

    PYTHONPATH=python conda run -n fluidpaper python compare_tamaas_periodic.py
"""

from __future__ import annotations

import argparse

import numpy as np
import tamaas as tm

import aspher


def periodic_surface(n: int) -> np.ndarray:
    x = np.arange(n) / n
    xx, yy = np.meshgrid(x, x, indexing="xy")
    surface = (
        np.cos(2 * np.pi * xx)
        + 0.37 * np.cos(4 * np.pi * yy + 0.2)
        + 0.19 * np.cos(2 * np.pi * (3 * xx + 2 * yy) + 0.7)
    )
    surface -= surface.mean()
    surface *= 0.01 / surface.std()
    surface -= surface.max()
    return np.ascontiguousarray(surface)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--pressure", type=float, default=0.05)
    parser.add_argument("--tol", type=float, default=1e-10)
    args = parser.parse_args()

    surface = periodic_surface(args.grid)
    model = tm.ModelFactory.createModel(
        tm.model_type.basic_2d, [1.0, 1.0], [args.grid, args.grid]
    )
    model.E = 1.0
    model.nu = 0.0
    tamaas_solver = tm.PolonskyKeerRey(model, surface, args.tol)
    tamaas_solver.max_iter = 10000
    # No dcfft override: Tamaas's default operator is the periodic reference.
    tamaas_solver.solve(args.pressure)
    p_tamaas = np.asarray(model.traction, dtype=float)

    solver = aspher.ContactSolver(
        args.grid, domain_size=1.0, E_star=1.0, backend="periodic"
    )
    result = solver.solve(
        -surface,
        args.pressure,
        tol=args.tol,
        max_iter=10000,
        precond="fourier-fft",
    )
    p_aspher = np.asarray(result.pressure)
    relative_l2 = np.linalg.norm(p_aspher - p_tamaas) / np.linalg.norm(p_tamaas)
    area_tamaas = float((p_tamaas > 1e-12).mean())
    area_aspher = float((p_aspher > 1e-12).mean())
    print(f"Tamaas contact fraction: {area_tamaas:.12f}")
    print(f"Aspher contact fraction: {area_aspher:.12f}")
    print(f"pressure relative L2:    {relative_l2:.6e}")
    assert result.converged
    assert area_aspher == area_tamaas
    assert relative_l2 < 1e-8
    print("PASS: periodic Aspher agrees with Tamaas")


if __name__ == "__main__":
    main()
