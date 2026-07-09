"""hc.solve_nested single entry point: same solution as a single-grid
preconditioned solve, with fewer (or equal) fine-grid iterations."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
import hmatrix_contact as hc
from precond_pcg import bandlimited_surface

P_BAR = 0.05


def test_solve_nested():
    for Ns in (256, 512):
        surf = bandlimited_surface(Ns)
        g0 = (-surf).ravel()

        # single-grid preconditioned reference
        single = hc.ContactSolver(grid_size=Ns, backend="h2", q=6)
        ref = single.solve(g0, P_BAR, tol=1e-8, precond="fourier")

        # single entry point: coarse->fine internally
        nest = hc.solve_nested(grid_size=Ns, gap=g0, p_nominal=P_BAR,
                               coarsest=64, q=6, tol=1e-8)

        d_area = abs(nest.contact_area - ref.contact_area)
        relL2 = (np.linalg.norm(np.asarray(nest.pressure) - np.asarray(ref.pressure))
                 / np.linalg.norm(np.asarray(ref.pressure)))
        print(f"Ns={Ns}: single(fourier)={ref.iterations} it, "
              f"nested fine={nest.iterations} it, dArea={d_area:.2e}, relL2={relL2:.1e}")
        assert nest.converged
        assert d_area < 1e-3, d_area
        assert relL2 < 1e-4, relL2
        assert nest.iterations <= ref.iterations

        # single precision + light result: converges and matches to float accuracy
        sp = hc.solve_nested(grid_size=Ns, gap=g0, p_nominal=P_BAR, coarsest=64,
                             q=6, single_precision=True, light_result=True)
        d_area_sp = abs(sp.contact_area - ref.contact_area)
        relL2_sp = (np.linalg.norm(np.asarray(sp.pressure) - np.asarray(ref.pressure))
                    / np.linalg.norm(np.asarray(ref.pressure)))
        print(f"Ns={Ns}: single+light={sp.iterations} it, conv={sp.converged}, "
              f"dArea={d_area_sp:.2e}, relL2={relL2_sp:.1e}, "
              f"disp_none={sp.displacement is None}")
        assert sp.converged           # reaches the float floor via stagnation guard
        assert d_area_sp < 2e-3, d_area_sp
        assert relL2_sp < 1e-3, relL2_sp
        assert sp.displacement is None  # light_result: displacement not stored

        # opt-in convergence history: off by default, populated on request
        assert nest.error_history.size == 0
        hist = hc.solve_nested(grid_size=Ns, gap=g0, p_nominal=P_BAR,
                               coarsest=64, q=6, tol=1e-8,
                               record_error_history=True)
        assert hist.error_history.size > 0
        assert abs(float(hist.error_history[-1]) - hist.error) < 1e-9, (
            hist.error_history[-1], hist.error)
        print(f"Ns={Ns}: error_history has {hist.error_history.size} entries, "
              f"last={hist.error_history[-1]:.3e}")


if __name__ == "__main__":
    test_solve_nested()
    print("OK")
