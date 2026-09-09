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

        # A09 precision policy. Float cannot drive the certificate to the
        # default 1e-8, so a float-only solve that is ASKED for it now reports
        # an honest `stagnated`/`precision_limit` instead of a relaxed
        # success. This is the documented behaviour change.
        sp = hc.solve_nested(grid_size=Ns, gap=g0, p_nominal=P_BAR, coarsest=64,
                             q=6, single_precision=True, light_result=True)
        print(f"Ns={Ns}: float-only asked for {sp.requested_tol:.0e} -> "
              f"{sp.status}/{sp.status_reason} at {sp.effective_tol:.0e}")
        assert not sp.converged
        assert sp.status == "stagnated" and sp.status_reason == "precision_limit"
        assert sp.requested_tol < sp.effective_tol

        # ... and succeeds when the caller explicitly accepts the float floor
        spr = hc.solve_nested(grid_size=Ns, gap=g0, p_nominal=P_BAR, coarsest=64,
                              q=6, single_precision=True, light_result=True,
                              allow_tolerance_relaxation=True)
        d_area_sp = abs(spr.contact_area - ref.contact_area)
        relL2_sp = (np.linalg.norm(np.asarray(spr.pressure) - np.asarray(ref.pressure))
                    / np.linalg.norm(np.asarray(ref.pressure)))
        print(f"Ns={Ns}: float+relaxed={spr.iterations} it, conv={spr.converged}, "
              f"dArea={d_area_sp:.2e}, relL2={relL2_sp:.1e}, "
              f"disp_none={spr.displacement is None}")
        assert spr.converged
        assert d_area_sp < 2e-3, d_area_sp
        assert relL2_sp < 1e-3, relL2_sp
        assert spr.displacement is None  # light_result: displacement not stored

        # float_then_double: identify in float, polish in double. Meets the
        # requested tolerance AND is markedly more accurate than float alone.
        ftd = hc.solve_nested(grid_size=Ns, gap=g0, p_nominal=P_BAR, coarsest=64,
                              q=6, precision="float_then_double")
        relL2_ftd = (np.linalg.norm(np.asarray(ftd.pressure) - np.asarray(ref.pressure))
                     / np.linalg.norm(np.asarray(ref.pressure)))
        names = [st["name"] for st in ftd.stage_stats]
        print(f"Ns={Ns}: float_then_double={ftd.iterations} it, "
              f"relL2={relL2_ftd:.1e}, stages={names[-2:]}")
        assert ftd.converged and ftd.status == "converged"
        assert ftd.fw_error <= ftd.requested_tol
        assert relL2_ftd < relL2_sp   # strictly better than the float floor
        assert names[-1].startswith("polish:")
        assert ftd.stage_stats[-2]["precision"] == "float"
        assert ftd.stage_stats[-1]["precision"] == "double"

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
