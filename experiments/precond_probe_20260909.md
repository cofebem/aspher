# Preconditioner probes, 2026-09-09 — raw records

Evidence behind `doc/specs/2026-09-09-stencil-preconditioner-design.md`.
Revision `77c4296`, pocketfft, 20 threads, `OMP_NUM_THREADS=20`,
`OPENBLAS_NUM_THREADS=1`, rough-H0.8 (rfgen 0.2.2, seed 42, H=0.8, rms=0.02,
k_low=12/Ns, k_high=0.33), `coarsest=64`, `q=4`, `leaf_side=16`, `tol=1e-8`.

Machine is a shared desktop; single reps, so ratios are more reliable than
absolutes. Every row below reached `status == converged` unless stated.

## P1 — preconditioner on/off, nested active-set path

Per-row JSON in `data/precond_probe.jsonl`. Wall seconds (iterations):

| Ns | f32 on | f32 off | f64 on | f64 off |
|---|---|---|---|---|
| 1024 | 0.33 (11) | 0.35 (23) | 0.51 (15) | 0.67 (31) |
| 2048 | 0.68 (14) | 0.58 (34) | 1.13 (19) | 0.95 (43) |
| 4096 | 2.19 (23) | 1.76 (80) | 3.87 (27) | 3.17 (92) |
| 8192 | 11.26 (39) | 7.83 (152) | 22.96 (43) | 21.21 (166) |
| 16384 | 68.26 (47) | 35.03 (213) | 183.99 (72) | 78.80 (233) |

Contact area agreed exactly within every pair.

Occupancy sweep, Ns=2048, float, nested active-set path:

| p_bar | contact | on | off |
|---|---|---|---|
| 0.0005 | 0.025% | 0.45 (12) | 0.24 (22) |
| 0.002 | 0.092% | 0.68 (14) | 0.58 (34) |
| 0.01 | 0.41% | 1.20 (27) | 1.26 (48) |
| 0.05 | 1.97% | 5.07 (80) | 7.08 (148) |
| 0.15 | 5.63% | 30.06 (241) | 35.63 (324) |

## P2 — band-limited symbol (throwaway `HMC_PRECOND_KCUT` build)

Ns=1024, **single-level** `ContactSolver(backend="h2", q=6)`, p_bar=0.002,
`max_iter=20000`. Symbol zeroed above `k_max/m`:

| m | iterations | status | fw_error |
|---|---|---|---|
| full | 27 | converged | 6.40e-09 |
| 2 | 3985 | stagnated | 9.88e-05 |
| 4 | 2261 | stagnated | 1.22e-03 |
| 8 | 2222 | stagnated | 4.63e-03 |
| 16 | 3437 | stagnated | 8.47e-03 |

`precond="none"` on the same case: 34 iterations, converged, every time.

## P3 — real-space truncated kernel (throwaway `HMC_PRECOND_RTRUNC` build)

Symbol replaced by that of `w` truncated to the disc `dx^2+dy^2 <= R^2`.
All **single-level**, `max_iter=4000`.

Ns=1024, p_bar=0.002 (0.11% contact):

| | full | R=1 | R=2 | R=4 | R=8 | R=16 | R=32 | R=64 |
|---|---|---|---|---|---|---|---|---|
| mass kept | 100% | 74.56% | 83.92% | 91.08% | 95.25% | 97.52% | 98.74% | 99.36% |
| iterations | 27 | 27 | 25 | 26 | 26 | 27 | 27 | 27 |

Other regimes:

| case | contact | full | R=1 | R=2 | R=4 | R=8 |
|---|---|---|---|---|---|---|
| Ns=2048, p=0.002 | 0.092% | 33 | 36 | 33 | 34 | — |
| Ns=1024, p=0.05 | 2.45% | 53 | 60 | 60 | 56 | — |
| Ns=512, p=0.15 | 8.70% | 78 | 83 | 83 | 77 | 81 |
| Ns=512, p=0.5 | 24.4% | 221 | 196 | 198 | 206 | 215 |

Contact area identical to the full-symbol run in every case.

Truncated-symbol min/max over non-DC modes (numpy, independent of the C++
build), showing the truncated preconditioner is *flatter* than `|k|`:

| Ns | R=1 | R=2 | R=3 | R=4 | R=8 | untruncated |
|---|---|---|---|---|---|---|
| 512 | 0.16044 | 0.13707 | 0.07050 | 0.06475 | 0.03064 | 2.8e-3 |
| 2048 | 0.16043 | 0.13706 | 0.07047 | 0.06472 | 0.03058 | 6.9e-4 |

## P4 — weight universality (numpy)

`w = ifft2(|k|)`, normalised by `w(0,0)`; converged to six figures by Ns=512:

| Ns | (1,0) | (1,1) | (2,0) | (2,1) | (2,2) | (3,0) | (4,0) | w(0,0)/Ns |
|---|---|---|---|---|---|---|---|---|
| 128 | -0.180889 | -0.032862 | 0.021211 | -0.003910 | -0.003198 | -0.015496 | 0.006300 | 0.382616 |
| 512 | -0.180876 | -0.032860 | 0.021190 | -0.003912 | -0.003198 | -0.015474 | 0.006279 | 0.382599 |
| 4096 | -0.180875 | -0.032860 | 0.021189 | -0.003912 | -0.003198 | -0.015473 | 0.006277 | 0.382598 |

## Note: a crash worth tracing

The band-limited build (P2) aborted the *nested/active* path with
`Eigen ... DenseCoeffsBase<Derived,1>::operator()(Index)` index assertion on a
`Matrix<float,-1,1>` — an out-of-range index, not a convergence failure. The
single-level path only converged slowly on the same symbol. Trigger was a
deliberately degenerate preconditioner, so it may be unreachable in practice.
Recorded as an open item in the spec.
