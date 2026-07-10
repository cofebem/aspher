# Active-set localization — prototype results and go/no-go

**Script**: `active_set_proto.py`. Restricted Polonsky–Keer PCG (full algorithm
— load constraint, implicit α, PR+, overlap correction on the candidate set)
with matvecs emulated by the full-grid C++ FFT backend; iteration counts and
correctness are the measurables, Python wall time is not. Self-affine surface
(H=0.8 nominal, rms 0.02, seed 42, spectrum to the fine Nyquist so coarse
grids cannot see the smallest asperities). tol 1e-8 everywhere.

## Q1 — Conditioning: restriction does NOT replace the preconditioner

Iterations, candidate set C = dilate_k(true active set):

| config | global none | global \|q\| | restricted none | restricted \|q\| | restricted capped qc=32 |
|---|---|---|---|---|---|
| Ns=1024, p̄=0.002 (N_c 0.76%, 362 islands) | 132 | 64 | 121–132 (k=1..4) | 56–57 | 104–110 |
| Ns=1024, p̄=0.01 (3.8%, 1241 islands) | 381 | 136 | 377 | 130 | 237 |
| Ns=2048, p̄=0.002 (0.49%, 996 islands) | 250 | 80 | 247 | 65 | 185 |

- Restricted-unpreconditioned ≈ global-unpreconditioned at every size/load:
  the contact islands span the whole domain, so the smooth 1/|q| inter-patch
  modes keep their domain-scale conditioning. The hypothesis "restriction
  shrinks κ to patch extent" is **refuted**.
- The |q| preconditioner keeps its full ~2–3× iteration advantage on the
  restricted solve, and its advantage *grows* with Ns (none: ×1.9 per grid
  doubling; fourier: ×1.25).
- The reduced-frequency preconditioner (capped symbol min(|q|,qc), the
  Kochmann/Gierden–Reese reduced-wave-vector idea, arXiv:2103.10203) lands in
  between and scales like the unpreconditioned arm — the |q| rescaling matters
  across the whole band, not just the smooth end. In its simple capped form:
  not worth the complexity. (A per-island-local + low-band-global split is
  possible future work; not needed for the go decision below.)

## Q2 — Discovery: gap-proximity halos are mandatory and sufficient

Candidate set from one coarser level (Ns/2): dilate₂(prolonged coarse contact)
∪ {prolonged coarse gap < δ}; verify-and-extend with a full-grid gap scan.

- **Orphan islands — fine contact patches with zero coarse-contact precursor —
  are ~30% of all islands in every config** (108/362, 380/1241, 313/996).
  Geometric dilation of coarse contact alone would miss all of them; the gap
  threshold catches them (physics: an asperity invisible on the coarse grid
  still sits at small coarse gap). This confirmed VY's prediction for rich
  spectra.
- δ = 0.03·rms (dilute) / 0.10·rms (p̄=0.01): **zero missed points, 1
  verification round, exact recovery** (rel-L2 vs reference ≈ solver tol),
  with |C| ≈ 2.2–5.3 N_c. δ = 0.01·rms is too tight: a few missed boundary
  points let verification pass (no negative gap beyond tolerance) while the
  pressure carries 0.5–0.9% error — **the certificate can be clean while the
  active set is subtly wrong**, so δ must be generous and should scale with
  load. Warm-starting from the prolonged coarse pressure also cuts ~15% of
  iterations.

## Q3 — Cost model at Ns=16384 (measured baseline: 1168 s / 38 it / ~24 GiB)

N_c ≈ 124k (4.6e-4 N). Per-iteration ingredients (double, this machine):
full H2 matvec ~5 s; masked H2 matvec est. 1–3% of full (occupied leaves +
ancestors; **the biggest unknown — measure first in C++**) ≈ 0.05–0.15 s;
full-FFT |q| precond ≈ 1–1.5 s; O(N_c) vector work negligible; verification =
1 full matvec + gap pass ≈ 5 s × 1–2 rounds.

Iteration extrapolation (warm-started, from the measured scaling):
restricted-|q| ≈ 40–120; restricted-none ≈ 700–1700 (×1.9/doubling continues).

| variant | projected solve | projected memory |
|---|---|---|
| current nested (measured) | 1168 s | ~24 GiB |
| restricted + full-FFT \|q\| precond | 40–120 × (0.15+1.3) s ≈ **60–175 s** | gap 2.1 + H2 0.5 + precond 4.3 + O(N_c) ≈ **7 GiB** |
| restricted, no precond | 700–1700 × 0.15 s ≈ **105–255 s** | ≈ **3 GiB** |

Both variants project **5–20× wall-time** and **3–8× memory** improvements;
the no-precond variant would even put Ns=32768 double (~4× memory) within
reach of this 31 GiB machine, at the price of √Ns iteration growth.

## Verdict: GO for a C++ masked-H2 implementation

Design sketch (next phase):
1. `H2Operator::matvec_masked` — per-level occupancy bitmaps from the active
   leaves (UniformQuadTree index ranges make this cheap): P2M/L2P/near only on
   occupied leaves, M2M/M2L/L2L only on boxes whose subtree is occupied.
   **Milestone 1: measure the masked-matvec constant** (the model's main
   unknown) before building the rest.
2. Active-set PK driver: candidate mask from the nested cascade (coarse
   contact ∪ gap<δ, δ ≈ 0.05–0.1·rms scaled with load), restricted CG vectors
   (index list, O(N_c) state), full-grid verification via one unmasked matvec
   per round, keep the existing |q| preconditioner (applied full-grid; drop it
   only in a memory-max mode).
3. The existing nested continuation stays as the outer loop — it already
   provides the warm start and the coarse gap field the candidate set needs.

Caveats: iteration counts here are Python-emulated (exact algorithm, exact
operator) — trustworthy; wall-time projections are modeled — not. The
certificate subtlety in Q2 (clean verification with slightly-wrong boundary
set at tight δ) must be respected in the C++ driver: generous δ, and treat
near-zero positive gaps outside C as suspects, not certified.
