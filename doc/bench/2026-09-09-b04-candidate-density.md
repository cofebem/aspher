# B04 — candidate density and preconditioner cost

Settles the promotion question left open by
[the A11 measurement](2026-09-09-a11-all-levels-restriction.md): the
all-levels restriction was measured favourable on one surface at one load
(0.045% occupancy), and the open risk was that it inverts as contact becomes
dense. Ledgers: `data/bench_b04.jsonl`, `data/bench_b04_tight.jsonl`.

Conditions: revision `0394bf5`+, pocketfft, 20 threads, float, `coarsest=64`,
`q=6`, `tol=1e-8`, five paired reps at Ns=1024 and three at Ns=2048, ABBA
order. Occupancy is reached through the load and the **achieved** contact
fraction is what is reported — the mapping is surface- and
resolution-dependent (the same load gives 1.005% / 0.768% / 0.610% at
Ns=512/1024/2048 on the self-affine surface).

## Wall time against achieved occupancy

Ns=1024, five paired reps, median seconds:

| occupancy | standard | active | active-all | best |
|---|---|---|---|---|
| **Hertz — clustered** | | | | |
| 0.11% | 0.227 | 0.154 | **0.144** | −36% |
| 0.95% | 0.245 | 0.160 | **0.136** | −45% |
| 10.4% | 0.270 | 0.179 | **0.164** | −39% |
| 48.2% | 0.287 | **0.251** | 0.280 | −13% |
| 100% | **0.778** | 1.049 | 0.920 | +18% *worse* |
| **Self-affine — fragmented** | | | | |
| 0.11% | 0.459 | 0.283 | **0.171** | −63% |
| 1.01% | 0.773 | 0.362 | **0.337** | −56% |
| 11.0% | 4.78 | 3.33 | **3.24** | −32% |
| 47.0% | **17.4** | 17.6 | 18.0 | +3.6% *worse* |
| 99.0% | **2.42** | 9.07 | 14.2 | **5.9× worse** |

Ns=2048, three paired reps:

| occupancy | standard | active | active-all | best |
|---|---|---|---|---|
| Hertz 0.95% | 0.680 | 0.354 | **0.332** | −51% |
| Hertz 10.4% | 0.846 | 0.455 | **0.407** | −52% |
| Hertz 48.2% | 0.961 | 0.817 | **0.816** | −15% |
| rough 0.81% | 5.44 | 1.63 | **1.18** | −78% |
| rough 9.06% | 44.5 | 25.6 | **23.7** | −47% |
| rough 41.2% | 119.4 | **115.8** | 116.8 | −3% |

**The benefit grows with resolution and the crossover moves up with it.** At
Ns=1024 the restriction is neutral by ~50% occupancy; at Ns=2048 it is still
15% ahead at 48% and neutral-to-favourable at 41%. Higher resolution means
more absolute work per level, so the restriction's fixed overhead is a smaller
share and the coarse cascade a larger one.

## Where the penalty comes from — not where expected

The hypothesis going in was that candidate-set *construction* would be the
overhead that inverts at high occupancy. It is not. `time_candidate` — mask
build, slot mapping and compressed gather, summed over all rounds and levels —
is **3–9 ms everywhere, flat across four decades of occupancy**:

| geometry | occupancy | total | matvec | candidate | verification |
|---|---|---|---|---|---|
| Hertz | 0.95% | 0.135 | 0.011 | 0.0038 | 0.0019 |
| Hertz | 100% | 0.917 | 0.584 | 0.0079 | 0.0081 |
| rough | 0.11% | 0.169 | 0.013 | 0.0033 | 0.0021 |
| rough | 99.0% | 14.21 | **11.30** | 0.0084 | 0.0107 |

The penalty is the **masked matvec**: at 99% rough occupancy it costs 11.3 s
against the unmasked path's 1.77 s, 6.4× slower. At full occupancy the mask
skips nothing, so the per-box occupancy guards in `matvec_masked_impl` and the
compressed gather/scatter indirection are pure overhead on every pass. If the
high-occupancy end is ever wanted, that is where the work is — not in how the
candidate set is built.

A related observation: the standard path is *faster* at 99% rough occupancy
than at 47% (2.42 vs 17.4 s). Near-full contact is nearly unconstrained, so
the active set stops churning and CG converges quickly; the restricted path
gets no such benefit because it still pays full masked-matvec cost.

## Memory

Same shape, smaller amplitude: −17% to −23% peak RSS below 10% occupancy
(Ns=1024), parity to slightly worse above. Memory is not the reason to choose
either path.

## Evolving C — VOID, the arm never ran the intended configuration

**Retracted 2026-09-09.** The `h2-f32-active-tight` variant sets
`active_delta=0.0`, but `active_delta` was not in the harness's allowlist of
variant keys forwarded to the solver, so it was **silently dropped** and every
"tight" run used the default `active_delta=0.05`. The ledger rows show it:
their `solver_args` carry `active_halo=0` and `active_max_rounds=8` and no
`active_delta` at all. Only the halo was actually tightened.

So the original conclusion — that "the candidate set taken from the prolonged
coarse pressure alone is already adequate", inferred from seeing
`active_rounds=1` everywhere — **is not supported by these runs**. The δ margin
that would have had to be absent was present throughout. Nothing here says
anything about a starved candidate set.

The harness bug is fixed (the allowlist now carries `active_delta`,
`active_occupancy_max` and the preconditioner keys, with a comment saying that
an unlisted key is silently dropped). The evolving-C measurement itself is
**still open** and needs re-running. The flat 3–9 ms `time_candidate` across
the whole sweep remains valid and is weak evidence that candidate construction
would not dominate even under expansion, since it never exceeds 0.3% of any
run — but that is an argument, not the measurement.

Nothing else in this document depends on the tight arm: the occupancy sweep,
the crossover, the masked-matvec attribution and the gate all come from the
`h2-f32-active` / `h2-f32-active-all` / standard arms, whose keys were all
forwarded correctly.

## Outcome: implemented as the default (2026-09-09)

The rule below is now in `solve_contact_nested` as `active_occupancy_max=0.4`,
with `active_all_levels` defaulting to true. Measured at Ns=1024 on the
self-affine surface with the gate active, median of 3:

| occupancy | levels restricted | standard | gated | ratio |
|---|---|---|---|---|
| 0.11% | 4/5 | 0.502 s | 0.222 s | **2.26×** |
| 11.0% | 4/5 | 4.333 s | 3.211 s | **1.35×** |
| 47.0% | **0/5** | 18.68 s | 18.57 s | 1.01× |
| 99.0% | **0/5** | 2.589 s | 2.600 s | 1.00× |

All of the upside below the threshold, and exact parity above it. One
refinement was needed to reach parity: a level whose successor the gate will
close no longer keeps its gap field, since keeping it forces the level off the
light path and materialises two N-sized arrays nothing will read. That alone
was a 19% penalty at 47% occupancy — the gate closed correctly but the
bookkeeping it implies had not been closed with it.

## Recommendation: a *measured* conditional default

The restriction should be on by default where it wins and off where it loses,
and the crossover is in occupancy — but occupancy is not known before solving.
The nested cascade already measures it: **each coarse level reports its
contact fraction, which predicts the next level's.** So the rule needs no
user input and no a-priori knowledge:

> Restrict a level when the level below it reported a contact fraction under a
> threshold; otherwise solve it in full.

From this data a threshold of **0.4** captures essentially all the benefit
while avoiding the penalty: every measured point below 41% occupancy favours
restriction at both resolutions, the only losses are at 47%+ (Ns=1024) and
they are small until near-full contact, and the one catastrophic case (99%,
5.9×) is far above it. The threshold is conservative at Ns=2048, where 48%
still wins by 15% — deliberately, since the cost of being wrong is asymmetric:
missing a 15% gain versus taking a 6× loss.

Caveats on the recommendation: measured in float only, at Ns=1024 and 2048,
on two geometries, with `coarsest=64` and `q=6`. Double precision showed the
same A11 behaviour at Ns=16384 but was not swept across occupancy here.

## Input to the preconditioner design (A13)

The preconditioner's share of the run, from the same sweep (`active-all` arm):

| geometry | occupancy | total | precond | share |
|---|---|---|---|---|
| rough | 0.11% | 0.169 | 0.020 | 12% |
| rough | 1.01% | 0.334 | 0.039 | 12% |
| rough | 11.0% | 3.240 | 0.310 | 10% |
| rough | 47.0% | 18.00 | 1.514 | 8% |

At Ns=1024 the preconditioner is a steady ~10% of the run regardless of
occupancy, because it is a full-grid FFT whose cost does not depend on the
contact set at all — which is exactly the point. Its *share* rises with grid
size, not with occupancy: at Ns=16384 with the cascade removed it reaches
~79%. So the A13 target regime is **large Ns at any occupancy**, and the
design should be judged at Ns≥8192 rather than on this sweep, where the
absolute FFT cost is still small.
