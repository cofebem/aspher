# A11: restricting every level, not only the finest — measured outcome

Companion to [the Ns=16384 rebaseline](2026-09-09-ns16384-rebaseline.md), which
identified the target. Ledgers: `data/bench_ledger.jsonl` (pre-fix) and
`data/bench_ledger_memfix.jsonl` (at the fixed revision `61b377a`).

## Where it came from

The rebaseline's per-level `stage_stats` showed the cascade cost is one level:
at Ns=16384 the coarse 8192 solve was **41.9% of the whole run in 16
iterations**, while the six levels below 2048 totalled 2%. The cause was that
`active_set` applied only to the finest level, so 8192 ran an unrestricted
full-grid solve and paid exactly the matvec cost the finest level no longer
pays.

Nothing in the restricted solver was ever finest-specific — it needs this
level's gap, the prolonged pressure from below, and that level's gap field —
so `active_finest` became `active_level`, called for any level with a coarser
one beneath it. The coarsest always solves in full.

## Result

Contact area is identical in every pairing (spread exactly 0), and the finest
level is untouched: the entire saving is the coarse cascade.

| pairing | metric | finest-only | all-levels | change |
|---|---|---|---|---|
| Ns=2048 f32, **5 paired samples** | wall | 0.730 s | 0.483 s | **−37.7%**, CI [−44.5, −22.3] |
| Ns=2048 f32, 5 samples | peak RSS | 0.2468 GiB | 0.2406 GiB | −2.6%, CI [−2.8, −2.1] |
| Ns=16384 f32, 1 sample | wall | 128.0 s | 66.98 s | −47.7% (indicative) |
| Ns=16384 f64, 1 sample | wall | 313.7 s | 179.2 s | −42.9% (indicative) |
| Ns=16384 f32, 1 sample | peak RSS | 9.116 GiB | 9.116 GiB | −0.01% |
| Ns=16384 f64, 1 sample | peak RSS | 9.314 GiB | 9.313 GiB | −0.01% |

Per-level mechanism at Ns=16384 (float), which is the whole story:

| stage | finest-only | all-levels | |
|---|---|---|---|
| coarse 4096 | 9.66 s (15 it) | **0.92 s (17 it)** | 10.5× |
| coarse 8192 | 47.95 s (16 it) | **3.98 s (19 it)** | 12× |
| finest 16384 | 54.90 s (47 it) | 55.24 s (47 it) | unchanged |
| total | 119.7 s | **65.1 s** | −45.6% |

The restricted coarse levels take slightly *more* iterations — the restriction
changes the iterate path — and still cost a twelfth as much. The gain is
precision-independent (−47.7% float, −42.9% double).

## A regression the paired A/B caught

The first all-levels measurement showed **+7.2% peak RSS** on the double pair
(9.98 vs 9.31 GiB). The restricted branch never released `gap[li]`, so every
level's double gap stayed resident for the whole cascade: Σ 8N/4ᵏ = 8N/3 =
0.71 GiB at Ns=16384, matching the 0.67 GiB observed. The standard branches
have always released it; the restricted one was missed when it was generalised
from finest-only, where a single coarse level made the cost invisible.

The two precisions need the release in different places, which is why one
blanket line would not have worked: the float path copies into `g0f`, so the
double can go as soon as the cast is done (lower peak *during* the solve); the
double path passes `glvl` straight through and `active_level` reads it for its
compressed gather, its streamed verification and its field materialisation, so
it can only be released after the call returns. After the fix, memory is at
parity (−0.01%) with the time gain intact.

## Promotion decision: NOT YET a default

The evidence is favourable and consistent — five-sample statistics at Ns=2048
clear the 10% bar with room, single-sample confirmations at Ns=16384 agree in
direction and magnitude across both precisions, memory is at parity, and the
answer is identical. But it stays **opt-in** (`active_all_levels=false`),
because the sweep covers one surface at one load and two cases could plausibly
turn it negative:

* **Dense contact.** As occupancy rises the candidate set approaches the full
  grid and the restriction becomes pure overhead — mask construction,
  compressed gather/scatter, and a streamed verification pass per round buying
  nothing. This sweep ran at 0.045% occupancy. The crossover is unknown.
* **Small grids.** At Ns=256 with `coarsest=64` there is one restricted coarse
  level and a solve measured in milliseconds; fixed overhead could dominate.
* **Fallback risk.** `active_delta` and `active_halo` were tuned at the finest
  level. A coarse level whose candidate set is too tight triggers the
  full-solve fallback and then costs *more* than today. Every run here reports
  `active_fallback=false`, on one surface.

B04 (candidate-density sweep, occupancy 0.1% / 1% / 10% / 50%, clustered and
fragmented) is the measurement that settles it. The likely outcome is a
*conditional* default — on above some Ns and below some occupancy — which is
the shape the plan anticipates: "If a variant is useful only at high q, dense
occupancy or a particular precision, select it conditionally and publish that
regime."

## What this does to the next target

After the change, the finest level is 85% of the Ns=16384 run and the full-grid
|q| preconditioner is ~51 s of the remaining 65 s — **roughly 79% of the whole
solve**. The two-level preconditioner (A13) went from targeting 41% to
targeting nearly four-fifths of the wall time, and A08/A10 — which optimise the
matvec — now address about 4%.

## Note on the analyzer

`analyze.py` initially reported "promote: NO" for the Ns=16384 pairs purely
because a single sample yields no confidence interval. That conflated "too few
samples to decide" with "the evidence says no", which would have read a
favourable single-rep result as evidence against the change. The verdict is now
three-valued, and large-grid rows — single-rep by design — report
`INSUFFICIENT EVIDENCE (direction favourable)` rather than a false negative.
