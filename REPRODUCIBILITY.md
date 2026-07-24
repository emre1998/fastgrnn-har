# Reproducibility audit (2026-07-24)

Every committed software result was re-run at the same seed and configuration in the
current environment. Code: `run_drift_probe.py`, comparison: `analyze_drift.py`.
30 paired measurements, 3 datasets x 3 compression routes x 2 seeds.

## Headline

**Thread count is part of the experimental configuration and was never recorded.**

PyTorch's CPU BLAS changes its reduction order with the thread count, which changes
floating-point results, which over 200 epochs changes where training lands. With the
thread count pinned to 1 — matching how the committed results were produced — the
FastGRNN and shrink-H routes reproduce **bit-exactly** on HAPT and WISDM:

| route | HAPT | WISDM |
|---|---|---|
| FastGRNN (low-rank + IHT + calibrated Q15) | 0.0000 | 0.0000 |
| GRU shrink-H | 0.0000 | 0.0000 |
| LSTM shrink-H | 0.0000 | 0.0000 |

Twelve measurements, all exactly zero drift. Over all 30 pairs the median |drift| is
0.0151; every non-zero entry belongs either to the pruned route or to PAMAP2.

An earlier probe that appeared to show non-reproducibility (HAPT GRU 0.9094 -> 0.9200)
was itself the artifact: it ran with the default multi-threaded BLAS. The committed
numbers were correct; the probe was not.

### Direct evidence

Four epochs of a GRU(H=16) on real HAPT windows, same seed, hashing the resulting
weights:

| run | threads | weight hash | weight sum |
|---|---|---|---|
| `OMP_NUM_THREADS=1` (how the committed results were produced) | 1 | `5f71d1d4…` | 1.5558724403 |
| `repro.pin_threads()` | 1 | `5f71d1d4…` | 1.5558724403 |
| unpinned | 8 | `7bfd1349…` | 1.5558724403 |

Two things follow. `pin_threads()` is bit-equivalent to the environment-variable
route, so pinning cannot shift results away from what is committed. And the
unpinned run genuinely produces different weights, which makes the cause direct
evidence rather than inference.

Note the third column: the weight *sum* is identical in all three runs. After four
epochs the difference lives below the precision of any summary statistic and only
appears under bitwise comparison — which is why it went unnoticed while compounding
over 200 epochs.

**Done:** `repro.py` pins the count (override with `FASTGRNN_THREADS=n` for a
deliberate experiment) and stamps torch version, thread count, platform and git
commit into every result JSON. Wired into the six scripts that produce paper
numbers.

## Two genuine instabilities

Neither is environmental. Both are properties of the experiment itself.

### 1. The magnitude-pruned route is unstable

Drift is never zero on this route, on any dataset:

| | seed 0 | seed 1 |
|---|---|---|
| HAPT GRU pruned | -0.0261 | +0.0012 |
| HAPT LSTM pruned | **+0.2479** | +0.0285 |
| WISDM GRU pruned | -0.0360 | -0.0607 |
| WISDM LSTM pruned | **-0.2678** | +0.0343 |
| PAMAP2 GRU pruned | -0.0145 | -0.0230 |
| PAMAP2 LSTM pruned | **+0.1421** | +0.0090 |

One sub-pattern is worth separating from the noise. **WISDM GRU pruned drifts
downward on both seeds** (-0.036, -0.061), and that is the exact number the
equal-byte comparison rests on: it is the GRU's winning route on WISDM, committed
at a 5-seed mean of 0.7671 against FastGRNN's 0.7997. If the remaining three seeds
move the same way, the baseline lands near 0.72 and FastGRNN's margin grows from
+0.033 to roughly +0.08. Two seeds are not enough to claim that, but the direction
means the committed WISDM comparison is, if anything, **too generous to the
baseline** rather than flattering to us.

This is not new information: the committed 5-seed standard deviations for this route
were already extreme (WISDM LSTM pruned +/-0.184, PAMAP2 +/-0.169). The re-runs confirm
that those numbers describe a genuinely unstable procedure rather than a wide but
stable distribution.

Consequence for the equal-byte comparison: `best_route_summary.json` reports
`max(shrink_mean, pruned_mean)` for each baseline, and on both HAPT and WISDM the GRU's
winning route is the pruned one. The maximum of two noisy estimates is biased upward,
so this control **over-credits the baselines** — it is conservative with respect to the
FastGRNN claim, not favourable to it. That direction is the acceptable one, but it must
be stated rather than left implicit.

### 2. PAMAP2 does not reproduce in any pipeline

| pipeline | FastGRNN | GRU | LSTM |
|---|---|---|---|
| deployment budget (seed 0) | -0.0624 | -0.0999 | -0.1293 |
| Tier-1 equal-H (seed 0) | -0.0744 | +0.0453 | -0.0850 |

Two independent pipelines, mixed signs in the second, so this is not a single bug in one
script. Contributing factors: 12 classes with macro-F1 in the 0.25-0.44 band, and input
values spanning -107 to +142 (HAPT spans -1.8 to 2.0) — unclipped sensor spikes. The
dataset sits at the edge of what these models can learn, and training is chaotic there.

FastGRNN on PAMAP2 is internally deterministic (1 thread and 3 threads give the identical
0.2759) yet does not match the committed 0.3382, so the divergence predates this audit
and is not thread-related.

**Recommendation:** report PAMAP2 numbers, but do not draw ranking conclusions from them.
A dataset whose results move by 0.06-0.15 between runs cannot support a 0.090 margin.

## What this does not touch

Every hardware result. Latency, energy, memory and the sensor-in-the-loop matrix are
physical measurements of fixed firmware, and the MCU takes the same time to multiply
regardless of the weight values. F1, F2 and F6 are unaffected, as is the feasibility
proof that carries claims C1, C3, C4 and C5.

Q15 losslessness (C7) is also unaffected: it compares FP32 against Q15 *within a single
trained model*, so no cross-run comparison is involved.

## Pruned route re-run at 5 seeds, threads pinned

Written to the tag `{ds}V2` so both sets exist side by side; compare with
`analyze_pruned_v2.py`. The FastGRNN column is unchanged in all three, because that
route reproduces bit-exactly — so every difference below comes from the baseline
side of the comparison.

| | GRU pruned, committed | re-run | FastGRNN margin |
|---|---|---|---|
| HAPT | 0.9023 ± 0.0263 | 0.9009 ± 0.0278 | $-$0.033 → $-$0.031 |
| WISDM | 0.7671 ± 0.0111 | 0.7325 ± 0.0267 | **+0.033 → +0.067** |
| PAMAP2 | 0.3392 ± 0.0262 | 0.3188 ± 0.0223 | +0.090 → +0.090 (shrink is the better route either way) |

HAPT is unchanged and the GRU still wins it. WISDM's margin doubles, and the
direction is what matters: the published comparison was **too generous to the
baseline**, not to us.

The earlier characterisation of "the pruned route is unstable" needs narrowing.
GRU with pruning is stable — 0.0015 across five seeds on HAPT. What is unstable is
LSTM with pruning: +0.064 on HAPT, $-$0.058 on WISDM, +0.073 on PAMAP2, with
standard deviations up to 0.17. The LSTM is not the winning cell anywhere, so this
does not reach the headline claims, but it does mean LSTM-pruned figures should
carry their spread rather than a bare mean.

Still open: which set becomes canonical. The re-run is the only one produced under
a documented configuration, which argues for it — but that decision replaces
published numbers and is not one to make silently.

## Open items

- [x] Pin thread count in run scripts; stamp environment into result JSONs.
- [x] Re-run the pruned route at 5 seeds with threads pinned.
- [ ] Decide whether the re-run replaces the committed pruned numbers; if so,
      regenerate `best_route_summary.json` and the ranking figure.
- [ ] Decide the PAMAP2 framing: demote to "reported, not ranked".
- [ ] If PAMAP2 is demoted, the equal-byte record becomes HAPT to the GRU and WISDM to
      FastGRNN. The ranking-reversal finding still holds on WISDM, where both sides of
      the comparison reproduce, but "FastGRNN wins 2 of 3" must go.
