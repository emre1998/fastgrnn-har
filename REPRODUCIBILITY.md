# Reproducibility audit (2026-07-24)

Every committed software result was re-run at the same seed and configuration in the
current environment. Code: `run_drift_probe.py`, comparison: `analyze_drift.py`.
28 paired measurements, 3 datasets x 3 compression routes x 2 seeds.

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

Twelve measurements, all exactly zero drift. Median |drift| over all 28 pairs is 0.0118.

An earlier probe that appeared to show non-reproducibility (HAPT GRU 0.9094 -> 0.9200)
was itself the artifact: it ran with the default multi-threaded BLAS. The committed
numbers were correct; the probe was not.

**Action:** pin `torch.set_num_threads(1)` in the run scripts and record the thread
count, torch version, device and commit in every result JSON.

## Two genuine instabilities

Neither is environmental. Both are properties of the experiment itself.

### 1. The magnitude-pruned route is unstable

Drift is never zero on this route, on any dataset:

| | seed 0 | seed 1 |
|---|---|---|
| HAPT GRU pruned | -0.0261 | +0.0012 |
| HAPT LSTM pruned | **+0.2479** | +0.0285 |
| WISDM GRU pruned | -0.0360 | -- |
| WISDM LSTM pruned | **-0.2678** | -- |

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

## Open items

- [ ] Pin thread count in run scripts; stamp environment into result JSONs.
- [ ] Re-run the pruned route at 5 seeds with threads pinned, or report both routes
      separately instead of best-of.
- [ ] Decide the PAMAP2 framing: demote to "reported, not ranked".
- [ ] If PAMAP2 is demoted, the equal-byte record becomes HAPT to the GRU and WISDM to
      FastGRNN. The ranking-reversal finding still holds on WISDM, where both sides of
      the comparison reproduce, but "FastGRNN wins 2 of 3" must go.
