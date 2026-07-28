# Bootstrap confidence intervals (5 seeds)

95% percentile bootstrap, 20000 resamples, RNG seed 0. Per-seed values are shown next to every interval: with n = 5 the interval is wide and approximate, and the raw distribution is the honest primary record.

## Equal capacity (Tier-1, H=16)

| Dataset | Cell | Mean [95% CI] | Per-seed |
|---|---|---|---|
| HAPT | GRU | 0.905 [0.876, 0.925] | [0.932, 0.904, 0.922, 0.849, 0.915] |
| HAPT | LSTM | 0.884 [0.863, 0.906] | [0.861, 0.908, 0.890, 0.911, 0.852] |
| HAPT | FastGRNN | 0.860 [0.838, 0.883] | [0.905, 0.862, 0.865, 0.828, 0.841] |
| WISDM | GRU | 0.772 [0.762, 0.782] | [0.784, 0.786, 0.771, 0.755, 0.764] |
| WISDM | LSTM | 0.746 [0.698, 0.789] | [0.780, 0.802, 0.781, 0.671, 0.697] |
| WISDM | FastGRNN | 0.739 [0.717, 0.765] | [0.785, 0.737, 0.702, 0.726, 0.743] |
| PAMAP2 | GRU | 0.327 [0.300, 0.367] | [0.406, 0.313, 0.292, 0.320, 0.302] |
| PAMAP2 | LSTM | 0.264 [0.246, 0.280] | [0.273, 0.276, 0.287, 0.245, 0.237] |
| PAMAP2 | FastGRNN | 0.290 [0.282, 0.299] | [0.299, 0.278, 0.280, 0.291, 0.304] |

## Equal byte budget — margin of FastGRNN over the best baseline

The margin is FastGRNN minus the stronger of the two baseline routes (shrink-H / pruned-H16). A 95% CI that excludes zero supports the ranking; one that straddles zero is a statistical tie.

| Dataset | FastGRNN | Best baseline | Margin [95% CI] | Verdict |
|---|---|---|---|---|
| HAPT | 0.869 [0.901, 0.708, 0.918, 0.902, 0.918] | GRU pruned 0.901 | -0.031 [-0.117, +0.028] | tie (CI spans 0) |
| WISDM | 0.800 [0.797, 0.809, 0.811, 0.788, 0.793] | GRU pruned 0.732 | +0.067 [+0.043, +0.091] | **supported** |
| PAMAP2 | 0.444 [0.338, 0.407, 0.541, 0.458, 0.477] | GRU shrink 0.354 | +0.090 [+0.024, +0.155] | **supported** |

## What the intervals say

- **HAPT**: margin -0.031, 95% CI [-0.117, +0.028] — tie (CI spans 0).
- **WISDM**: margin +0.067, 95% CI [+0.043, +0.091] — **supported**.
- **PAMAP2**: margin +0.090, 95% CI [+0.024, +0.155] — **supported**.

The equal-byte win should be reported at the strength the intervals actually support: where the CI excludes zero it is a result, where it spans zero it is a tie and must be called one. HAPT FastGRNN's wide interval comes from a single collapsing seed (the low-rank instability, characterized in the ablation), not from broad noise — which is why the per-seed column, not the mean alone, is the honest record.

