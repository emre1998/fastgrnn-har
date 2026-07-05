You are a brutally honest senior reviewer for ACM Transactions on Embedded Computing
Systems (TECS). Help me structure a journal paper revision. Be terse, technical, no
compliments. Challenge the framing. I want a paper SCHEMA recommendation.

== CONTEXT ==
- arXiv v1 already public + indexed (NASA ADS, Scholar, ResearchGate, 9 platforms).
  v1 claimed "FastGRNN is the best/superior cell for ultra-constrained MCU HAR" --
  but that claim was BORROWED from Kusupati et al., not measured by us.
- Now revising for a v2 (arXiv) + first journal submission (ACM TECS).
- Philosophy: truth over publication. We ran controlled experiments; if data contradicts
  the premise, we report the truth.
- Domain: Human Activity Recognition (HAR) on microcontrollers (Arduino Uno ATmega328P
  2KB SRAM; MSP430G2553 512B SRAM, no hardware multiplier). Deployed model ~283 nonzero
  params / 566 bytes, Q15 fixed-point.

== WHAT WE MEASURED (all 5 seeds, subject-disjoint splits) ==

THREE datasets: HAPT (6 class, 50Hz), WISDM (6 class, 20Hz), PAMAP2 (12 class, 100->50Hz,
single wrist accelerometer -- intentionally hard).

EXPERIMENT 1 -- Equal CAPACITY (matched hidden size H=16, dense, FP32):
              GRU            LSTM           FastGRNN   (params: FastGRNN 440 vs GRU 1110)
  HAPT    0.917+/-0.013  0.886+/-0.027  0.864+/-0.018
  WISDM   0.764+/-0.023  0.744+/-0.073  0.748+/-0.033
  PAMAP2  0.389+/-0.051  0.329+/-0.027  0.351+/-0.017
  -> At equal H, GRU wins on means all 3, but statistically clear (2-sigma) only on HAPT;
     WISDM/PAMAP2 are ties. FastGRNN competitive #2 with 2.5x fewer params. LSTM weakest.

EXPERIMENT 2 -- Equal BYTE BUDGET (deployment; each cell its NATURAL compression to ~equal
bytes; FastGRNN = low-rank r_w2/r_u8 + IHT sparsity 0.5 + calibrated Q15; GRU/LSTM = shrink
hidden size to fit budget + weight-Q15). 200-epoch fair training for all:
              GRU            LSTM           FastGRNN
  HAPT    0.880+/-0.043  0.844+/-0.025  0.869+/-0.081
  WISDM   0.683+/-0.045  0.674+/-0.015  0.800+/-0.009
  PAMAP2  0.354+/-0.032  0.306+/-0.029  0.444+/-0.068
  -> At equal BYTES, FastGRNN WINS 2/3 (WISDM +0.117, PAMAP2 +0.090), ties HAPT.
     REVERSAL of Experiment 1. Mechanism: to fit the budget GRU/LSTM must shrink to H=5-7
     (small dense), losing capacity (WISDM GRU H16->H6 drops 0.764->0.683); FastGRNN keeps
     H=16 (low-rank+sparse). On capacity-hungry tasks this is decisive.
  -> We RULED OUT the confound "FastGRNN trained 2x longer" by retraining GRU/LSTM at the
     same 200 epochs (fairness check). Win survives.

EXPERIMENT 3 -- Pareto (HAPT only, dense, accuracy vs params, H in {4..12}): GRU frontier
on top across budgets; confirms at EQUAL CAPACITY GRU is the better cell.

DIAGNOSIS -- the L-S-Q (low-rank -> sparse -> quantize) failure mode: on HAPT one seed
collapses (macro-F1 0.708, std 0.081). Root cause traced: the low-rank (L) warm-start has
seed variance; irreversible IHT hard-thresholding (S) amplifies it; the capacity-starved
model has no redundancy to recover the dynamic-gait classes. BUT this instability is
HAPT-SPECIFIC: on WISDM FastGRNN is the MOST stable cell (std 0.009).

QUANTIZATION: calibrated Q15 (+LUT for sigmoid/tanh) is NEAR-LOSSLESS on all 3 datasets,
all 3 cells (delta-F1 ~ 0.000 vs FP32).

== OUR GENUINE ORIGINAL CONTRIBUTIONS (cell-independent) ==
1. Calibrated Q15 + lookup-table activation recipe -> near-lossless quantization (measured
   on 3 datasets now).
2. Cross-platform BIT-EXACT deterministic inference: same integer outputs on Arduino (8-bit)
   and MSP430 (16-bit, no HW multiplier).
3. Measured energy via INA226 (real measurement, not estimation like ZIP-CNN).
4. Warm-up analysis (streaming inference startup behavior).
5. Full reproducibility (public GitHub, 5-seed variance, 3 datasets).

== THE INTELLECTUAL BACKBONE I THINK WE HAVE ==
"FastGRNN does NOT have a better CELL (at equal capacity GRU wins), but it has a better
COMPRESSION-CAPACITY tradeoff. At the real byte budget -- the actual MCU constraint --
this is what matters: GRU must shrink its hidden state to fit, FastGRNN compresses and
keeps it. So the equal-capacity vs equal-budget DISTINCTION is the paper's core idea.
The original compressibility thesis is vindicated, but for the right (measured) reason."

== WHAT I NEED FROM YOU ==
1. Is the "equal-capacity vs equal-budget distinction" strong enough to be the central
   contribution of a TECS paper, or is it too thin / already known in the compression lit?
2. What is the SINGLE strongest framing/title direction? Is FastGRNN the hero, the
   hypothesis-under-test, or a co-equal player?
3. Recommend a concrete SECTION STRUCTURE (Intro / Related / Method / Experiments /
   Discussion) with what each section must contain and what to LEAD with in the abstract.
4. What will Reviewer #2 attack hardest? Rank the 3 biggest risks and how to defend each.
5. Is anything MISSING that a TECS reviewer would demand before acceptance? (e.g. latency
   numbers, more cells, a real baseline from literature, ablations).
6. How should we honestly handle the v1->v2 narrative shift (claim corrected by our own
   experiments) without it reading as a retraction?

Give me a decisive recommendation, not a menu. Where you're uncertain, say so.
