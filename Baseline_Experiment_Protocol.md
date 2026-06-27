# Baseline Experiment Protocol (Pre-Registration)

**Status:** pre-registered — thresholds fixed *before* any result is observed.
**Date:** 2026-07-01
**Author:** Emre Can Kızılateş

This document is written and frozen *before* running the experiment. Its purpose
is to prevent post-hoc rationalization: the decision rule is committed in advance,
so the numbers, once produced, interpret themselves.

---

## 1. The claim under test

The paper's premise is that **FastGRNN is the right recurrent cell for
ultra-constrained microcontrollers** because it reaches accuracy comparable to
GRU/LSTM at a fraction of the parameter count.

In the current manuscript, Table `tab:baselines` reports **measured** F1 only for
the MLP baseline and FastGRNN; the LSTM and GRU rows carry an empty F1 column
("---") and their accuracy is *borrowed* from the original FastGRNN paper
(Kusupati et al., 2018), not measured on our HAPT split.

A reviewer can legitimately object: *the superiority claim rests on a citation,
not on our own measurement.* This experiment removes that gap.

> **Integrity gate (committed in advance):** if the evidence contradicts the
> premise (Outcome 3 below), we do **not** push the "FastGRNN is superior" framing
> through peer review. We either reframe the paper honestly or hold submission.
> Truth takes precedence over publication.

---

## 2. Tier 1 — matched-hidden-size, FP32 head-to-head

The minimal honest control: train GRU and LSTM at the same hidden size as the
deployed FastGRNN, under identical conditions, and measure F1.

### Models (all at H = 16, input D = 3, 6 classes)
- **FastGRNN** (full-rank cell) — re-run in this harness for an identical-conditions
  reference (not reused from prior logs, to remove any harness-difference confound).
- **GRU** — `torch.nn.GRU`, single layer, final hidden state → `Linear(16, 6)`.
- **LSTM** — `torch.nn.LSTM`, single layer, final hidden state → `Linear(16, 6)`.

No low-rank, no sparsity, no quantization at Tier 1. (Compression survival is
Tier 2.)

### Fixed protocol (identical across all three models)
- Data: `data/processed/hapt_windows.npz`, labels shifted 1–6 → 0–5.
- Subject-aware validation: last 4 training subjects held out.
- Normalization: train-set per-channel mean/std.
- Loaders: batch 64 (train, shuffled), 256 (val/test).
- Optimizer: Adam, lr = 1e-3. Loss: cross-entropy. Grad-clip: max-norm 5.0.
- Epochs: 120, checkpoint selected by best validation macro-F1.
- Seeds: {0, 1, 2, 3, 4} — the same five seeds used throughout the paper.

### Reported metrics
- Macro-F1 on the 3,399-window test split: **5-seed mean ± std**, plus every
  individual seed (no seed hidden, including any outlier).
- Measured parameter count (cell-only and total), reported as-is — these may
  differ slightly from the paper's theoretical counts; the measured value is the
  correction.

---

## 3. Pre-registered decision rule

Let `σ_pool` be the larger across-seed standard deviation of the two cells being
compared. A difference is "meaningful" only if it exceeds **2 × σ_pool**.

| Outcome | Condition | Decision |
|---------|-----------|----------|
| **1 — FastGRNN dominant** | FastGRNN mean F1 ≥ GRU and LSTM means (within noise), and FastGRNN has fewer params | Premise proven empirically. Keep current framing; fill Table 4 with measured numbers. |
| **2 — Comparable** | GRU/LSTM mean F1 overlaps FastGRNN within ~1 σ_pool, FastGRNN ~3–4× smaller | Premise holds on size. Reframe claim honestly: *comparable accuracy at 3–4× fewer parameters.* Submit. |
| **3 — FastGRNN clearly worse** | GRU or LSTM mean F1 exceeds FastGRNN by > 2 σ_pool | "FastGRNN superiority" framing fails. Either reframe to a neutral *compact-RNN deployment study*, or hold submission (integrity gate). |

Notes committed in advance:
- Seed 1 is a known convergence outlier for FastGRNN. The decision uses the
  **all-five-seed mean**. A with/without-seed-1 sensitivity may be *reported* but
  must **not** change which outcome is triggered.
- Thresholds above are frozen. They will not be adjusted after seeing results.
- The deployment/energy/determinism/warm-up contributions stand regardless of
  which cell wins — they concern the deployment methodology, not cell choice.

---

## 4. Tier 2 — equal-budget comparison (FROZEN 2026-07, after Tier 1)

Tier 1 verdict: at matched H=16, GRU beats FastGRNN by >2σ (Outcome 3); the naive
"FastGRNN is most accurate" framing is refuted. The paper pivots to a mature
framing: **a systematic study of which recurrent cell is best for ultra-constrained
MCUs**, carried by our own contributions (calibrated Q15+LUT recipe, cross-platform
bit-equivalent determinism, INA226 energy characterization, warm-up analysis, and
this cell comparison). Under this framing any winner is a publishable result.

### The question
At an **equal deployment budget (~283 nonzero params / 566 B at Q15)**, which cell
is most accurate? Each cell reaches the budget by the compression natural to it
(fair to each cell's own nature), but the budget is held strictly equal.

### Models at the budget (FP32, 5 seeds {0..4}, identical data/protocol)
- **FastGRNN** — its deployed recipe: low-rank (r_w=2, r_u=8) + IHT sparsity 0.5.
  181 cell + 102 head = 283 params. Reference: sparse-stage 5-seed F1 = 0.856 ± 0.099
  (FP32; Q15 ≈ FP32, 0.853 ± 0.107 — calibrated Q15 is near-lossless here).
- **GRU** — shrink hidden size to the budget: H=7 (≈300 params, dense).
- **LSTM** — shrink hidden size to the budget: H=6 (≈306 params, dense).

GRU/LSTM land slightly **over** budget (300/306 vs 283) — i.e., the baselines get a
hair *more* capacity, a conservative choice against our own cell. Measured param
counts reported as-is.

### Pre-registered decision rule (σ_pool = larger across-seed std of the pair)
| Outcome | Condition | Decision |
|---------|-----------|----------|
| **1 — FastGRNN best at budget** | FastGRNN mean F1 ≥ GRU and LSTM (within noise) | Mature thesis confirmed: "GRU wins at matched H, but FastGRNN dominates at the deployment budget." Strong paper, submit. |
| **2 — Comparable at budget** | means within ~2 σ_pool | "Cells are comparable at the budget; selection turns on size/stability." Honest comparative study, submit. |
| **3 — Baseline best at budget** | GRU or LSTM mean F1 > FastGRNN by > 2 σ_pool | FastGRNN not best at any operating point. Neutral framing: "GRU/LSTM preferable for this budget." Still publishable, FastGRNN not the hero. |

### Honest caveats committed in advance
- **Variance asymmetry.** FastGRNN's budget variance is large (±0.099, driven by the
  seed-1 collapse to 0.66); GRU at matched H was tight (±0.013). The 2σ rule with
  FastGRNN's large σ makes Outcome 3 hard to trigger — so a near-tie on the *mean*
  may still hide a large **reliability** gap. Reliability (std, worst-seed) is
  therefore reported as a first-class, deployment-relevant axis, not a footnote: a
  cell that collapses on 1-in-5 seeds is worse for deployment even at equal mean.
- Decision uses the all-five-seed mean; a seed-1 sensitivity may be *reported* but
  must not change which outcome triggers.
- Primary evaluation is FP32 at equal budget (Q15 ≈ FP32 already shown for FastGRNN);
  a full Q15 cross-check for GRU/LSTM is a later refinement, not a blocker.
- If GRU/LSTM shrink-H proves unexpectedly weak, a same-H pruned variant may be added
  to give each baseline its genuine best shot (can only help baselines, never FastGRNN).
