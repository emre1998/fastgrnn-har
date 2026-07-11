# FastGRNN-HAR v2 — Complete Experimental Briefing (AI Agent Handoff)

> **Purpose.** This document hands off the *entire* experimental campaign for the v2 revision of
> the paper "Real-Time Human Activity Recognition on Multiplier-less Microcontrollers." It is written
> for AI agents (and the author, Emre) who will help write the paper after a short break. It is
> self-contained: an agent should be able to read only this file and understand the thesis, the
> evidence, the honest caveats, where the data lives, and what to do next.
>
> **Author works in Turkish; the paper is in English.** This brief is in English so it can feed the
> writing directly. Governing philosophy throughout: **truth over publication.** No claim ships
> unless the data supports it; every counterintuitive result was chased down, not smoothed over.

---

## 0. TL;DR (read this first)

- **The protagonist is NOT an algorithm.** It is a *proof*: that an ultra-constrained, **multiplier-less,
  no-FPU** MCU (TI MSP430G2553 — 16 KB Flash, 512 B SRAM, no hardware multiplier, no floating-point unit)
  can run **real-time (50 Hz) HAR**. GRU, LSTM, and FastGRNN are three candidate *tools*, compared
  objectively. **No cell is advocated.**
- **Thesis (refined, measured):** real-time HAR on such an MCU is feasible, **but conditionally** — it
  requires **BOTH** (a) LUT-based activations (sigmoid/tanh lookup tables) **AND** (b) a fast enough
  sensor I2C bus (100 kHz). Remove either and it fails. This is stronger and more honest than "it works."
- **Three evidence layers, all complete:** (1) accuracy on 3 HAR datasets, (2) bench hardware
  deployment (no sensor), (3) real-world hardware with the actual MPU6050 sensor in the loop.
- **Data collection is DONE.** Remaining work is desk-work: compile the master table (B3), then write
  the paper (Phase C), then publish (Phase D: arXiv v2 + ACM TECS).
- **Repo:** `github.com/emre1998/fastgrnn-har` (branch `master`). Key files listed in §7.

---

## 1. The intellectual arc — from v1 to v2

**v1 (already on arXiv, indexed):** claimed, on borrowed grounds, that "FastGRNN is the best cell for
MCU-HAR." The v2 campaign was a **reproduction → investigation → extension → publication** effort.

What the investigation found (this is the *story*):

1. **Equal-capacity (equal hidden size H=16):** GRU is the best cell — it wins accuracy on **all 3**
   datasets. The naive "FastGRNN is best" claim **died** here.
2. **Equal-byte (equal deployment footprint ~283 params / ~566 B):** FastGRNN wins **2 of 3** datasets,
   because its structural compression (low-rank + sparsity + calibrated Q15) preserves capacity at a
   fixed byte budget, while GRU/LSTM must shrink H and lose capacity. FastGRNN's original thesis is
   **vindicated — but only under the byte-budget question.**
3. **The reconciliation (the paper's intellectual spine):** *equal-capacity ≠ equal-byte = different
   questions.* GRU is the better cell; FastGRNN is the better compression-capacity tradeoff. Which one
   "wins" depends entirely on which constraint you hold fixed. Neither is universally best.
4. **The mature pivot:** since no cell wins universally, the paper's contribution is not "cell X is
   best" but **the deployment framework + the feasibility proof** — a framework (calibrated Q15+LUT,
   bit-exact dual-platform inference, INA226 energy, warm-up) that is **cell-independent** and makes
   real-time HAR possible on a multiplier-less MCU at all.

**v2 is a REFINEMENT, not a retraction.** v1 was an honest reproduction whose comparison claim was
under-specified; v2 measures it properly. arXiv v2 will carry a transparent changelog. This self-
correction is a *strength* (reviewers value it), not a weakness.

---

## 2. Layer 1 — Accuracy (software, 3 datasets, 5 seeds, macro-F1)

Datasets: **HAPT** (smartphone, 6 activities), **WISDM** (phone accel, 20 Hz), **PAMAP2** (IMU, 50 Hz).
All pipelines reproduce HAPT behavior exactly; code parametric on `--data/--tag`.

**Tier 1 — equal H=16 (architecture quality):**

| Dataset | GRU | LSTM | FastGRNN |
|---|---|---|---|
| HAPT | **0.917 ± 0.013** | 0.886 ± 0.027 | 0.864 ± 0.018 |
| WISDM | **0.764 ± 0.023** | 0.744 ± 0.073 | 0.748 ± 0.033 |
| PAMAP2 | **0.389 ± 0.051** | 0.329 ± 0.027 | 0.351 ± 0.017 |

→ **GRU wins all 3.** LSTM is weakest + most unstable despite most params. FastGRNN is compact
(440 params vs GRU 1110), stable, competitive #2 on WISDM/PAMAP2.

**Deployment budget — ~equal byte (~283 params / 566 B), 200 epochs, Q15, each cell's own compression:**

| Dataset | GRU | LSTM | FastGRNN |
|---|---|---|---|
| HAPT | 0.880 | 0.844 | 0.869 |
| WISDM | 0.683 | 0.674 | **0.800** |
| PAMAP2 | 0.354 | 0.306 | **0.444** |

→ **FastGRNN wins 2/3** (WISDM +0.12, PAMAP2 +0.11); HAPT ~tie (GRU 0.880 vs FastGRNN 0.869).
Mechanism: at a fixed byte budget GRU collapses H16→H6 (WISDM 0.764→0.683); FastGRNN keeps H=16 via
low-rank+sparse. **Fairness controls done:** GRU/LSTM re-run at 200 epochs (equal training), two
compression routes each (shrink-H and pruning), best-of taken. FastGRNN win survives all.

**Compression ablation (dense→low-rank→+sparse→+Q15):** instability originates at the **low-rank** step
(HAPT-specific, seed-sensitive), NOT at IHT/sparsity; **Q15 is near-lossless everywhere** (the paper's
own contribution). Compression is often *regularizing*, not purely destructive.

---

## 3. Layer 2 — Bench deployment (hardware, NO sensor)

Hardware: MSP-EXP430G2ET (MSP430G2553), 16 MHz calibrated DCO, CCS bare-metal firmware, INA226 energy
rig read by an Arduino Uno. Deployed configs: **GRU H6, LSTM H5, FastGRNN H16(compressed)**, HAPT seed0,
bit-exact fixed-point C (verified 100% vs Python).

**Memory (production firmware — no UART/debug/test code; the paper's authoritative footprint):**

| Cell | Flash | SRAM |
|---|---|---|
| GRU | 5392 B | 308 B |
| FastGRNN | 5544 B | 348 B |
| LSTM | 5742 B | 324 B |

All fit the 16 KB Flash / 512 B SRAM budget. (Analytical weight-only footprint, for fair model-size
comparison: GRU 480 B, LSTM 472 B, FastGRNN 566 B — a *different* quantity from firmware Flash; keep
them distinct in the paper.)

**Latency (per-step, zero input, µs→ms), 50 Hz budget = 20 ms/step:**

| Cell | LUT | no-LUT |
|---|---|---|
| GRU | 12.12 ms ✅ | 19.27 ms (96% util, marginal) |
| LSTM | 12.37 ms ✅ | 22.10 ms ❌ |
| FastGRNN | 13.9 ms ✅ | ~26.1 ms ❌ |

→ **With LUT all three meet 50 Hz; without LUT LSTM/FastGRNN fail, GRU is marginal.** LUT is a
**cell-independent prerequisite**, not a FastGRNN trick.

**Energy (INA226, bench, zero input):** ~17.7 mW for all cells, **LUT-independent** (active power is
platform/clock-dominated). **Derived** energy/window = P×t → LUT cuts energy by **−37% (GRU), −44%
(LSTM), −46% (FastGRNN)** purely via latency. Keep the **measured (power, latency) vs derived (energy)**
distinction explicit — reviewers will ask "how did you get energy?"

**Compiler -O sweep (36 measurements: 3 cells × LUT{0,1} × -O{off,0,1,2,3,4}):** compiler optimization is
**nearly irrelevant** (no-LUT 0–5%, LUT 5–9% saturating at -O2). **Root cause — the thesis's own
mechanism:** the MSP430 has **no FPU**, so *every* float op (MACs and expf/tanhf) is a precompiled RTS
soft-float library call the project's -O flag cannot touch. The same fact that makes the MCU slow (no
hardware multiplier/FPU) makes -O ineffective. `use_hw_mpy=none` confirmed.

**Retired claim — the "54 s" figure:** an old v1-era number (FastGRNN no-LUT ~54 s/window, 30.5×
speedup, 96.7% energy saving). We proved it is **not** a "-O off" result (current code at -O off is
~27 ms/step). It was a **Week-8 live-mode experiment with the MPU6050 sensor in the loop** (different,
pre-Q15 code). It is now a labeled legacy footnote, removed from the main tables. Do **not** put it
beside the −37/−44/−46% figures.

---

## 4. Layer 3 — Real-world (MPU6050 sensor IN the loop) — the crown jewel

The key move that makes this paper's feasibility claim real. Live firmware (`TEST_MODE 4` = latency,
`5` = energy) reads the actual MPU6050 over USCI_B0 I2C at 50 Hz, times **sensor read + inference**
end-to-end. `I2C_FAST` macro selects 10 kHz (conservative) or 100 kHz (standard). Sensor powered from
the INA226-measured rail → energy is true **system** power. **24 measurements** (3 cells × LUT{0,1} ×
I2C{10k,100k} × {latency, energy}).

**End-to-end latency (ms), REAL-TIME @ 20 ms:**

| Cell | LUT | I2C | e2e | verdict |
|---|---|---|---|---|
| GRU | 1 | 100k | **12.01** | ✅ OK |
| LSTM | 1 | 100k | **12.98** | ✅ OK |
| FastGRNN | 1 | 100k | **13.98** | ✅ OK |
| GRU | 1 | 10k | 20.43 | ❌ FAIL (marginal) |
| LSTM | 1 | 10k | 20.71 | ❌ FAIL |
| FastGRNN | 1 | 10k | 22.26 | ❌ FAIL |
| GRU | 0 | 100k | 20.03 | ❌ FAIL (barely) |
| LSTM | 0 | 100k | 22.49 | ❌ FAIL |
| FastGRNN | 0 | 100k | 27.45 | ❌ FAIL |
| GRU/LSTM/FastGRNN | 0 | 10k | 27.48 / 30.12 / 34.98 | ❌ FAIL |

**System power (INA226, MCU+I2C+MPU6050):** cell- AND LUT-independent → **~32 mW @100 kHz, ~34 mW
@10 kHz**. That is **~2× the MCU-only bench (17.7 mW)** — the MPU6050 (~3.9 mA) roughly doubles system
power and halves battery life. This is the honest real-world deployment energy.

**The four findings that carry the paper:**
1. **Real-time is achievable ONLY with LUT + 100 kHz I2C** (GRU 12 < LSTM 13 < FastGRNN 14 ms). No
   other combination works.
2. **Two independent preconditions:** LUT **AND** fast sensor I2C. This is the refined feasibility claim.
3. **The sensor acquisition path is first-order:** at 10 kHz the read alone is 8.4 ms and sinks even
   the fastest LUT config; at 100 kHz it is ~0.8 ms. Feasibility is not just about inference.
4. **no-LUT cannot be real-time at any I2C speed** (inference alone ≥ 19 ms).

**Cross-validation (why we trust everything):** every live inference latency matched the bench
(zero-input) number within ~5% (e.g. GRU 12.0≈12.1, FastGRNN 13.97≈13.9, LSTM no-LUT 21.6≈22.1). This
validates *both* the live harness *and* the earlier bench figures.

---

## 5. Honesty & rigor log (what makes this defensible)

These are the "içime sinmedi" (this doesn't sit right) moments — each one strengthens the paper:

- **Energy was re-measured from scratch** after a Debug/Flash staleness scare; strict procedure
  (Save→Clean→Build→Debug) adopted for consistency.
- **Measured vs derived** kept explicit for energy (power & latency measured; energy = P×t derived).
- **Busy-wait upper bound:** the firmware busy-waits (no LPM sleep) between samples, so reported energy
  is an *active-regime upper bound*. Future work: LPM0/3 duty-cycling would lower system energy roughly
  in proportion to the duty cycle (= latency %util).
- **-O sweep** was run at *every* level (author is thorough) — which is exactly what revealed the
  no-FPU mechanism and buried the 54 s myth.
- **Test-harness vs production memory:** the author caught that early memory numbers included UART/debug
  code; genuinely stripped production firmware was built for the authoritative footprint.
- **Flaky-contact catch:** one GRU-100k energy run gave a reproducible ~40 mW oscillation. The physical
  contradiction (100 kHz drawing more than 10 kHz, and more than FastGRNN) was flagged and **not
  recorded**; root cause = marginal I2C contact making the CPU spin in `i2c_read` 50000-count timeout
  loops. After reseating, it read a stable ~32 mW. Breadboard contacts degrade untouched — always
  sanity-check against physics.
- **Deployed accuracy** (single-config firmware, HAPT seed0, dense shrink-H): GRU F1 0.915, LSTM 0.818.
  These are the bit-exact deployed models; report alongside the 5-seed distributions, not instead of.

**Known limitation to state plainly:** most experiments center on HAPT for hardware; the 3-dataset
sweep is software-only. The hardware feasibility proof is dataset-independent (compute cost is
data-independent in structure), but say so explicitly.

---

## 6. Where everything lives (repo map)

Repo: `github.com/emre1998/fastgrnn-har`, branch `master`.

- **`B2_RESULTS.md`** — all hardware measurements: bench latency, energy (derived), -O sweep matrix,
  production memory, AND the full sensor-in-loop 24-run matrix. **Primary hardware data source.**
- **`EXECUTION_PLAN.md`** — the sequential plan (Phase A software / B hardware / C writing / D publish)
  with inline results and decisions. Phases A & B marked complete.
- **`docs/energy_measurement.md`** — FastGRNN reference energy/latency protocol (INA226 rig, wiring).
- **`msp/ccs_{gru,lstm,fastgrnn}_har/`** — CCS bare-metal firmware. `main.cpp` has all modes:
  `TEST_MODE` 1=bench-latency, 3=bench-energy, 4=live-latency (sensor), 5=live-energy (sensor);
  `USE_LUT` (in `{cell}.cpp`), `I2C_FAST` (10/100 kHz), all at deployed default -O3 / LUT=1.
- **`msp/ccs_{cell}_production/`** — minimal production firmware (authoritative memory footprint).
- **`experiments/*.json`** — software results (tier1/tier2/pareto/deploy-budget summaries).
- **`run_*.py` / `analyze_*.py`** — experiment + analysis scripts (parametric on dataset).
- **`analyze_footprint.py`** — analytical SRAM working-set + Flash weight footprint.
- **Persistent notes:** the author's private memory file `project_fastgrnn_journal_notes.md` holds the
  full running narrative (not in repo).

**Deployed configs:** GRU H6, LSTM H5, FastGRNN H16-compressed; HAPT seed0; Q15 weights + 256-entry
sigmoid/tanh LUT; streaming inference API `{cell}_reset/step/predict`.

---

## 7. What's next (the work after the break)

Data collection is finished. Remaining:

- **B3 — master deployment table (desk-work, no runs).** Merge the three layers into one per-cell table:
  accuracy · Flash · SRAM · latency · energy · real-time verdict, each labeled **measured / derived /
  analytical**. Handle the two Flash numbers (production firmware 5.4–5.7 KB vs analytical weight-only
  472–566 B) distinctly. Prove "real-time" with latency << window budget.
- **Phase C — writing.** The three-layer structure IS the paper skeleton:
  - Title/Abstract: feasibility + reversal-to-refinement, objective tone.
  - Intro: corrected premise; the hook "Can multiplier-less sub-kilobyte MCUs do real-time HAR?";
    say the equal-H GRU-ahead result early; ~5 contributions.
  - Related Work: organize around *evaluation regimes* (equal-H / equal-param / equal-byte / measured).
  - Methods: Regime A (equal capacity) + Regime B (equal byte); compression recipes; two-part budget;
    MCU deployment; live-sensor protocol.
  - Experiments: E1 equal-capacity → E2 equal-byte → E3 mechanism → E4 Pareto → E5 quantization →
    E6 deployment (bench latency/energy/memory) → **E7 real-world sensor-in-loop (NEW, strongest)**.
  - Failure Analysis (dedicated section): low-rank instability, seed variance, HAPT-specificity,
    no-LUT real-time failure, the LUT+I2C double precondition.
  - Discussion: objective; equal-capacity ≠ equal-byte; FastGRNN advantage is conditional; compression
    isn't free; sensor path is first-order; no-FPU explains why -O can't help.
  - Conclusion + a **"Relation to Prior Version"** note (under-specified → refined, not a retraction).
- **Phase D — publish.** arXiv v2 (transparent changelog) → ACM TECS (primary target; IEEE Trans.
  Computers is the full-journal backup). Full journal only, no letters, no venue-specific reframing.
- **Pre-submission — repo reproducibility pass (task D0):** clean stray files, README with "script →
  table/figure" mapping, seed/config transparency, firmware build settings (-O3, USE_LUT=1,
  use_hw_mpy=none). Goal: a reviewer clones and reproduces the headline results with one command.

**Open decisions for the writing sprint:** final title wording; how prominently to feature the sensor
layer (recommendation: it is the strongest, lead with it in deployment); whether to add the LPM
duty-cycled energy estimate as future-work vs a measured extension.

---

## 8. The one-paragraph narrative (for the abstract's spirit)

We ask whether a multiplier-less, no-FPU, sub-kilobyte microcontroller can run real-time human activity
recognition, and we answer with measurement rather than assertion. Evaluating three compact recurrent
cells (GRU, LSTM, FastGRNN) objectively, we show that the best cell depends on the constraint: GRU wins
at equal capacity, FastGRNN wins at equal byte-budget via structural compression — so the contribution
is not a winning algorithm but a cell-independent deployment framework (calibrated Q15 + activation
LUTs, bit-exact dual-platform inference, measured INA226 energy) and, above all, the demonstration that
real-time HAR on such an MCU is feasible **only** under two independent engineering preconditions:
lookup-table activations and a fast (100 kHz) sensor bus. With the real MPU6050 in the loop we measure
end-to-end latency and system energy directly, and find that the sensor acquisition path is as decisive
as the inference itself — a result invisible to inference-only benchmarks.

---

*Prepared 2026-07-10 at the close of the experimental campaign. Next session: the writing sprint —
ideas, data, narrative, motivation, all discussed in depth. Rest well; the hard, rewarding part is next.*
