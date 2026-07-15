# PAPER_THESIS_LOCK — v2 Framing (locked 2026-07-11)

> **Status: LOCKED.** After a three-way refinement (author's instinct + GPT's narrative + Claude's
> rigor/consistency), the scientific framing is fixed. From here the work is **authorship, not idea
> generation** — editing, ordering, figure placement, prose. Do **not** re-open the framing debate;
> use this file as the single source of truth for every section. Governing rule: **truth over
> publication** — every claim must map to an experiment we actually ran (see §5–6). If a sentence
> can't be traced to evidence here, it doesn't go in the paper.

---

## 0. Motivation (the philosophy — the intro hook, ~2 sentences of vision)

The prevailing assumption is that *better AI needs bigger hardware.* This paper questions that — **not**
by claiming hardware doesn't matter, but by showing that **with sufficient software and deployment
engineering, meaningful real-time inference is feasible on far smaller, cheaper, multiplier-less
hardware than assumed.** The stakes: bringing real-time neural sensing to the cheapest, most
mass-produced compute (wearables and commodity ultra-constrained devices) — a **software** frontier,
not a silicon one. *(Framing word: "enables/extends," never "democratizes.")*

## 1. Research Question (one sentence)

> On ultra-constrained, **multiplier-less** microcontrollers, **what actually determines** whether
> real-time recurrent neural inference is feasible — and how do deployment constraints reshape the way
> models should be evaluated?

## 2. Scientific Thesis (one paragraph)

> Real-time neural inference on extremely constrained, multiplier-less microcontrollers is **not merely
> a model-selection problem but a deployment-engineering problem.** Through systematic, measured
> evaluation on real hardware — including the sensor in the loop — we show that deployment constraints
> can fundamentally **alter model rankings and real-time feasibility**, and that some of the deciding
> factors (the activation implementation and the sensor-acquisition path) are **invisible to
> inference-only benchmarks.** Model choice matters, but it is **not sufficient alone**: feasibility is
> decided jointly by algorithmic performance, software optimizations, and hardware constraints. HAR is
> the case study through which this is demonstrated, not the object of the paper.

## 3. Non-goals (what we explicitly DO NOT claim — prevents scope creep, cite in intro)

- ✗ We do **not** propose a new RNN cell.
- ✗ We do **not** propose a new compression algorithm (low-rank + IHT sparsity are FastGRNN's; Q15 +
  LUT are engineering recipes applied rigorously, not novel algorithms).
- ✗ We do **not** claim to invent or advance TinyML as a field.
- ✗ We do **not** crown a single "best" model — for HAR or in general.
- ✗ We do **not** claim generalization to other domains (speech, vision, vibration). The framework is
  *applicable* to other sequential-sensing workloads; it is not *demonstrated* on them.
- ✗ We do **not** claim "hardware doesn't matter." Hardware constraints are real and inevitable; we
  show software extends the feasible envelope more than expected.
- ✗ We do **not** claim a deployed wearable product. The MSP430 HAR system is an **existence proof.**
- ✗ We do **not** claim a novel energy-measurement methodology — but we do emphasize that our energy is
  **measured, not estimated** (a rigor differentiator vs prior estimation-based work).

## 4. Contributions (claimed novelty — LEAD with findings, not "we propose a methodology")

1. An **empirical demonstration** that real-time RNN inference is feasible on a multiplier-less,
   no-FPU, sub-kilobyte MCU — **conditionally**, and we characterize the conditions.
2. The finding that **deployment constraints reverse model rankings** (GRU wins at equal capacity;
   FastGRNN wins at equal byte-budget) — model selection is regime-dependent.
3. The finding that **real-time feasibility has two independent preconditions** — LUT-based
   activations **and** a fast (≥100 kHz) sensor bus — and that the **sensor-acquisition path is
   first-order and invisible to inference-only benchmarks.**
4. A **mechanistic explanation** of why compiler optimization is nearly useless here: no FPU / no
   hardware multiplier → all float arithmetic is precompiled soft-float; software architecture, not
   the toolchain, decides feasibility.
5. A **deployment-aware, real-hardware, sensor-in-the-loop evaluation** with **measured** (not
   estimated) latency, energy, Flash, and SRAM — the methodological vehicle that reveals 1–4.

## 5. Claims → Evidence map (every claim must trace to a real experiment)

| # | Claim | Supporting evidence |
|---|-------|---------------------|
| C1 | Real-time RNN inference is feasible on the MCU (conditionally) | Sensor-in-loop matrix: GRU 12.01 / LSTM 12.98 / FastGRNN 13.98 ms end-to-end @ LUT+100 kHz, all < 20 ms (`B2_RESULTS.md`) |
| C2 | Model ranking reverses with the deployment constraint | Tier-1 equal-H: GRU wins all 3 datasets; deployment-budget equal-byte: FastGRNN wins 2/3 (`experiments/*.json`) |
| C3 | Real-time needs LUT **and** 100 kHz I2C (two preconditions) | 24-run sensor matrix: only LUT+100 kHz passes; LUT+10 kHz all fail (sensor read 8.4 ms); no-LUT all fail |
| C4 | Sensor path is first-order, invisible to inference-only bench | Bench (inference-only) shows all LUT configs "OK"; with the real sensor at 10 kHz they all FAIL — the 8.4 ms read is the deciding factor |
| C5 | Compiler -O is nearly ineffective (no-FPU mechanism) | 36-run -O sweep: no-LUT 0–5%, LUT 5–9%, plateau at -O2; `use_hw_mpy=none` |
| C6 | LUT is a cell-independent real-time prerequisite | Bench latency LUT vs no-LUT, all 3 cells: no-LUT LSTM/FastGRNN fail, GRU marginal |
| C7 | Q15 quantization is near-lossless | Compression ablation: dF1 ≈ 0 across 3 datasets × 3 cells |
| C8 | System energy is platform-dominated; sensor ≈ doubles it | Sensor energy matrix: ~32 mW @100 k / ~34 mW @10 k, cell/LUT-independent; vs 17.7 mW MCU-only bench |
| C9 | Energy tracks latency (constant power); LUT cuts energy via latency | Derived energy (measured P × measured t): LUT saves 37–46% per window |
| C10 | Fairness of the ranking reversal (not an artifact of the budget) | Two compression routes per baseline (shrink-H + pruning), best-of; GRU/LSTM re-run at equal 200 epochs; FastGRNN win survives |

## 6. Limitations (state plainly — earns reviewer trust)

- Hardware experiments center on **HAPT**; the 3-dataset sweep is software-only. The hardware
  feasibility result is dataset-independent in structure (compute cost is data-independent), stated as
  such — not silently generalized.
- Single MCU family (**MSP430G2553**); not tested on Cortex-M or other architectures.
- Firmware **busy-waits** between samples (no LPM sleep) → reported energy is an **active-regime upper
  bound**; LPM duty-cycling is future work.
- The sensor-vs-inference latency **split** is unresolvable below ~1 ms (1 ms timer); the **end-to-end**
  figure is reliable.
- Deployed firmware is a **single config** (HAPT seed0, dense shrink-H); reported alongside 5-seed
  distributions, not instead of them.
- I2C tested at **10 kHz and 100 kHz**; higher (400 kHz fast-mode) not tested.
- Compression **structure** differs per cell by design (fairness): FastGRNN low-rank+IHT+Q15;
  GRU/LSTM shrink-H or pruning + weight-Q15. Q15 quantization is the **common final step** — this is
  NOT the same as "all cells compressed identically." State the distinction explicitly to preempt the
  "you rigged the budget" critique.

## 7. Case study

**HAR** (Human Activity Recognition) — the canonical wearable sequential-sensing task. It is the
vehicle that instantiates the framework, not the paper's object.

## 8. Editor's checklist (the work from here)

- [ ] Title: keep **"multiplier-less"**; hook on the finding, not the generic "deployment-aware."
- [ ] Abstract: LEAD with C1–C3 (findings), then say how (the methodology). Preserve the
      "invisible to inference-only benchmarks" edge.
- [ ] Every section serves the §2 thesis; cut experiments that don't.
- [ ] Elevate the **no-FPU mechanism (C5)** to a contribution, not a Discussion aside.
- [ ] Front-load the **fairness controls (C10)** next to the ranking-reversal claim (C2).
- [ ] "Relation to Prior Version" note: under-specified → refined, not a retraction.
- [ ] Repo reproducibility pass before submission (task D0): script→table mapping, seed/config
      transparency, firmware build settings.

---

*Locked after ~two months of reproduction → investigation → refinement. The single biggest change from
v1 is not "more experiments" — it is that the work now has a **scientific thesis** and a **purpose**,
and every claim is traceable to measured evidence. From here: editor's hat on.*
