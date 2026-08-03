# v2 — Direction & Handoff (for continuing with Codex)

*Status as of 2026-08-02. This note carries the strategic decisions made while drafting v2,
so an assistant (e.g. Codex) can continue without the prior conversation. The full Turkish
working draft is `paper/v2_draft_tr.md` (8 sections, sealed).*

## Where we are

- v1 (on arXiv, `2606.17249`): "From Compression to Deployment: Real-Time and Energy-Efficient
  FastGRNN on Ultra-Constrained Microcontrollers" — an end-to-end open-source **reproduction of
  FastGRNN** on bare-metal MCUs (Arduino AVR + MSP430), with a LUT recipe, cross-platform
  bit-equivalence, warm-up, and INA226 energy. Its weak point: it borrowed the claim "FastGRNN is
  the best cell."
- v2 work added: a GRU/LSTM/FastGRNN comparison across two regimes (equal-capacity vs equal-byte),
  bootstrap CIs, a one-device CPU reproduction, and a 3rd platform (STM32G070, Cortex-M0+).
- A full Turkish v2 draft was written (Abstract → Conclusion), then we did an honest novelty check.

## Honest assessment (the pivot)

**v2 as-is is not ready for a peer-reviewed journal (JSA), and reframing will not fix that.**
Two neighbor papers shadow most of it:
- **Daghero et al., ACM TECS 2022 (10.1145/3542819)** — HAR on MCUs, **same datasets** (HAPT, WISDM),
  a richer size-accuracy Pareto (<60 B → kB), sub-byte + mixed-precision quantization, RISC-V
  deployment with latency+energy. CNNs; adaptive inference is their novelty. This shadows our
  Pareto (E4), quantization (E5), and bench deployment (E6).
- **Shakerian et al., Sensors 2023** — sensor-embedded real-time HAR on ESP32 (capable, FPU),
  1D-CNN. A neighbor, not a duplicate.

**Neither does our one distinct thing:** the **sensor-in-the-loop verdict flip** (a config that
passes an inference-only benchmark FAILS end-to-end once the real sensor is in the loop) plus the
**controlled necessity ablation** (LUT activations AND a fast sensor bus are jointly required),
on a **multiplier-less, no-FPU** MCU.

## Decision

- **Do not rush to a journal.** Rest, then revise with the direction below.
- **Two venue paths:** (1) close v2 now as a strong **arXiv** revision (the self-correction + the
  sensor-boundary observation is a real contribution to the record); (2) later, a journal/workshop
  paper earned by focused new work on the one finding.

## New identity (the "recipe")

Pivot back to the v1 spine and treat the cell comparison as seasoning:

- **Main dish (~65%): a rigorous, open-source end-to-end reproduction + deployment characterization
  of FastGRNN** for HAR on ultra-constrained, multiplier-less MCUs — LUT recipe, bit-exact
  cross-platform inference, **conditional real-time feasibility with the sensor in the loop**
  (the two preconditions), measured (INA226) energy, warm-up. Nobody has reproduced FastGRNN
  end-to-end on a bare-metal, multiplier-less MCU at this rigor — that is the anchor.
- **Seasoning (~20%): the GRU/LSTM comparison** as an honest scoping note — "is FastGRNN actually
  the best cell? It is regime-dependent (GRU at equal capacity; FastGRNN at equal byte where
  supported, WISDM; a tie on HAPT), so we make no universal-best claim." This also elegantly
  discharges the v1 self-correction. NOT a main contribution — it put us in Daghero's shadow.
- **Garnish (~15%):** 3-architecture cross-platform, the no-FPU/-O mechanism, bootstrap CIs —
  rigor evidence, footnote weight.

One-line identity: *"A rigorous, open-source end-to-end reproduction and deployment
characterization of FastGRNN for HAR on ultra-constrained, multiplier-less MCUs, including the
conditions under which real-time operation holds."*

## Gaps to fill (focused next work, ranked)

1. **n=1 → n≥2 (most critical).** The sensor-in-loop verdict-flip is shown on ONE MCU + ONE task.
   Strengthen it: bring the real sensor into the loop on a **second MCU** (we ran G070/E8 at
   prediction+latency only, NOT sensor-in-loop) and/or a **second sensing task**.
2. **Make it non-obvious.** Reviewer risk: "a bigger loop obviously flips some configs." Quantify
   *how much* inference-only misleads; show it changes the **decision/design choice**, not just
   the number (inference-only would pick the wrong design).
3. **"Why RNN?"** Daghero + Shakerian chose CNN (arguably better for HAR-on-MCU). Best fix: frame
   the sensor-boundary finding as **model-family-agnostic** (it is about the acquisition path, not
   the cell) so RNN needs no defense and the finding generalizes.
4. **Related-work niche.** Cite Daghero + Shakerian explicitly; state what they do and the single
   thing we uniquely add.
5. **Demote the shadowed parts.** E4 (Pareto), E5 (quantization), E1/E2 (accuracy) → context /
   reproduction, not headline contributions. Don't fight Daghero on his turf.

## Verified citations to add (all real, web-confirmed 2026-08-01)

- `hasanpour2025edgemark` — EdgeMark, Journal of Systems Architecture 167:103488 (2025).
- `bartoli2025benchmarking` — Benchmarking Energy and Latency in TinyML, IJCNN 2025 (pre/inference/
  post phase decomposition — echoes our sensor-path point).
- `daghero2022har` — HAR on Microcontrollers with Quantized and Adaptive DNNs, ACM TECS 21(4):46
  (2022). **The key neighbor to position against.**

## Key files

- `paper/v2_draft_tr.md` — full Turkish v2 draft (8 sections).
- `B2_RESULTS.md` — all MSP430 hardware measurements (bench latency, energy, -O sweep, memory,
  sensor-in-loop 24-run matrix). Primary hardware data.
- `BOOTSTRAP.md` — bootstrap CIs (equal-capacity + equal-byte margins).
- `stm32/g070_har/E5_RESULTS.md` — cross-platform (3 cells, GRU/LSTM/FastGRNN).
- `MASTER_TABLE.md`, `experiments/*.json` — accuracy + deployment summaries.
- `PAPER_THESIS_LOCK.md`, `AI_HANDOFF_BRIEFING.md` — earlier (pre-pivot) framing; read with the
  pivot above in mind.
