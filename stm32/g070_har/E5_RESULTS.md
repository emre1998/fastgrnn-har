# E5 — STM32G070 Cross-Platform Validation Results
*All three cells (GRU, LSTM, FastGRNN) run on a third architecture class (ARM Cortex-M0+, 32-bit),
extending the v1 Arduino Uno (AVR 8-bit) + MSP430G2553 (16-bit) portability line. This is a
side-validation: the backbone remains MSP430 + the deployment-engineering thesis.
For the MSP430/Arduino numbers this compares against, see [B2_RESULTS.md](../../B2_RESULTS.md).*

## Setup

- **Board:** NUCLEO-G070RB — STM32G070RB, Cortex-M0+, **hardware 32×32 multiplier (MULS), no FPU**.
- **Clock:** HSI **16 MHz** (reset default). Chosen deliberately to match the MSP430G2553's
  16 MHz so the only variable in the head-to-head is the **architecture** (presence of a
  hardware multiplier), not clock speed. The part can run to 64 MHz; not used here.
- **Firmware:** bare-metal, register-level (no HAL, no CubeMX — that wizard is absent in
  CubeIDE 2.2.0). Staged bring-up: LED blink → raw USART2 (PA2/PA3, VCP, BRR=139@16MHz) → model.
  One `main.cpp` (register/UART skeleton) serves every cell; only the cell header include, the three
  `*_reset/step/predict` calls, and the banner string change between cells.
- **Model code:** each cell's `*.cpp`, `*.h`, `model_weights.h`, `lut.h` copied **byte-for-byte**
  from the MSP430 deployment build (`msp/ccs_{cell}_production/`). Compiled as C++, exactly as on
  MSP430 (the Q15 weight tables rely on C++ const internal linkage). Deployed hidden sizes:
  GRU H=6, LSTM H=5, FastGRNN H=16 (shrink-to-budget / low-rank models).
- **Reference:** a host pre-flight (`g++`) computes each cell's `C_PRED` on the five embedded
  windows **before** any flash. The reference is **per-cell** — the three cells disagree on the
  ambiguous walking/stairs windows (1 and 4), so each cell is checked against its own host output,
  never against another cell.

## Equivalence (cross-platform identity)

Five embedded test windows, G070 prediction vs that cell's host-C reference (`C_PRED`), both LUT knobs:

| Cell | host-C `C_PRED` | USE_LUT=1 | USE_LUT=0 |
|------|-----------------|-----------|-----------|
| GRU (H=6) | `{4,1,4,5,1}` | 5/5 IDENTICAL | 5/5 IDENTICAL |
| LSTM (H=5) | `{4,0,4,5,0}` | 5/5 IDENTICAL | 5/5 IDENTICAL |
| FastGRNN (H=16) | `{4,2,4,5,1}` | 5/5 IDENTICAL | 5/5 IDENTICAL |

The three references differ on the confusable windows (window 1: GRU→UPSTAIRS, LSTM→WALKING,
FastGRNN→DOWNSTAIRS) — a concrete illustration of why each cell needs its own reference. Since
MSP430 also matches host-C ([B2 §B1 bit-exact](../../B2_RESULTS.md)), transitively
**G070 == host-C == MSP430** at the prediction (argmax) level, for all three cells. Dropping the
LUT does not change the argmax on these windows either.

**Claim level:** *same predictions (argmax)*, **not** bit-identical hidden state — MSP430 (TI RTS
soft-float) and G070 (GCC libgcc soft-float) use different soft-float libraries, so intermediate
floats can differ in the low bits while the classification is identical. Same two-claim distinction
as VERIFICATION.md #5.

## Latency (pure inference, 16 MHz, per-step ms; %util vs the 20 ms @50Hz budget)

| Cell | USE_LUT | per-step | per-window (128) | %util | real-time? |
|------|---------|----------|------------------|-------|------------|
| GRU (H=6) | 1 | **7.239 ms** | 926 ms | 36 % | ✅ OK |
| GRU (H=6) | 0 | **12.023 ms** | 1539 ms | 60 % | ✅ OK |
| LSTM (H=5) | 1 | **7.111 ms** | 910 ms | 36 % | ✅ OK |
| LSTM (H=5) | 0 | **13.501 ms** | 1728 ms | 68 % | ✅ OK |
| FastGRNN (H=16) | 1 | **10.835 ms** | 1386 ms | 54 % | ✅ OK |
| FastGRNN (H=16) | 0 | **17.000 ms** | 2176 ms | 85 % | ✅ OK |

### Head-to-head with MSP430 (same 16 MHz, byte-identical model, pure inference)

| Cell | MSP430 LUT=1 | MSP430 LUT=0 | G070 LUT=1 | G070 LUT=0 |
|------|--------------|--------------|------------|------------|
| GRU | 12.116 (61 %) | 19.273 (96 %, marginal) | **7.239 (36 %)** | **12.023 (60 %)** |
| LSTM | 12.370 (62 %) | **22.097 (FAIL)** | **7.111 (36 %)** | **13.501 (68 %)** |
| FastGRNN | 13.900 (70 %) | **26.100 (FAIL)** | **10.835 (54 %)** | **17.000 (85 %)** |

*MSP430 figures: [B2_RESULTS.md](../../B2_RESULTS.md) latency table, pure inference.*

**Findings**

1. **The hardware multiplier turns every no-LUT config real-time.** On the multiplier-less MSP430,
   without the LUT, GRU is marginal (96 %), and LSTM and FastGRNN outright fail (110 %, 26 ms). On
   G070 the same byte-identical models all clear 20 ms with the LUT off (60 % / 68 % / 85 %). The
   worst case anywhere in the matrix — FastGRNN no-LUT — still passes at 17 ms.
2. **Multiplier ≈ LUT as a lever (holds across cells).** GRU no-LUT on G070 (12.02 ms) ≈ GRU LUT-on
   on MSP430 (12.12 ms), within 0.8 %. Two different levers — a hardware multiplier vs an activation
   table — buy back roughly the same real-time margin, same clock.
3. **The LUT still matters on G070, just less urgently.** LUT gives GRU ~1.66×, LSTM ~1.90×,
   FastGRNN ~1.57×; G070 clears real-time with the LUT off for all three, where MSP430 is marginal
   or failing. So the LUT recipe is a comfort margin on M0+ and a hard requirement on the 16-bit part.
4. **Gate-count vs hidden-size roughly cancels:** LSTM (H=5, 4 gates) ≈ GRU (H=6, 3 gates) in latency
   (7.11 ≈ 7.24 ms with LUT), on both platforms — the cell ordering is preserved across silicon.

## Memory footprint

*This is a **validation/test build** — it embeds five 128×3 float test windows (7 680 B in Flash),
the equivalence + latency harness, and USART banner code, none of which exist in a real deployment.
Compare against the MSP430 **test-harness** column, not production. The MSP430 production footprints
(GRU 5 392 B / LSTM 5 742 B / FastGRNN 5 544 B Flash) are the authoritative deployment numbers; a
stripped G070 production build was not measured — RAM is the meaningful portability figure here.*

| Cell | USE_LUT | text | data | bss | **Flash** | **RAM** |
|------|---------|------|------|-----|-----------|---------|
| GRU | 1 | 18 956 | 0 | 1 616 | 18 956 B (18.5 KB) | 1 616 B (1.58 KB) |
| GRU | 0 | 18 648 | 80 | 1 928 | 18 728 B (18.3 KB) | 2 008 B (1.96 KB) |
| LSTM | 1 | 18 952 | 0 | 1 632 | 18 952 B (18.5 KB) | 1 632 B (1.59 KB) |
| LSTM | 0 | 18 644 | 80 | 1 944 | 18 724 B (18.3 KB) | 2 024 B (1.98 KB) |
| FastGRNN | 1 | 19 228 | 0 | 1 656 | 19 228 B (18.8 KB) | 1 656 B (1.62 KB) |
| FastGRNN | 0 | 18 920 | 80 | 1 968 | 19 000 B (18.6 KB) | 2 048 B (2.00 KB) |

RAM stays **≤2 KB of the G070's 36 KB (≤5.6 %)** for all three cells — the models stream comfortably
on a mainstream M0+.

**Finding — the LUT is essentially free in Flash on this MCU (all three cells).** Removing the 2 KB
LUT tables barely changes Flash (−228 B / −228 B / −228 B, not −2 KB): with `USE_LUT=0` the no-LUT
path pulls in libm's soft-float `expf`/`tanhf` (G070 has no FPU), whose code roughly replaces the
tables it removed, and RAM even rises slightly (libm internal state). So on an FPU-less M0+ the LUT
costs ~nothing in Flash while buying the 1.5–1.9× latency — a clean argument for the LUT recipe.

## Toolchain / reproducibility notes

- The `USE_LUT` knob lives in one place per cell — the cell header (`gru.h` / `lstm.h` /
  `fastgrnn.h`, `#ifndef USE_LUT / #define USE_LUT 1`). Both the model (`*.cpp`) and the firmware
  banner (`main.cpp`) include that header, so a single edit drives both. A per-file `#define` in
  `main.cpp` would **not** reach the cell `.cpp` (separate translation unit): the model would
  silently keep the default while the banner claimed otherwise. During bring-up this exact trap
  produced a byte-identical binary for both knob values until the knob was moved into the shared
  header — recorded here so it is not repeated per cell.
- **The `C_PRED` reference is per-cell** and must be regenerated by a host pre-flight for each cell
  (the three cells differ on windows 1 and 4). Reusing another cell's `C_PRED` produces a false MISMATCH.
- Toggling `USE_LUT` requires **Project → Clean → Build** (a full rebuild); an incremental build did
  not recompile the cell `.cpp` after the header change.
- Only one `model_weights.h` / `lut.h` may be in the project at a time (shared filename + guard +
  colliding `const` arrays) — the current cell's set, with the other cells' `.cpp/.h` removed.
- Build: `arm-none-eabi-g++`, `-mcpu=cortex-m0plus`, `-mfloat-abi=soft`, `-O0` (Debug),
  `--specs=nano.specs`, `-lm -lstdc++`.

## Status

E5 portability minimum is **complete for all three cells (GRU, LSTM, FastGRNN)**: equivalence
(5/5, both knobs, per-cell reference) + latency (all six rows real-time) + memory. The cross-platform
result now spans **three cells across three architecture classes** — AVR 8-bit (Arduino), MSP430
16-bit, and ARM Cortex-M0+ 32-bit — symmetric with the MSP430 cell matrix in B2.
