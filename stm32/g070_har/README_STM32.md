# STM32G070RB — cross-platform validation (E5)

Third architecture class for the portability story: after Arduino Uno (AVR 8-bit) and
MSP430G2553 (16-bit), the Nucleo-G070RB (ARM Cortex-M0+, 32-bit, hardware multiplier).
Results and analysis: [E5_RESULTS.md](E5_RESULTS.md).

The portability minimum (output-equivalence + latency + memory) needs **only the USB
cable** — the on-board ST-LINK supplies power, flashing, and the virtual COM port (UART)
over that one cable. No breadboard, no sensor.

## What actually built this (not HAL/CubeMX)

CubeIDE 2.2.0 here has **no CubeMX / board wizard**, so the firmware is **bare-metal,
register-level** — no HAL, no `.ioc`. The whole application is one file, [`main.cpp`](main.cpp),
which pokes RCC + GPIOA + USART2 + SysTick directly (raw addresses, see the file header).
Bring-up was staged: LED blink → raw USART2 banner → model. `app.cpp`/`app.h` are an earlier
HAL-based scaffold, **superseded** and kept only for reference — the flashed firmware is
`main.cpp`.

The model files (`gru.cpp`, `gru.h`, `lut.h`, `model_weights.h`, and the LSTM/FastGRNN
equivalents) are the MSP430 deployment build **byte-for-byte unchanged** — that identity is the
whole point. They compile as **C++** (the Q15 tables in `model_weights.h` rely on C++ const
internal linkage). A host pre-flight (`g++`) confirms each cell's `C_PRED` before any flash, so
when the G070 prints the same, the only variable that changed is the silicon.

## Build it (per cell)

1. CubeIDE → **File → New → STM32 Project** → **Empty Project** for the NUCLEO-G070RB, language
   **C++**. (No peripheral init — we do it in `main.cpp`.)
2. Put into the project: `main.cpp` + the cell's `<cell>.cpp` → `Core/Src`; the cell's
   `<cell>.h`, `model_weights.h`, `lut.h`, and `test_windows.h` → `Core/Inc`. **Exactly one**
   `model_weights.h` / `lut.h` may be present (shared filename + include guard + colliding `const`
   arrays), so keep only the current cell's set.
3. **Project → Build**. Clean build, no HAL.
4. **Run → Run** (flashes over ST-LINK). Open a serial monitor at **115200** on the VCP COM port.

Expected output:
```
=== STM32G070RB HAR - cross-platform validation ===
cell=GRU  H=6  clock=16MHz  USE_LUT=1
[equivalence] ...
  window 0..4: G070=X  host-C=X  OK
[equivalence] 5/5  -> CROSS-PLATFORM IDENTICAL
[latency] window ~926 ms   per-step ~7.239 ms
[latency] real-time @50Hz (step<20ms): OK
done.
```

## Switching cells (GRU ↔ LSTM ↔ FastGRNN)

`main.cpp` is cell-agnostic except for four things:
1. the cell header include — `#include "gru.h"` → `lstm.h` / `fastgrnn.h`
2. the three calls — `gru_reset/step/predict` → `lstm_*` / `fastgrnn_*`
3. the banner string — `cell=GRU`
4. `test_windows.h` line for `C_PRED[]` — set to that cell's host reference:
   **GRU `{4,1,4,5,1}` · LSTM `{4,0,4,5,0}` · FastGRNN `{4,2,4,5,1}`** (they differ on the
   confusable walking/stairs windows 1 and 4 — each cell must use its own reference or the
   equivalence check reports a false MISMATCH).

Then swap the cell's `*.cpp/*.h/model_weights.h/lut.h` in the project and rebuild.

FastGRNN's header uses different macro names internally (`WINDOW_LEN`, `INPUT_CHANNELS`,
`HIDDEN_STATE_SIZE`) but its `model_weights.h` still exports `HIDDEN_SIZE`/`WINDOW_T`/`INPUT_DIM`,
which is what `main.cpp` uses — so no renames are needed in `main.cpp`.

## The LUT knob and the no-LUT row

`USE_LUT` lives in **one place per cell — the cell header** (`gru.h`/`lstm.h`/`fastgrnn.h`,
`#ifndef USE_LUT / #define USE_LUT 1`). Both the model `.cpp` and `main.cpp`'s banner include that
header, so one edit drives both. **Do not** `#define USE_LUT` in `main.cpp`: it would not reach the
cell `.cpp` (separate translation unit), the model would keep the default while the banner claimed
otherwise, and both builds come out byte-identical (this trap cost time during bring-up).

For the no-LUT latency row: flip the cell header to `#define USE_LUT 0`, then **Project → Clean →
Build** (a full rebuild — an incremental build does not recompile the cell `.cpp`), reflash.

## Memory

After a build the Console's `arm-none-eabi-size` line gives `text data bss`:
**Flash = text + data, RAM = data + bss.** Record per cell (see E5_RESULTS.md). Note this is a
validation build (embeds test windows + harness); the authoritative deployment footprint is the
MSP430 production number.
