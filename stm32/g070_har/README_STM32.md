# STM32G070RB — cross-platform validation (Phase 1–3)

The portability minimum (output-equivalence + latency + memory) needs **only the
USB cable**. The on-board ST-LINK supplies power, flashing, and the virtual COM
port (UART) over that one cable — no breadboard, no INA226, no sensor.

Board facts (Nucleo-G070RB): user LED **LD4 = PA5**; virtual COM = **USART2
(PA2/PA3)**; ST-LINK/V2-1 embedded.

The model files here (`gru.cpp`, `gru.h`, `lut.h`, `model_weights.h`) are the MSP430
deployment build **byte-for-byte unchanged** — that identity is the whole point of
the check. They compile as **C++**, exactly as the MSP430 build did (the weight
tables in `model_weights.h` rely on C++ const internal linkage). A host pre-flight
already confirms `gru.cpp` produces `C_PRED = [4, 1, 4, 5, 1]` on these windows, so
when the G070 prints the same, the only variable that changed is the silicon.

---

## Phase 0 — install (one-time)

Install **STM32CubeIDE** (free, from ST). Plug the board in via USB; Windows should
enumerate an ST-LINK and a "STMicroelectronics STLink Virtual COM Port" — note its
COM number (Device Manager → Ports).

## Phase 1 — create the project

1. CubeIDE → **File → New → STM32 Project**.
2. **Board Selector** tab → search **NUCLEO-G070RB** → select it → **Next**.
   Name it `g070_har`, and set **Targeted Language = C++** (the model compiles as
   C++; if you forget, you can right-click the project → **Convert to C++** later).
   **Finish**. When asked "initialize peripherals to default mode?", answer
   **Yes** — this auto-configures the clock, LD4 (PA5), and USART2 as the VCP.
3. In the `.ioc` view, confirm:
   - **USART2** = Asynchronous, **115200** baud, 8N1 (it is the VCP).
   - **PA5** = GPIO_Output (labelled LD4).
   - SysTick is on by default (HAL time base — we use `HAL_GetTick()`).
   - *(optional)* Clock Configuration → set HCLK to **64 MHz**; the default HSI
     16 MHz also works, and the banner prints whichever is active.
4. **Project → Generate Code** (or Ctrl+S on the `.ioc`).

## Phase 2 — add the model + app code

5. Copy into the generated project:
   - `gru.cpp`, `app.cpp` → `Core/Src/`
   - `gru.h`, `lut.h`, `model_weights.h`, `test_windows.h`, `app.h` → `Core/Inc/`
6. Open `Core/Src/main.c` and add, in the marked user regions only:
   ```c
   /* USER CODE BEGIN Includes */
   #include "app.h"
   /* USER CODE END Includes */
   ```
   ```c
   /* USER CODE BEGIN 2 */
   app_run();                 /* never returns */
   /* USER CODE END 2 */
   ```
   (`USER CODE BEGIN 2` sits right after `MX_USART2_UART_Init()`.)
7. **Project → Build** (Ctrl+B). It should build clean. `gru.h` has `extern "C"`
   guards, so the C++ model links cleanly with the C `main.c`.

## Phase 3 — flash and read

8. **Run → Run** (flashes over ST-LINK).
9. Open a serial monitor at **115200** on the VCP COM port (CubeIDE has one:
   Window → Show View → Console → the "Open a Terminal" / or use PuTTY/TeraTerm).
10. You should see:
    ```
    === STM32G070RB HAR - cross-platform validation ===
    cell=GRU  H=6  clock=64MHz  USE_LUT=1
    [equivalence] G070 prediction must equal C_PRED (host-C reference):
      window 0: G070=4  host-C=4  OK
      window 1: G070=1  host-C=1  OK
      window 2: G070=4  host-C=4  OK
      window 3: G070=5  host-C=5  OK
      window 4: G070=1  host-C=1  OK
    [equivalence] 5/5 match  ->  CROSS-PLATFORM IDENTICAL
    [latency] per-step ~X.XXX ms   window ~XXX ms
    [latency] real-time @50Hz (step < 20 ms): OK
    done. LED now blinks ...
    ```

**Expected equivalence:** `C_PRED = [4, 1, 4, 5, 1]`. Five OKs means G070 == host-C,
and since MSP430 also matches host-C, G070 == MSP430 — the cross-platform result.

## Latency rows and memory

- **no-LUT row:** Project → Properties → C/C++ Build → Settings → Preprocessor →
  add symbol `USE_LUT=0`, rebuild, reflash. The banner will show `USE_LUT=0`.
- **memory footprint:** after a build, CubeIDE prints the linker summary
  (`text` / `data` / `bss`) in the Console; Flash = text+data, RAM = data+bss.
  Record it per cell.

## Other cells (LSTM, FastGRNN)

Same steps with that cell's files (to be copied when we get there). GRU first
because it is the smallest and is the one already deployed on the MSP430, so the
equivalence reference is exact.
