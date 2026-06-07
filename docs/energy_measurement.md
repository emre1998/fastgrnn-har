# Energy Measurement Protocol

Measurement methodology used to fill in the energy numbers in
`paper/en/sections/05_results.tex` (Section "Energy Consumption").

## Equipment

| Item | Purpose | Notes |
|------|---------|-------|
| **USB inline ammeter** (UM25C, UM34C, KEWEISI, or similar) | Reads current, voltage, and accumulated mWh in real time | Any tester with mA resolution and a stable display works. UM25C is the most common. |
| Arduino Uno R3 + USB cable | One target | Powered through the on-board USB-to-serial bridge |
| MSP-EXP430G2 LaunchPad + USB cable | Second target | Same idea; the LaunchPad regulates to 3.3 V on board |

> The on-board USB-to-serial bridges add their own ~10 mA. The published
> values in the paper subtract a per-board idle baseline measured with
> the firmware looping in `__WFI()` / `LPM3` so the reported numbers
> reflect the CPU+inference cost, not the dev-kit overhead.

## Firmware: the energy benchmark mode

A dedicated `TEST_MODE 3` ("ENERGY") is added to both the Arduino and
MSP430 firmware. It loops on `fastgrnn_step()` for a fixed wall-clock
window with **no UART output**, **no sensor I/O**, and **no LED toggle**.
Three sub-modes select the workload:

| `BENCH_MODE` | Behavior | What it isolates |
|---|---|---|
| `0` | Idle: `delay(20)` between empty samples | Baseline current of the MCU + dev-kit USB bridge |
| `1` | 50 Hz streaming: `fastgrnn_step(zero_input)` every 20 ms, idle the rest | Real-world 50 Hz HAR power draw (matches paper §5.7) |
| `2` | Continuous: tight loop of `fastgrnn_step(zero_input)` | Worst-case "always-on" inference power |

Build, flash, and run each sub-mode for at least 60 s before reading the
ammeter so the moving average has stabilized.

## Step-by-step

1. Plug `USB host -> ammeter -> board`.
2. Flash the firmware with `TEST_MODE 3` and the chosen `BENCH_MODE`.
3. Wait 60 s; record the displayed current (mA) and voltage (V).
4. Repeat for the three `BENCH_MODE` values, on each board.
5. Fill the table in `paper/en/sections/05_results.tex` (TODO markers).

## Derived quantities

For the per-sample and per-window numbers, combine the measurements
with the latencies already reported in paper §5.7
(9.21 ms/sample on Arduino, 13.0 ms/sample on MSP430).

```
P_active   = V * I_active      (continuous inference)
P_idle     = V * I_idle
P_50Hz     = V * I_50Hz        (directly measured, no model needed)

E_per_sample (model)   = t_inference * P_active
                       = 9.21 ms * P_active  (Arduino)
                       = 13.0 ms * P_active  (MSP430)

E_per_sample (deployed) = (20 ms) * P_50Hz
                        i.e. integrates the active + idle within one period

E_per_window = 128 * E_per_sample (deployed)
```

A 2000 mAh, 3.7 V Li-ion battery delivers approximately
`2000 mAh * 3.7 V = 7.4 Wh`. The deployed battery life is
`7.4 Wh / P_50Hz`.

## Expected ranges (sanity check)

Based on the ATmega328P and MSP430G2553 datasheets:

| Platform | Vcc | I_idle (typ.) | I_active (typ.) | Notes |
|---|---|---|---|---|
| Arduino Uno (ATmega328P @ 16 MHz) | 5.0 V | ~30-40 mA | ~45-55 mA | Includes USB-to-serial bridge (~10 mA) |
| MSP430G2553 LaunchPad @ 16 MHz | 3.3 V | ~5-8 mA | ~10-15 mA | TUSB3410 bridge dominates on the LaunchPad; bare MSP430 < 5 mA |

If measurements are outside these bands by more than 2x, double-check
the wiring, the `BENCH_MODE` flag, and that the USB bridge has stopped
spamming heartbeats.

## Reporting

In the paper, energy is reported as `mJ/sample` and `mJ/window`
alongside the existing latency numbers. The numbers fold directly into
the motivation in §1 (chip shortage + carbon framing): a deployed
50 Hz HAR pipeline on the MSP430 should consume on the order of
microjoules per inference, well below any cloud-streaming baseline.
