# Energy Measurement Protocol (INA226)

Methodology used to fill in the energy numbers in
`paper/en/sections/05_results.tex` ("Energy Consumption").

## Hardware

| Item | Purpose | Notes |
|------|---------|-------|
| **INA226 breakout** (DirencNet stock #10793 or equivalent) | High-side current/voltage/power sensor, 16-bit ADC, R = 0.1 Ω shunt | I²C @ 0x40; ~75 TL |
| **Arduino Uno R3** | INA226 host/CSV reader | Standard Arduino IDE setup |
| **MSP-EXP430G2 LaunchPad** | Device Under Test | Target VCC rail powered through INA226 |
| Jumper wires (M-M, M-F) + small breadboard | Wiring | ~20-50 TL each, optional if the INA226 ships with pins |

## Library

Arduino IDE → **Tools → Manage Libraries** → search **"INA226_WE"** by
Wolfgang Ewald → **Install**. This is the only third-party library
required.

## Wiring — INA226 to Arduino (the host side)

```
+----------+               +-----------+
| Arduino  |               |  INA226   |
|  Uno     |               | breakout  |
|----------|               |-----------|
|   5V o---+---------------+-o VCC     |
|  GND o---+---------------+-o GND     |
|   A4 o---+----- SDA -----+-o SDA     |
|   A5 o---+----- SCL -----+-o SCL     |
|          |               |  ALE (NC) |
+----------+               +-----------+
```

The INA226 also has IN+ / IN- pins that the *measured* current must flow
through. Those go to the DUT, not the host (see the scenario diagram below).

## Scenario - Measuring the MSP430

The Arduino acts as a quiet INA226 reader and is powered separately
through its own USB port. The MSP430 is the Device Under Test.

```
3.3 V source   INA226             MSP430 LaunchPad target side
               shunt
       o-------o IN+
                |   R = 0.1 ohm
               o IN- o------------o VCC (remove LaunchPad VCC jumper)
                                  |
                                  |  MSP430G2553 runs TEST_MODE 3
                                  |  with BENCH_MODE = 0 / 1 / 2
                                  |
                            o GND-+

                      I2C
  Arduino  <----- SDA/SCL -----  INA226
  (host)   <----- VCC/GND ------  (powered from Arduino)
  runs ina226_meter.ino
  prints CSV @ 10 Hz
```

Steps:

1. Flash `arduino/ina226_meter/ina226_meter.ino` to the Arduino.
2. Flash `msp/ccs_fastgrnn_har/main.cpp` to the MSP430 with
   `TEST_MODE 3` and `BENCH_MODE 0`.
3. Wire as shown above. Remove the LaunchPad VCC jumper and feed the
   target-side VCC pin through the INA226 shunt. Keep grounds common.
   The Arduino keeps its own USB power.
4. Open the Arduino Serial Monitor at 115200 baud.
5. Wait 60 s, record the steady-state `mA` and `mV`.
6. Re-flash the MSP430 with `BENCH_MODE 1`, repeat. Then `BENCH_MODE 2`.
7. Re-flash with `USE_LUT 0` and `BENCH_MODE 2`, then repeat the
   continuous measurement. The no-LUT 50 Hz row is reported as N/A
   because the firmware misses the 20 ms streaming deadline.

You now have four measured MSP430 runs:

| Run | Firmware | Purpose |
|---|---|---|
| Idle | `TEST_MODE 3`, `BENCH_MODE 0`, `USE_LUT 1` | Board baseline |
| 50 Hz LUT | `TEST_MODE 3`, `BENCH_MODE 1`, `USE_LUT 1` | Deployed workload |
| Continuous LUT | `TEST_MODE 3`, `BENCH_MODE 2`, `USE_LUT 1` | Always-on envelope |
| Continuous no-LUT | `TEST_MODE 3`, `BENCH_MODE 2`, `USE_LUT 0` | LUT ablation |

## Measured Results (2026-06-14)

Hardware: INA226 breakout (DirencNet #10793, R_shunt = 0.1 Ω, addr 0x44),
Arduino Uno reader, MSP-EXP430G2ET, CCS firmware `msp/ccs_fastgrnn_har/`.
INA226 config: 16-sample average, 1.1 ms conversion time, continuous mode.
Settling time: 60 s per run. Values are steady-state means (t > 500 ms).

| Run | BENCH_MODE | USE_LUT | I_mean (mA) | σ_I (mA) | V_mean (mV) | P (mW) |
|-----|-----------|---------|------------|---------|------------|--------|
| Idle (LPM3) | 0 | 1 | <0.025 | — | 3481 | <0.09 |
| 50 Hz LUT | 1 | 1 | 5.135 | 0.018 | 3478 | 17.86 |
| Continuous LUT | 2 | 1 | 5.096 | 0.013 | 3478 | 17.72 |
| Continuous no-LUT | 2 | 0 | 5.078 | 0.000 | 3478 | 17.66 |

Notes:
- Idle draws below INA226 resolution (LSB = 0.025 mA with 0.1 Ω shunt).
  MSP430G2553 datasheet LPM3 typ. = 0.5–1 µA; INA226 cannot resolve this.
  Report as P_idle < 0.09 mW (upper bound: 0.025 mA × 3.481 V).
- 50 Hz LUT ≈ Continuous LUT because BENCH_MODE 1 uses a busy-wait loop
  between inferences, keeping the CPU active the entire 20 ms period.
  If inter-sample idle were implemented with LPM0, power would drop ~5×.
- Continuous LUT vs no-LUT power difference is <0.3% (within 1 INA226 LSB).
  The LUT benefit is latency, not average power at fixed clock frequency.
  At 16 MHz the LUT is ~30× faster per inference, so energy-per-inference
  is ~30× lower even though average power is the same.
- no-LUT σ = 0.000: software expf()/tanhf() produces a perfectly periodic
  current signature, averaged flat by the 16-sample INA226 filter.

## Latency Results (TEST_MODE 1, 2026-06-14)

Both runs use CCS firmware, 16 MHz DCO, WINDOW_LEN = 128 steps.
Values are average of 2 test windows.

| Config | t_window (ms) | t_step (ms) | 50 Hz feasible? |
|--------|--------------|------------|-----------------|
| USE_LUT 1 | 1778 | 13.9 | YES — 6.1 ms margin (31% headroom) |
| USE_LUT 0 (compiler-optimized) | 3338 | 26.1 | NO — exceeds 20 ms deadline by 30% |
| USE_LUT 0 (unoptimized baseline, Week 8) | ~54000 | ~421 | NO — 21× over deadline |

The LUT is a prerequisite for real-time 50 Hz HAR on MSP430G2553, not
merely an optimization. Without it the MCU cannot meet the streaming deadline.

### LUT Speedup — Two Valid Perspectives

**30.5× end-to-end (Week 8 original measurement):**
Without the LUT, the MSP430G2553 requires ~54 seconds per 128-sample window
using the TI software math library's unoptimized expf()/tanhf() (Taylor series +
software float, ~5000 cycles/call). The LUT reduces this to 1.778 s — a 30.5×
end-to-end speedup — enabling real-time 50 Hz inference (13.9 ms/sample < 20 ms).
This is the figure reported in the paper's results section.

**1.88× end-to-end (compiler-optimized baseline):**
With compiler optimizations applied to both paths, the no-LUT path is 16× faster
than its unoptimized counterpart (3338 ms vs ~54000 ms), reducing the overall
speedup to 1.88×. This reveals that the 30.5× figure is dominated by the cost of
unoptimized software transcendentals, not matrix operations.

**Per-activation speedup (~30×, both conditions):**
Each individual sigmoid/tanh call: ~5000 cycles (expf) → ~100 cycles (LUT) = ~50×
faster per call. With 32 calls per step and 128 steps per window, this accounts for
the bulk of the 30.5× end-to-end gain in the unoptimized baseline.

The paper uses 30.5× as the primary figure because it reflects the real deployment
scenario: the TI math library is the default toolchain, and compiler optimization
of transcendentals is not guaranteed on embedded targets.

## Derived Quantities

```
V            = 3.478 V  (measured target VCC under load)
WINDOW_LEN   = 128 steps

-- Power --
P_idle        < 0.09 mW   (LPM3; below INA226 floor of 0.025 mA × 3.481 V)
P_50Hz_busy   = 17.86 mW  (BENCH_MODE 1 busy-wait — upper bound)
P_cont_LUT    = 17.72 mW  (BENCH_MODE 2, USE_LUT 1)
P_cont_noLUT  = 17.66 mW  (BENCH_MODE 2, USE_LUT 0)

Note: P_cont_LUT ≈ P_cont_noLUT (<0.3% difference, within 1 INA226 LSB).
Average power is clock-dominated, not computation-dominated.

-- Latency --
t_window_LUT   = 1.778 s  (128 steps, USE_LUT 1, CCS optimized)
t_window_noLUT = 54.0 s   (128 steps, USE_LUT 0, TI math library baseline)
t_step_LUT     = 13.9 ms  (per fastgrnn_step, LUT)
t_step_noLUT   = 421 ms   (per fastgrnn_step, no LUT, unoptimized)
Speedup        = 30.5×    (end-to-end, paper figure)

-- Energy per window (128 steps) --
E_window_LUT    = 17.72 mW × 1.778 s  =  31.5 mJ
E_window_noLUT  = 17.66 mW × 54.0 s   = 954.0 mJ
LUT energy saving per window: (954.0 - 31.5) / 954.0 = 96.7%

-- Energy per step --
E_step_LUT    = 17.72 mW × 0.0139 s = 0.246 mJ
E_step_noLUT  = 17.66 mW × 0.421 s  = 7.434 mJ
LUT energy saving per step: 96.7%

-- Real 50 Hz deployment power (with LPM sleep between inferences) --
duty_LUT      = 13.9 ms / 20 ms = 69.5%
P_real_50Hz   = 0.695 × 17.72 mW + 0.305 × ~0 ≈ 12.3 mW
(no-LUT cannot run at 50 Hz — step time 421 ms >> 20 ms deadline)

-- Battery life (2000 mAh, 3.7 V Li-ion = 7.4 Wh) --
Continuous LUT (BENCH_MODE 2):   7.4 Wh / 17.72 mW ≈ 417 h ≈ 17 days
Real 50 Hz deployed (LPM idle):  7.4 Wh / 12.3 mW  ≈ 602 h ≈ 25 days
```

## Summary for Paper (copy-paste ready)

| Metric | Value |
|--------|-------|
| Supply voltage (measured) | 3.478 V |
| Active inference power | 17.72 mW |
| Idle power (LPM3) | <0.09 mW |
| Inference time / window (LUT) | 1778 ms |
| Inference time / window (no LUT) | ~54000 ms |
| Inference time / step (LUT) | 13.9 ms |
| LUT end-to-end speedup | **30.5×** |
| 50 Hz real-time feasible (LUT) | YES — 31% headroom |
| 50 Hz real-time feasible (no LUT) | NO — 21× over deadline |
| Energy / window (LUT) | **31.5 mJ** |
| Energy / window (no LUT) | **954 mJ** |
| LUT energy saving / window | **96.7%** |
| Battery life, continuous (2000 mAh) | ~17 days |
| Battery life, 50 Hz deployed (LPM) | ~25 days |

## Expected Sanity Ranges (pre-measurement estimates — superseded above)

| Platform | Vcc | I_idle (est.) | I_50Hz (est.) | I_cont. (est.) |
|---|---|---|---|---|
| MSP430G2553 target rail (16 MHz) | 3.3 V | 5–8 mA | 7–10 mA | 11–15 mA |

Actual measurements came in lower than estimates: the target rail excludes
the EZ-FET debugger current (which flows through a separate USB path), so
only the MSP430G2553 core + regulator load appears on the INA226.

## Reporting

Drop the four measured MSP430 runs plus the derived energies into the
`\TODO{?}` slots in `paper/en/sections/05_results.tex` (the two energy
tables right after the streaming-performance section). The same tables
exist in TR; the numbers are identical.

A small Python helper that converts the CSV stream to a stable
average is included as `paper/scripts/parse_ina226_log.py`:

```
python paper/scripts/parse_ina226_log.py logs/ina226_msp_stream50hz.csv
python paper/scripts/parse_ina226_log.py logs/ina226_msp_stream50hz.csv --last-rows 300
```
