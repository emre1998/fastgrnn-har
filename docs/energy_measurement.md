# Energy Measurement Protocol (INA226)

Methodology used to fill in the energy numbers in
`paper/en/sections/05_results.tex` ("Energy Consumption").

## Hardware

| Item | Purpose | Notes |
|------|---------|-------|
| **INA226 breakout** (DirencNet stock #10793 or equivalent) | High-side current/voltage/power sensor, 16-bit ADC, R = 0.1 Ω shunt | I²C @ 0x40; ~75 TL |
| **Arduino Uno R3** | INA226 host (Scenario A) **or** Device Under Test (Scenario B) | Standard Arduino IDE setup |
| **MSP-EXP430G2 LaunchPad** | Device Under Test (Scenario A) | Powered through the on-board USB-to-serial bridge |
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
through. Those go to the DUT, not the host (see scenario diagrams below).

## Scenario A - Measuring the MSP430

The Arduino acts as a quiet INA226 reader and is powered separately
through its own USB port. The MSP430 is the Device Under Test.

```
USB power      INA226             MSP430 LaunchPad
  source        shunt
  (5V) o-------o IN+
                |   R = 0.1 ohm
               o IN- o------------o VBUS (5V on the LaunchPad)
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
3. Wire as shown above. Power the MSP430 through the INA226 shunt
   (cut/reroute the LaunchPad's USB +5V line, or use the J6 power
   header). The Arduino keeps its own USB power.
4. Open the Arduino Serial Monitor at 115200 baud.
5. Wait 60 s, record the steady-state `mA` and `mV`.
6. Re-flash the MSP430 with `BENCH_MODE 1`, repeat. Then `BENCH_MODE 2`.

You now have three (mA, mV) pairs for the MSP430.

## Scenario B - Measuring the Arduino itself

The Arduino is both DUT and INA226 reader. There is no second board.

```
USB power      INA226            Arduino Uno
  source        shunt
  (5V) o-------o IN+
                |   R = 0.1 ohm
               o IN- o------------o 5V
                                  |
                                  |  Arduino runs fastgrnn_har.ino
                                  |  with TEST_MODE 3, BENCH_MODE = 0/1/2,
                                  |  and INA226_SELF_READ 1.
                                  |
                            o GND-+
                                  |
                       I2C        |
                 INA226 <----- A4/A5
                 VCC/GND <--- 5V/GND
```

The Arduino keeps Serial up, reads its own INA226 once per second,
and prints `t_ms, mA, mV, mW` to the Serial Monitor. The INA226's own
quiescent current (~0.5 mA) plus the periodic I²C and Serial cost
(~5 mA averaged) add a constant bias that applies to every
`BENCH_MODE` and therefore cancels out when comparing modes.

Steps:

1. Flash `arduino/fastgrnn_har/fastgrnn_har.ino` to the Arduino with
   `TEST_MODE 3`, `BENCH_MODE 0`, `INA226_SELF_READ 1`.
2. Wire as shown.
3. Open the Serial Monitor at 115200 baud.
4. Wait 60 s. The CSV lines stabilize within ~5 s; take the average of
   the last 30 lines.
5. Re-flash with `BENCH_MODE 1`, repeat. Then `BENCH_MODE 2`.

## Expected sanity ranges

If readings fall outside these bands by more than 2x, double-check
the wiring polarity and that the right `BENCH_MODE` was flashed.

| Platform | Vcc | I_idle (typ.) | I_50Hz (typ.) | I_cont. (typ.) | Notes |
|---|---|---|---|---|---|
| Arduino Uno (ATmega328P, 16 MHz) | 5.0 V | 30-40 mA | 35-45 mA | 45-55 mA | Includes USB-to-serial bridge (~10 mA) |
| MSP430G2553 LaunchPad (16 MHz) | 3.3 V | 5-8 mA | 7-10 mA | 11-15 mA | TUSB3410 bridge dominates the baseline |

## Derived quantities

```
P_active = V * I_active           (continuous inference)
P_idle   = V * I_idle
P_50Hz   = V * I_50Hz             (directly measured, includes idle + active)

E_per_sample_model    = t_inf * P_active
                      = 9.21 ms * P_active   (Arduino)
                      = 13.0 ms * P_active   (MSP430)

E_per_sample_deployed = 20 ms * P_50Hz        (integrates active + idle)
E_per_window          = 128 * E_per_sample_deployed   (2.56 s window)
```

For battery life: a 2000 mAh, 3.7 V Li-ion pack stores
`2000 mAh * 3.7 V = 7.4 Wh`. Hours of continuous operation at the
deployed 50 Hz workload: `7.4 Wh / P_50Hz`.

## Reporting

Drop the six (mA, mV) tuples plus the derived energies into the
`\TODO{?}` slots in
`paper/en/sections/05_results.tex` (the two energy tables right after
the streaming-performance section). The same tables exist in TR; the
numbers are identical.

A small Python helper that converts the CSV stream to a stable
average is included as
`paper/scripts/parse_ina226_log.py` (to be added).
