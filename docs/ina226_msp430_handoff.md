# INA226 + MSP430 Energy Measurement Handoff

This note summarizes the current state of the FastGRNN-HAR energy
measurement setup so another assistant can continue debugging without
reconstructing the whole conversation.

## Goal

Measure MSP430G2553 target-rail energy for the paper's MSP430-only
energy section using:

- Arduino Uno as INA226 I2C/CSV reader
- INA226 breakout as high-side current sensor
- MSP-EXP430G2 LaunchPad as the device under test

The intended measurement matrix is:

| Run | Firmware |
|---|---|
| Idle | `TEST_MODE 3`, `BENCH_MODE 0`, `USE_LUT 1` |
| 50 Hz LUT | `TEST_MODE 3`, `BENCH_MODE 1`, `USE_LUT 1` |
| Continuous LUT | `TEST_MODE 3`, `BENCH_MODE 2`, `USE_LUT 1` |
| Continuous no-LUT | `TEST_MODE 3`, `BENCH_MODE 2`, `USE_LUT 0` |

MPU6050 should not be connected for these measurements.

## Repository Changes Already Made

- `arduino/ina226_meter/ina226_meter.ino`
  - INA226 address changed to `0x44`.
  - New `INA226_WE` constants are used:
    - `INA226_AVERAGE_16`
    - `INA226_CONV_TIME_1100`
    - `INA226_CONTINUOUS`
- `arduino/i2c_scanner/i2c_scanner.ino`
  - Added to verify the INA226 I2C address.
- `msp/ccs_fastgrnn_har/main.cpp`
  - Has `TEST_MODE 3` energy benchmark mode and `BENCH_MODE`.
- `msp/ccs_fastgrnn_har/fastgrnn.cpp`
  - Has `USE_LUT` switch.
- `paper/scripts/parse_ina226_log.py`
  - Added CSV summarizer.
- `docs/energy_measurement.md`
  - Updated for MSP430-only measurements.

## Confirmed Working

The INA226 I2C connection works.

I2C scanner found:

```text
Found I2C device at 0x44
```

After changing `INA226_ADDR` to `0x44`, `ina226_meter.ino` prints:

```text
INA226 ready.
CSV output: t_ms,mA,mV,mW
```

With an earlier wiring attempt, the INA226 showed plausible current:

```text
mV ~= 3478
mA ~= 5 to 6.4
```

This proves the INA226 and Arduino reader can work.

## Current Problem

After rewiring through the MSP430 LaunchPad `3V3` jumper area, the INA226
shows voltage but near-zero current:

```text
mV ~= 2770 to 2860
mA ~= 0.000 or -0.122
```

This is not valid for `BENCH_MODE 2` continuous inference. It means the
INA226 bus-voltage sense path sees some voltage, but the actual current
path is not feeding the MSP430 target rail correctly.

Do not record paper data until continuous mode shows nonzero current.

## Correct Electrical Intent

The LaunchPad `3V3` jumper normally shorts two pins:

```text
3V3 source pin -> jumper -> MSP430 target VCC pin
```

For current measurement, remove the jumper and insert INA226 in series:

```text
3V3 source pin -> INA226 IN+ -> INA226 IN- -> MSP430 target VCC pin
```

Also:

```text
INA226 VBS -> INA226 IN-
LaunchPad GND -> INA226 GND -> Arduino GND
Arduino 5V -> INA226 VCC
Arduino A4 -> INA226 SDA
Arduino A5 -> INA226 SCL
```

Important:

- Only one of the two removed `3V3` jumper pins goes to `IN+`.
- The other removed `3V3` jumper pin goes to `IN-`.
- `IN+` and `IN-` must never share the same breadboard row.
- `VBS` and `IN-` should share the same node.
- LaunchPad GND does not go through `IN+` or `IN-`; it goes directly to
  INA226/Arduino GND.

## Physical Board Confusion

The user has an MSP-EXP430G2ET LaunchPad. There are multiple `3V3` labels:

1. The removed upper jumper block labeled:

   ```text
   GND | 5V | 3V3 | RXD | TXD | ...
   ```

   This is the intended place to insert the INA226 in series.

2. Other header pins labeled `3V3`, such as lower-side target/header pins.

To reduce confusion, prefer using only the two pins from the removed upper
`3V3` jumper for the high-side measurement path.

## STATUS: COMPLETE (2026-06-14)

All four measurement runs were successfully collected. Final results are in
`docs/energy_measurement.md` (Measured Results section). Do not re-run
unless hardware changes.

| Run | I (mA) | P (mW) |
|-----|--------|--------|
| Idle (LPM3) | <0.025 | <0.09 |
| 50 Hz LUT | 5.135 | 17.86 |
| Continuous LUT | 5.096 | 17.72 |
| Continuous no-LUT | 5.078 | 17.66 |

---

## Recommended Next Debug Step (historical — problem is solved)

Do not guess by photos. Use one of these checks:

### With Multimeter

1. Remove the `3V3` jumper.
2. Plug in LaunchPad USB.
3. Measure each of the two exposed `3V3` jumper pins relative to LaunchPad
   GND.
4. The pin reading about `3.3 V` is the source pin.
5. The pin reading `0 V` is the MSP430 target VCC pin.
6. Power off.
7. Wire:

   ```text
   source pin -> INA226 IN+
   target VCC pin -> INA226 IN-
   INA226 VBS -> INA226 IN-
   ```

### Without Multimeter

1. Remove all extra target-rail wires.
2. Use only:

   ```text
   upper 3V3 jumper pin A -> INA226 IN+
   upper 3V3 jumper pin B -> INA226 IN-
   INA226 VBS -> INA226 IN-
   LaunchPad GND -> INA226 GND
   ```

3. Flash `TEST_MODE 3`, `BENCH_MODE 2`.
4. Observe INA226 current.
5. If current is negative but nonzero, swap `IN+` and `IN-`.
6. If current remains zero, the target VCC pin is not actually connected
   through INA226 or the firmware is not in continuous mode.

## Firmware Sanity Check

The old MSP code the user pasted did not include energy mode. The correct
`msp/ccs_fastgrnn_har/main.cpp` must contain:

```cpp
//   3 = ENERGY (current/power benchmark - no UART, no I2C, no LED)
#ifndef TEST_MODE
#define TEST_MODE 0
#endif

#ifndef BENCH_MODE
#define BENCH_MODE 1
#endif
```

And an energy block:

```cpp
#elif TEST_MODE == 3
    serial_print("Mode: ENERGY BENCHMARK\n");
    UCA0CTL1 |= UCSWRST;
    P1OUT &= ~BIT0;
    fastgrnn_reset();
    static const float zero[3] = {0.0f, 0.0f, 0.0f};
    while (1) {
  #if BENCH_MODE == 0
        __bis_SR_register(LPM3_bits + GIE);
  #elif BENCH_MODE == 1
        unsigned long t0 = millis_ccs();
        fastgrnn_step(zero);
        while ((millis_ccs() - t0) < 20) { }
  #else
        fastgrnn_step(zero);
  #endif
    }
```

For the immediate debug test, use:

```cpp
#define TEST_MODE 3
#define BENCH_MODE 2
```

Expected in continuous mode:

```text
mV ~= 3000 to 3500
mA = clearly nonzero, likely several mA
```

Near-zero current means the power path is still wrong or the firmware did
not actually enter continuous benchmark mode.

