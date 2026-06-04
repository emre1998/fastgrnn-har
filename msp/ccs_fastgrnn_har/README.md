# FastGRNN HAR - CCS MSP430G2553 Test

CCS bare-metal MSP430G2553 test package. This version uses a calibrated
16 MHz DCO clock and 9600 baud UART.

## Files

- `main.cpp`: CCS test runner, UART, Timer_A, LED heartbeat.
- `fastgrnn.cpp`, `fastgrnn.h`: shared inference engine.
- `model_weights.h`: Q15 weights and scales.
- `test_data.h`: two embedded test windows.

## Build Notes

- Target: `MSP430G2553`
- Linker stack: `256`
- Linker heap: `0`
- Serial: `9600 baud`

## Live MPU6050 Mode

The CCS live mode uses GPIO I2C on `P1.6=SCL` and `P1.7=SDA`. Remove the
LaunchPad `P1.6/LED` jumper so the onboard LED does not load the clock line.
Connect `VCC=3.3V`, `GND=GND`, and `AD0=GND`. Add 4.7 kOhm pull-ups from SCL
and SDA to 3.3V if the GY-521 board does not provide suitable pull-ups.

## Expected Result

Both embedded tests should match Python:

- Test 0: `pred=4`, `expected=4` -> PASS
- Test 1: `pred=2`, `expected=2` -> PASS
