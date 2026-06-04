/*
 * main.cpp - CCS/MSP430G2553 bare-metal test runner for FastGRNN HAR.
 *
 * Direct MSP430 setup:
 * - DCO: 16 MHz calibrated clock
 * - UART: USCI_A0, 9600 baud, P1.1 RX / P1.2 TX
 * - Timer_A: 1 ms tick for latency measurement
 * - LED: P1.0 heartbeat after tests
 */

#include <msp430.h>
#include <stdint.h>

#include "fastgrnn.h"
#include "model_weights.h"
#include "test_data.h"

// Mode selection:
//   1 = TEST   (embedded test windows, full-window batch)
//   2 = STREAM (embedded test windows, 50 Hz paced streaming sim)
//   0 = LIVE   (MPU6050 sensor)
#define TEST_MODE 0

static volatile unsigned long g_millis = 0;

static void clock_init(void) {
    if (CALBC1_16MHZ == 0xFF) {
        while (1) {
        }
    }
    DCOCTL = 0;
    BCSCTL1 = CALBC1_16MHZ;
    DCOCTL = CALDCO_16MHZ;
}

static void uart_init(void) {
    P1SEL |= BIT1 | BIT2;
    P1SEL2 |= BIT1 | BIT2;

    UCA0CTL1 = UCSWRST;
    UCA0CTL1 |= UCSSEL_2;      // SMCLK = 16 MHz
    UCA0BR0 = 0x82;            // 16 MHz / 9600
    UCA0BR1 = 0x06;
    UCA0MCTL = UCBRS_6;
    UCA0CTL1 &= ~UCSWRST;
}

static void timer_init(void) {
    TA0CCTL0 = CCIE;
    TA0CCR0 = 15999;           // 1 ms at 16 MHz SMCLK
    TA0CTL = TASSEL_2 | MC_1 | TACLR;
}

#pragma vector=TIMER0_A0_VECTOR
__interrupt void timer0_a0_isr(void) {
    g_millis++;
}

static unsigned long millis_ccs(void) {
    unsigned long value;
    __disable_interrupt();
    value = g_millis;
    __enable_interrupt();
    return value;
}

static void serial_write_char(char c) {
    while (!(IFG2 & UCA0TXIFG)) {
    }
    UCA0TXBUF = (unsigned char)c;
}

static void serial_print(const char* s) {
    while (*s) {
        if (*s == '\n') serial_write_char('\r');
        serial_write_char(*s++);
    }
}

static void serial_print_uint(unsigned long value) {
    char buf[11];
    uint8_t i = 0;

    if (value == 0) {
        serial_write_char('0');
        return;
    }
    while (value > 0 && i < sizeof(buf)) {
        buf[i++] = (char)('0' + (value % 10));
        value /= 10;
    }
    while (i > 0) serial_write_char(buf[--i]);
}

static void serial_print_int(long value) {
    if (value < 0) {
        serial_write_char('-');
        value = -value;
    }
    serial_print_uint((unsigned long)value);
}

static void serial_print_float3(float value) {
    if (value < 0.0f) {
        serial_write_char('-');
        value = -value;
    }

    long whole = (long)value;
    long frac = (long)((value - (float)whole) * 1000.0f + 0.5f);
    if (frac >= 1000) {
        whole++;
        frac -= 1000;
    }

    serial_print_int(whole);
    serial_write_char('.');
    if (frac < 100) serial_write_char('0');
    if (frac < 10) serial_write_char('0');
    serial_print_uint((unsigned long)frac);
}

static const float* test_row(uint8_t sample, uint16_t t) {
    return (sample == 0) ? TEST_WINDOW_0[t] : TEST_WINDOW_1[t];
}

static void run_embedded_tests(void) {
    serial_print("\nEmbedded test windows: ");
    serial_print_uint(N_TEST_SAMPLES);
    serial_print("\n\n");

    for (uint8_t s = 0; s < N_TEST_SAMPLES; s++) {
        serial_print("--- Test ");
        serial_print_uint(s);
        serial_print(" ---\n");

        fastgrnn_reset();
        unsigned long t0 = millis_ccs();

        for (uint16_t t = 0; t < WINDOW_LEN; t++) {
            const float* row = test_row(s, t);
            float x[3] = { row[0], row[1], row[2] };
            fastgrnn_step(x);
        }

        uint8_t pred = fastgrnn_predict();
        unsigned long elapsed_ms = millis_ccs() - t0;

        serial_print("Prediction: ");
        serial_print_uint(pred);
        serial_print(" (");
        serial_print(CLASS_NAMES[pred]);
        serial_print(")\n");

        serial_print("Expected (Python): ");
        serial_print_uint(TEST_EXPECTED[s]);
        serial_print(" (");
        serial_print(CLASS_NAMES[TEST_EXPECTED[s]]);
        serial_print(")\n");

        serial_print("Ground truth:      ");
        serial_print_uint(TEST_TRUE[s]);
        serial_print(" (");
        serial_print(CLASS_NAMES[TEST_TRUE[s]]);
        serial_print(")\n");

        serial_print("Inference time: ");
        serial_print_uint(elapsed_ms);
        serial_print(" ms\n");

        serial_print((pred == TEST_EXPECTED[s]) ? "[PASS] matches Python\n" : "[FAIL] differs from Python\n");

        const float* lg = fastgrnn_get_logits();
        serial_print("Logits: ");
        for (uint8_t c = 0; c < NUM_CLASSES; c++) {
            serial_print_float3(lg[c]);
            serial_write_char(' ');
        }
        serial_print("\n\n");
    }

    serial_print("Tests complete.\n");
}

// ============================================================================
// STREAMING SIMULATION (equivalent of Arduino TEST_MODE == 2)
// Feed the embedded test window at 50 Hz pacing; measure per-sample latency.
// ============================================================================
static void run_streaming_simulation(void) {
    const uint16_t SAMPLE_PERIOD_MS = 20;   // 50 Hz
    const uint16_t PREDICT_EVERY = 25;       // predict every 0.5 s
    const uint8_t  USE_WINDOW = 0;

    serial_print("\nSTREAM SIM (50 Hz, window ");
    serial_print_uint(USE_WINDOW);
    serial_print(", ");
    serial_print_uint(WINDOW_LEN);
    serial_print(" samples)\n");
    serial_print("Headers: t, sample_lat_ms, h0, pred\n");
    serial_print("---------------------------------\n");

    fastgrnn_reset();
    unsigned long total_start = millis_ccs();
    unsigned long max_lat = 0, sum_lat = 0;
    uint16_t over_budget = 0;

    for (uint16_t t = 0; t < WINDOW_LEN; t++) {
        unsigned long t0 = millis_ccs();

        const float* row = (USE_WINDOW == 0) ? TEST_WINDOW_0[t] : TEST_WINDOW_1[t];
        float x[3] = { row[0], row[1], row[2] };
        fastgrnn_step(x);
        unsigned long lat = millis_ccs() - t0;

        if (lat > max_lat) max_lat = lat;
        sum_lat += lat;
        if (lat > SAMPLE_PERIOD_MS) over_budget++;

        if ((t + 1) % PREDICT_EVERY == 0 || t == WINDOW_LEN - 1) {
            uint8_t pred = fastgrnn_predict();
            const float* h = fastgrnn_get_hidden_state();
            serial_print_uint(t + 1);
            serial_write_char(',');
            serial_print_uint(lat);
            serial_write_char(',');
            serial_print_float3(h[0]);
            serial_write_char(',');
            serial_print_uint(pred);
            serial_write_char(' ');
            serial_print(CLASS_NAMES[pred]);
            serial_write_char('\n');
        }

        // Pace to 50 Hz
        while ((millis_ccs() - t0) < SAMPLE_PERIOD_MS) {
            // busy wait
        }
    }

    unsigned long total_ms = millis_ccs() - total_start;
    serial_print("---------------------------------\n");
    serial_print("Total: ");
    serial_print_uint(total_ms);
    serial_print(" ms (expected ~");
    serial_print_uint(WINDOW_LEN * SAMPLE_PERIOD_MS);
    serial_print(" ms)\n");
    serial_print("Avg sample latency: ");
    serial_print_uint(sum_lat / WINDOW_LEN);
    serial_print(" ms\n");
    serial_print("Max sample latency: ");
    serial_print_uint(max_lat);
    serial_print(" ms\n");
    serial_print("Over-budget (>");
    serial_print_uint(SAMPLE_PERIOD_MS);
    serial_print(" ms): ");
    serial_print_uint(over_budget);
    serial_print(" / ");
    serial_print_uint(WINDOW_LEN);
    serial_write_char('\n');
    serial_print("Final prediction: ");
    serial_print_uint(fastgrnn_predict());
    serial_print(" (");
    serial_print(CLASS_NAMES[fastgrnn_predict()]);
    serial_print(")\n");
}

// ============================================================================
// USCI_B0 I2C + MPU6050 driver (LIVE mode, TEST_MODE == 0)
// Wiring (MSP-EXP430G2):
//   VCC      -> 3.3 V (J6 or VCC pin)
//   GND      -> GND
//   P1.6 SCL -> MPU6050 SCL  (on the LaunchPad, REMOVE THE J5 JUMPER!)
//   P1.7 SDA -> MPU6050 SDA
//   GND      -> AD0 (so the address stays at 0x68)
// ============================================================================
#define MPU6050_ADDR        0x68
#define MPU_PWR_MGMT_1      0x6B
#define MPU_ACCEL_CONFIG    0x1C
#define MPU_ACCEL_XOUT_H    0x3B

// Bus recovery done correctly: drive OUT bits HIGH before flipping DIR so we
// do not accidentally generate a false START condition.
static void i2c_bus_recover(void) {
    // OUT bits HIGH first (so flipping DIR later does not cause a glitch)
    P1OUT |= BIT6 | BIT7;

    P1SEL  &= ~(BIT6 | BIT7);
    P1SEL2 &= ~(BIT6 | BIT7);
    P1REN  &= ~(BIT6 | BIT7);

    // SCL = output (starts HIGH because P1OUT[6]=1), SDA = input
    P1DIR |=  BIT6;
    P1DIR &= ~BIT7;

    __delay_cycles(1600);

    // If SDA is stuck low, give nine SCL clocks
    for (uint8_t i = 0; i < 9; i++) {
        if (P1IN & BIT7) break;
        P1OUT &= ~BIT6;
        __delay_cycles(800);
        P1OUT |=  BIT6;
        __delay_cycles(800);
    }

    // Clean STOP: SCL low -> SDA driven low -> SCL high -> SDA released high
    P1OUT &= ~BIT6;                      // SCL low
    __delay_cycles(800);
    P1OUT &= ~BIT7;                      // P1OUT[7] = 0 first
    P1DIR |=  BIT7;                      // SDA output (LOW)
    __delay_cycles(800);
    P1OUT |=  BIT6;                      // SCL high
    __delay_cycles(800);
    P1OUT |=  BIT7;                      // SDA goes HIGH while SCL HIGH = STOP
    __delay_cycles(800);

    // Hand back to inputs; the USCI peripheral will take over
    P1DIR &= ~(BIT6 | BIT7);
}

static void i2c_init(void) {
    // Bus recovery disabled - it was generating false STARTs on some boards.
    // i2c_bus_recover();

    P1SEL  |= BIT6 | BIT7;
    P1SEL2 |= BIT6 | BIT7;

    UCB0CTL1 |= UCSWRST;
    UCB0CTL0 = UCMST | UCMODE_3 | UCSYNC;     // I2C master, 7-bit address
    UCB0CTL1 = UCSSEL_2 | UCSWRST;            // SMCLK source
    UCB0BR0 = 0x40;                           // 10 kHz @ 16 MHz (safe for clones)
    UCB0BR1 = 0x06;
    UCB0I2CSA = MPU6050_ADDR;
    UCB0CTL1 &= ~UCSWRST;
}

// Returns 0 on success, -1 on timeout
static int i2c_write_reg(uint8_t reg, uint8_t val) {
    UCB0CTL1 |= UCTR | UCTXSTT;

    uint16_t to = 50000;
    while (!(IFG2 & UCB0TXIFG) && --to);
    if (!to) { UCB0CTL1 |= UCTXSTP; return -1; }
    UCB0TXBUF = reg;

    to = 50000;
    while (!(IFG2 & UCB0TXIFG) && --to);
    if (!to) { UCB0CTL1 |= UCTXSTP; return -1; }
    UCB0TXBUF = val;

    to = 50000;
    while (!(IFG2 & UCB0TXIFG) && --to);
    if (!to) { UCB0CTL1 |= UCTXSTP; return -1; }

    UCB0CTL1 |= UCTXSTP;
    to = 50000;
    while ((UCB0CTL1 & UCTXSTP) && --to);
    return to ? 0 : -1;
}

static int i2c_read_bytes(uint8_t reg, uint8_t* buf, uint8_t n) {
    // Write register address
    UCB0CTL1 |= UCTR | UCTXSTT;
    uint16_t to = 50000;
    while (!(IFG2 & UCB0TXIFG) && --to);
    if (!to) { UCB0CTL1 |= UCTXSTP; return -1; }
    UCB0TXBUF = reg;

    to = 50000;
    while (!(IFG2 & UCB0TXIFG) && --to);
    if (!to) { UCB0CTL1 |= UCTXSTP; return -1; }

    // Repeated START for read
    UCB0CTL1 &= ~UCTR;
    UCB0CTL1 |= UCTXSTT;
    to = 50000;
    while ((UCB0CTL1 & UCTXSTT) && --to);
    if (!to) { UCB0CTL1 |= UCTXSTP; return -1; }

    for (uint8_t i = 0; i < n; i++) {
        if (i == n - 1) UCB0CTL1 |= UCTXSTP;
        to = 50000;
        while (!(IFG2 & UCB0RXIFG) && --to);
        if (!to) return -1;
        buf[i] = UCB0RXBUF;
    }
    to = 50000;
    while ((UCB0CTL1 & UCTXSTP) && --to);
    return to ? 0 : -1;
}

// Bus health check - temporarily release USCI and read the lines as GPIO.
// Can be called whether USCI is enabled or disabled.
static int bus_health_ok(void) {
    // Quiesce the USCI peripheral
    UCB0CTL1 |= UCSWRST;

    P1SEL  &= ~(BIT6 | BIT7);
    P1SEL2 &= ~(BIT6 | BIT7);
    P1DIR  &= ~(BIT6 | BIT7);   // input
    P1REN  &= ~(BIT6 | BIT7);   // no internal pullup
    __delay_cycles(1600);        // ~100 us to settle (capacitive bus)
    uint8_t scl_high = (P1IN & BIT6) ? 1 : 0;
    uint8_t sda_high = (P1IN & BIT7) ? 1 : 0;
    serial_print("  BUS HEALTH: SCL=");
    serial_print_uint(scl_high);
    serial_print(" SDA=");
    serial_print_uint(sda_high);
    serial_print(" (both must read 1)\n");

    // Hand back to USCI
    P1SEL  |= BIT6 | BIT7;
    P1SEL2 |= BIT6 | BIT7;
    UCB0CTL1 &= ~UCSWRST;

    return (scl_high && sda_high) ? 1 : 0;
}

// ============================================================================
// SOFTWARE I2C - completely bypass USCI, pure GPIO bit-bang
// Useful when the USCI peripheral itself is misbehaving.
// ============================================================================
#define SCL_BIT BIT6
#define SDA_BIT BIT7

#define SW_DELAY() __delay_cycles(160)   // ~10 us = ~50 kHz I2C (slow but reliable)

static inline void scl_high(void) { P1DIR &= ~SCL_BIT; }  // input = pulled high
static inline void scl_low(void)  { P1DIR |=  SCL_BIT; }  // output, drives low (OUT=0)
static inline void sda_high(void) { P1DIR &= ~SDA_BIT; }
static inline void sda_low(void)  { P1DIR |=  SDA_BIT; }
static inline uint8_t sda_read(void) { return (P1IN & SDA_BIT) ? 1 : 0; }

static void sw_i2c_init(void) {
    P1SEL  &= ~(SCL_BIT | SDA_BIT);     // GPIO mode
    P1SEL2 &= ~(SCL_BIT | SDA_BIT);
    P1OUT  &= ~(SCL_BIT | SDA_BIT);     // OUT=0 (drives low when DIR=output)
    P1DIR  &= ~(SCL_BIT | SDA_BIT);     // input mode (high-Z, the pull-up wins)
    P1REN  &= ~(SCL_BIT | SDA_BIT);     // sensor provides the pull-up
    SW_DELAY();
}

static void sw_i2c_start(void) {
    sda_high(); SW_DELAY();
    scl_high(); SW_DELAY();
    sda_low();  SW_DELAY();             // SDA falls while SCL high = START
    scl_low();  SW_DELAY();
}

static void sw_i2c_stop(void) {
    sda_low();  SW_DELAY();
    scl_high(); SW_DELAY();
    sda_high(); SW_DELAY();             // SDA rises while SCL high = STOP
}

// Returns 1 if ACK, 0 if NACK
static int sw_i2c_write_byte(uint8_t b) {
    for (int i = 0; i < 8; i++) {
        if (b & 0x80) sda_high(); else sda_low();
        SW_DELAY();
        scl_high(); SW_DELAY();
        scl_low();  SW_DELAY();
        b <<= 1;
    }
    sda_high();          // release SDA so the slave can ACK
    SW_DELAY();
    scl_high(); SW_DELAY();
    int ack = !sda_read();
    scl_low(); SW_DELAY();
    return ack;
}

// Software ping - returns 1 if 0x68 acks, 0 otherwise
static int sw_ping_mpu(void) {
    sw_i2c_init();
    sw_i2c_start();
    int ack = sw_i2c_write_byte((MPU6050_ADDR << 1) | 0);   // write mode
    sw_i2c_stop();
    return ack;
}


// I2C ping - just START + addr+W to 0x68 and check for ACK.
static void i2c_scan(void) {
    serial_print("I2C bus scan...\n");

    int bus_ok = bus_health_ok();
    if (!bus_ok) {
        serial_print("  WARNING: bus appears dead, still trying the SW ping\n");
    }

    // ---- SOFTWARE I2C PING (USCI bypass) ----
    serial_print("  SW Ping 0x68: ");
    if (sw_ping_mpu()) {
        serial_print("ACK! (software I2C works - USCI is broken)\n");
    } else {
        serial_print("NACK (even software gets no reply - sensor is dead)\n");
    }
    SW_DELAY();
    SW_DELAY();

    // ---- FULL RE-INIT - reset USCI and bring it back up cleanly ----
    UCB0CTL1 = UCSWRST;                           // full reset
    P1SEL  |= BIT6 | BIT7;
    P1SEL2 |= BIT6 | BIT7;
    UCB0CTL0 = UCMST | UCMODE_3 | UCSYNC;
    UCB0CTL1 = UCSSEL_2 | UCSWRST;
    UCB0BR0  = 0x40;                              // 10 kHz @ 16 MHz (some clones cannot do 100 kHz)
    UCB0BR1  = 0x06;
    UCB0I2CSA = MPU6050_ADDR;
    UCB0STAT = 0;                                 // clear all flags
    UCB0CTL1 &= ~UCSWRST;
    __delay_cycles(16000);                        // 1 ms for USCI to settle

    // ---- 0x68 ping ----
    UCB0CTL1 |= UCTR | UCTXSTT;
    uint16_t to = 50000;
    while ((UCB0CTL1 & UCTXSTT) && --to);
    int start_completed = (to != 0);
    uint8_t stat = UCB0STAT;
    int ack = start_completed && !(stat & UCNACKIFG);

    UCB0CTL1 |= UCTXSTP;
    to = 50000;
    while ((UCB0CTL1 & UCTXSTP) && --to);
    UCB0STAT &= ~UCNACKIFG;

    serial_print("  Ping 0x68: ");
    if (ack) {
        serial_print("ACK!\n");
    } else if (!start_completed) {
        serial_print("START timeout, UCB0STAT=0x");
        uint8_t hi = (stat >> 4) & 0xF;
        uint8_t lo = stat & 0xF;
        serial_write_char(hi < 10 ? '0' + hi : 'A' + hi - 10);
        serial_write_char(lo < 10 ? '0' + lo : 'A' + lo - 10);
        serial_print(" (UCBBUSY=");
        serial_print_uint((stat & UCBBUSY) ? 1 : 0);
        serial_print(" UCSCLLOW=");
        serial_print_uint((stat & UCSCLLOW) ? 1 : 0);
        serial_print(" UCALIFG=");
        serial_print_uint((stat & UCALIFG) ? 1 : 0);
        serial_print(")\n");
    } else {
        serial_print("NACK (wrong address or no device)\n");
    }
}

static int mpu6050_init_dev(void) {
    if (i2c_write_reg(MPU_PWR_MGMT_1, 0x00)) return -1;     // wake from sleep
    if (i2c_write_reg(MPU_ACCEL_CONFIG, 0x00)) return -1;   // +/-2 g range
    return 0;
}

static int mpu6050_read_accel(int16_t* x, int16_t* y, int16_t* z) {
    uint8_t buf[6];
    if (i2c_read_bytes(MPU_ACCEL_XOUT_H, buf, 6)) return -1;
    *x = (int16_t)((((uint16_t)buf[0]) << 8) | buf[1]);
    *y = (int16_t)((((uint16_t)buf[2]) << 8) | buf[3]);
    *z = (int16_t)((((uint16_t)buf[4]) << 8) | buf[5]);
    return 0;
}

// ============================================================================
// LIVE MODE - MPU6050 streaming, 50 Hz, predict every 128 samples
// ============================================================================
static void run_live_mode(void) {
    serial_print("Initializing I2C...\n");
    i2c_init();
    __delay_cycles(1600000);    // 100 ms sensor boot wait
    i2c_scan();

    serial_print("Initializing MPU6050... ");
    if (mpu6050_init_dev() != 0) {
        serial_print("FAIL\n");
        serial_print("  Check hardware: J5 jumper removed, VCC=3.3 V, wiring, AD0 to GND\n");
        while (1) {
            P1OUT ^= BIT0;
            __delay_cycles(800000);
        }
    }
    serial_print("OK\n\n");

    fastgrnn_reset();
    uint16_t sample_count = 0;
    unsigned long last_sample_ms = millis_ccs();
    int16_t prev_raw[3] = {1, 1, 1};
    uint8_t freeze_count = 0;

    while (1) {
        unsigned long now = millis_ccs();
        if (now - last_sample_ms >= 20) {   // 50 Hz
            last_sample_ms = now;

            int16_t raw[3];
            if (mpu6050_read_accel(&raw[0], &raw[1], &raw[2]) != 0) {
                continue;
            }

            // Freeze detection (bit-exact identical reads = stale)
            if (raw[0] == prev_raw[0] && raw[1] == prev_raw[1] && raw[2] == prev_raw[2]) {
                freeze_count++;
                if (freeze_count == 10) {
                    serial_print("[I2C freeze, re-init]\n");
                    i2c_init();
                    mpu6050_init_dev();
                    freeze_count = 0;
                }
            } else {
                freeze_count = 0;
            }
            prev_raw[0] = raw[0]; prev_raw[1] = raw[1]; prev_raw[2] = raw[2];

            const float ACCEL_SCALE = 1.0f / 16384.0f;
            float x[3] = {
                raw[0] * ACCEL_SCALE,
                raw[1] * ACCEL_SCALE,
                raw[2] * ACCEL_SCALE
            };

            // Debug print every 25 samples (0.5 s)
            if (sample_count % 25 == 0) {
                serial_print("Raw: ");
                serial_print_float3(x[0]); serial_write_char(' ');
                serial_print_float3(x[1]); serial_write_char(' ');
                serial_print_float3(x[2]); serial_write_char('\n');
            }

            fastgrnn_step(x);
            sample_count++;

            // After a 128-sample window: predict + RESET
            // (matches the training pipeline, where the hidden state is zeroed
            // at every window boundary).
            if (sample_count >= WINDOW_LEN) {
                sample_count = 0;
                uint8_t cls = fastgrnn_predict();
                serial_print("[t=");
                serial_print_uint(now / 1000);
                serial_print("s] Activity: ");
                serial_print(CLASS_NAMES[cls]);
                serial_write_char('\n');
                fastgrnn_reset();
            }
        }
    }
}

int main(void) {
    WDTCTL = WDTPW | WDTHOLD;

    clock_init();
    uart_init();
    timer_init();

    P1DIR |= BIT0;
    P1OUT &= ~BIT0;

    __enable_interrupt();

    serial_print("=================================\n");
    serial_print(" FastGRNN HAR - MSP430G2553 CCS 16 MHz\n");
    serial_print("=================================\n");
#if TEST_MODE == 1
    serial_print("Mode: TEST (embedded data, batch)\n");
    run_embedded_tests();
#elif TEST_MODE == 2
    serial_print("Mode: STREAM (embedded data, 50 Hz paced)\n");
    run_streaming_simulation();
#else
    serial_print("Mode: LIVE (MPU6050 streaming, 50 Hz)\n");
    run_live_mode();
#endif

    while (1) {
        P1OUT ^= BIT0;
        __delay_cycles(8000000);
    }
}
