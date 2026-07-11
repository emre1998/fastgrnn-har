/*
 * main.cpp - CCS/MSP430G2553 bare-metal test runner for LSTM HAR.
 *
 * Mirrors msp/ccs_gru_har/main.cpp and the FastGRNN CCS runner (16 MHz DCO,
 * USCI_A0 9600 baud UART, Timer_A 1 ms tick). Latency + energy modes only.
 *
 * TEST_MODE: 1 = LATENCY (default), 3 = ENERGY (BENCH_MODE 0 idle / 1 stream / 2 cont),
 *            4 = LIVE-LATENCY (MPU6050 + lstm_step, end-to-end 50 Hz over UART),
 *            5 = LIVE-ENERGY  (MPU6050 sensor loop, UART silent, for INA226 V/I/W).
 */
#include <msp430.h>
#include <stdint.h>
#include "lstm.h"
#include "model_weights.h"

#ifndef TEST_MODE
#define TEST_MODE 1
#endif
#ifndef BENCH_MODE
#define BENCH_MODE 1
#endif
// I2C bus speed for LIVE modes (TEST_MODE 4/5): 0 = 10 kHz (conservative), 1 = 100 kHz (standard)
#ifndef I2C_FAST
#define I2C_FAST 0
#endif
#define N_TIMING_WINDOWS 10

static volatile unsigned long g_millis = 0;

static void clock_init(void) {
    if (CALBC1_16MHZ == 0xFF) { while (1) {} }
    DCOCTL = 0; BCSCTL1 = CALBC1_16MHZ; DCOCTL = CALDCO_16MHZ;
}
static void uart_init(void) {
    P1SEL |= BIT1 | BIT2; P1SEL2 |= BIT1 | BIT2;
    UCA0CTL1 = UCSWRST; UCA0CTL1 |= UCSSEL_2;
    UCA0BR0 = 0x82; UCA0BR1 = 0x06; UCA0MCTL = UCBRS_6;
    UCA0CTL1 &= ~UCSWRST;
}
static void timer_init(void) {
    TA0CCTL0 = CCIE; TA0CCR0 = 15999; TA0CTL = TASSEL_2 | MC_1 | TACLR;
}
#pragma vector=TIMER0_A0_VECTOR
__interrupt void timer0_a0_isr(void) { g_millis++; }
static unsigned long millis_ccs(void) {
    unsigned long v; __disable_interrupt(); v = g_millis; __enable_interrupt(); return v;
}
static void sputc(char c) { while (!(IFG2 & UCA0TXIFG)) {} UCA0TXBUF = (unsigned char)c; }
static void sprint(const char* s) { while (*s) { if (*s == '\n') sputc('\r'); sputc(*s++); } }
static void sprint_u(unsigned long v) {
    char b[11]; uint8_t i = 0;
    if (v == 0) { sputc('0'); return; }
    while (v > 0 && i < sizeof(b)) { b[i++] = (char)('0' + (v % 10)); v /= 10; }
    while (i > 0) sputc(b[--i]);
}
static void sprint_f3(float v) {
    if (v < 0) { sputc('-'); v = -v; }
    long w = (long)v; long f = (long)((v - (float)w) * 1000.0f + 0.5f);
    if (f >= 1000) { w++; f -= 1000; }
    sprint_u(w); sputc('.');
    if (f < 100) sputc('0'); if (f < 10) sputc('0'); sprint_u(f);
}

static const float ZERO[3] = {0.0f, 0.0f, 0.0f};

// ============================================================================
// LIVE MODES: real MPU6050 sensor in the loop, 50 Hz.
//   TEST_MODE == 4  LIVE-LATENCY: end-to-end per-sample latency (I2C read + lstm_step),
//                   split into sensor vs inference, reported over UART.
//   TEST_MODE == 5  LIVE-ENERGY:  same loop run continuously, UART silenced after init
//                   so the INA226 reads clean steady-state system power (MCU+I2C+sensor).
// USCI_B0 I2C driver ported from ccs_fastgrnn_har (proven). Wiring (MSP-EXP430G2):
//   VCC->3.3V (from the INA226-measured rail), GND->GND, P1.6 SCL, P1.7 SDA
//   (REMOVE J5 jumper!), AD0->GND (addr 0x68).
// ============================================================================
#define MPU6050_ADDR     0x68
#define MPU_PWR_MGMT_1   0x6B
#define MPU_ACCEL_CONFIG 0x1C
#define MPU_ACCEL_XOUT_H 0x3B
#define N_LIVE_WINDOWS   4

static void i2c_init(void) {
    P1SEL  |= BIT6 | BIT7;
    P1SEL2 |= BIT6 | BIT7;
    UCB0CTL1 |= UCSWRST;
    UCB0CTL0 = UCMST | UCMODE_3 | UCSYNC;      // I2C master, 7-bit addr
    UCB0CTL1 = UCSSEL_2 | UCSWRST;             // SMCLK
#if I2C_FAST
    UCB0BR0 = 0xA0; UCB0BR1 = 0x00;            // ~100 kHz @ 16 MHz (standard mode)
#else
    UCB0BR0 = 0x40; UCB0BR1 = 0x06;            // ~10 kHz @ 16 MHz (safe for clones)
#endif
    UCB0I2CSA = MPU6050_ADDR;
    UCB0CTL1 &= ~UCSWRST;
}
static int i2c_write_reg(uint8_t reg, uint8_t val) {
    UCB0CTL1 |= UCTR | UCTXSTT;
    uint16_t to = 50000; while (!(IFG2 & UCB0TXIFG) && --to); if (!to) { UCB0CTL1 |= UCTXSTP; return -1; }
    UCB0TXBUF = reg;
    to = 50000; while (!(IFG2 & UCB0TXIFG) && --to); if (!to) { UCB0CTL1 |= UCTXSTP; return -1; }
    UCB0TXBUF = val;
    to = 50000; while (!(IFG2 & UCB0TXIFG) && --to); if (!to) { UCB0CTL1 |= UCTXSTP; return -1; }
    UCB0CTL1 |= UCTXSTP;
    to = 50000; while ((UCB0CTL1 & UCTXSTP) && --to);
    return to ? 0 : -1;
}
static int i2c_read_bytes(uint8_t reg, uint8_t* buf, uint8_t n) {
    UCB0CTL1 |= UCTR | UCTXSTT;
    uint16_t to = 50000; while (!(IFG2 & UCB0TXIFG) && --to); if (!to) { UCB0CTL1 |= UCTXSTP; return -1; }
    UCB0TXBUF = reg;
    to = 50000; while (!(IFG2 & UCB0TXIFG) && --to); if (!to) { UCB0CTL1 |= UCTXSTP; return -1; }
    UCB0CTL1 &= ~UCTR;                         // repeated START for read
    UCB0CTL1 |= UCTXSTT;
    to = 50000; while ((UCB0CTL1 & UCTXSTT) && --to); if (!to) { UCB0CTL1 |= UCTXSTP; return -1; }
    for (uint8_t i = 0; i < n; i++) {
        if (i == n - 1) UCB0CTL1 |= UCTXSTP;
        to = 50000; while (!(IFG2 & UCB0RXIFG) && --to); if (!to) return -1;
        buf[i] = UCB0RXBUF;
    }
    to = 50000; while ((UCB0CTL1 & UCTXSTP) && --to);
    return to ? 0 : -1;
}
static int mpu6050_init_dev(void) {
    if (i2c_write_reg(MPU_PWR_MGMT_1, 0x00)) return -1;    // wake
    if (i2c_write_reg(MPU_ACCEL_CONFIG, 0x00)) return -1;  // +/-2 g
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

static void run_live_latency(void) {
    sprint("Mode: LIVE end-to-end latency (MPU6050 + lstm_step, 50Hz)\n");
    sprint("H="); sprint_u(HIDDEN_SIZE); sprint(" window="); sprint_u(WINDOW_T);
    sprint(" (LUT = lstm.cpp ayari; ELLE not al)\n");
    sprint("Wiring: P1.6 SCL, P1.7 SDA, VCC 3.3V, AD0->GND, J5 jumper REMOVED\n");

    sprint("Init I2C + MPU6050... ");
    i2c_init();
    __delay_cycles(1600000);          // ~100 ms sensor boot
    if (mpu6050_init_dev() != 0) {
        sprint("FAIL (wiring/J5/VCC/AD0 kontrol)\n");
        while (1) { P1OUT ^= BIT0; __delay_cycles(800000); }
    }
    sprint("OK\n");

    const float ACCEL_SCALE = 1.0f / 16384.0f;
    unsigned long sum_lat = 0, sum_sensor = 0, max_lat = 0;
    uint16_t over = 0, n = 0;

    for (uint8_t w = 0; w < N_LIVE_WINDOWS; w++) {
        lstm_reset();
        for (uint16_t t = 0; t < WINDOW_T; t++) {
            unsigned long t0 = millis_ccs();
            int16_t raw[3];
            if (mpu6050_read_accel(&raw[0], &raw[1], &raw[2]) != 0) continue;
            unsigned long t_sensor = millis_ccs();
            float x[3] = { raw[0] * ACCEL_SCALE, raw[1] * ACCEL_SCALE, raw[2] * ACCEL_SCALE };
            lstm_step(x);
            unsigned long lat = millis_ccs() - t0;       // sensor + inference
            sum_lat += lat; sum_sensor += (t_sensor - t0);
            if (lat > max_lat) max_lat = lat;
            if (lat > 20) over++;
            n++;
            while ((millis_ccs() - t0) < 20) { /* pace 50 Hz */ }
        }
        (void)lstm_predict();
    }

    float avg = (n ? (float)sum_lat / n : 0.0f);
    float avg_sensor = (n ? (float)sum_sensor / n : 0.0f);
    sprint("Samples timed:   "); sprint_u(n); sprint("\n");
    sprint("Avg end-to-end:  "); sprint_f3(avg); sprint(" ms (sensor+inference)\n");
    sprint("Avg sensor read: "); sprint_f3(avg_sensor); sprint(" ms\n");
    sprint("Avg inference:   "); sprint_f3(avg - avg_sensor); sprint(" ms\n");
    sprint("Max end-to-end:  "); sprint_u(max_lat); sprint(" ms\n");
    sprint("Over-budget(>20): "); sprint_u(over); sprint(" / "); sprint_u(n); sprint("\n");
    sprint(avg < 20.0f ? "REAL-TIME: OK\n" : "REAL-TIME: FAIL (avg over 20 ms)\n");
    while (1) { P1OUT ^= BIT0; __delay_cycles(8000000); }
}

// LIVE-ENERGY (TEST_MODE == 5): continuous 50 Hz sensor+inference loop, UART silenced
// after init so the INA226 reads clean steady-state system power (MCU + I2C + MPU6050).
static void run_live_energy(void) {
    sprint("Mode: LIVE ENERGY (sensor loop, UART silent after init)\n");
    i2c_init();
    __delay_cycles(1600000);
    if (mpu6050_init_dev() != 0) {
        sprint("MPU FAIL (wiring/J5/VCC/AD0)\n");
        while (1) { P1OUT ^= BIT0; __delay_cycles(800000); }
    }
    sprint("OK - going silent for the ammeter\n");
    UCA0CTL1 |= UCSWRST;               // silence UART; clean current for INA226
    P1OUT &= ~BIT0;
    const float ACCEL_SCALE = 1.0f / 16384.0f;
    lstm_reset();
    uint16_t sc = 0;
    while (1) {
        unsigned long t0 = millis_ccs();
        int16_t raw[3];
        if (mpu6050_read_accel(&raw[0], &raw[1], &raw[2]) == 0) {
            float x[3] = { raw[0] * ACCEL_SCALE, raw[1] * ACCEL_SCALE, raw[2] * ACCEL_SCALE };
            lstm_step(x);
            if (++sc >= WINDOW_T) { sc = 0; (void)lstm_predict(); lstm_reset(); }
        }
        while ((millis_ccs() - t0) < 20) { /* pace 50 Hz */ }
    }
}

int main(void) {
    WDTCTL = WDTPW | WDTHOLD;
    clock_init(); uart_init(); timer_init();
    P1DIR |= BIT0; P1OUT &= ~BIT0;
    __enable_interrupt();

    sprint("=================================\n");
    sprint(" LSTM HAR - MSP430G2553 CCS 16 MHz\n");
    sprint("=================================\n");

#if TEST_MODE == 3
    sprint("Mode: ENERGY BENCHMARK\n");
    UCA0CTL1 |= UCSWRST;
    P1OUT &= ~BIT0;
    lstm_reset();
    while (1) {
  #if BENCH_MODE == 0
        __bis_SR_register(LPM3_bits + GIE);
  #elif BENCH_MODE == 1
        unsigned long t0 = millis_ccs();
        lstm_step(ZERO);
        while ((millis_ccs() - t0) < 20) { /* busy idle */ }
  #else
        lstm_step(ZERO);
  #endif
    }
#elif TEST_MODE == 4
    run_live_latency();
#elif TEST_MODE == 5
    run_live_energy();
#else
    sprint("Mode: LATENCY  H="); sprint_u(HIDDEN_SIZE);
    sprint(" window="); sprint_u(WINDOW_T);
    sprint(" classes="); sprint_u(NUM_CLASSES); sprint("\n");

    lstm_reset();
    for (uint16_t t = 0; t < WINDOW_T; t++) lstm_step(ZERO);   // warm-up

    unsigned long t0 = millis_ccs();
    for (uint8_t w = 0; w < N_TIMING_WINDOWS; w++) {
        lstm_reset();
        for (uint16_t t = 0; t < WINDOW_T; t++) lstm_step(ZERO);
    }
    unsigned long total_ms = millis_ccs() - t0;

    float win_ms = (float)total_ms / N_TIMING_WINDOWS;
    float step_us = win_ms * 1000.0f / WINDOW_T;

    sprint("Windows timed:   "); sprint_u(N_TIMING_WINDOWS);
    sprint(" ("); sprint_u(total_ms); sprint(" ms total)\n");
    sprint("Per-window:      "); sprint_f3(win_ms); sprint(" ms\n");
    sprint("Per-step:        "); sprint_f3(step_us); sprint(" us  (");
    sprint_f3(step_us / 1000.0f); sprint(" ms)\n");
    sprint("50Hz step budget: 20000 us (20 ms)\n");
    sprint("Utilization:     "); sprint_f3(100.0f * step_us / 20000.0f); sprint(" %\n");
    sprint(step_us < 20000.0f ? "REAL-TIME: OK\n" : "REAL-TIME: FAIL (over 20 ms)\n");

    while (1) { P1OUT ^= BIT0; __delay_cycles(8000000); }
#endif
}
