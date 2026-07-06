/*
 * main.cpp - CCS/MSP430G2553 bare-metal test runner for LSTM HAR.
 *
 * Mirrors msp/ccs_gru_har/main.cpp and the FastGRNN CCS runner (16 MHz DCO,
 * USCI_A0 9600 baud UART, Timer_A 1 ms tick). Latency + energy modes only.
 *
 * TEST_MODE: 1 = LATENCY (default), 3 = ENERGY (BENCH_MODE 0 idle / 1 stream / 2 cont).
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
