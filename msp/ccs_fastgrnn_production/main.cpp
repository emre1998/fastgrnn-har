/*
 * main.cpp - FastGRNN HAR PRODUCTION firmware (MSP430G2553, CCS bare-metal).
 * Minimal real-deployment build: no UART, no debug prints, no repeated
 * timing loop -- symmetric with ccs_gru_production / ccs_lstm_production
 * so the three cells' true Flash/RAM footprint is directly comparable.
 */
#include <msp430.h>
#include <stdint.h>
#include "fastgrnn.h"

static volatile unsigned long g_millis = 0;

static void clock_init(void) {
    if (CALBC1_16MHZ == 0xFF) { while (1) {} }
    DCOCTL = 0; BCSCTL1 = CALBC1_16MHZ; DCOCTL = CALDCO_16MHZ;
}
static void timer_init(void) {
    TA0CCTL0 = CCIE; TA0CCR0 = 15999; TA0CTL = TASSEL_2 | MC_1 | TACLR;
}
#pragma vector=TIMER0_A0_VECTOR
__interrupt void timer0_a0_isr(void) { g_millis++; }
static unsigned long millis_ccs(void) {
    unsigned long v; __disable_interrupt(); v = g_millis; __enable_interrupt(); return v;
}

static const float ZERO[INPUT_CHANNELS] = {0.0f, 0.0f, 0.0f};

int main(void) {
    WDTCTL = WDTPW | WDTHOLD;
    clock_init(); timer_init();
    P1DIR |= BIT0; P1OUT &= ~BIT0;
    __enable_interrupt();

    while (1) {
        fastgrnn_reset();
        for (uint16_t t = 0; t < WINDOW_LEN; t++) {
            unsigned long t0 = millis_ccs();
            fastgrnn_step(ZERO);
            while ((millis_ccs() - t0) < 20) {}
        }
        uint8_t cls = fastgrnn_predict();
        if (cls & 1) P1OUT |= BIT0; else P1OUT &= ~BIT0;
    }
}
