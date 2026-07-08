/*
 * main.cpp - GRU HAR PRODUCTION firmware (MSP430G2553, CCS bare-metal).
 *
 * This is the MINIMAL real-deployment build: no UART, no debug prints, no
 * repeated timing/averaging loop. It measures the TRUE Flash/RAM footprint
 * an actual deployed classifier would need, as opposed to the test-harness
 * builds (ccs_gru_har) which include UART banners, per-step diagnostics, and
 * a 10x timing-repeat loop purely for measurement convenience.
 *
 * Behavior: stream at 50 Hz (paced by Timer_A), classify every WINDOW_T
 * samples, and drive the LED (P1.0) based on the predicted class as the
 * "real" downstream action (a real deployment would replace this with an
 * actuator, radio packet, etc. -- the classification IS the deliverable).
 */
#include <msp430.h>
#include <stdint.h>
#include "gru.h"
#include "model_weights.h"

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

static const float ZERO[INPUT_DIM] = {0.0f, 0.0f, 0.0f};

int main(void) {
    WDTCTL = WDTPW | WDTHOLD;
    clock_init(); timer_init();
    P1DIR |= BIT0; P1OUT &= ~BIT0;   // LED = downstream action indicator
    __enable_interrupt();

    while (1) {
        gru_reset();
        for (uint16_t t = 0; t < WINDOW_T; t++) {
            unsigned long t0 = millis_ccs();
            gru_step(ZERO);                       // real deployment reads a sensor here
            while ((millis_ccs() - t0) < 20) {}    // pace to 50 Hz
        }
        uint8_t cls = gru_predict();
        if (cls & 1) P1OUT |= BIT0; else P1OUT &= ~BIT0;   // minimal real action
    }
}
