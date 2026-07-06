/*
 * main.cpp - Minimal UART sanity test for MSP430G2553 (CCS, bare-metal).
 *
 * Prints a counter line once per second forever. No model, no timing loop --
 * isolates whether the UART/clock setup itself is correct before layering
 * the GRU/LSTM firmware on top.
 *
 * Identical clock_init/uart_init to msp/ccs_gru_har and msp/ccs_fastgrnn_har,
 * so if THIS prints cleanly at 9600 baud, the bug is in the GRU/LSTM firmware
 * or timing loop, not the UART setup. If THIS is also garbled, the bug is in
 * board config (jumpers / COM port / DCO calibration), not our code.
 */
#include <msp430.h>
#include <stdint.h>

static volatile unsigned long g_millis = 0;

static void clock_init(void) {
    if (CALBC1_16MHZ == 0xFF) { while (1) {} }   // calibration missing -- would hang here
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

int main(void) {
    WDTCTL = WDTPW | WDTHOLD;
    clock_init(); uart_init(); timer_init();
    P1DIR |= BIT0; P1OUT &= ~BIT0;
    __enable_interrupt();

    sprint("=== UART TEST - MSP430G2553 @ 9600 baud ===\n");
    sprint("If you can read this clean, UART setup is OK.\n");

    unsigned long count = 0;
    unsigned long last = 0;
    while (1) {
        unsigned long now = millis_ccs();
        if (now - last >= 1000) {
            last = now;
            sprint("tick ");
            sprint_u(count++);
            sprint("\n");
            P1OUT ^= BIT0;   // blink LED so you can SEE it's alive even w/o serial
        }
    }
}
