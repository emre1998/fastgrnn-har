#pragma once

#include <stdint.h>

#define __interrupt
#define BIT0 0x01
#define BIT1 0x02
#define BIT2 0x04
#define BIT6 0x40
#define BIT7 0x80
#define WDTPW 0x5A00
#define WDTHOLD 0x0080
#define UCSWRST 0x01
#define UCSSEL_2 0x80
#define UCBRS_6 0x0C
#define CCIE 0x0010
#define TASSEL_2 0x0200
#define MC_1 0x0010
#define TACLR 0x0004
#define UCA0TXIFG 0x02

static volatile uint8_t CALBC1_16MHZ = 0;
static volatile uint8_t CALDCO_16MHZ = 0;
static volatile uint8_t DCOCTL = 0;
static volatile uint8_t BCSCTL1 = 0;
static volatile uint8_t P1SEL = 0;
static volatile uint8_t P1SEL2 = 0;
static volatile uint8_t P1DIR = 0;
static volatile uint8_t P1OUT = 0;
static volatile uint8_t P1IN = 0;
static volatile uint8_t P1REN = 0;
static volatile uint8_t UCA0CTL1 = 0;
static volatile uint8_t UCA0BR0 = 0;
static volatile uint8_t UCA0BR1 = 0;
static volatile uint8_t UCA0MCTL = 0;
static volatile uint8_t UCA0TXBUF = 0;
static volatile uint8_t IFG2 = 0;
static volatile uint16_t TA0CCTL0 = 0;
static volatile uint16_t TA0CCR0 = 0;
static volatile uint16_t TA0CTL = 0;
static volatile uint16_t WDTCTL = 0;

static inline void __disable_interrupt(void) {}
static inline void __enable_interrupt(void) {}
static inline void __delay_cycles(unsigned long) {}

