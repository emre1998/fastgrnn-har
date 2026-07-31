/*
 * main.cpp - STM32G070RB HAR cross-platform validation (bare-metal, register-level).
 *
 * No HAL, no CubeMX (that wizard is absent in CubeIDE 2.2.0) - raw register
 * access only: RCC + GPIOA + USART2 (VCP over ST-LINK) + SysTick 1 ms.
 *
 * Runs two checks, then idles blinking LD4 (PA5):
 *   [equivalence]  the 5 embedded windows must reproduce this cell's host-C
 *                  reference C_PRED[] exactly  ->  G070 == host-C == MSP430.
 *   [latency]      per-step inference time (zero input), vs the 20 ms @50Hz budget.
 *
 * This file is CELL-AGNOSTIC except for four lines. To switch cells, change:
 *   1) the cell header include below            (gru.h / lstm.h / fastgrnn.h)
 *   2) the three  *_reset / *_step / *_predict  calls
 *   3) the "cell=GRU" banner string
 * and in the project: swap the cell's *.cpp/*.h/model_weights.h/lut.h, and set
 * C_PRED[] in test_windows.h to that cell's host reference (see README_STM32.md).
 * Measured references: GRU {4,1,4,5,1} · LSTM {4,0,4,5,0} · FastGRNN {4,2,4,5,1}.
 *
 * Board: NUCLEO-G070RB. LD4=PA5, VCP=USART2 (PA2/PA3). HSI 16 MHz, 115200 8N1.
 */
#include <stdint.h>
#include "gru.h"             // cell API + USE_LUT (single source of truth)
#include "model_weights.h"   // HIDDEN_SIZE, WINDOW_T, INPUT_DIM, NUM_CLASSES
#include "test_windows.h"    // N_TEST, TEST_X, C_PRED

#define REG(a)       (*(volatile uint32_t *)(a))
#define RCC_IOPENR   REG(0x40021000u + 0x34u)
#define RCC_APBENR1  REG(0x40021000u + 0x3Cu)
#define GPIOA_MODER  REG(0x50000000u + 0x00u)
#define GPIOA_ODR    REG(0x50000000u + 0x14u)
#define GPIOA_AFRL   REG(0x50000000u + 0x20u)
#define USART2_CR1   REG(0x40004400u + 0x00u)
#define USART2_BRR   REG(0x40004400u + 0x0Cu)
#define USART2_ISR   REG(0x40004400u + 0x1Cu)
#define USART2_TDR   REG(0x40004400u + 0x28u)
#define SYST_CSR     REG(0xE000E010u)
#define SYST_RVR     REG(0xE000E014u)
#define SYST_CVR     REG(0xE000E018u)

static volatile uint32_t g_ms = 0;
extern "C" void SysTick_Handler(void) { g_ms++; }

static void uart_init(void)
{
    RCC_IOPENR  |= (1u << 0);              /* GPIOA clock */
    RCC_APBENR1 |= (1u << 17);             /* USART2 clock */
    GPIOA_MODER &= ~((3u << (2*2)) | (3u << (3*2)));
    GPIOA_MODER |=  ((2u << (2*2)) | (2u << (3*2)));   /* PA2/PA3 = AF */
    GPIOA_AFRL  &= ~((0xFu << (2*4)) | (0xFu << (3*4)));
    GPIOA_AFRL  |=  ((1u << (2*4)) | (1u << (3*4)));    /* AF1 = USART2 */
    USART2_BRR = 139u;                     /* 16 MHz / 115200 */
    USART2_CR1 = (1u << 3) | (1u << 0);    /* TE | UE */
}
static void uart_print(const char *s)
{ while (*s) { while (!(USART2_ISR & (1u<<7))) {} USART2_TDR = (uint8_t)*s++; } }
static void uart_uint(uint32_t v)
{ char b[11]; int i=0; if(!v){uart_print("0");return;} while(v){b[i++]=(char)('0'+v%10);v/=10;} while(i){char c[2]={b[--i],0};uart_print(c);} }

int main(void)
{
    RCC_IOPENR  |= (1u << 0);
    GPIOA_MODER &= ~(3u << (5*2));
    GPIOA_MODER |=  (1u << (5*2));         /* LED PA5 = output */
    uart_init();
    SYST_RVR = 16000u - 1u; SYST_CVR = 0;
    SYST_CSR = (1u<<0)|(1u<<1)|(1u<<2);    /* SysTick 1 ms, core clock, IRQ */

    uart_print("\r\n=== STM32G070RB HAR - cross-platform validation ===\r\n");
    uart_print("cell=GRU  H="); uart_uint(HIDDEN_SIZE);
    uart_print("  clock=16MHz  USE_LUT="); uart_uint(USE_LUT); uart_print("\r\n\r\n");

    uart_print("[equivalence] G070 pred must equal C_PRED (host-C ref):\r\n");
    int ok = 0;
    for (int i = 0; i < N_TEST; i++) {
        gru_reset();
        for (int t = 0; t < WINDOW_T; t++) gru_step(TEST_X[i][t]);
        uint8_t pred = gru_predict();
        int m = (pred == C_PRED[i]); ok += m;
        uart_print("  window "); uart_uint((uint32_t)i);
        uart_print(": G070="); uart_uint(pred);
        uart_print("  host-C="); uart_uint(C_PRED[i]);
        uart_print(m ? "  OK\r\n" : "  MISMATCH\r\n");
    }
    uart_print("[equivalence] "); uart_uint((uint32_t)ok); uart_print("/"); uart_uint((uint32_t)N_TEST);
    uart_print(ok==N_TEST ? "  -> CROSS-PLATFORM IDENTICAL\r\n\r\n" : "  -> DIVERGENCE\r\n\r\n");

    const float zero[INPUT_DIM] = {0};
    const int N_REPEAT = 20;
    uint32_t t0 = g_ms;
    for (int r = 0; r < N_REPEAT; r++) {
        gru_reset();
        for (int t = 0; t < WINDOW_T; t++) gru_step(zero);
        (void)gru_predict();
    }
    uint32_t dt = g_ms - t0;
    uint32_t us = (dt * 1000u) / (uint32_t)(N_REPEAT * WINDOW_T);
    uart_print("[latency] window ~"); uart_uint(dt / N_REPEAT);
    uart_print(" ms   per-step ~"); uart_uint(us/1000u); uart_print(".");
    { uint32_t f=us%1000u; char d[4]={(char)('0'+f/100),(char)('0'+(f/10)%10),(char)('0'+f%10),0}; uart_print(d); }
    uart_print(" ms\r\n");
    uart_print("[latency] real-time @50Hz (step<20ms): ");
    uart_print(us < 20000u ? "OK\r\n" : "FAIL\r\n");

    uart_print("\r\ndone.\r\n");
    while (1) { GPIOA_ODR ^= (1u << 5); for (volatile uint32_t i=0;i<800000u;i++){} }
}
