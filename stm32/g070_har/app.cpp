/*
 * app.c - STM32G070RB HAR cross-platform validation.
 *
 * One flash runs all three portability-minimum checks and reports over the
 * ST-LINK virtual COM port (USART2, 115200 8N1):
 *
 *   1. bring-up      : banner + LED on  -> confirms clock + UART + toolchain
 *   2. equivalence   : GRU on 5 embedded test windows; each prediction must
 *                      equal C_PRED[] (the host-C reference). Matching here means
 *                      G070 == host-C == MSP430, since MSP430 also matches host-C.
 *   3. latency       : time N_REPEAT full windows of gru_step, report per-step ms
 *                      against the 20 ms (50 Hz) budget.
 *
 * Only the model files (gru.c / gru.h / lut.h / model_weights.h) and this file
 * are added to the CubeMX project; nothing in the model code changes from the
 * MSP430 deployment build -- that identity is the point.
 *
 * Toggle USE_LUT (below, or a project symbol) to get the no-LUT latency row.
 */
#include "main.h"          /* CubeMX-generated: HAL + huart2 + GPIO defines   */
#include "app.h"
#include "gru.h"
#include "test_windows.h"
#include <string.h>
#include <stdio.h>

/* USART2 is the ST-LINK VCP on the Nucleo-G070RB (PA2/PA3). CubeMX names the
 * handle huart2 when USART2 is enabled. */
extern UART_HandleTypeDef huart2;

/* User LED LD4 = PA5 on the Nucleo-G070RB. */
#define LED_PORT  GPIOA
#define LED_PIN   GPIO_PIN_5

#define N_REPEAT  20        /* windows to average for the latency figure        */
#define WINDOW_MS 20        /* 50 Hz sampling period, for the real-time verdict  */

static void tx(const char *s)
{
    HAL_UART_Transmit(&huart2, (uint8_t *)s, (uint16_t)strlen(s), 100);
}

static void txln(const char *s) { tx(s); tx("\r\n"); }

void app_run(void)
{
    char line[96];

    /* ---- 1. bring-up ---------------------------------------------------- */
    txln("");
    txln("=== STM32G070RB HAR - cross-platform validation ===");
    snprintf(line, sizeof line,
             "cell=GRU  H=%d  clock=%luMHz  USE_LUT=%d",
             (int)HIDDEN_SIZE, (unsigned long)(HAL_RCC_GetHCLKFreq() / 1000000u),
             (int)
#if defined(USE_LUT)
             USE_LUT
#else
             1
#endif
    );
    txln(line);
    HAL_GPIO_WritePin(LED_PORT, LED_PIN, GPIO_PIN_SET);   /* alive */

    /* ---- 2. equivalence vs the host-C reference ------------------------- */
    txln("");
    txln("[equivalence] G070 prediction must equal C_PRED (host-C reference):");
    int ok = 0;
    for (int i = 0; i < N_TEST; i++) {
        gru_reset();
        for (int t = 0; t < WINDOW_T; t++)
            gru_step(TEST_X[i][t]);
        uint8_t pred = gru_predict();
        int match = (pred == C_PRED[i]);
        ok += match;
        snprintf(line, sizeof line, "  window %d: G070=%u  host-C=%u  %s",
                 i, (unsigned)pred, (unsigned)C_PRED[i], match ? "OK" : "MISMATCH");
        txln(line);
    }
    snprintf(line, sizeof line, "[equivalence] %d/%d match  ->  %s",
             ok, (int)N_TEST, (ok == N_TEST) ? "CROSS-PLATFORM IDENTICAL" : "DIVERGENCE");
    txln(line);

    /* ---- 3. latency ----------------------------------------------------- */
    txln("");
    const float zero[INPUT_DIM] = {0};                 /* input-independent    */
    uint32_t t0 = HAL_GetTick();
    for (int r = 0; r < N_REPEAT; r++) {
        gru_reset();
        for (int t = 0; t < WINDOW_T; t++)
            gru_step(zero);
        (void)gru_predict();
    }
    uint32_t dt = HAL_GetTick() - t0;                  /* ms for N_REPEAT windows */
    /* integer math to avoid pulling in soft-float printf just for the report  */
    uint32_t per_step_us = (dt * 1000u) / (uint32_t)(N_REPEAT * WINDOW_T);
    uint32_t window_ms   = dt / N_REPEAT;
    snprintf(line, sizeof line, "[latency] per-step ~%lu.%03lu ms   window ~%lu ms",
             (unsigned long)(per_step_us / 1000u), (unsigned long)(per_step_us % 1000u),
             (unsigned long)window_ms);
    txln(line);
    snprintf(line, sizeof line, "[latency] real-time @50Hz (step < %d ms): %s",
             WINDOW_MS, (per_step_us < (uint32_t)WINDOW_MS * 1000u) ? "OK" : "FAIL");
    txln(line);

    txln("");
    txln("done. LED now blinks to show the board is alive.");

    /* ---- idle: heartbeat ----------------------------------------------- */
    while (1) {
        HAL_GPIO_TogglePin(LED_PORT, LED_PIN);
        HAL_Delay(500);
    }
}
