/*
 * app.h - STM32G070 HAR cross-platform validation entry point.
 *
 * The application layer is kept separate from the CubeMX-generated main.c so
 * regenerating the peripheral init never overwrites it. main.c calls app_run()
 * once, after the HAL and USART2 are initialized.
 *
 * Integration (see README_STM32.md):
 *   in main.c USER CODE INCLUDES:  #include "app.h"
 *   in main.c USER CODE 2 (after MX_USART2_UART_Init):  app_run();
 */
#ifndef APP_H
#define APP_H

/* app.cpp compiles as C++ (the model headers rely on C++ const-linkage, exactly
 * as the MSP430 build did), but main.c is C and calls app_run(), so give it C
 * linkage. */
#ifdef __cplusplus
extern "C" {
#endif

void app_run(void);   // never returns; runs the check, then idles blinking the LED

#ifdef __cplusplus
}
#endif

#endif
