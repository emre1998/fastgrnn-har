/*
 * fastgrnn.h — FastGRNN HAR inference engine
 *
 * Ağırlıklar Q15 PROGMEM'de, hesaplama float. Streaming + tek pencere modu.
 *
 * Kullanim:
 *   fastgrnn_reset();             // h_state'i sifirla, yeni pencere baslat
 *   for (int t = 0; t < 128; t++) {
 *       float x[3] = { ax, ay, az };   // ham g cinsinden
 *       fastgrnn_step(x);              // streaming: tek adim
 *   }
 *   uint8_t cls = fastgrnn_predict(); // sinif index (0-5)
 *
 * VEYA tek seferde:
 *   uint8_t cls = fastgrnn_classify_window(X);  // (128, 3) float pencere
 */

#ifndef FASTGRNN_H
#define FASTGRNN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WINDOW_LEN 128
#define INPUT_CHANNELS 3
#define HIDDEN_STATE_SIZE 16
#define NUM_OUTPUT_CLASSES 6

// Streaming API
void fastgrnn_reset(void);                              // h_state sifirla
void fastgrnn_step(const float x_raw[INPUT_CHANNELS]);  // bir ornek
uint8_t fastgrnn_predict(void);                         // mevcut h_state'ten sinif

// Convenience: tek pencere
uint8_t fastgrnn_classify_window(const float X[WINDOW_LEN][INPUT_CHANNELS]);

// Debug yardimcilari
const float* fastgrnn_get_hidden_state(void);           // h_state pointer (read-only kullan)
const float* fastgrnn_get_logits(void);                 // son predict'ten logits

#ifdef __cplusplus
}
#endif

#endif // FASTGRNN_H
