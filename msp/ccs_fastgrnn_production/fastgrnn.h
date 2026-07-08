/*
 * fastgrnn.h - FastGRNN HAR inference engine
 *
 * Weights are Q15 in PROGMEM; compute is float. Supports streaming
 * (one sample at a time) and full-window classification.
 *
 * Usage:
 *   fastgrnn_reset();              // zero h_state, start a new window
 *   for (int t = 0; t < 128; t++) {
 *       float x[3] = { ax, ay, az };   // raw acceleration in g
 *       fastgrnn_step(x);              // streaming: one sample
 *   }
 *   uint8_t cls = fastgrnn_predict(); // class index (0-5)
 *
 * Or in one shot:
 *   uint8_t cls = fastgrnn_classify_window(X);  // (128, 3) float window
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
void fastgrnn_reset(void);                              // zero h_state
void fastgrnn_step(const float x_raw[INPUT_CHANNELS]);  // one sample
uint8_t fastgrnn_predict(void);                         // classify from current h_state

// Convenience: full-window prediction
uint8_t fastgrnn_classify_window(const float X[WINDOW_LEN][INPUT_CHANNELS]);

// Debug helpers
const float* fastgrnn_get_hidden_state(void);           // pointer to h_state (read-only)
const float* fastgrnn_get_logits(void);                 // logits from the most recent predict

#ifdef __cplusplus
}
#endif

#endif // FASTGRNN_H
