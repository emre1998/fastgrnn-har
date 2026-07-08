/*
 * lstm.h - LSTM HAR inference engine (deployment baseline)
 *
 * Mirrors the FastGRNN/GRU firmware (Q15 weights in PROGMEM, float compute,
 * LUT sigmoid/tanh, streaming API) so latency/energy isolates the CELL
 * structure (4 gates + cell state). Dense small hidden size (shrink-to-budget).
 *
 * Usage:
 *   lstm_reset();
 *   for (int t = 0; t < WINDOW_T; t++) { float x[3]={ax,ay,az}; lstm_step(x); }
 *   uint8_t cls = lstm_predict();
 */
#ifndef LSTM_H
#define LSTM_H

#include <stdint.h>
#include "model_weights.h"   // HIDDEN_SIZE, INPUT_DIM, NUM_CLASSES, WINDOW_T

#ifdef __cplusplus
extern "C" {
#endif

void lstm_reset(void);                              // zero h_state and c_state
void lstm_step(const float x_raw[INPUT_DIM]);       // one streaming sample
uint8_t lstm_predict(void);
uint8_t lstm_classify_window(const float X[WINDOW_T][INPUT_DIM]);

const float* lstm_get_hidden_state(void);
const float* lstm_get_logits(void);

#ifdef __cplusplus
}
#endif

#endif // LSTM_H
