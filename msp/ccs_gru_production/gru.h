/*
 * gru.h - GRU HAR inference engine (deployment baseline)
 *
 * Mirrors the FastGRNN firmware exactly (Q15 weights in PROGMEM, float compute,
 * LUT sigmoid/tanh, streaming one-sample-at-a-time API) so the latency/energy
 * comparison isolates the CELL structure (3 gates) rather than implementation.
 *
 * Dense small hidden size (shrink-to-budget route). HIDDEN_SIZE comes from
 * model_weights.h (exported by export_gru_to_c.py).
 *
 * Usage (identical to fastgrnn.h):
 *   gru_reset();
 *   for (int t = 0; t < WINDOW_T; t++) { float x[3]={ax,ay,az}; gru_step(x); }
 *   uint8_t cls = gru_predict();
 */
#ifndef GRU_H
#define GRU_H

#include <stdint.h>
#include "model_weights.h"   // defines HIDDEN_SIZE, INPUT_DIM, NUM_CLASSES, WINDOW_T

#ifdef __cplusplus
extern "C" {
#endif

void gru_reset(void);                              // zero h_state
void gru_step(const float x_raw[INPUT_DIM]);       // one streaming sample
uint8_t gru_predict(void);                         // classify from current h_state
uint8_t gru_classify_window(const float X[WINDOW_T][INPUT_DIM]);

const float* gru_get_hidden_state(void);
const float* gru_get_logits(void);

#ifdef __cplusplus
}
#endif

#endif // GRU_H
