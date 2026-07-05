/*
 * gru.cpp - GRU HAR inference, weight-only Q15 deployment (dense, shrink-H)
 *
 * PyTorch nn.GRU semantics, gate order [r, z, n] (reset, update, new):
 *   r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
 *   z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
 *   n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
 *   h = (1 - z) * n + z * h_prev
 *
 * Weight layout (model_weights.h), Q15 int16 in Flash:
 *   W_IH: (3*H, D)  W_HH: (3*H, H)  B_IH: (3*H,)  B_HH: (3*H,)
 *   CLS_W: (C, H)   CLS_B: (C,)
 * Gate row offsets: r = 0*H, z = 1*H, n = 2*H.
 */
#include "gru.h"
#include "model_weights.h"
#include <math.h>
#include <string.h>

#ifndef USE_LUT
#define USE_LUT 1
#endif
#if USE_LUT
#include "lut.h"
#endif

#ifdef __AVR__
  #include <avr/pgmspace.h>
  #define READ_INT16(ptr) ((int16_t)pgm_read_word(ptr))
  #define READ_LUT(arr, idx) pgm_read_float(&(arr)[idx])
#else
  #define READ_INT16(ptr) (*(const int16_t*)(ptr))
  #define READ_LUT(arr, idx) ((arr)[idx])
#endif

static float h_state[HIDDEN_SIZE];
static float last_logits[NUM_CLASSES];

// --- Q15 weight readers ---
static inline float w_ih(uint16_t r, uint8_t c) { return (float)READ_INT16(&W_IH[r][c]) * W_IH_SCALE; }
static inline float w_hh(uint16_t r, uint8_t c) { return (float)READ_INT16(&W_HH[r][c]) * W_HH_SCALE; }
static inline float b_ih(uint16_t i)            { return (float)READ_INT16(&B_IH[i]) * B_IH_SCALE; }
static inline float b_hh(uint16_t i)            { return (float)READ_INT16(&B_HH[i]) * B_HH_SCALE; }
static inline float cls_w(uint8_t c, uint8_t i) { return (float)READ_INT16(&CLS_W[c][i]) * CLS_W_SCALE; }
static inline float cls_b(uint8_t c)            { return (float)READ_INT16(&CLS_B[c]) * CLS_B_SCALE; }

// --- Activations (identical LUT scheme to FastGRNN) ---
#if USE_LUT
static inline float sigmoid_f(float x) {
    if (x <= LUT_INPUT_MIN) return 0.0f;
    if (x >= LUT_INPUT_MAX) return 1.0f;
    int idx = (int)((x - LUT_INPUT_MIN) * LUT_INPUT_SCALE);
    if (idx < 0) idx = 0; if (idx >= LUT_SIZE) idx = LUT_SIZE - 1;
    return READ_LUT(SIGMOID_LUT, idx);
}
static inline float tanh_f(float x) {
    if (x <= LUT_INPUT_MIN) return -1.0f;
    if (x >= LUT_INPUT_MAX) return 1.0f;
    int idx = (int)((x - LUT_INPUT_MIN) * LUT_INPUT_SCALE);
    if (idx < 0) idx = 0; if (idx >= LUT_SIZE) idx = LUT_SIZE - 1;
    return READ_LUT(TANH_LUT, idx);
}
#else
static inline float sigmoid_f(float x) {
    if (x <= -8.0f) return 0.0f; if (x >= 8.0f) return 1.0f;
    return 1.0f / (1.0f + expf(-x));
}
static inline float tanh_f(float x) {
    if (x <= -8.0f) return -1.0f; if (x >= 8.0f) return 1.0f;
    return tanhf(x);
}
#endif

void gru_reset(void) {
    memset(h_state, 0, sizeof(h_state));
    memset(last_logits, 0, sizeof(last_logits));
}

void gru_step(const float x_raw[INPUT_DIM]) {
    // 1) normalize input
    float xn[INPUT_DIM];
    for (uint8_t k = 0; k < INPUT_DIM; k++)
        xn[k] = (x_raw[k] - INPUT_MEAN[k]) / INPUT_STD[k];

    // 2) all gates read OLD h_state; write to h_new, then commit
    float h_new[HIDDEN_SIZE];
    const uint16_t R = 0, Z = HIDDEN_SIZE, N = 2 * HIDDEN_SIZE;  // gate row offsets
    for (uint8_t i = 0; i < HIDDEN_SIZE; i++) {
        // reset gate r
        float rr = b_ih(R + i) + b_hh(R + i);
        for (uint8_t k = 0; k < INPUT_DIM; k++)     rr += xn[k]      * w_ih(R + i, k);
        for (uint8_t k = 0; k < HIDDEN_SIZE; k++)   rr += h_state[k] * w_hh(R + i, k);
        float r = sigmoid_f(rr);

        // update gate z
        float zz = b_ih(Z + i) + b_hh(Z + i);
        for (uint8_t k = 0; k < INPUT_DIM; k++)     zz += xn[k]      * w_ih(Z + i, k);
        for (uint8_t k = 0; k < HIDDEN_SIZE; k++)   zz += h_state[k] * w_hh(Z + i, k);
        float z = sigmoid_f(zz);

        // new gate n: tanh(W_in x + b_in + r*(W_hn h + b_hn))
        float n_ih = b_ih(N + i);
        for (uint8_t k = 0; k < INPUT_DIM; k++)     n_ih += xn[k]      * w_ih(N + i, k);
        float n_hh = b_hh(N + i);
        for (uint8_t k = 0; k < HIDDEN_SIZE; k++)   n_hh += h_state[k] * w_hh(N + i, k);
        float n = tanh_f(n_ih + r * n_hh);

        h_new[i] = (1.0f - z) * n + z * h_state[i];
    }
    memcpy(h_state, h_new, sizeof(h_state));
}

uint8_t gru_predict(void) {
    uint8_t best_c = 0; float best_v = -1e30f;
    for (uint8_t c = 0; c < NUM_CLASSES; c++) {
        float s = cls_b(c);
        for (uint8_t i = 0; i < HIDDEN_SIZE; i++) s += cls_w(c, i) * h_state[i];
        last_logits[c] = s;
        if (s > best_v) { best_v = s; best_c = c; }
    }
    return best_c;
}

uint8_t gru_classify_window(const float X[WINDOW_T][INPUT_DIM]) {
    gru_reset();
    for (uint16_t t = 0; t < WINDOW_T; t++) gru_step(X[t]);
    return gru_predict();
}

const float* gru_get_hidden_state(void) { return h_state; }
const float* gru_get_logits(void)       { return last_logits; }
