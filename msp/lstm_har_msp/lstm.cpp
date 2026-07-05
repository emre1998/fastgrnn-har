/*
 * lstm.cpp - LSTM HAR inference, weight-only Q15 deployment (dense, shrink-H)
 *
 * PyTorch nn.LSTM semantics, gate order [i, f, g, o]:
 *   i = sigmoid(W_ii x + b_ii + W_hi h + b_hi)
 *   f = sigmoid(W_if x + b_if + W_hf h + b_hf)
 *   g = tanh   (W_ig x + b_ig + W_hg h + b_hg)
 *   o = sigmoid(W_io x + b_io + W_ho h + b_ho)
 *   c = f * c_prev + i * g
 *   h = o * tanh(c)
 *
 * Weight layout (model_weights.h), Q15 int16 in Flash:
 *   W_IH: (4*H, D)  W_HH: (4*H, H)  B_IH: (4*H,)  B_HH: (4*H,)
 *   CLS_W: (C, H)   CLS_B: (C,)
 * Gate row offsets: i = 0*H, f = 1*H, g = 2*H, o = 3*H.
 */
#include "lstm.h"
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
static float c_state[HIDDEN_SIZE];
static float last_logits[NUM_CLASSES];

static inline float w_ih(uint16_t r, uint8_t c) { return (float)READ_INT16(&W_IH[r][c]) * W_IH_SCALE; }
static inline float w_hh(uint16_t r, uint8_t c) { return (float)READ_INT16(&W_HH[r][c]) * W_HH_SCALE; }
static inline float b_ih(uint16_t i)            { return (float)READ_INT16(&B_IH[i]) * B_IH_SCALE; }
static inline float b_hh(uint16_t i)            { return (float)READ_INT16(&B_HH[i]) * B_HH_SCALE; }
static inline float cls_w(uint8_t c, uint8_t i) { return (float)READ_INT16(&CLS_W[c][i]) * CLS_W_SCALE; }
static inline float cls_b(uint8_t c)            { return (float)READ_INT16(&CLS_B[c]) * CLS_B_SCALE; }

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

void lstm_reset(void) {
    memset(h_state, 0, sizeof(h_state));
    memset(c_state, 0, sizeof(c_state));
    memset(last_logits, 0, sizeof(last_logits));
}

void lstm_step(const float x_raw[INPUT_DIM]) {
    float xn[INPUT_DIM];
    for (uint8_t k = 0; k < INPUT_DIM; k++)
        xn[k] = (x_raw[k] - INPUT_MEAN[k]) / INPUT_STD[k];

    // gates read OLD h_state; c_state[i] updated in place (only unit i touches it);
    // h written to h_new then committed (later units need old h).
    float h_new[HIDDEN_SIZE];
    const uint16_t I = 0, F = HIDDEN_SIZE, G = 2 * HIDDEN_SIZE, O = 3 * HIDDEN_SIZE;
    for (uint8_t u = 0; u < HIDDEN_SIZE; u++) {
        float ii = b_ih(I + u) + b_hh(I + u);
        float ff = b_ih(F + u) + b_hh(F + u);
        float gg = b_ih(G + u) + b_hh(G + u);
        float oo = b_ih(O + u) + b_hh(O + u);
        for (uint8_t k = 0; k < INPUT_DIM; k++) {
            float x = xn[k];
            ii += x * w_ih(I + u, k); ff += x * w_ih(F + u, k);
            gg += x * w_ih(G + u, k); oo += x * w_ih(O + u, k);
        }
        for (uint8_t k = 0; k < HIDDEN_SIZE; k++) {
            float hk = h_state[k];
            ii += hk * w_hh(I + u, k); ff += hk * w_hh(F + u, k);
            gg += hk * w_hh(G + u, k); oo += hk * w_hh(O + u, k);
        }
        float ig = sigmoid_f(ii);
        float fg = sigmoid_f(ff);
        float cg = tanh_f(gg);
        float og = sigmoid_f(oo);
        c_state[u] = fg * c_state[u] + ig * cg;     // in-place: unit u only
        h_new[u]   = og * tanh_f(c_state[u]);
    }
    memcpy(h_state, h_new, sizeof(h_state));
}

uint8_t lstm_predict(void) {
    uint8_t best_c = 0; float best_v = -1e30f;
    for (uint8_t c = 0; c < NUM_CLASSES; c++) {
        float s = cls_b(c);
        for (uint8_t i = 0; i < HIDDEN_SIZE; i++) s += cls_w(c, i) * h_state[i];
        last_logits[c] = s;
        if (s > best_v) { best_v = s; best_c = c; }
    }
    return best_c;
}

uint8_t lstm_classify_window(const float X[WINDOW_T][INPUT_DIM]) {
    lstm_reset();
    for (uint16_t t = 0; t < WINDOW_T; t++) lstm_step(X[t]);
    return lstm_predict();
}

const float* lstm_get_hidden_state(void) { return h_state; }
const float* lstm_get_logits(void)       { return last_logits; }
