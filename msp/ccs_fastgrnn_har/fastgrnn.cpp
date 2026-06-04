/*
 * fastgrnn.cpp — FastGRNN HAR inference, weights-only Q15 deployment
 *
 * Mimari: low-rank (r_w=2, r_u=8) + sparse %50 + Q15 ağırlıklar
 * Forward: x_t (B, D) → h_t (B, H), T=128 adim, classifier ile sinif
 *
 * Q15 dequantize: ham int16 değer × per-tensor scale = float ağırlık
 * Sparsity: 0 olan ağırlıklar zaten 0'a çarpar → ekstra is yok
 */

#include "fastgrnn.h"
#include "model_weights.h"
#include "lut.h"
#include <math.h>
#include <string.h>

// ============================================================================
// Platform abstraction: AVR'da PROGMEM, host'ta direkt erisim
// ============================================================================
#ifdef __AVR__
  #include <avr/pgmspace.h>
  #define READ_INT16(ptr) ((int16_t)pgm_read_word(ptr))
  #define READ_LUT(arr, idx) pgm_read_float(&(arr)[idx])
#else
  // Host (PC) ve MSP430: PROGMEM stub — direkt erisim
  #define READ_INT16(ptr) (*(const int16_t*)(ptr))
  #define READ_LUT(arr, idx) ((arr)[idx])
#endif

// ============================================================================
// Persistent state (SRAM): streaming inference icin h_t saklanir
// ============================================================================
static float h_state[HIDDEN_STATE_SIZE];   // 64 byte SRAM
static float last_logits[NUM_OUTPUT_CLASSES];  // 24 byte SRAM

// ============================================================================
// Yardımcı: Q15 ağırlık matrisinden float değer oku
// ============================================================================
// 2D matris (rows × cols) için: W[i][j] → float
static inline float read_W1(uint8_t i, uint8_t j) {
    return (float)READ_INT16(&W1[i][j]) * W1_SCALE;
}
static inline float read_W2(uint8_t i, uint8_t j) {
    return (float)READ_INT16(&W2[i][j]) * W2_SCALE;
}
static inline float read_U1(uint8_t i, uint8_t j) {
    return (float)READ_INT16(&U1[i][j]) * U1_SCALE;
}
static inline float read_U2(uint8_t i, uint8_t j) {
    return (float)READ_INT16(&U2[i][j]) * U2_SCALE;
}
static inline float read_BZ(uint8_t i)  { return (float)READ_INT16(&B_Z[i])  * BZ_SCALE; }
static inline float read_BH(uint8_t i)  { return (float)READ_INT16(&B_H[i])  * BH_SCALE; }
static inline float read_CLS_W(uint8_t c, uint8_t i) {
    return (float)READ_INT16(&CLS_W[c][i]) * CLS_W_SCALE;
}
static inline float read_CLS_B(uint8_t c) { return (float)READ_INT16(&CLS_B[c]) * CLS_B_SCALE; }

// ============================================================================
// Aktivasyon fonksiyonlari — LUT tabanli (expf/tanhf yerine, ~3-5x hizli)
// ============================================================================
// LUT input araligi: [-8, 8], 256 bucket. Disinda saturate.
static inline float sigmoid_f(float x) {
    if (x <= LUT_INPUT_MIN) return 0.0f;
    if (x >= LUT_INPUT_MAX) return 1.0f;
    int idx = (int)((x - LUT_INPUT_MIN) * LUT_INPUT_SCALE);
    if (idx < 0) idx = 0;
    if (idx >= LUT_SIZE) idx = LUT_SIZE - 1;
    return READ_LUT(SIGMOID_LUT, idx);
}

static inline float tanh_f(float x) {
    if (x <= LUT_INPUT_MIN) return -1.0f;
    if (x >= LUT_INPUT_MAX) return  1.0f;
    int idx = (int)((x - LUT_INPUT_MIN) * LUT_INPUT_SCALE);
    if (idx < 0) idx = 0;
    if (idx >= LUT_SIZE) idx = LUT_SIZE - 1;
    return READ_LUT(TANH_LUT, idx);
}

// ============================================================================
// Streaming API
// ============================================================================
void fastgrnn_reset(void) {
    memset(h_state, 0, sizeof(h_state));
    memset(last_logits, 0, sizeof(last_logits));
}

// Tek bir zaman adimi: x_t (ham g cinsinden) -> h_state guncelle
//
// Matematik:
//   xn = (x_raw - mean) / std       // normalize (per-channel)
//   xW = xn @ W2 @ W1.T              // (H,) intermediate (low-rank distribute)
//   hU = h_prev @ U2 @ U1.T          // (H,)
//   pre = xW + hU
//   z = sigmoid(pre + b_z)           // kapi
//   h_tilde = tanh(pre + b_h)        // aday hafiza
//   h_t = (zeta*(1-z) + nu) * h_tilde + z * h_prev
void fastgrnn_step(const float x_raw[INPUT_CHANNELS]) {
    // --- 1) Input normalize ---
    float xn[INPUT_CHANNELS];
    for (uint8_t i = 0; i < INPUT_CHANNELS; i++) {
        xn[i] = (x_raw[i] - INPUT_MEAN[i]) / INPUT_STD[i];
    }

    // --- 2) Low-rank intermediate: xz = xn @ W2 → (R_W,) ---
    // xn: (3,), W2: (3, 2), xz: (2,)
    float xz[R_W];
    for (uint8_t j = 0; j < R_W; j++) {
        float s = 0.0f;
        for (uint8_t k = 0; k < INPUT_CHANNELS; k++) {
            s += xn[k] * read_W2(k, j);
        }
        xz[j] = s;
    }
    // --- 3) Hidden-to-hidden: hz = h_state @ U2 → (R_U,) ---
    float hz[R_U];
    for (uint8_t j = 0; j < R_U; j++) {
        float s = 0.0f;
        for (uint8_t k = 0; k < HIDDEN_STATE_SIZE; k++) {
            s += h_state[k] * read_U2(k, j);
        }
        hz[j] = s;
    }

    // --- 4) FastGRNN birlestirme ---
    // xW[16] ve hU[16] gecici dizilerini stack'te tutmak yerine her hidden
    // unit icin dogrudan hesapla. Bu kucuk MCU'larda stack baskisini ~128 byte azaltir.
    for (uint8_t i = 0; i < HIDDEN_STATE_SIZE; i++) {
        float xw = 0.0f;
        for (uint8_t j = 0; j < R_W; j++) {
            xw += xz[j] * read_W1(i, j);
        }

        float hu = 0.0f;
        for (uint8_t j = 0; j < R_U; j++) {
            hu += hz[j] * read_U1(i, j);
        }

        float pre = xw + hu;
        float z = sigmoid_f(pre + read_BZ(i));
        float h_tilde = tanh_f(pre + read_BH(i));
        float coef = ZETA * (1.0f - z) + NU;
        h_state[i] = coef * h_tilde + z * h_state[i];
    }
}

// Mevcut h_state'ten sinif tahmini
uint8_t fastgrnn_predict(void) {
    // logits = CLS_W @ h_state + CLS_B
    // CLS_W: (NUM_CLASSES, HIDDEN), CLS_B: (NUM_CLASSES,)
    uint8_t best_c = 0;
    float best_v = -1e30f;
    for (uint8_t c = 0; c < NUM_OUTPUT_CLASSES; c++) {
        float s = read_CLS_B(c);
        for (uint8_t i = 0; i < HIDDEN_STATE_SIZE; i++) {
            s += read_CLS_W(c, i) * h_state[i];
        }
        last_logits[c] = s;
        if (s > best_v) {
            best_v = s;
            best_c = c;
        }
    }
    return best_c;
}

// Convenience: tek pencerede tahmin
uint8_t fastgrnn_classify_window(const float X[WINDOW_LEN][INPUT_CHANNELS]) {
    fastgrnn_reset();
    for (uint16_t t = 0; t < WINDOW_LEN; t++) {
        fastgrnn_step(X[t]);
    }
    return fastgrnn_predict();
}

const float* fastgrnn_get_hidden_state(void) { return h_state; }
const float* fastgrnn_get_logits(void)       { return last_logits; }
