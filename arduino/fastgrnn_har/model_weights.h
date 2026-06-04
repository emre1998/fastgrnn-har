/*
 * model_weights.h — FastGRNN HAR model, auto-generated
 *
 * Pipeline: low-rank (r_w=2, r_u=8) + sparsity 50% + Q15
 * Source checkpoint: sparse_h16_rw2_ru8_sp50_s0_e100_best.pt
 * Total params (dense kabul): 430
 * Nonzero params: 281
 * Flash size (Q15): 562 byte
 *
 * PORTABLE: AVR (Arduino) ve MSP430 (Energia) icin uygun.
 *   - AVR: PROGMEM macrosu kullanilir
 *   - MSP430 ve diðerleri: const ile Flash'a otomatik
 */

#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#include <stdint.h>

#ifdef __AVR__
  #include <avr/pgmspace.h>
#else
  #ifndef PROGMEM
    #define PROGMEM
  #endif
#endif

// --- Mimari sabitleri ---
#define HIDDEN_SIZE  16
#define INPUT_DIM    3
#define NUM_CLASSES  6
#define WINDOW_T     128
#define R_W          2
#define R_U          8

// --- Sabit-nokta scale'leri ---
const float W1_SCALE       = 4.23673642e-05f;
const float W2_SCALE       = 2.99582127e-05f;
const float U1_SCALE       = 2.87499885e-05f;
const float U2_SCALE       = 3.33665954e-05f;
const float BZ_SCALE       = 3.39441349e-05f;
const float BH_SCALE       = 2.22018707e-05f;
const float CLS_W_SCALE    = 3.67618682e-05f;
const float CLS_B_SCALE    = 4.02555768e-05f;

// --- Aktivasyon scale'leri (kalibrasyondan, headroom %10) ---
const float Z_MAX_ABS       = 1.1000f;   // sigmoid output range
const float H_TILDE_MAX_ABS = 1.1000f;  // tanh output range
const float H_T_MAX_ABS     = 68.1297f;   // h_t actual range (genis!)

// --- Skaler parametreler (sigmoid sonrasi) ---
const float ZETA = 0.543981f;
const float NU   = 0.502244f;

// --- Input normalizasyon ---
const float INPUT_MEAN[INPUT_DIM] = { 0.814323f, 0.004648f, 0.058871f };
const float INPUT_STD[INPUT_DIM]  = { 0.404806f,  0.414163f,  0.336902f };

// --- Sinif isimleri (UART output icin) ---
const char* const CLASS_NAMES[NUM_CLASSES] = {
  "WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"
};

// ============================================================================
// AÐIRLIKLAR — int16_t, PROGMEM (Flash)
// ============================================================================

// W1: (HIDDEN_SIZE, R_W) = (16, 2)
const int16_t W1[HIDDEN_SIZE][R_W] PROGMEM = {
  { -14739,      0 },
  {      0,      0 },
  {  10441,  20976 },
  {      0, -32767 },
  {      0, -19192 },
  {      0,      0 },
  {      0,      0 },
  {  28354,      0 },
  { -23685, -21931 },
  {  17471,      0 },
  {      0, -20857 },
  { -17829,  19430 },
  {      0,      0 },
  {      0,      0 },
  {  20685,  21598 },
  {  15411, -17022 },
};

// W2: (INPUT_DIM, R_W) = (3, 2)
const int16_t W2[INPUT_DIM][R_W] PROGMEM = {
  { -30772,      0 },
  { -32154, -32767 },
  {      0,      0 },
};

// U1: (HIDDEN_SIZE, R_U) = (16, 8)
const int16_t U1[HIDDEN_SIZE][R_U] PROGMEM = {
  {      0,  -7711,  13465,   7103,      0,  11315,   6214,      0 },
  {      0,  -8715, -13306,  10646,  -6469,   5537,  15396,  24238 },
  {  10193,  14868, -18359,  -7871,      0,  25173,      0,  12177 },
  { -21090,  -5891,      0,   8388,  -9422,      0,      0,      0 },
  {      0,      0, -12749,      0,      0,      0,      0,      0 },
  { -11500,      0,      0,  17195,      0,  20435, -20336,      0 },
  {  32767,      0,      0,      0,  12766, -13801,   9586,      0 },
  {      0,  -6307,      0,      0,      0,      0,      0,  10856 },
  { -14812, -13174,      0, -11314,      0, -16950,      0,      0 },
  {      0,      0,      0, -10087,      0,      0,  17126,  -5806 },
  {      0,      0,      0,  -9461, -15554,      0,      0,   7029 },
  {      0,      0,      0,  -5982,      0,  10528,  -8743, -15100 },
  {  -3474,  18093,  32309,   6272,      0,      0,      0,   9803 },
  {  32242,   9268,      0,   7369,      0, -11058, -10956,      0 },
  { -13704,  -4636,      0,      0,  16223,  -9677,      0,  -8916 },
  {      0,      0,   4573, -17991,      0,      0,      0,      0 },
};

// U2: (HIDDEN_SIZE, R_U) = (16, 8)
const int16_t U2[HIDDEN_SIZE][R_U] PROGMEM = {
  { -16343,  -5761,      0,  -7309,      0,      0, -13391,      0 },
  {      0,   8720,      0,      0, -11163,      0,   8102,   6553 },
  {      0,  -8795,      0, -18569,  -6809,      0,      0,      0 },
  {   6699,  -8601,  17378,      0,      0,  -6843,      0,      0 },
  {  -4684, -10423,      0, -19433,      0,      0,  12132,  12989 },
  {      0,   5997,      0,  -6846,      0,  11656,      0,      0 },
  {      0, -17171, -19826,   6847,   6213,  14914,      0,      0 },
  {   5731,  14424,      0,      0,      0,      0,      0,   7913 },
  {      0,   7849,      0,   8935,  -6415, -10437,      0, -10288 },
  {  -7987,  -7361,      0,   9818,  -7483,      0,   8125,      0 },
  {  11201,      0,      0,   9481,      0,      0,      0,      0 },
  {  32767,  -8317, -23412,      0,      0,  11308,      0,      0 },
  {  14450,  10138,  -6076,      0,      0,  12711,      0,      0 },
  { -10222,      0,      0,  -5144,      0,   9543,  11838,      0 },
  {  -6735, -17819,      0,      0,      0,      0,      0,      0 },
  {  -9488,      0,   8383,   9039,  -5780,  11378,  -8564,  11606 },
};

// b_z, b_h: (HIDDEN_SIZE,)
const int16_t B_Z[HIDDEN_SIZE] PROGMEM = {
    2479,   6556,   8711,  -8176,  32767,   5401,  14995,   -551,  -3003,  12046,  -5946,  11096, 
    3959,  14589,   3295,  10170
};

const int16_t B_H[HIDDEN_SIZE] PROGMEM = {
  -14208,   1091, -32767,  -2204,  13029, -28239,   9127,    509,   4595, -31225,  19816, -10270, 
    3179,  -5378,  -1453, -11599
};

// Classifier W: (NUM_CLASSES, HIDDEN_SIZE)
const int16_t CLS_W[NUM_CLASSES][HIDDEN_SIZE] PROGMEM = {
  {  -8743,   8814,  -3647, -15604,   1598, -32767,  20305,   1878,  -1374, -11640,   9462,  -2487,   4093,  -4701,   3049,   2034 },
  {  -7132,   3222,   7066,  -9594,  -4875,   5916,  12525,   1783,   8451,  -1283,   5983,  -1207,    191,  -1862,   3914,  -1572 },
  {  -3500,  19868, -13483,   5985,   1413,   4879,   6099,  -4228, -20581,   6498,   6789,   6504,  -2538,  -2544, -20180,  -7921 },
  {   1635,  -6165,   7473,   9651,  -6900,  10905, -17201,   4435,   3486,  13117,  -5921,    577,   3223,  -4303,  -4862,  11054 },
  {   5146, -12001,  12079,  -2321,  -6694,  -3136, -21206,   8936,    377,  -4566,  -4534,   1475,   1168,   8182,   6259,  10192 },
  {  11557,  -8238,   1283,  12451,  15393,  17629, -12687,  14398,  -5245,   8589,   -815,  -1662,  -2463,   4910, -18144,   8171 },
};

// Classifier bias: (NUM_CLASSES,)
const int16_t CLS_B[NUM_CLASSES] PROGMEM = {
   32767,  15777,  -5858,  -8691,  -7375, -25572
};

#endif // MODEL_WEIGHTS_H
