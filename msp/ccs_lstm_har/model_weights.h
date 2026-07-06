/* model_weights.h - LSTM HAR, auto-generated (build_deploy_firmware.py)
 * Dense shrink-H deployment model | H=5 | 4 gates | Q15 weights
 * Nonzero params: 236 | Flash (Q15): 472 bytes | F1 0.8176
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

#define HIDDEN_SIZE  5
#define INPUT_DIM    3
#define NUM_CLASSES  6
#define WINDOW_T     128
#define N_GATES      4

const float W_IH_SCALE  = 7.37791666e-05f;
const float W_HH_SCALE  = 4.84888760e-05f;
const float B_IH_SCALE  = 5.56229010e-05f;
const float B_HH_SCALE  = 3.94744462e-05f;
const float CLS_W_SCALE = 1.08764988e-04f;
const float CLS_B_SCALE = 3.82410973e-05f;

const float INPUT_MEAN[INPUT_DIM] = { 0.826186f, 0.003943f, 0.046434f };
const float INPUT_STD[INPUT_DIM]  = { 0.398807f, 0.404674f, 0.330410f };
const char* const CLASS_NAMES[NUM_CLASSES] = { "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING" };

const int16_t W_IH[N_GATES*HIDDEN_SIZE][INPUT_DIM] PROGMEM = {
  { -14913,  -2231,   -242 },
  { -14104,  -7717,   7391 },
  {  -4341,  26910,  -2903 },
  {  -9192,   -937,  -2734 },
  {  -3066, -11355,   1809 },
  {   -259,  -3711,   1877 },
  { -15101,   7010,   6971 },
  {  -3996,   7562,   3422 },
  {   5946,   -853,   -606 },
  {  -9735,  -7736,    839 },
  { -16782,   5047,   2620 },
  {  -5847,  -4164,  -3057 },
  {  -7573,  -3691,    188 },
  {   6314,  -2617,    644 },
  {   3339,  -8998,  -1248 },
  {  -4289,     13,  -1056 },
  {  32767,   3442,   5377 },
  {   5595,   6411,   5453 },
  { -18082,   1347, -14273 },
  {   4776,   9208, -11251 },
};
const int16_t W_HH[N_GATES*HIDDEN_SIZE][HIDDEN_SIZE] PROGMEM = {
  {     56,  14759,  -2518,  -4689,  19753 },
  {  10692,  -2056,  -9512,   4282,  12147 },
  {  -1320, -26115,   3725,  -1865,   4959 },
  {   3288,  15975,  -9855,  15245,   1455 },
  {   6972,   2362,   9116,   -681,  -7909 },
  {  10950,   9125,  -9517,  -7948,   8328 },
  {  20928,  15198,  -2867,     17,   1570 },
  {  15779,  12591, -15051, -13837,   9903 },
  {  13598,  11325,  -7062,    719,  -8449 },
  {  10503,   1766,  -8747,   7270,   7481 },
  {  10662,   4253,   4701,  -3106,   6978 },
  {  -8783,  14454,   3586,   2128,   5719 },
  {   -466,   7153,  13750,   1171,  -7739 },
  { -17717, -14615,  -3964,   5329, -12635 },
  {   5453,   6101, -17103,  -1422,  12491 },
  {   6053,  12489,  -3411,  -3150,  27295 },
  {  11777,   5897,   -515,  15762,  -2235 },
  {  19693,   8150, -18886, -15366,   1600 },
  { -20554, -32767,  -7465,   9195, -10787 },
  {  18366,   -124,  10709,  10960, -15544 },
};
const int16_t B_IH[N_GATES*HIDDEN_SIZE] PROGMEM = {
    2912,   5058,  -5508,   8017,   5258,  12652,  13331,  16139,  13162,   6633,   4266,  -2797,   6370,    561,  -3266,  32767,  16730,  10764,  27536,   7590
};
const int16_t B_HH[N_GATES*HIDDEN_SIZE] PROGMEM = {
    7501,   6980,  10495,  24242,  14626,  21682,  21206,  16091,  19407,  16249,   5043,   8432,  -9245,  -2808,  -2364,  32767,  18280,  24096,  22991,  19602
};
const int16_t CLS_W[NUM_CLASSES][HIDDEN_SIZE] PROGMEM = {
  {   1822,  14519, -13826,  -1559,    720 },
  {   5638,   4038,  14631,  -5816,  12918 },
  {   1990,  -4850, -17400,  -6213,    847 },
  {  -5220,  -1201,  -5245,  18679, -16858 },
  { -24018,  16378,   3471,  17301,   1122 },
  {   9982, -19488,  26645, -19272, -32767 },
};
const int16_t CLS_B[NUM_CLASSES] PROGMEM = {
   -5839,   2768, -14380,  32767,  16172, -13621
};
#endif // MODEL_WEIGHTS_H
