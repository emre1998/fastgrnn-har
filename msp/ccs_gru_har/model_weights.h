/* model_weights.h - GRU HAR, auto-generated (build_deploy_firmware.py)
 * Dense shrink-H deployment model | H=6 | 3 gates | Q15 weights
 * Nonzero params: 240 | Flash (Q15): 480 bytes | F1 0.9153
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

#define HIDDEN_SIZE  6
#define INPUT_DIM    3
#define NUM_CLASSES  6
#define WINDOW_T     128
#define N_GATES      3

const float W_IH_SCALE  = 5.56868841e-05f;
const float W_HH_SCALE  = 8.04967249e-05f;
const float B_IH_SCALE  = 4.30636874e-05f;
const float B_HH_SCALE  = 4.17052901e-05f;
const float CLS_W_SCALE = 2.39106411e-04f;
const float CLS_B_SCALE = 6.57848208e-05f;

const float INPUT_MEAN[INPUT_DIM] = { 0.826186f, 0.003943f, 0.046434f };
const float INPUT_STD[INPUT_DIM]  = { 0.398807f, 0.404674f, 0.330410f };
const char* const CLASS_NAMES[NUM_CLASSES] = { "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING" };

const int16_t W_IH[N_GATES*HIDDEN_SIZE][INPUT_DIM] PROGMEM = {
  {   -982,  -1579, -16391 },
  { -23461, -12925,  -3774 },
  { -11732,  16601, -27428 },
  {  32767,  -2360,  12667 },
  {  -7916, -14165, -24713 },
  { -16385,   2394,   -136 },
  { -15541, -20814,  -3194 },
  { -13212,   6211, -23624 },
  { -24420,  21544,   1483 },
  { -14246,   7515,   9343 },
  {   8035,  -2743,  14050 },
  { -15640,  -7276,  -7621 },
  { -10584,   2571,   6566 },
  {  13063,  -4203,  -1969 },
  {  16440,   1497,  -3795 },
  { -15251,   4808,   8034 },
  { -18650,  -4571,  -4320 },
  {   6546,   8571,    799 },
};
const int16_t W_HH[N_GATES*HIDDEN_SIZE][HIDDEN_SIZE] PROGMEM = {
  {   -489,   -647, -20246,  -3652, -32767,   -981 },
  {    673,    982, -16655,  -5286,  -3827,    795 },
  {    947,   2303,  -7280,   2854,  -6984,  -8632 },
  {   4689,   2486,    324,   6928, -10203,  -1980 },
  {  -6947,   7546,  -1816,   1599,  -6443, -11971 },
  {   2068,  -3831,  -6229, -17507,   5927,  -2029 },
  {  -7478,   8411, -10113,  21390, -10941, -18183 },
  {  -3608,   6814,  -9087,   9650, -15311,  -5474 },
  {  -8463,   1769,  21857,   3421,   1006, -16462 },
  {  -9095,   9090,  -2727,   6956,  -2165,     78 },
  {  -9542,   4740, -12070, -26242,  15936,  -5089 },
  {    144,   1434,   3634,   1143,  23640,  -9137 },
  {   8476,  -4570, -11749,  -7390,  -6021,   5911 },
  {  -7806,   9443,   5353,  -3538,  -7976,  -4885 },
  {   -455,   2957,  14646,   2105,   4562,     13 },
  {  -7023,  -2344,  11351,  12854,  -2817, -12626 },
  {  -1824,  -8604,    236,  12261,  19598,  -3421 },
  {   7360, -11910,  -5706,   2642,   2206,   8108 },
};
const int16_t B_IH[N_GATES*HIDDEN_SIZE] PROGMEM = {
    1825,   2153,   3276,  15764,  10979,  12387,   4142,  12122,  20226,   7871,  32767,   4761,   3549,    912,  -4823,  -4713,   6604,  -4970
};
const int16_t B_HH[N_GATES*HIDDEN_SIZE] PROGMEM = {
   11719,   8626,  11905,  18883,  20487,  12560,  10734,  11119,   4210,  17424,  32767,   8069,   -591,  -2796, -12038,  -7000,  -6649,   4910
};
const int16_t CLS_W[NUM_CLASSES][HIDDEN_SIZE] PROGMEM = {
  {  -7675,   4352,   3589,  11556, -32767,  -2795 },
  { -11476,   8974,    344,   4470,   3447,  -7187 },
  {   5310,  -7211,  13303,  -6693,  31434,   -240 },
  {   2475,  -9783,  -1694,  -8225,  -3241,   9127 },
  {   6764,   5180,  -9060,  -5211,  -8938,  -1914 },
  {   4900,  -6179, -14814,  20923,  14457,  12721 },
};
const int16_t CLS_B[NUM_CLASSES] PROGMEM = {
   -8368,   8414,  -3051,   2352,  -2973, -32767
};
#endif // MODEL_WEIGHTS_H
