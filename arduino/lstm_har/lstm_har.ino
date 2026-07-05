/*
 * lstm_har.ino - LSTM HAR latency + energy measurement harness (Arduino Uno)
 *
 * Mirrors fastgrnn_har.ino / gru_har.ino benchmark modes for an apples-to-apples
 * latency/energy comparison. Input values do not affect compute cost, so the
 * benchmarks feed a zero sample.
 *
 * Target: Arduino Uno R3 (ATmega328P, 16 MHz, 32 KB Flash, 2 KB SRAM), 115200 baud.
 *
 * TEST_MODE: 0 = LATENCY (default), 3 = ENERGY (BENCH_MODE 0 idle / 1 stream / 2 continuous).
 */
#include "lstm.h"
#include "model_weights.h"

#ifndef TEST_MODE
#define TEST_MODE 0
#endif
#ifndef BENCH_MODE
#define BENCH_MODE 1
#endif
#define LED_PIN 13
#define SAMPLE_PERIOD_MS 20   // 50 Hz

static const float ZERO[INPUT_DIM] = {0.0f, 0.0f, 0.0f};

void setup() {
#if TEST_MODE == 3
    pinMode(LED_PIN, OUTPUT);
    lstm_reset();
    return;
#else
    Serial.begin(115200);
    delay(500);
    Serial.println(F("=== LSTM HAR - latency (Arduino Uno) ==="));
    Serial.print(F("H=")); Serial.print(HIDDEN_SIZE);
    Serial.print(F("  window=")); Serial.print(WINDOW_T);
    Serial.print(F("  classes=")); Serial.println(NUM_CLASSES);

    lstm_reset();
    for (uint16_t t = 0; t < WINDOW_T; t++) lstm_step(ZERO);   // warm-up

    lstm_reset();
    unsigned long t0 = micros();
    for (uint16_t t = 0; t < WINDOW_T; t++) lstm_step(ZERO);
    unsigned long window_us = micros() - t0;

    unsigned long tc = micros();
    volatile uint8_t cls = lstm_predict();
    unsigned long predict_us = micros() - tc;
    (void)cls;

    Serial.print(F("Per-step latency:   "));
    Serial.print((float)window_us / WINDOW_T, 2); Serial.println(F(" us/step"));
    Serial.print(F("Full window (")); Serial.print(WINDOW_T); Serial.print(F(" steps): "));
    Serial.print(window_us); Serial.println(F(" us"));
    Serial.print(F("Classifier:         "));
    Serial.print(predict_us); Serial.println(F(" us"));
    Serial.print(F("Window budget @50Hz: "));
    Serial.print((unsigned long)WINDOW_T * SAMPLE_PERIOD_MS * 1000UL); Serial.println(F(" us"));
    float util = 100.0f * window_us / ((float)WINDOW_T * SAMPLE_PERIOD_MS * 1000.0f);
    Serial.print(F("Compute utilization: ")); Serial.print(util, 3); Serial.println(F(" %"));
    Serial.println(F("(real-time OK if per-step << 20000 us)"));
#endif
}

void loop() {
#if TEST_MODE == 3
  #if BENCH_MODE == 0
    delay(20);
  #elif BENCH_MODE == 1
    unsigned long t0 = millis();
    lstm_step(ZERO);
    while ((millis() - t0) < 20) { /* idle */ }
  #else
    lstm_step(ZERO);
  #endif
#endif
}
