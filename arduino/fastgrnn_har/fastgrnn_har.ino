/*
 * fastgrnn_har.ino - Arduino Uno HAR demo
 *
 * Mode 0 (TEST_MODE 0): Live mode - read MPU6050, streaming inference
 * Mode 1 (TEST_MODE 1): No MPU6050 - run inference on embedded test data
 * Mode 2 (TEST_MODE 2): Embedded test data fed at 50 Hz, streaming simulation
 *
 * Target hardware: Arduino Uno R3 (ATmega328P, 16 MHz, 32 KB Flash, 2 KB SRAM)
 *
 * UART output: 115200 baud
 * I2C (Wire): SDA = A4, SCL = A5 (standard Arduino Uno pin assignment)
 * MPU6050 I2C address: 0x68 (default, AD0 tied to GND)
 *
 * Required Arduino libraries:
 *   - I2Cdev (Jeffrey Rowberg)
 *   - MPU6050 (Jeff Rowberg)
 *   Install: Sketch -> Include Library -> Manage Libraries
 *   or from GitHub as a ZIP: https://github.com/jrowberg/i2cdevlib
 */

#include "fastgrnn.h"
#include "model_weights.h"
#include "test_data.h"
#include <avr/pgmspace.h>

// MPU6050 I2C libraries
#include "I2Cdev.h"
#include "MPU6050.h"
#include <Wire.h>

// Mode selection:
//   0 = LIVE   (MPU6050 streaming, 50 Hz sampling)
//   1 = TEST   (embedded test data, full-window batch inference)
//   2 = STREAM (embedded test data, 50 Hz paced streaming simulation)
//   3 = ENERGY (current/power benchmark - no UART, no LED, see BENCH_MODE)
#ifndef TEST_MODE
#define TEST_MODE 0
#endif

// Energy benchmark sub-mode (only used when TEST_MODE == 3):
//   0 = IDLE       (just delay(20), measures baseline + USB bridge)
//   1 = STREAM50HZ (fastgrnn_step every 20 ms, idle the rest - realistic HAR)
//   2 = CONTINUOUS (tight loop of fastgrnn_step - worst case always-on)
#ifndef BENCH_MODE
#define BENCH_MODE 1
#endif

// INA226 self-measurement (only used when TEST_MODE == 3):
//   0 = silent benchmark (use this when an external INA226 host is reading)
//   1 = Arduino reads its own INA226 every 1 s and prints a CSV line.
//       The Serial transmit + I2C transaction add ~5 mA of constant bias to
//       the measurement; the bias is the same in every BENCH_MODE so it
//       cancels out in the inter-mode delta.
//
// REQUIREMENTS when 1:
//   - INA226 breakout wired with its shunt in series with the Arduino's
//     5V rail (see docs/energy_measurement.md for the diagram).
//   - Library "INA226_WE" by Wolfgang Ewald installed via Arduino IDE.
#ifndef INA226_SELF_READ
#define INA226_SELF_READ 0
#endif

#define STRINGIFY_VALUE(x) #x
#define STRINGIFY(x) STRINGIFY_VALUE(x)

#if TEST_MODE == 3 && INA226_SELF_READ
  // Pulled in only for the energy self-measurement build to avoid forcing
  // the dependency on every user. Install via Library Manager.
  #include <INA226_WE.h>
  static INA226_WE ina226_self(0x40);
#endif

// LED for visual feedback (Arduino Uno pin 13)
#define LED_PIN 13

// Sampling configuration (live mode)
#define SAMPLE_RATE_HZ 50
#define SAMPLE_PERIOD_MS (1000 / SAMPLE_RATE_HZ)  // 20 ms @ 50 Hz

// ============================================================================
// Input normalization - training data statistics
// Computed from data/processed/hapt_windows.npz (train split)
// Formula: x_norm = (x - mean) / std
// ============================================================================
const float NORM_MEAN[3] = {0.8143f, 0.0047f, 0.0589f};
const float NORM_STD[3]  = {0.4048f, 0.4142f, 0.3369f};

// ============================================================================
// MPU6050 global instance and state (LIVE mode)
// ============================================================================
MPU6050 mpu(0x68);  // I2C address: 0x68 (AD0 = GND)

// MPU6050 raw acceleration (16-bit)
int16_t accel_raw[3] = {0, 0, 0};  // [x, y, z]

// Sampling state
static uint16_t sample_count = 0;
static unsigned long last_sample_ms = 0;
static unsigned long last_predict_ms = 0;

void setup() {
    Serial.begin(115200);
    pinMode(LED_PIN, OUTPUT);
    delay(500);

    Serial.println(F("============================="));
    Serial.println(F(" FastGRNN HAR - Arduino Uno"));
    Serial.println(F("============================="));
    Serial.print(F("Mode: "));
#if TEST_MODE == 1
    Serial.println(F("TEST (embedded data, batch)"));
    run_embedded_tests();
#elif TEST_MODE == 2
    Serial.println(F("STREAM (embedded data, 50 Hz paced)"));
    run_streaming_simulation();
#elif TEST_MODE == 3
    Serial.println(F("ENERGY BENCHMARK (mode " STRINGIFY(BENCH_MODE) ")"));
  #if INA226_SELF_READ
    // Self-measurement: keep UART up so we can print INA226 readings.
    Wire.begin();
    if (!ina226_self.init()) {
        Serial.println(F("[ERROR] INA226 not found at 0x40. Check wiring."));
        while (1) { digitalWrite(LED_PIN, (millis() / 100) % 2); }
    }
    ina226_self.setAverage(INA226_AVERAGE_16);
    ina226_self.setConversionTime(INA226_CONV_TIME_1100);
    ina226_self.setMeasureMode(INA226_CONTINUOUS);
    ina226_self.setResistorRange(0.1f, 0.8f);
    Serial.println(F("INA226 self-read enabled. CSV: t_ms,mA,mV,mW"));
  #else
    // External-host measurement: go fully silent so the bench reflects only
    // the inference + idle cost.
    Serial.println(F("UART will go silent after this line."));
    Serial.flush();
    Serial.end();
  #endif
    fastgrnn_reset();
#else
    Serial.println(F("LIVE (MPU6050 streaming, 50 Hz)"));
    Serial.print(F("Initializing MPU6050..."));
    setup_mpu6050();
    Serial.println(F("[OK]"));
    fastgrnn_reset();
    last_sample_ms = millis();
#endif
}

void loop() {
#if TEST_MODE == 3
    // ============================================================
    // ENERGY BENCHMARK - tight workload loop for ammeter measurement
    // ============================================================
    static const float zero[3] = {0.0f, 0.0f, 0.0f};

  #if BENCH_MODE == 0
    // IDLE: sleep one 20-ms tick at a time, never call the model.
    delay(20);
  #elif BENCH_MODE == 1
    // STREAM50HZ: pace exactly like deployed HAR
    unsigned long t0 = millis();
    fastgrnn_step(zero);
    while ((millis() - t0) < 20) { /* busy idle */ }
  #else // BENCH_MODE == 2
    // CONTINUOUS: pin the CPU on inference, worst-case envelope
    fastgrnn_step(zero);
  #endif

  #if INA226_SELF_READ
    // Emit one CSV reading per second. This adds a constant ~35 ms of
    // I2C + ~5 ms of UART work per second (~4% duty cycle bias) that
    // applies equally to every BENCH_MODE.
    static unsigned long last_log_ms = 0;
    unsigned long now_ms = millis();
    if (now_ms - last_log_ms >= 1000) {
        last_log_ms = now_ms;
        ina226_self.readAndClearFlags();
        Serial.print(now_ms);
        Serial.print(',');
        Serial.print(ina226_self.getCurrent_mA(), 3);
        Serial.print(',');
        Serial.print(ina226_self.getBusVoltage_V() * 1000.0f, 1);
        Serial.print(',');
        Serial.println(ina226_self.getBusPower(), 2);
    }
  #endif
    return;
#endif

#if TEST_MODE == 0
    unsigned long now = millis();

    if (now - last_sample_ms >= SAMPLE_PERIOD_MS) {
        last_sample_ms = now;

        float ax, ay, az;
        read_mpu6050(&ax, &ay, &az);

        // DEBUG: print sensor values periodically (every ~0.5 s)
        if (sample_count % 25 == 0) {
            Serial.print("Raw: ");
            Serial.print(ax, 3); Serial.print(" ");
            Serial.print(ay, 3); Serial.print(" ");
            Serial.println(az, 3);
        }

        float x[3] = { ax, ay, az };
        fastgrnn_step(x);
        sample_count++;

        // After a complete 128-sample window: predict + RESET.
        // The training pipeline used independent 128-sample windows; if we
        // never reset h_state, it accumulates indefinitely, blows up the
        // logits, and pins the prediction to LAYING.
        if (sample_count >= WINDOW_LEN) {  // 128 samples = 2.56 s
            sample_count = 0;
            uint8_t cls = fastgrnn_predict();
            print_prediction(cls);
            fastgrnn_reset();  // CRITICAL: zero h for the next window
        }
    }
#endif
}

// ============================================================================
// TEST MODE: run the embedded test windows
// ============================================================================
void run_embedded_tests() {
    Serial.println();
    Serial.print(F("Embedded test windows: "));
    Serial.println(N_TEST_SAMPLES);
    Serial.println();

    for (uint8_t s = 0; s < N_TEST_SAMPLES; s++) {
        Serial.print(F("--- Test "));
        Serial.print(s);
        Serial.println(F(" ---"));

        // Read the window from PROGMEM and run inference
        fastgrnn_reset();
        unsigned long t0 = millis();

        // Streaming style: feed sample by sample (matches real deployment)
        for (uint16_t t = 0; t < WINDOW_LEN; t++) {
            float x[3];
            // Float read from PROGMEM array
            const float* row;
            if (s == 0) row = TEST_WINDOW_0[t];
            else        row = TEST_WINDOW_1[t];
            x[0] = pgm_read_float(&row[0]);
            x[1] = pgm_read_float(&row[1]);
            x[2] = pgm_read_float(&row[2]);
            fastgrnn_step(x);
        }
        uint8_t pred = fastgrnn_predict();
        unsigned long elapsed_ms = millis() - t0;

        Serial.print(F("Prediction: "));
        Serial.print(pred);
        Serial.print(F(" ("));
        Serial.print(CLASS_NAMES[pred]);
        Serial.println(F(")"));

        Serial.print(F("Expected (Python): "));
        Serial.print(TEST_EXPECTED[s]);
        Serial.print(F(" ("));
        Serial.print(CLASS_NAMES[TEST_EXPECTED[s]]);
        Serial.println(F(")"));

        Serial.print(F("Ground truth:      "));
        Serial.print(TEST_TRUE[s]);
        Serial.print(F(" ("));
        Serial.print(CLASS_NAMES[TEST_TRUE[s]]);
        Serial.println(F(")"));

        Serial.print(F("Inference time: "));
        Serial.print(elapsed_ms);
        Serial.println(F(" ms"));

        if (pred == TEST_EXPECTED[s]) {
            Serial.println(F("[PASS] matches Python"));
        } else {
            Serial.println(F("[FAIL] differs from Python - bug!"));
        }

        // Logits
        const float* lg = fastgrnn_get_logits();
        Serial.print(F("Logits: "));
        for (uint8_t c = 0; c < NUM_CLASSES; c++) {
            Serial.print(lg[c], 3);
            Serial.print(F(" "));
        }
        Serial.println();
        Serial.println();
    }

    Serial.println(F("Tests complete."));
}

// ============================================================================
// STREAMING SIMULATION (TEST_MODE == 2)
// Feed the embedded test window at 50 Hz pacing to simulate real MPU6050
// streaming. Measures: per-sample latency, prediction drift, total time,
// over-budget count.
// ============================================================================
void run_streaming_simulation() {
    // Use the global SAMPLE_PERIOD_MS macro (1000 / 50 = 20 ms = 50 Hz)
    const uint16_t PREDICT_EVERY = 25;       // predict every 0.5 s
    const uint8_t  USE_WINDOW = 0;           // 0 = STANDING, 1 = UPSTAIRS

    Serial.println();
    Serial.print(F("STREAM SIM (50 Hz, window "));
    Serial.print(USE_WINDOW);
    Serial.print(F(", "));
    Serial.print(WINDOW_LEN);
    Serial.println(F(" samples)"));
    Serial.println(F("Headers: t, sample_lat_ms, h0, pred"));
    Serial.println(F("---------------------------------"));

    fastgrnn_reset();
    unsigned long total_start = millis();
    unsigned long max_lat = 0, sum_lat = 0;
    uint16_t over_budget = 0;

    for (uint16_t t = 0; t < WINDOW_LEN; t++) {
        unsigned long t0 = millis();

        // "Sensor read" - actually reads from the embedded PROGMEM window
        float x[3];
        const float* row = (USE_WINDOW == 0) ? TEST_WINDOW_0[t] : TEST_WINDOW_1[t];
        x[0] = pgm_read_float(&row[0]);
        x[1] = pgm_read_float(&row[1]);
        x[2] = pgm_read_float(&row[2]);

        fastgrnn_step(x);
        unsigned long lat = millis() - t0;

        if (lat > max_lat) max_lat = lat;
        sum_lat += lat;
        if (lat > SAMPLE_PERIOD_MS) over_budget++;

        // Periodic emit
        if ((t + 1) % PREDICT_EVERY == 0 || t == WINDOW_LEN - 1) {
            uint8_t pred = fastgrnn_predict();
            const float* h = fastgrnn_get_hidden_state();
            Serial.print(t + 1);
            Serial.print(F(","));
            Serial.print(lat);
            Serial.print(F(","));
            Serial.print(h[0], 2);
            Serial.print(F(","));
            Serial.print(pred);
            Serial.print(F(" "));
            Serial.println(CLASS_NAMES[pred]);
        }

        // Pace to 50 Hz - wait for the next sample period
        while ((millis() - t0) < SAMPLE_PERIOD_MS) {
            // busy wait (more precise than delay())
        }
    }

    unsigned long total_ms = millis() - total_start;
    Serial.println(F("---------------------------------"));
    Serial.print(F("Total: "));
    Serial.print(total_ms);
    Serial.print(F(" ms (expected ~"));
    Serial.print(WINDOW_LEN * SAMPLE_PERIOD_MS);
    Serial.println(F(" ms)"));
    Serial.print(F("Avg sample latency: "));
    Serial.print((float)sum_lat / WINDOW_LEN, 2);
    Serial.println(F(" ms"));
    Serial.print(F("Max sample latency: "));
    Serial.print(max_lat);
    Serial.println(F(" ms"));
    Serial.print(F("Over-budget (>"));
    Serial.print(SAMPLE_PERIOD_MS);
    Serial.print(F("ms): "));
    Serial.print(over_budget);
    Serial.print(F(" / "));
    Serial.println(WINDOW_LEN);
    Serial.print(F("Final prediction: "));
    Serial.print(fastgrnn_predict());
    Serial.print(F(" ("));
    Serial.print(CLASS_NAMES[fastgrnn_predict()]);
    Serial.println(F(")"));
}

// ============================================================================
// LIVE MODE: MPU6050 I2C driver
// ============================================================================

/**
 * setup_mpu6050()
 *
 * Initialize the I2C bus and wake the MPU6050.
 * - I2C speed: 100 kHz (more robust than 400 kHz with long jumper wires)
 * - Accelerometer range: +/-2g (ACCEL_CONFIG = 0x00)
 * - Wake from sleep mode
 *
 * Note: an I2C failure is not deeply diagnosed here - the LED blinks fast
 * and the serial output prints the error.
 */
void setup_mpu6050() {
    Wire.begin();
    Wire.setClock(100000);  // 100 kHz is safer than 400 kHz when cabling is questionable

    // --- I2C bus scan ---
    Serial.println(F("I2C bus scan..."));
    int found = 0;
    for (uint8_t addr = 1; addr < 127; addr++) {
        Wire.beginTransmission(addr);
        if (Wire.endTransmission() == 0) {
            Serial.print(F("  Device at 0x"));
            if (addr < 0x10) Serial.print('0');
            Serial.println(addr, HEX);
            found++;
        }
    }
    if (found == 0) {
        Serial.println(F("  ERROR: no I2C devices on the bus!"));
        Serial.println(F("  Check: VCC->5V, GND->GND, SDA->A4, SCL->A5, AD0->GND"));
        while (1) {
            digitalWrite(LED_PIN, (millis() / 200) % 2);  // fast LED = error
            delay(50);
        }
    }

    // --- MPU6050 init ---
    mpu.initialize();
    mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);

    // --- Skip strict testConnection(), rely on getAcceleration() ---
    Serial.println(F("MPU6050 init OK (sensor responding, skipping testConnection)"));
}


/**
 * read_mpu6050(float* ax, float* ay, float* az)
 *
 * Read acceleration from the MPU6050 over I2C and convert to g.
 * Does NOT normalize - the cell normalizes internally with INPUT_MEAN/INPUT_STD.
 *
 * Freeze detection: jrowberg's library silently returns the previous value
 * after an I2C failure. Ten consecutive bit-exact reads -> reset the bus
 * once. Spamming a re-init never helps.
 */
static int16_t prev_raw[3] = {0, 0, 0};
static uint8_t freeze_count = 0;
static uint16_t total_resets = 0;

void read_mpu6050(float* ax, float* ay, float* az) {
    mpu.getAcceleration(&accel_raw[0], &accel_raw[1], &accel_raw[2]);

    // Bit-exact identical reads = freeze (a real sensor always adds noise)
    bool identical = (accel_raw[0] == prev_raw[0] &&
                      accel_raw[1] == prev_raw[1] &&
                      accel_raw[2] == prev_raw[2]);

    if (identical) {
        freeze_count++;
        // Single re-init attempt after 10 consecutive identical samples
        if (freeze_count == 10) {
            Serial.print(F("\n[I2C freeze @ "));
            Serial.print(millis());
            Serial.println(F(" ms - trying re-init once]"));

            Wire.end();
            delay(10);
            Wire.begin();
            Wire.setClock(100000);
            mpu.initialize();
            mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);

            if (mpu.testConnection()) {
                Serial.println(F("[Re-init OK]"));
                total_resets++;
            } else {
                Serial.println(F("[Re-init FAIL - sensor is dead. Check the wiring.]"));
                Serial.println(F("[Suppressing further spam; reset the Arduino if needed.]"));
            }
        }
        // No spam: do not retry on subsequent frames
    } else {
        freeze_count = 0;
    }
    prev_raw[0] = accel_raw[0];
    prev_raw[1] = accel_raw[1];
    prev_raw[2] = accel_raw[2];

    const float ACCEL_SCALE = 1.0f / 16384.0f;
    *ax = accel_raw[0] * ACCEL_SCALE;
    *ay = accel_raw[1] * ACCEL_SCALE;
    *az = accel_raw[2] * ACCEL_SCALE;
}


/**
 * print_prediction(uint8_t cls)
 *
 * Predicted class + logits (for debugging).
 */
void print_prediction(uint8_t cls) {
    unsigned long t_sec = millis() / 1000;
    Serial.print(F("[t="));
    Serial.print(t_sec);
    Serial.print(F("s] Activity: "));
    Serial.print(CLASS_NAMES[cls]);

    // Logits
    const float* lg = fastgrnn_get_logits();
    Serial.print(F(" | logits=["));
    for (uint8_t c = 0; c < NUM_CLASSES; c++) {
        Serial.print(lg[c], 1);
        if (c < NUM_CLASSES - 1) Serial.print(F(" "));
    }
    Serial.println(F("]"));
}
