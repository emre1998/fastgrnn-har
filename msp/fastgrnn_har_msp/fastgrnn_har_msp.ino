/*
 * fastgrnn_har_msp.ino - MSP430G2553 (LaunchPad) HAR demo
 *
 * Target: TI MSP-EXP430G2 LaunchPad + MSP430G2553
 *   - 16-bit RISC, 16 MHz
 *   - 16 KB Flash, 512 B SRAM
 *   - NO HARDWARE MULTIPLIER (software 16x16 multiply)
 *   - Toolchain: Energia 1.8.x (energia.nu)
 *
 * Mode 1 (TEST_MODE 1): inference on embedded test data, no MPU6050 needed
 * Mode 0 (TEST_MODE 0): live MPU6050 streaming
 *
 * UART: 9600 baud (MSP430 USCI_A0, P1.1/P1.2)
 *
 * SRAM warning: only ~512 B available; the model state + scratch use ~300 B.
 * The embedded test data (3 KB float) lives in Flash (const), not SRAM.
 */

#include "fastgrnn.h"
#include "model_weights.h"
#include "test_data.h"
#include <Wire.h>

#define TEST_MODE 0

// MSP-EXP430G2 LaunchPad LEDs:
//   P1.0 = RED_LED, P1.6 = GREEN_LED (Energia constants)
// P1.6 is also USCI_B0 SCL - if you use I2C you MUST remove the J5 jumper.
#define LED_PIN RED_LED

#define SAMPLE_RATE_HZ 50
#define SAMPLE_PERIOD_MS (1000 / SAMPLE_RATE_HZ)

static uint16_t sample_count = 0;
static unsigned long last_predict_ms = 0;
static float last_ax = 0.0f;
static float last_ay = 0.0f;
static float last_az = 0.0f;

void setup() {
    Serial.begin(9600);            // MSP430 default UART speed
    pinMode(LED_PIN, OUTPUT);
    delay(500);

    Serial.println("=================================");
    Serial.println(" FastGRNN HAR - MSP430G2553");
    Serial.println("=================================");
    Serial.print("Mode: ");
#if TEST_MODE
    Serial.println("TEST (embedded data)");
    run_embedded_tests();
#else
    Serial.println("LIVE (MPU6050 streaming)");
    setup_mpu6050();
    fastgrnn_reset();
#endif
}

void loop() {
#if TEST_MODE
    digitalWrite(LED_PIN, (millis() / 500) % 2);
    delay(100);
#else
    unsigned long now = millis();
    if (now - last_predict_ms >= SAMPLE_PERIOD_MS) {
        last_predict_ms = now;
        float ax, ay, az;
        read_mpu6050(&ax, &ay, &az);
        last_ax = ax;
        last_ay = ay;
        last_az = az;
        float x[3] = { ax, ay, az };
        fastgrnn_step(x);
        sample_count++;
        if (sample_count >= SAMPLE_RATE_HZ) {
            sample_count = 0;
            uint8_t cls = fastgrnn_predict();
            print_prediction(cls);
        }
    }
#endif
}

void run_embedded_tests() {
    Serial.println();
    Serial.print("Embedded test windows: ");
    Serial.println(N_TEST_SAMPLES);
    Serial.println();

    for (uint8_t s = 0; s < N_TEST_SAMPLES; s++) {
        Serial.print("--- Test ");
        Serial.print(s);
        Serial.println(" ---");

        fastgrnn_reset();
        unsigned long t0 = millis();

        for (uint16_t t = 0; t < WINDOW_LEN; t++) {
            float x[3];
            // MSP430: const data already lives in Flash; no PROGMEM read needed.
            const float* row;
            if (s == 0) row = TEST_WINDOW_0[t];
            else        row = TEST_WINDOW_1[t];
            x[0] = row[0]; x[1] = row[1]; x[2] = row[2];
            fastgrnn_step(x);
        }
        uint8_t pred = fastgrnn_predict();
        unsigned long elapsed_ms = millis() - t0;

        Serial.print("Prediction: ");
        Serial.print(pred);
        Serial.print(" (");
        Serial.print(CLASS_NAMES[pred]);
        Serial.println(")");

        Serial.print("Expected (Python): ");
        Serial.print(TEST_EXPECTED[s]);
        Serial.print(" (");
        Serial.print(CLASS_NAMES[TEST_EXPECTED[s]]);
        Serial.println(")");

        Serial.print("Ground truth:      ");
        Serial.print(TEST_TRUE[s]);
        Serial.print(" (");
        Serial.print(CLASS_NAMES[TEST_TRUE[s]]);
        Serial.println(")");

        Serial.print("Inference time: ");
        Serial.print(elapsed_ms);
        Serial.println(" ms");

        if (pred == TEST_EXPECTED[s]) {
            Serial.println("[PASS] matches Python");
        } else {
            Serial.println("[FAIL] differs from Python");
        }

        const float* lg = fastgrnn_get_logits();
        Serial.print("Logits: ");
        for (uint8_t c = 0; c < NUM_CLASSES; c++) {
            Serial.print(lg[c], 3);
            Serial.print(" ");
        }
        Serial.println();
        Serial.println();
    }

    Serial.println("Tests complete.");
}

// ============================================================================
// LIVE MODE (MPU6050 sensor)
// MSP430 I2C: USCI_B0 module, P1.6 = SCL, P1.7 = SDA
// Wire.h ships with Energia and targets MSP430 correctly.
// ============================================================================
void setup_mpu6050() {
    Serial.println(F("Initializing MPU6050..."));
    Wire.begin();

    Serial.println(F("I2C bus scan..."));
    bool found = false;
    Wire.beginTransmission(0x68);
    if (Wire.endTransmission() == 0) {
        Serial.println(F("  Device at 0x68"));
        found = true;
    }
    if (!found) {
        Serial.println(F("  ERROR: MPU6050 not found!"));
        Serial.println(F("  Check: J5 jumper removed? P1.6=SCL, P1.7=SDA, VCC=3.3V/5V, GND=GND, AD0=GND"));
        while (1) {
            digitalWrite(LED_PIN, (millis() / 200) % 2);
            delay(100);
        }
    }

    Wire.beginTransmission(0x68);
    Wire.write(0x6B);  // PWR_MGMT_1
    Wire.write(0x00);  // wake up
    Wire.endTransmission();

    Wire.beginTransmission(0x68);
    Wire.write(0x1C);  // ACCEL_CONFIG
    Wire.write(0x00);  // +/-2g
    Wire.endTransmission();

    Serial.println(F("MPU6050 init OK"));
    Serial.println(F("time_s,ax,ay,az,class,logit0,logit1,logit2,logit3,logit4,logit5"));
}

void read_mpu6050(float* ax, float* ay, float* az) {
    const float ACCEL_SCALE = 1.0f / 16384.0f;

    Wire.beginTransmission(0x68);
    Wire.write(0x3B);  // ACCEL_XOUT_H
    if (Wire.endTransmission(false) != 0) {
        Serial.println(F("[I2C ERROR] read_mpu6050 beginTransmission fail"));
        *ax = *ay = *az = 0.0f;
        return;
    }

    Wire.requestFrom(0x68, 6);
    if (Wire.available() < 6) {
        Serial.println(F("[I2C ERROR] accel bytes missing"));
        *ax = *ay = *az = 0.0f;
        return;
    }

    int16_t raw_x = (Wire.read() << 8) | Wire.read();
    int16_t raw_y = (Wire.read() << 8) | Wire.read();
    int16_t raw_z = (Wire.read() << 8) | Wire.read();

    *ax = raw_x * ACCEL_SCALE;
    *ay = raw_y * ACCEL_SCALE;
    *az = raw_z * ACCEL_SCALE;
}

void print_prediction(uint8_t cls) {
    unsigned long t_sec = millis() / 1000;
    Serial.print(F("[t="));
    Serial.print(t_sec);
    Serial.print(F("s] Activity: "));
    Serial.println(CLASS_NAMES[cls]);

    const float* lg = fastgrnn_get_logits();
    Serial.print(t_sec);
    Serial.print(",");
    Serial.print(last_ax, 4);
    Serial.print(",");
    Serial.print(last_ay, 4);
    Serial.print(",");
    Serial.print(last_az, 4);
    Serial.print(",");
    Serial.print(CLASS_NAMES[cls]);
    Serial.print(",");
    for (uint8_t c = 0; c < NUM_CLASSES; c++) {
        Serial.print(lg[c], 2);
        if (c < NUM_CLASSES - 1) Serial.print(F(","));
    }
    Serial.println();
}
