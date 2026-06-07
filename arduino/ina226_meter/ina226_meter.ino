/*
 * ina226_meter.ino - INA226 host reader for the FastGRNN-HAR energy
 *                    measurement protocol.
 *
 * Hardware:
 *   Arduino Uno (or compatible)
 *   INA226 breakout (e.g. DirencNet stock #10793, R100 = 0.1 ohm shunt)
 *
 * Wiring (INA226 -> Arduino):
 *   VCC  -> Arduino 5V         (the breakout has its own 3.3V LDO; 5V is fine)
 *   GND  -> Arduino GND
 *   SCL  -> Arduino A5
 *   SDA  -> Arduino A4
 *   ALE  -> (not connected)
 *   IN+  -> USB +5V from the host port (power source)
 *   IN-  -> VCC of the Device Under Test (Arduino #2 or MSP430 LaunchPad)
 *   VBS  -> (optionally tie to IN- to monitor bus voltage at the load)
 *
 * The 0.1 ohm shunt on the breakout sits between IN+ and IN-, so the DUT's
 * supply current must flow through the shunt to reach its VCC.
 *
 * Output:
 *   Streams CSV to Serial at 10 Hz:   t_ms, mA, mV, mW
 *   Open the Serial Monitor at 115200 baud, leave it open for 60 s, and
 *   read the steady-state current after the moving average has settled.
 *
 * Library:
 *   Arduino IDE -> Tools -> Manage Libraries -> "INA226_WE" (by Wolfgang Ewald)
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <Wire.h>
#include <INA226_WE.h>

// I2C address - INA226 breakouts default to 0x40 with A0/A1 tied to GND
#define INA226_ADDR 0x40

INA226_WE ina226 = INA226_WE(INA226_ADDR);

// Heartbeat LED so we can see the reader is alive even with Serial Monitor closed
#define LED_PIN 13

void setup() {
    Serial.begin(115200);
    pinMode(LED_PIN, OUTPUT);

    Wire.begin();

    if (!ina226.init()) {
        Serial.println(F("[ERROR] INA226 not found. Check wiring."));
        while (1) {
            digitalWrite(LED_PIN, (millis() / 100) % 2);  // fast blink = error
        }
    }

    // --- INA226 configuration -----------------------------------------------
    //   Average:           16-sample moving average (datasheet table 4)
    //   Conv. time bus/sh: 1.1 ms each (datasheet table 5)
    //   Total per reading: 16 * 2 * 1.1 ms = 35.2 ms
    //   Effective rate:    ~28 Hz; we throttle to 10 Hz for human-readable CSV
    //
    //   Shunt R = 0.1 ohm (DirencNet "R100" on the breakout)
    //   Max expected current = 0.8 A  (full envelope; we stay <100 mA in practice)
    ina226.setAverage(AVERAGE_16);
    ina226.setConversionTime(CONV_TIME_1100);
    ina226.setMeasureMode(CONTINUOUS);
    ina226.setResistorRange(0.1, 0.8);

    Serial.println(F("INA226 ready."));
    Serial.println(F("CSV output: t_ms,mA,mV,mW"));
    Serial.println(F("(Wait ~60 s after a workload change for the average to settle.)"));
}

void loop() {
    static unsigned long last_blink = 0;
    static bool led_state = false;

    // 10 Hz heartbeat (visual confirmation the reader is alive)
    if (millis() - last_blink >= 500) {
        last_blink = millis();
        led_state = !led_state;
        digitalWrite(LED_PIN, led_state);
    }

    // INA226 sample
    ina226.readAndClearFlags();
    float mA = ina226.getCurrent_mA();
    float mV = ina226.getBusVoltage_V() * 1000.0f;
    float mW = ina226.getBusPower();

    unsigned long t = millis();
    Serial.print(t);
    Serial.print(',');
    Serial.print(mA, 3);  // mA with three decimals -> ~1 uA resolution on screen
    Serial.print(',');
    Serial.print(mV, 1);
    Serial.print(',');
    Serial.println(mW, 2);

    delay(100);  // 10 Hz CSV stream
}
