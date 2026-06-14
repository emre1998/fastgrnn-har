/*
 * Minimal I2C scanner for debugging INA226 wiring.
 *
 * Expected INA226 address is usually 0x40. If this sketch prints another
 * address, update INA226_ADDR in arduino/ina226_meter/ina226_meter.ino.
 */

#include <Wire.h>

void setup() {
    Serial.begin(115200);
    while (!Serial) {
    }

    Wire.begin();
    Serial.println(F("I2C scanner starting..."));
}

void loop() {
    byte count = 0;

    Serial.println(F("Scanning I2C bus..."));
    for (byte address = 1; address < 127; address++) {
        Wire.beginTransmission(address);
        byte error = Wire.endTransmission();

        if (error == 0) {
            Serial.print(F("Found I2C device at 0x"));
            if (address < 16) {
                Serial.print('0');
            }
            Serial.println(address, HEX);
            count++;
        } else if (error == 4) {
            Serial.print(F("Unknown error at 0x"));
            if (address < 16) {
                Serial.print('0');
            }
            Serial.println(address, HEX);
        }
    }

    if (count == 0) {
        Serial.println(F("No I2C devices found."));
    } else {
        Serial.print(F("Found "));
        Serial.print(count);
        Serial.println(F(" device(s)."));
    }

    Serial.println();
    delay(3000);
}
