# Arduino Uno + MPU6050 Kablolama

## Donanım
- **Arduino Uno R3** (ATmega328P)
- **MPU6050** 6-axis IMU (3-axis accelerometer + 3-axis gyroscope)
- **USB kablo** (programlama + seri haberleşme)
- **I2C kablolar** (jumper wires: 4x)

---

## Pin Harası

### Arduino Uno → MPU6050 (I2C)

```
Arduino Uno (ATmega328P)              MPU6050 Module
============================================
A4 (SDA) ──────────────────────────── SDA
A5 (SCL) ──────────────────────────── SCL
GND ───────────────────────────────── GND
5V ────────────────────────────────── VCC

İsteğe bağlı (interrupt, henüz kullanılmıyor):
D2 (INT0) ───────────────────────── INT
```

### Kablolama Diyagramı

```
┌─────────────────────────────────────────────────────┐
│                    Arduino Uno R3                   │
│  (ATmega328P, 16MHz, 32KB Flash, 2KB SRAM)        │
│                                                     │
│   5V ──────┐                               ┌─────── 5V
│   GND ─────┤                               ├─────── GND
│            │                               │
│   A5 (SCL) ┼───────┐              ┌───────┼─ SCL
│   A4 (SDA) ┼───────┼──┐      ┌────┼───────┼─ SDA
│   D2  (INT)┼───────┼──┼──┐   │    │       │
│            │       │  │  │   │    │       │
│            │       │  │  │  GND  VCC      │
│            │       │  │  │   │    │       │
│            │       │  │  └─┬─────┘       │
│            │       │  │    │             │
│            └───┬───┴──┼────┴─────────────┘
│                │      │
│                │      └────────────────────────────┐
│                │                                   │
│                └──────────────────────────┐        │
│                                           │        │
│        ┌─────────────────────────────────┴────┐   │
│        │                                      │   │
│     ┌──┴──┬─────┬─────┬─────┬─────┬──────────┴──┐│
│     │ GND │ SCL │ SDA │ EDA │ CLK │ INT  │ VCC  ││
│     └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬───┴──┬───┘│
│        │     │     │     │     │     │      │    │
│     ┌──┴─────┴─────┴─────┴────┐│    │      │    │
│     │    MPU6050 Module        ││    │      │    │
│     │ (3-axis accel + gyro)   ││    │      │    │
│     │ Sensor: ICM-20602 chip  ││    │      │    │
│     └───────────────────────────┘│    │      │    │
│                                  └────┴──────┴───┘
│                                     (Pinout)
└─────────────────────────────────────────────────────┘
```

---

## MPU6050 Pinleri Detaylı

| Pin | Adı   | Açıklama                    | Arduino Uno |
|-----|-------|-----------------------------|-----------  |
| 1   | VCC   | +3.3V güç (veya 5V tolerant)| 5V          |
| 2   | GND   | Toprak                      | GND         |
| 3   | SDA   | I2C Veri Hattı              | A4 (SDA)    |
| 4   | SCL   | I2C Saat Hattı              | A5 (SCL)    |
| 5   | EDA   | I2C yardımcı (pin 7 ile tümleyen) | bağlı değil |
| 6   | CLK   | Harici saat (isteğe bağlı)  | bağlı değil |
| 7   | INT   | Interrupt çıkışı (DMP)      | (isteğe bağlı: D2) |

---

## I2C Harita

- **I2C Adresi**: `0x68` (varsayılan, AD0=GND)
- **I2C Hızı**: 400 kHz (standard I2C, Arduino `Wire` kütüphanesi ile)

### Önemli Registerler

| Register | Adres | Açıklama          |
|----------|-------|---  --------------|
| PWR_MGMT_1 | 0x6B | Power management, wake-up |
| ACCEL_CONFIG | 0x1C | Accelerometer range (±2g, ±4g, ±8g, ±16g) |
| ACCEL_XOUT_H | 0x3B | X-axis accel (raw, 16-bit) |
| ACCEL_YOUT_H | 0x3D | Y-axis accel (raw) |
| ACCEL_ZOUT_H | 0x3F | Z-axis accel (raw) |

---

## Yazılım Gereksinimleri

### Arduino IDE Kütüphaneleri

**Sketch → Include Library → Manage Libraries** (Ctrl+Shift+I) açarak:

1. **I2Cdev** (Jeffrey Rowberg)
   - Aynı dizinine kurun: `C:\Users\<user>\Documents\Arduino\libraries\I2Cdev\`
   - GitHub: https://github.com/jrowberg/i2cdevlib

2. **MPU6050** (Jeff Rowberg)
   - Aynı dizinine: `C:\Users\<user>\Documents\Arduino\libraries\MPU6050\`
   - GitHub: https://github.com/jrowberg/MPU6050

**Veya**: GitHub'dan ZIP indir → Arduino IDE'de Sketch → Include Library → Add .ZIP Library

### Kodda Include Etme

```cpp
#include "I2Cdev.h"
#include "MPU6050.h"
#include <Wire.h>

MPU6050 mpu;

void setup() {
    Wire.begin();
    mpu.initialize();
    // ... chip check + range setup
}

void loop() {
    int16_t ax_raw, ay_raw, az_raw;
    mpu.getAcceleration(&ax_raw, &ay_raw, &az_raw);
    
    // ±2g range için: 1 LSB = 2g / 32768 = 61.04 μg
    // Dönüştürme: float_value = raw_value * 2.0 / 32768.0
}
```

---

## Kontrol Listesi

- [ ] MPU6050 modülü alındı
- [ ] Arduino IDE 1.8.19+ yüklü
- [ ] I2Cdev + MPU6050 kütüphaneleri kuruldu
- [ ] USB kablo Arduino'ya bağlı
- [ ] Kablolama yapıldı (SDA, SCL, VCC, GND)
- [ ] Kodu compile ettim ve yükledim (TEST_MODE=0)
- [ ] Serial Monitor'u 115200 baud'da açtım
- [ ] Cihaz çalışıyor (sensör değerleri geliyor)

---

## Sorun Giderme

| Sorun | Çözüm |
|-------|-------|
| "I2Cdev.h not found" | I2Cdev kütüphanesini kur, klasör adını kontrol et |
| "MPU6050 not found" | MPU6050 kütüphanesini kur |
| "Wire.h not found" | Arduino IDE versiyonunu güncelle |
| Serial'de veri yok | USB baud 115200 kontrolü, kablo bağlantısı kontrol et |
| MPU6050 yanıt vermiyor | I2C bağlantısı, voltaj (3.3V/5V), AD0 pin seviyesi kontrol et |
| Sensor değerleri sabit/saçma | Sensor başlatması başarısız, PWR_MGMT ve range ayarları kontrol et |

---

## İlk Test Adımları

1. **Kütüphaneleri yükle**
2. **Kodu Arduino IDE'ye paste et** (aşağıya bakınız)
3. **Tools → Board: Arduino Uno, Port: COM?**
4. **Upload et**
5. **Serial Monitor (115200) açarak çıktıyı gözle**
6. **Sensörü hareket ettir (döndür, salla, eğ)**

Çıktı şöyle görünmeli:
```
=============================
 FastGRNN HAR — Arduino Uno
=============================
Mode: LIVE (MPU6050 streaming)
[OK] MPU6050 initialized (ID: 0x68)
[OK] Sampling at 50 Hz...

[t=0s] Accel: 0.10g, -0.05g, 0.99g
[t=1s] Accel: 0.12g, -0.03g, 0.98g
[t=1s] Activity: STANDING

[t=2s] Accel: 0.30g, 0.20g, 0.85g
[t=3s] Activity: WALKING
...
```
