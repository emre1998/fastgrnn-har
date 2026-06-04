# Arduino + MPU6050 Kurulum Özeti

## 🎯 Tamamlandı

### 1. Hardware Kablolama (Vizüel)
```
┌─────────────────────────────────────────────────────┐
│                    Arduino Uno R3                   │
│                                                     │
│   5V ─────────┬─────────────────────────┬─ 5V      │
│   GND ────────┤                         ├─ GND      │
│   A5 (SCL) ───┼────────────┐        ┌──┼─ SCL      │
│   A4 (SDA) ───┼───┐        │  ┌─────┤  ├─ SDA      │
│               │   │        │  │     │  │            │
│               │   │ USB    │  │     │  │ I2C        │
│               │   │        │  │     │  │            │
│        ┌──────┴───┴────────┘  │     │  │            │
│        │                      │     │  │            │
│        │  ┌────────────────────┴─────┴──┘            │
│        │  │                                          │
│        └──┼──────────────────────────────────────────┘
│           │
│           └───────────────────────┬─────────────────┐
│                                   │ MPU6050 Module  │
│                              ┌────┴────────┐        │
│                              │ GND SDA SDA │        │
│                              │ VCC SCL CLK │        │
│                              └─────────────┘        │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 2. Yazılım Entegrasyonu
- ✅ I2Cdev + MPU6050 kütüphaneleri included
- ✅ Wire.begin() + I2C 400 kHz init
- ✅ mpu.initialize() + mpu.testConnection()
- ✅ Accelerometer range: ±2g
- ✅ 50Hz polling loop (20 ms periyot)
- ✅ Raw 16-bit → float g dönüştürme
- ✅ FastGRNN streaming inference (128 sample ring buffer)
- ✅ 1 Hz tahmin output

### 3. Kod Dağılımı
```
fastgrnn-har/arduino/fastgrnn_har/
├── fastgrnn_har.ino        ← Main sketch (750 lines)
│   ├── setup() → Wire.begin(), mpu.initialize()
│   ├── loop() → 50Hz polling, fastgrnn_step(), 1Hz predict
│   ├── setup_mpu6050() → I2C init, ±2g range
│   ├── read_mpu6050() → Raw → float g
│   └── print_prediction() → Serial output + logits
├── fastgrnn.h              ← Inference API
├── fastgrnn.cpp            ← Q15 arithmetic
├── model_weights.h         ← Sparse Q15 weights (281 non-zero)
└── test_data.h             ← Embedded test vectors (PROGMEM)
```

---

## ✅ Kontrol Listesi — Test Öncesi

```
DONANIM
- [ ] Arduino Uno R3 hazır
- [ ] MPU6050 sensörü hazır
- [ ] USB kablo (USB-A to USB-B)
- [ ] 4x jumper kablolar
- [ ] Breadboard (isteğe bağlı)
- [ ] Kablolama kontrol edildi:
      - A4 (SDA) ↔ MPU6050 SDA
      - A5 (SCL) ↔ MPU6050 SCL
      - 5V ↔ MPU6050 VCC
      - GND ↔ MPU6050 GND

YAZILIM
- [ ] Arduino IDE 1.8.19+ yüklü
- [ ] I2Cdev kütüphanesi yüklü
      (Sketch → Include Library → Manage Libraries → "I2Cdev")
- [ ] MPU6050 kütüphanesi yüklü
      (Sketch → Include Library → Manage Libraries → "MPU6050")
- [ ] Tüm .h ve .cpp dosyaları aynı dizinde (fastgrnn_har/)
- [ ] fastgrnn_har.ino'da #define TEST_MODE 0 ayarlanmış
- [ ] Arduino IDE → Tools → Board: Arduino Uno
- [ ] Arduino IDE → Tools → Port: COM? (Arduino'ya bağlı port)
- [ ] Sketch → Verify (Ctrl+R) başarılı
      (Beklenen: ~11 KB Flash, ~400 B SRAM)

KURULUM
- [ ] USB kablo Arduino Uno'ya takıldı
- [ ] Arduino IDE → Sketch → Upload (Ctrl+U)
- [ ] "Done uploading" mesajını gözlemle
- [ ] Tools → Serial Monitor açıldı
- [ ] Serial Monitor baud: 115200 seçildi

BAŞLANGIC KONTROLÜ
- [ ] Serial Monitor'da:
      ```
      =============================
       FastGRNN HAR — Arduino Uno
      =============================
      Mode: LIVE (MPU6050 streaming, 50Hz)
      Initializing MPU6050...
      [OK] MPU6050 init OK (ID=0x68)
      ```
- [ ] Sensörü hareketsiz tut → STANDING tahminleri
- [ ] Sensörü yana salla → WALKING tahminleri
- [ ] Tahminler değişiyor → ✓ Başarılı!
```

---

## 🚀 İlk Test Protokolü

### A. Sistem Kontrolü (1 dakika)
```
1. Arduino başlama:
   - Çıktıyı oku (başlantı kontrolü)
   - MPU6050 init mesajını gözlemle

2. Sensör yanıt kontrolü:
   - Sensörü hareketsiz tut → log değerleri sabit olmalı
   - Sensörü salla → log değerleri değişmeli
```

### B. Aktivite Tanıma (2-3 dakika)
```
1. STANDING (Durma — sınıf 4)
   - Sensörü hareketsiz tut, 10 saniye
   - Beklenen: "Activity: STANDING" sık görülmeli
   - Log: logits içinde sınıf 4 en yüksek olmalı

2. WALKING (Yürüme — sınıf 2)
   - Sensörü yana salla (~1-2 Hz), 10 saniye
   - Beklenen: "Activity: WALKING" sık görülmeli
   - Log: sınıf 2 logit en yüksek

3. UPSTAIRS (Merdiven — sınıf 0)
   - Sensörü yukarı-aşağı salla (~1 Hz), 10 saniye
   - Beklenen: "Activity: UPSTAIRS"

4. (İsteğe bağlı) JOGGING (Koşu — sınıf 1)
   - Hızlı salla (~2-3 Hz), 10 saniye
   - Beklenen: "Activity: JOGGING"
```

### C. Latency Ölçümü
```
Serial output format:
[t=5s] Activity: WALKING | logits=[0.1 -5.2 -8.3 -4.1 2.1 -3.0]
[t=6s] Activity: WALKING | logits=[0.2 -4.9 -8.1 -4.0 2.2 -2.9]

Latency = Son [t=Xs] ile mevcut zaman arası

Beklenen: ~1000 ms (1 Hz output = her 50 sample = 50×20ms = 1000ms)
```

---

## 📊 Beklenen Sonuçlar

| Metrik | Min | Expected | Max |
|--------|-----|----------|-----|
| **Accuracy (test seti)** | 85% | 91-92% | 95% |
| **Inference latency (per-sample)** | 0.5 ms | 1-2 ms | 5 ms |
| **Sampling rate** | 45 Hz | 50 Hz | 55 Hz |
| **Prediction rate** | 0.9 Hz | 1.0 Hz | 1.1 Hz |
| **MPU6050 init time** | 50 ms | 100 ms | 200 ms |
| **Flash usage** | 10 KB | 11-12 KB | 15 KB |
| **SRAM working set** | 250 B | 350-400 B | 500 B |

---

## 🔍 Sorun Giderme Hızlı Başvuru

| Sorun | Belirtiler | Çözüm |
|-------|-----------|-------|
| **I2C bağlantı hatası** | "ERROR: MPU6050 not found" | SDA/SCL voltaj kontrol, kablo bağlantısı kontrol |
| **Kütüphane eksik** | "I2Cdev.h not found" | IDE yeniden başlat, kütüphane kurulumunu kontrol |
| **Port seçimi yanlış** | Serial Monitor'da hiç çıktı yok | Tools → Port yeniden seç |
| **Baud yanlış** | Garip karakterler | Serial Monitor baud 115200 seç |
| **Sensör saçmış** | Logit değerleri çok büyük | Sensör hareket ettir, başlatma kontrol et |

---

## 📝 Not Almak

İlk testi yaptıktan sonra rapora eklemek için:

```
Test Tarihi: [DATE]
Arduino Model: Arduino Uno R3 (ATmega328P)
MPU6050 Modüle: [KÜTÜPHANE VERSİYON]

Sonuçlar:
- MPU6050 init: [ms] (beklenen: 100 ms)
- Sampling rate: [Hz] (beklenen: 50 Hz)
- Tahmin hızı: [Hz] (beklenen: 1 Hz)
- STANDING doğruluk: [%] (manuel 10 test)
- WALKING doğruluk: [%]
- UPSTAIRS doğruluk: [%]

Notlar:
- [Gözlemler, sorunlar, iyileştirmeler]
```

---

## 🔗 Referanslar

- [WIRING_MPU6050.md](./WIRING_MPU6050.md) — Detaylı kablolama
- [SETUP_INSTRUCTIONS.md](./SETUP_INSTRUCTIONS.md) — Adım adım kurulum
- [fastgrnn_har.ino](./fastgrnn_har/fastgrnn_har.ino) — Kaynak kodu
- Arduino I2C Wire API: https://www.arduino.cc/en/Reference/Wire
- MPU6050 Datasheet: https://invensense.tdk.com/products/motion-tracking/6-axis/mpu-6050/
- I2Cdev library: https://github.com/jrowberg/i2cdevlib
