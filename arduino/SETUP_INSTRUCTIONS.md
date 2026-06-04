# Arduino Uno + MPU6050 — Kurulum ve Test Talimatları

## 1. Donanım Kurulumu

### Gerekli Komponentler
- Arduino Uno R3 (ATmega328P, 16 MHz)
- MPU6050 6-axis IMU sensörü (I2C modülü)
- USB kablo (USB-A to USB-B, programlama için)
- 4x jumper kablolar (I2C: SDA, SCL, VCC, GND)

### Kablolama

| Arduino Uno | MPU6050 |
|-------------|---------|
| 5V          | VCC     |
| GND         | GND     |
| A4 (SDA)    | SDA     |
| A5 (SCL)    | SCL     |

Detaylı diyagram için bkz: [WIRING_MPU6050.md](./WIRING_MPU6050.md)

---

## 2. Yazılım Kurulumu

### 2.1 Arduino IDE

1. **Arduino IDE 1.8.19 veya daha yeni indir**
   - https://www.arduino.cc/en/software
   - Windows installer'ı indirip çalıştır

2. **Tahtayı seç**
   - Tools → Board: Arduino AVR Boards → **Arduino Uno**
   - Tools → Port: **COM?** (Arduino'nun bağlı olduğu port)

### 2.2 Gerekli Kütüphaneleri Yükle

Arduino IDE'de:
1. Sketch → Include Library → Manage Libraries (Ctrl+Shift+I)
2. Aşağıdaki kütüphaneleri ara ve yükle:
   - **I2Cdev** (Jeff Rowberg)
   - **MPU6050** (Jeff Rowberg)

Alternatif olarak GitHub'dan manuel yükle:
- https://github.com/jrowberg/i2cdevlib/tree/master/Arduino/I2Cdev
- https://github.com/jrowberg/i2cdevlib/tree/master/Arduino/MPU6050

İndirdikten sonra:
1. ZIP'i çıkart
2. Klasörü `Documents\Arduino\libraries\` altına taşı
3. Arduino IDE'yi yeniden başlat

---

## 3. Kod Hazırlığı

### 3.1 Gerekli Dosyaları Kopyala

Arduino IDE'de yeni bir sketch aç veya mevcut dosyaları kullan:

```
fastgrnn-har/
├── arduino/
│   └── fastgrnn_har/
│       ├── fastgrnn_har.ino       ← Main sketch
│       ├── fastgrnn.h             ← Inference engine
│       ├── fastgrnn.cpp           ← Inference impl
│       ├── model_weights.h         ← Q15 weights
│       └── test_data.h             ← Embedded test vectors
```

Tüm `.h` ve `.cpp` dosyaları fastgrnn_har dizininde olmalı.

### 3.2 TEST_MODE Seçimi

fastgrnn_har.ino dosyasında:
```cpp
#define TEST_MODE 0  // 0 = LIVE (MPU6050), 1 = TEST (embedded), 2 = STREAM
```

- **TEST_MODE = 0**: MPU6050'den canlı okuma (bu kez aktif)
- **TEST_MODE = 1**: Gömülü test verisi (sensör olmadan test etmek için)
- **TEST_MODE = 2**: Gömülü test verisi 50Hz benzetim (latency ölçmek için)

### 3.3 Kod Kontrol

Arduino IDE'de **Sketch → Verify (Ctrl+R)** ile compile et:
- "Sketch uses 11,XXX bytes of Flash"
- "Global variables use XXX bytes of SRAM"
- Hata yok mu? OK.

---

## 4. Upload ve İlk Test

### 4.1 Arduino Uno'ya Yükle

1. USB kablı Arduino'ya bağla
2. Arduino IDE: Sketch → Upload (Ctrl+U)
3. Şöyle görülmeli:
   ```
   Compiling sketch...
   Uploading...
   Done uploading.
   ```

### 4.2 Serial Monitor

1. Tools → Serial Monitor (Ctrl+Shift+M)
2. Sağ altta: **115200 baud** seç
3. Arduino Uno LED'i kısa ışıl + şöyle çıktı görülmeli:

```
=============================
 FastGRNN HAR — Arduino Uno
=============================
Mode: LIVE (MPU6050 streaming, 50Hz)
Initializing MPU6050...
[OK] MPU6050 init OK (ID=0x68)

[t=0s] Activity: STANDING | logits=[0.1 -5.2 -8.3 -4.1 2.1 -3.0]
[t=1s] Activity: STANDING | logits=[0.2 -4.9 -8.1 -4.0 2.2 -2.9]
[t=2s] Activity: WALKING  | logits=[-2.1 3.5 -1.2 1.4 8.1 -1.5]
...
```

### 4.3 Hareketleri Test Et

1. **STANDING** (Durma)
   - Sensörü hareketsiz tut
   - Beklenen output: Sınıf 4 (STANDING) sık sık çıkmalı

2. **WALKING** (Yürüme)
   - Sensörü yana salla (perde itmek gibi, ~1-2 Hz)
   - Beklenen output: Sınıf 2 (WALKING) çıkmalı

3. **UPSTAIRS** (Merdiven çıkma)
   - Sensörü dikey eksende (yukarı-aşağı) salla (~1 Hz, daha hızlı)
   - Beklenen output: Sınıf 0 (UPSTAIRS)

4. **DOWNSTAIRS** (Merdiven inme)
   - Upstairs'e benzer ama şekil/timing değişir
   - Beklenen output: Sınıf 2 (DOWNSTAIRS)

5. **SITTING** (Oturma)
   - Sensörü düz tut ve hareketli tut
   - Beklenen output: Sınıf 3 (SITTING)

6. **JOGGING** (Koşu)
   - Hızlı perde itme (~2-3 Hz)
   - Beklenen output: Sınıf 1 (JOGGING)

---

## 5. Sorun Giderme

| Sorun | Çözüm |
|-------|-------|
| "I2Cdev.h: No such file" | Kütüphaneleri yükle (bkz 2.2) |
| "MPU6050 not found" | Kütüphaneleri yükle ve IDE yeniden başlat |
| "Arduino port göz vermez" | USB kablı değiş, Arduino IDE restart, Driver yükle |
| Serial'de çıktı yok | Baud 115200 mu? USB bağlı mı? COM port doğru mu? |
| MPU6050 bağlantı hatası | I2C kablolama (SDA/SCL) kontrol et, voltaj kontrol et |
| Sabit sınıf değeri (sürekli aynı) | Model sorun yok, sensor hamming iyi diş değiş |
| Garip / saçma tahminler | Sensör hareketi tut, hareket şeklini değiştir |

---

## 6. Performans Metrikleri

### Beklenen Sonuçlar

| Metrik | Beklenen Değer |
|--------|---|
| Flash Kullanım | ~35% (11-12 KB) |
| SRAM Kullanım | ~17% (300-400 B) |
| I2C Hızı | 400 kHz |
| Sampling Hızı | 50 Hz (20 ms periyot) |
| Tahminin Latency | ~1-3 ms (per-sample) |
| Çıkış Hızı | 1 Hz (her 50 sample'da tahmin) |
| Doğruluk (Eğitim Seti) | ~91-92% (test seti) |

### Ölçümler

Serial output'ta her satırın başında `[t=Xs]` var — başlangıçtan geçen saniye.
Zamanlama manuel ölç:
```
[t=5s] Activity: WALKING
[t=6s] Activity: WALKING
[t=7s] Activity: SITTING
```
= 3 tahmin = 3 saniye → 1 tahmin/saniye ✓

---

## 7. Araştırma Notları

### Sensör Verisi Üflemeleri

- MPU6050'nin ±2g aralığında ölçüm hassasiyet: 1 LSB = 61 μg
- Gravity default = 9.81 m/s² (Arduino float precision: ~7 decimal)
- I2C noise direnç var ise low-pass filtre eklenebilir (software)

### Tahmin Güvenliği (Logits)

- 6 çıkışlı softmax model
- Logitler scale'i: genellikle -20 ile +10 arasında
- confidence = exp(best_logit) / sum(exp(all_logits)) (softmax)
- threshold setlemek mümkün (ör: confidence < 0.5 ise "uncertain")

---

## 8. Sonraki Adımlar

1. **MSP430 versiyonu** (msp/ klasöründe, Energia ile)
2. **Veri toplaması** (gerçek aktivite verileri kaydı)
3. **Model iyileştirme** (yeni train setler)
4. **Deployed optimizasyon** (daha düşük latency, daha az SRAM)
