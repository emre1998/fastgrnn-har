# MSP430G2553 Deployment — Setup ve Kullanım

Bu klasör, FastGRNN HAR modelinin **MSP430G2553** üzerinde deploy'unu hazırlar.
Manşet iddiası: *"FastGRNN'i 512 byte RAM'li 16-bit MCU'da, donanım çarpıcısı olmadan çalıştırdım."*

---

## 1. Donanım

- **Kart:** TI MSP-EXP430G2 LaunchPad (eski, ucuz, yaygın)
- **Çip:** MSP430G2553 (kart ile gelir)
- **USB:** LaunchPad üstünde onboard emulator var, doğrudan USB ile programlama

### Çip özellikleri (önemli)
| Özellik | Değer |
|---|---|
| Mimari | 16-bit MSP430 RISC |
| Saat | 16 MHz (DCO) |
| Flash | 16 KB |
| SRAM | 512 byte |
| Donanım çarpıcı | **YOK** (yazılım 16×16 multiply, ~100 cycle) |
| FPU | YOK (float software emulation) |
| I2C | USCI_B0 modülü (P1.6=SCL, P1.7=SDA) |

---

## 2. Energia Kurulumu (Önerilen)

**Energia**, Arduino IDE'nin MSP430 için forklanmış halidir. Aynı syntax,
aynı serial monitor, aynı Wire.h. Arduino bildiğin kadarıyla MSP430'da yazabilirsin.

### Adımlar

1. **İndir:** [energia.nu](http://energia.nu) → Windows için Energia 1.8.10E23 indir
2. **Çıkart:** ZIP'i bir klasöre (örn. `C:\Energia`)
3. **Çalıştır:** `energia.exe`
4. **Board:** Tools → Board → "MSP-EXP430G2 w/ MSP430G2553 (16 MHz)"
5. **Port:** Tools → Port → COM portu (USB takıldığında görünür, "MSP Application UART")
   - **USB sürücüsü** otomatik kurulmazsa: TI'nin MSP-EXP430G2 driver paketi (energia ile birlikte gelir, `drivers/` klasöründe)

### Sketch'i aç ve derle
1. File → Open → `msp/fastgrnn_har_msp/fastgrnn_har_msp.ino`
2. Ctrl+R (Verify) — derler
3. Çıktıyı kontrol et:
   ```
   sketch ... uses ~6500 bytes of program memory (40% of 16384 max)
   ... uses ~280 bytes of dynamic memory (54% of 512 max)
   ```
4. Ctrl+U (Upload) — karta yükle

### Serial Monitor
- Ctrl+Shift+M
- Baud: **9600** (MSP430 varsayılan, Arduino'nun 115200'ünden farklı)
- TEST_MODE 1'de 2 test penceresi çalışıp PASS/FAIL yazacak

---

## 3. SRAM Bütçesi (Kritik!)

MSP430G2553'te yalnız **512 byte SRAM** var. Hesabımız:

| Bileşen | SRAM |
|---|---:|
| h_state[16] (float) | 64 byte |
| last_logits[6] (float) | 24 byte |
| fastgrnn_step lokal scratch (xz, xW, hz, hU) | ~168 byte |
| Stack frame + Serial buffer | ~50 byte |
| **Toplam tahmin** | **~310 byte** |
| **Kullanılabilir** | 512 byte |
| **Headroom** | ~200 byte ✓ |

Eğer compile'da "Not enough memory" hatası alırsan optimize ederiz:
- xW ve hU'yu aynı buffer'da tut (sırayla kullan)
- last_logits'i lokal yap (her predict'te yeniden hesaplanır)

## 4. Flash Bütçesi

| Bileşen | Flash |
|---|---:|
| Q15 ağırlıklar (281 nonzero × 2) | 562 byte |
| Test verileri (2 pencere × 128 × 3 × 4) | 3072 byte |
| Inference kodu (yaklaşık) | ~2500 byte |
| Energia runtime + Serial | ~3000 byte |
| **Toplam tahmin** | **~9 KB / 16 KB** = %56 |

Rahat sığar. Test verisi en büyük kullanıcı — sensör gelince çıkarırız, %30'a düşer.

## 5. MPU6050 Bağlantısı (1 Haziran sonrası)

MSP430G2553 I2C pinleri:
- **P1.6 → SCL** (MPU6050 SCL)
- **P1.7 → SDA** (MPU6050 SDA)
- VCC → 3.3V (LaunchPad J6 header)
- GND → GND

**ÖNEMLİ:** P1.6 bazı LaunchPad varyantlarında kart üstündeki LED'e de bağlıdır.
I2C kullanırken **P1.6/LED jumperini çıkar**. Eski MSP-EXP430G2 kartlarında bu
bağlantı J5 üzerindedir; MSP-EXP430G2ET kartında baskı devre üzerindeki P1.6/LED
etiketini izle.

GY-521 kartındaki pull-up dirençleri yoksa veya yetersizse SCL ve SDA hatlarının
her birinden 3.3V hattına birer 4.7 kΩ direnç bağla.

Energia'da `Wire.h` MSP430 için çalışır:
```cpp
#include <Wire.h>
Wire.begin();  // USCI_B0 master olarak başlat
```

CCS bare-metal canlı sürüm, kart üstündeki USCI kilitlenmesine karşı GPIO tabanlı
I2C kullanır. Fiziksel pinler yine P1.6=SCL ve P1.7=SDA'dır.

## 6. Performans Beklentisi

Donanım çarpıcı olmadığı için her float multiply ~500-1000 cycle (yazılım).
Tahmini inference süresi:
- **Streaming 1 sample:** ~5-10 ms (Arduino'nun ~1-2 ms'i)
- **Full window (128 sample):** ~700-1200 ms (Arduino'nun ~150-250 ms'i)
- **50Hz örnekleme periyodu:** 20 ms → streaming **rahatça yetişir**

Eğer optimizasyon gerekirse: pure Q15 integer inference (no float) ~10× hızlandırma getirir. Ama önce float ile çalıştığını doğrulayalım.

## 7. Test Akışı

1. Energia'yı kur
2. Board ve port seç
3. `fastgrnn_har_msp.ino` derle (~30 sn)
4. Karta yükle (~10 sn)
5. Serial Monitor 9600 baud
6. İki test penceresi çıktısı:
   - Tahmin sınıfı
   - Python'la eşit mi?
   - Inference süresi (ms)
7. Eğer her ikisi de PASS → MSP430'da çalışıyor demek

Arduino sürümüyle aynı test verilerini kullanıyoruz, bu nedenle:
- **PASS** = MSP430 ve Arduino aynı tahminleri üretiyor
- **FAIL** = Float math, derleme veya bellek sorunu

## Sorun Giderme

| Sorun | Çözüm |
|---|---|
| Energia "board not found" | TI MSP430 USB drivers manuel kur |
| Compile hatası "out of memory" | Stack size azalt veya buffer'ları paylaştır |
| Serial garbage karakter | Baud 9600 olduğundan emin ol, COM port doğru |
| Hiç çıktı yok | Reset düğmesine bas (S1), Serial Monitor'u yeniden aç |
| Inference süresi >2000 ms | Beklenen MSP430 yavaş; Q15 integer için optimize edilir |

## Sonraki Adım

Senin **iki** plug-and-play projen var:
1. `arduino/fastgrnn_har/` — Arduino Uno (R3 clone)
2. `msp/fastgrnn_har_msp/` — MSP-EXP430G2 (MSP430G2553)

İkisini de senin elinde derleyip yükleyebilirsin. MPU6050 sensörü gelince
her ikisine de I2C sürücüsü ekleriz, **iki donanımda canlı demo** verirsin.
