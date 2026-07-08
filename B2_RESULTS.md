# B2 — Donanım Ölçüm Sonuçları
*Latency + enerji + bellek, GRU/LSTM × MSP430/Arduino. FastGRNN referansı için: docs/energy_measurement.md.*

## Latency (TEST_MODE 0, µs)
| Hücre | Platform | LUT | per-step µs | window µs | %util @50Hz | real-time? |
|-------|----------|-----|-------------|-----------|-------------|------------|
| FastGRNN | MSP430 | 1 | 13900 | 1778000 | 69.5 | ✅ (ref) |
| GRU  | MSP430 | 1 | 12116 | 1550900 | 60.6 | ✅ OK |
| GRU  | MSP430 | 0 | 19273 | 2467000 | 96.4 | ✅ OK (marginal) |
| LSTM | MSP430 | 1 | 12370 | 1583400 | 61.9 | ✅ OK |
| LSTM | MSP430 | 0 | 22097 | 2828400 | 110.5 | ❌ FAIL |
| GRU  | Arduino| 1 | ? | ? | ? | ? |
| LSTM | Arduino| 1 | ? | ? | ? | ? |

## Enerji (TEST_MODE 3, INA226, steady-state)
*NOT: Önceki tur bir Debug/Flash karışıklığı yüzünden şüpheli bulundu ve TAMAMEN SIFIRLANDI (6 Tem 2026).
Yeni prosedür: her #define değişikliğinde Save → Project Clean → Build → Debug ile çalıştır (hep aynı yöntem, tutarlılık için).
Sıra: önce LSTM (baştan), sonra GRU (baştan).*
| Hücre | Platform | BENCH | LUT | I (mA) | V (mV) | P (mW) |
|-------|----------|-------|-----|--------|--------|--------|
| (idle platform-bağımsız — MSP430 idle <0.025 mA / <0.09 mW, referans FastGRNN'den) |
| LSTM | MSP430 | 1 (50Hz) | 1 | 5.058 | 3479.2 | 17.70 |
| LSTM | MSP430 | 2 (cont) | 1 | 5.077 | 3478.9 | 17.70 |
| LSTM | MSP430 | 2 (cont) | 0 | 5.078 | 3478.9 | 17.70 |
| GRU  | MSP430 | 1 (50Hz) | 1 | 5.066 | 3478.9 | 17.65 |
| GRU  | MSP430 | 2 (cont) | 1 | 5.054 | 3478.9 | 17.70 |
| GRU  | MSP430 | 2 (cont) | 0 | 5.054 | 3478.8 | 17.70 |

## Bellek — TEST HARNESS build (referans, test kapsamı dahil: UART banner + sprint_* + 10x timing tekrarı)
*Tutarlı ölçüm: üçü de AYNI ayarda — TEST_MODE=1, BENCH_MODE=1, USE_LUT=1 (9 Tem 2026).
Bu sayılar GERÇEK deployment'ı DEĞİL, test/ölçüm kodunu da içeren build'i yansıtır — aşağıdaki
"PRODUCTION" tablosu asıl deployment footprint'i, makalede o kullanılacak.*
| Hücre | Platform | Flash (B) | SRAM (B) |
|-------|----------|-----------|----------|
| FastGRNN | MSP430 | 10214 | 348 |
| GRU  | MSP430 | 6546 | 308 |
| LSTM | MSP430 | 6908 | 324 |

## Bellek — PRODUCTION build (ASIL deployment footprint, UART/debug/tekrar YOK)
*msp/ccs_{cell}_production/ — sadece init + gerçek streaming döngü + LED aksiyonu. 9 Tem 2026.*
| Hücre | Platform | Flash (B) | SRAM (B) |
|-------|----------|-----------|----------|
| FastGRNN | MSP430 | 5544 | 348 |
| GRU  | MSP430 | 5392 | 308 |
| LSTM | MSP430 | 5742 | 324 |

### Production testinin metodolojisi (neye göre ölçüldü)

**Amaç:** Yukarıdaki "test harness" tablosu (satır 29-37), bir hakem tarafından haklı olarak
itiraz edilebilecek şişirilmiş sayılardı — çünkü UART banner metni, `sprint_u`/`sprint_f3`
debug yazdırma fonksiyonları ve latency ortalaması almak için 10× tekrarlanan zamanlama
döngüsü, gerçek dağıtılmış bir cihazda **hiç bulunmaz.** "Production" build bu farkı kapatır.

**Her hücre için ölçüm koşulları (üçü de birebir aynı):**
- **Donanım:** MSP-EXP430G2ET (MSP430G2553), 16 MHz kalibre DCO clock
- **Derleyici:** TI cl430 (ti-cgt-msp430_21.6.1.LTS), `-O3`, aynı proje ayarları (heap=0, stack=256B)
- **Firmware kapsamı — DAHİL OLANLAR (gerçek dağıtımda da gerekli):**
  - `clock_init()` + `timer_init()` (Timer_A, 1ms tik) — gerçek 50Hz örnekleme temposu için
  - Hücrenin çıkarım motoru (`gru.cpp`/`lstm.cpp`/`fastgrnn.cpp`, değişmemiş, B1'de bit-tam doğrulanan aynı kod)
  - Q15 ağırlıklar (`model_weights.h`) + sigmoid/tanh LUT tabloları (`lut.h`, 256-giriş, USE_LUT=1)
  - Sonsuz döngü: pencere boyunca (WINDOW_T adım) 20ms'de bir örnek besle (`{cell}_step`) → pencere bitince sınıflandır (`{cell}_predict`) → sınıfa göre LED (P1.0) aç/kapat (gerçek bir "aşağı akış aksiyonu" temsili — gerçekte bir aktüatör/radyo paketi olurdu)
- **Firmware kapsamı — HARİÇ TUTULANLAR (test-only, gerçek dağıtımda gereksiz):**
  - UART başlatma (`uart_init`) ve TÜM `sprint_*`/`sputc` yazdırma kodu — hiç UART çıktısı yok
  - Banner metinleri ("GRU HAR - MSP430G2553...") ve ilişkili string sabitleri
  - Latency ölçümü için 10× tekrar döngüsü (N_TIMING_WINDOWS) — sadece bir kez, gerçek zamanlı çalışır
  - Girdi: sabit sıfır (`{0,0,0}`) — çünkü Flash/RAM boyutu koddan gelir, veriden değil (bkz. latency bölümündeki aynı gerekçe); MPU6050/I2C sürücüsü bilerek yok (GRU/LSTM firmware'i sensöre bağlanmadı, sadece hesaplama+bellek izole edildi)
- **Derleme doğrulaması:** Her üç `main.cpp` de host'ta `cl430 --compile_only` ile önce sözdizimi hatasız derlendi (exit 0), sonra CCS'te gerçek MSP430'a flash edilip Debug Output'taki linker özeti (`Flash/FRAM usage is X bytes. RAM usage is Y bytes.`) okundu.
- **Proje izolasyonu:** Karışıklığı önlemek için her hücre **ayrı, yepyeni bir CCS projesinde** (eski FastGRNN/GRU/LSTM projelerinin yeniden kullanılması değil) derlendi — B1/B2'de birkaç kez yaşanan "main bulunamıyor" / stale-build hatalarından ders alınarak.

**Sonuç yorumu:** Test-harness'e göre düşüş (FastGRNN -%46, GRU -%18, LSTM -%17) beklenen yöndeydi
ve FastGRNN'in en büyük düşüşü, orijinal `ccs_fastgrnn_har` harness'inin (MPU6050+I2C+live-mode
kodu içeren, GRU/LSTM'in minimal harness'inden daha zengin) diğerlerinden daha fazla test-only
kod taşımasıyla açıklanıyor. Production sayıları (5392-5742B Flash) makaledeki **tek yetkili**
deployment footprint iddiası olacak; test-harness sayıları sadece şeffaflık için referans olarak duruyor.
